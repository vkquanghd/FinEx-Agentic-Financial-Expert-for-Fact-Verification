-- Consolidated PostgreSQL schema (2026-07-26 migration off MySQL+pgvector-as-
-- two-services onto a single Postgres/pgvector database - see
-- docs/PROGRESS.md dated entry "Gộp MySQL + pgvector thành một PostgreSQL
-- duy nhất" for the full rationale).
--
-- This file REPLACES, as the single source of truth:
--   infra/mysql/migrations/001_init_schema.sql   (companies, financial_line_items,
--                                                  financial_ratios, disclosures,
--                                                  ingestion_log)
--   infra/mysql/migrations/002_macro_indicators.sql (macro_indicators)
--   infra/mysql/migrations/003_market_snapshot.sql  (company_price_snapshot)
--   infra/pgvector/migrations/001_disclosure_chunks.sql (disclosure_chunks)
-- Those four files are kept on disk for historical reference (each now has a
-- deprecation notice at the top pointing here) but are NOT applied anymore -
-- docker-compose.yml's single "postgres" service mounts ONLY this file into
-- /docker-entrypoint-initdb.d.
--
-- Auto-applied by Postgres on first container startup, on an EMPTY data
-- volume only (same one-shot behavior the old MySQL init script had).
--
-- Duplicate-safety design unchanged from the MySQL version: every table
-- loaded repeatedly by the ETL has a UNIQUE constraint on its natural key.
-- The ETL must INSERT using "ON CONFLICT (...) DO UPDATE SET ..." (Postgres
-- upsert - replaces MySQL's "ON DUPLICATE KEY UPDATE") so re-running it, or
-- loading exports with overlapping periods, never creates duplicate rows -
-- it just overwrites the old value with the newer one.
--
-- Design choices made while porting off MySQL syntax (documented once here
-- instead of repeating per-table):
--   * AUTO_INCREMENT -> BIGSERIAL. Same "auto-incrementing bigint" behavior.
--   * ENUM(...) -> VARCHAR + CHECK constraint, NOT Postgres's native
--     CREATE TYPE ... AS ENUM. Reason: the old MySQL migrations' own comments
--     show this schema's enum-like columns (file_type, period_type) were
--     widened MORE THAN ONCE as new data sources were added (see the ALTER
--     COLUMN notes that used to live in 002/003_*.sql). Postgres native ENUM
--     types require "ALTER TYPE ... ADD VALUE" (can't run inside the same
--     transaction as other DDL in older Postgres, and can't remove values at
--     all) - a CHECK constraint can be dropped and re-added with a plain
--     ALTER TABLE, matching how this schema has actually evolved so far.
--   * TIMESTAMP ... ON UPDATE CURRENT_TIMESTAMP has no direct Postgres
--     equivalent - replaced with a shared trigger function
--     (set_updated_at(), defined once below) attached to every table that
--     had this MySQL clause.
--   * DECIMAL(p,s) needed NO change - Postgres accepts DECIMAL as a direct
--     alias for NUMERIC.
--   * UNIQUE KEY name (...) -> CONSTRAINT name UNIQUE (...), same constraint
--     names kept so existing references in docs/code comments still make
--     sense.
--   * Vector column/index (disclosure_chunks, was a separate "vectorstore"
--     service) now lives in the SAME database as the structured tables -
--     this is the actual point of the migration: one connection string, one
--     migration history, one backup, and SQL can JOIN disclosure_chunks
--     against companies/financial_line_items directly if ever needed.

CREATE EXTENSION IF NOT EXISTS vector;

-- Shared trigger function for every `updated_at` column below (replaces
-- MySQL's inline "ON UPDATE CURRENT_TIMESTAMP").
CREATE OR REPLACE FUNCTION set_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- =====================================================================
-- From infra/mysql/migrations/001_init_schema.sql
-- =====================================================================

CREATE TABLE IF NOT EXISTS companies (
    ticker          VARCHAR(10)  NOT NULL PRIMARY KEY,
    company_name    VARCHAR(255) NOT NULL,
    exchange        VARCHAR(10)  NOT NULL,   -- HOSE / HNX / UPCOM
    sector          VARCHAR(100) NOT NULL,   -- thesis sample sector label
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS financial_line_items (
    id              BIGSERIAL PRIMARY KEY,
    ticker          VARCHAR(10)  NOT NULL REFERENCES companies(ticker),
    fiscal_year     SMALLINT     NOT NULL,
    period          VARCHAR(10)  NOT NULL,   -- 'Q1','Q2','Q3','Q4','YEAR','H1','9M'
    ms_code         VARCHAR(10)  NOT NULL,   -- account code per Circular 200 (e.g. '100').
                                              -- NOTE: MS codes reset per statement (e.g. '01'
                                              -- exists in income_statement AND cash_flow_indirect
                                              -- with completely different meanings) - statement_type
                                              -- MUST stay part of the unique key below, never drop it.
    item_name       VARCHAR(255) NOT NULL,
    statement_type  VARCHAR(20)  NOT NULL
                    CHECK (statement_type IN ('balance_sheet','income_statement','cash_flow_indirect','cash_flow_direct')),
    consolidation   VARCHAR(20)  NOT NULL DEFAULT 'consolidated',
    value           DECIMAL(24,0),
    unit            VARCHAR(10)  NOT NULL DEFAULT 'VND',
    updated_at      TIMESTAMPTZ  NOT NULL DEFAULT now(),
    CONSTRAINT uq_line_item UNIQUE (ticker, fiscal_year, period, statement_type, ms_code, consolidation)
);
CREATE TRIGGER trg_financial_line_items_updated_at
    BEFORE UPDATE ON financial_line_items
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();

CREATE TABLE IF NOT EXISTS financial_ratios (
    id              BIGSERIAL PRIMARY KEY,
    ticker          VARCHAR(10)  NOT NULL REFERENCES companies(ticker),
    fiscal_year     SMALLINT     NOT NULL,
    period          VARCHAR(10)  NOT NULL,
    ratio_group     VARCHAR(100) NOT NULL,   -- 'Định giá','Sinh lợi','Tăng trưởng','Thanh khoản',
                                              -- 'Hiệu quả hoạt động','Đòn bẩy tài chính','Dòng tiền'
    ratio_name      VARCHAR(255) NOT NULL,   -- e.g. 'P/E', 'ROEA', 'Beta'
    value           DECIMAL(18,4),
    unit            VARCHAR(10),             -- 'Lần','%','VNĐ'
    updated_at      TIMESTAMPTZ  NOT NULL DEFAULT now(),
    CONSTRAINT uq_ratio UNIQUE (ticker, fiscal_year, period, ratio_name)
);
CREATE TRIGGER trg_financial_ratios_updated_at
    BEFORE UPDATE ON financial_ratios
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();

CREATE TABLE IF NOT EXISTS disclosures (
    id                BIGSERIAL PRIMARY KEY,
    ticker            VARCHAR(10)  NOT NULL REFERENCES companies(ticker),
    fiscal_year       SMALLINT     NOT NULL,
    period            VARCHAR(10)  NOT NULL,
    consolidation     VARCHAR(20)  NOT NULL DEFAULT 'consolidated',
    report_type       VARCHAR(20)  NOT NULL
                      CHECK (report_type IN ('nam','ban_nien','quy')),
    audit_status      VARCHAR(50),            -- 'Kiểm toán' / 'Soát xét' / 'Chưa kiểm toán'
    auditor           VARCHAR(255),
    audit_opinion     VARCHAR(255),
    disclosure_date   DATE,
    signed_date       DATE,
    source_pdf_path   VARCHAR(500),           -- path under data/raw/pdf_disclosures
    structure_valid   BOOLEAN,                -- output of report_structure_validator.py
    updated_at        TIMESTAMPTZ  NOT NULL DEFAULT now(),
    CONSTRAINT uq_disclosure UNIQUE (ticker, fiscal_year, period, consolidation)
);
CREATE TRIGGER trg_disclosures_updated_at
    BEFORE UPDATE ON disclosures
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();

-- pdf_documents - added 2026-07-29 evening (see
-- infra/postgres/migrations/006_pdf_documents.sql for the full rationale).
-- One row per PHYSICAL PDF file in the pdf_ingest_router.py corpus -
-- deliberately a SEPARATE table from `disclosures` above (which is
-- structured-filing metadata from vietstock_etl.py, a different grain: one
-- row per ticker x fiscal_year x period x consolidation, no file identity
-- at all). Conflating the two would mean a ticker with several real PDFs
-- for the same fiscal year (annual report, audited BCTC, a later
-- correction, a governance report) has no way to be told apart by id -
-- giving every PDF its own row via the UNIQUE source_file key removes that
-- ambiguity entirely. Populated by the NEW scripts/register_pdf_documents.py,
-- which runs AFTER pdf_ingest_router.py and BEFORE text_splitter.py - so
-- disclosure_chunks.pdf_document_id (see below) is assigned at ingest time,
-- not guessed later per-chunk.
CREATE TABLE IF NOT EXISTS pdf_documents (
    id                   BIGSERIAL PRIMARY KEY,
    source_file          VARCHAR(255) NOT NULL UNIQUE,  -- original PDF filename - the real, unambiguous identity key
    pdf_hash             CHAR(64)     NOT NULL,         -- SHA-256 of the PDF's bytes (scripts/pdf_ingest_router.py's _file_hash())
    ticker               VARCHAR(16),                   -- best-effort, see scripts/pdf_ingest_router.py's guess_ticker()
    doctype              VARCHAR(32),                   -- best-effort, see guess_doctype()
    fiscal_year          SMALLINT,                       -- best-effort, extracted from filename (src/retrieval/vector_store.py's extract_year())
    n_pages_total        INT,
    extraction_version   VARCHAR(100) NOT NULL,          -- scripts/pdf_ingest_router.py's INGEST_VERSION at extraction time
    extracted_at         TIMESTAMPTZ  NOT NULL DEFAULT now(),
    document_version     INT          NOT NULL DEFAULT 1,  -- bumped when the SAME source_file reappears with a DIFFERENT pdf_hash (a re-exported/corrected file under the same name) - a genuinely different filename is just a new row, no version bump needed
    -- VERSIONING POLICY (decided 2026-07-30, methodology review after v3):
    -- this is the "simple" policy, chosen deliberately over a fuller
    -- pdf_documents/pdf_document_versions history split. A changed
    -- source_file UPDATES this row in place and bumps document_version -
    -- the OLD pdf_hash/extraction_version/n_pages_total values are
    -- OVERWRITTEN, not preserved as a separate historical row. Historical
    -- PDF versions are NOT retained; only the active document state and a
    -- monotonically increasing version number are stored. This is
    -- sufficient for this thesis's scope (demo + reproducibility of the
    -- CURRENT corpus state) - it is NOT sufficient for a research question
    -- that specifically studies how corrected/restated disclosures change
    -- over time, which would need a real pdf_document_versions table
    -- (one immutable row per historical version) instead. Do not build
    -- that fuller table unless a concrete RQ actually needs it.
    is_active            BOOLEAN      NOT NULL DEFAULT TRUE,
    linked_disclosure_id BIGINT REFERENCES disclosures(id),  -- SEPARATE, still best-effort (ticker+fiscal_year+doctype heuristic) link to the structured-filing table above - NULL if no confident match, never a hard guarantee
    updated_at           TIMESTAMPTZ  NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_pdf_documents_ticker ON pdf_documents (ticker);
CREATE INDEX IF NOT EXISTS idx_pdf_documents_linked_disclosure ON pdf_documents (linked_disclosure_id);
CREATE TRIGGER trg_pdf_documents_updated_at
    BEFORE UPDATE ON pdf_documents
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();

-- Audit trail of what's been loaded - not required for correctness (the
-- UNIQUE constraints above already make loading idempotent), but useful for
-- the course's "research conduct / reproducibility" assessment: shows
-- exactly which source file produced which rows, and when.
--
-- file_type CHECK list already includes 'market_snapshot' (originally added
-- to MySQL via a hand-run ALTER documented in 003_market_snapshot.sql's
-- comments) so this consolidated schema starts from the FINAL widened list,
-- not the original 5-value one - no separate widening migration needed here.
CREATE TABLE IF NOT EXISTS ingestion_log (
    id              BIGSERIAL PRIMARY KEY,
    file_name       VARCHAR(255) NOT NULL,
    file_type       VARCHAR(30)  NOT NULL
                    CHECK (file_type IN ('bao_cao_tai_chinh','chi_so_tai_chinh','du_lieu_doanh_nghiep','pdf_disclosure','du_lieu_vi_mo','market_snapshot')),
    ticker          VARCHAR(10),
    row_count       INT,
    ingested_at     TIMESTAMPTZ  NOT NULL DEFAULT now(),
    notes           VARCHAR(500)
);

-- =====================================================================
-- From infra/mysql/migrations/002_macro_indicators.sql
-- =====================================================================

-- period_type CHECK list already includes 'DAILY' and 'YTD' (originally
-- added via hand-run ALTERs documented in the old 002_macro_indicators.sql
-- comments) - same "start from the final widened list" approach as
-- ingestion_log.file_type above. unit is VARCHAR(50) for the same reason
-- (the old file's comment: 'Số bé trai/100 bé gái' = 22 chars overflowed the
-- original VARCHAR(20)).
CREATE TABLE IF NOT EXISTS macro_indicators (
    id              BIGSERIAL PRIMARY KEY,
    category        VARCHAR(50)  NOT NULL,   -- 'CPI','PPI','GDP_REAL_GROWTH','IIP_BY_INDUSTRY', etc.
    indicator_name  VARCHAR(255) NOT NULL,   -- row label as reported, e.g. 'Sản xuất kim loại'
    unit            VARCHAR(50),
    period_type     VARCHAR(10)  NOT NULL
                    CHECK (period_type IN ('QUARTER','MONTH','YEAR','DAILY','YTD')),
    fiscal_year     SMALLINT     NOT NULL,
    period          VARCHAR(10)  NOT NULL,   -- 'Q1'..'Q4' / 'M1'..'M12' / 'YEAR' / 'DmmDD' / 'YTD3'..'YTD12'
    value           DECIMAL(24,4),
    source_file     VARCHAR(255),
    release_date    DATE,                    -- date the figure was actually published (vs fiscal_year/period, the period it DESCRIBES) - added 2026-07-29 for leakage control in any future forecasting model; NULL means "unknown, don't trust for leakage-sensitive joins", not "safe to use"
    updated_at      TIMESTAMPTZ  NOT NULL DEFAULT now(),
    CONSTRAINT uq_macro UNIQUE (category, indicator_name, period_type, fiscal_year, period)
    -- Deliberately NOT (category, indicator_name, period_type, fiscal_year, period, source_file) -
    -- src/causal/macro_lookup.py reads a single scalar value per period, so widening this key would
    -- let two source files legitimately coexist as two rows for the same period and break that
    -- assumption - see infra/postgres/migrations/003_disclosure_lineage_and_versions.sql's comment.
);
CREATE TRIGGER trg_macro_indicators_updated_at
    BEFORE UPDATE ON macro_indicators
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();

-- =====================================================================
-- From infra/mysql/migrations/003_market_snapshot.sql
-- =====================================================================

CREATE TABLE IF NOT EXISTS company_price_snapshot (
    id              BIGSERIAL PRIMARY KEY,
    ticker          VARCHAR(10)  NOT NULL REFERENCES companies(ticker),
    trade_date      DATE         NOT NULL,
    open_price      DECIMAL(18,2),
    high_price      DECIMAL(18,2),
    low_price       DECIMAL(18,2),
    close_price     DECIMAL(18,2),
    volume          BIGINT,
    source          VARCHAR(20)  NOT NULL DEFAULT 'vnstock_vci',  -- data provider, in case a ticker's source ever changes (see vnstock VCI/KBS choice in market_snapshot.py)
    updated_at      TIMESTAMPTZ  NOT NULL DEFAULT now(),
    CONSTRAINT uq_price UNIQUE (ticker, trade_date)
);
CREATE TRIGGER trg_company_price_snapshot_updated_at
    BEFORE UPDATE ON company_price_snapshot
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();

CREATE INDEX IF NOT EXISTS idx_price_trade_date ON company_price_snapshot (trade_date);

-- =====================================================================
-- From infra/pgvector/migrations/001_disclosure_chunks.sql
-- (unchanged from the 2026-07-26 HNSW migration - see that file's original
-- comments, preserved there, for the ivfflat->hnsw rationale. Only change
-- here is that this table now lives in the SAME database/connection as
-- everything above, instead of a separate "vectorstore" service.)
-- =====================================================================

CREATE TABLE IF NOT EXISTS disclosure_chunks (
    id              BIGSERIAL PRIMARY KEY,
    ticker          VARCHAR(16)  NOT NULL,   -- best-effort from filename, see scripts/pdf_ingest_router.py guess_ticker() - cross-check against companies.ticker before trusting it for joins
    doctype         VARCHAR(32)  NOT NULL,   -- annual_report / financial_statement / kqkd_explanation / agm_resolution / governance_report / board_resolution / other
    source_file     VARCHAR(255) NOT NULL,   -- original PDF filename (path.stem from the router)
    page            INT          NOT NULL,
    chunk_index     INT          NOT NULL,   -- 0-based index of this chunk within its page
    chunk_text      TEXT         NOT NULL,
    char_count      INT          NOT NULL,
    embedding       VECTOR(1024),            -- NULL until src/retrieval/embedder.py fills it in - lets chunking and embedding run as separate, independently re-runnable steps
    chunk_hash      CHAR(64),                -- SHA-256 of chunk_text (see src/retrieval/text_splitter.py's chunk_hash()) - added 2026-07-29 so a content change under the SAME key (e.g. a corrected/restated disclosure) is detected and re-embedded instead of silently kept stale, see src/retrieval/vector_store.py's _needs_embedding()
    embedding_version VARCHAR(255),          -- e.g. "BAAI/bge-m3_normalized_v1" (src/retrieval/vector_store.py's EMBEDDING_VERSION) - added 2026-07-29 alongside chunk_hash so an embedding-model/config change is also detected, not just a text change
    pdf_document_id BIGINT NOT NULL REFERENCES pdf_documents(id),  -- renamed from disclosure_id 2026-07-29 evening; locked NOT NULL 2026-07-30 (migration 010) - exact (not fuzzy) link to the pdf_documents row this chunk's PDF page came from (see src/retrieval/vector_store.py's resolve_pdf_document_id()/UnregisteredDocumentError). A chunk with no registered pdf_documents row is now a hard failure at write time (see vector_store.py), never silently stored as NULL - fresh installs go straight to the locked-down schema, only pre-migration-010 databases ever had NULLs here.
    chunking_version VARCHAR(100),           -- src/retrieval/text_splitter.py's CHUNKING_VERSION at the time this chunk was produced - added 2026-07-29
    normalization_version VARCHAR(100),      -- src/retrieval/text_splitter.py's NORMALIZATION_VERSION at the time this chunk was produced - added 2026-07-29
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT now(),
    -- Key locked 2026-07-30 (migration 010): (pdf_document_id, page, chunk_index)
    -- REPLACES the original (ticker, source_file, page, chunk_index) -
    -- ticker/source_file are already reachable via pdf_document_id's FK to
    -- pdf_documents, so they're redundant as key material once every chunk
    -- is guaranteed to have a real pdf_document_id (enforced by the NOT
    -- NULL above + UnregisteredDocumentError at write time).
    CONSTRAINT uq_disclosure_chunk_v2 UNIQUE (pdf_document_id, page, chunk_index)
);

CREATE INDEX IF NOT EXISTS idx_disclosure_chunks_ticker  ON disclosure_chunks (ticker);
CREATE INDEX IF NOT EXISTS idx_disclosure_chunks_doctype ON disclosure_chunks (doctype);
CREATE INDEX IF NOT EXISTS idx_disclosure_chunks_pdf_document_id ON disclosure_chunks (pdf_document_id);

-- Approximate nearest-neighbor index for cosine similarity search. HNSW, not
-- IVFFlat - see docs/PROGRESS.md ("Đổi chỉ mục pgvector: IVFFlat -> HNSW")
-- for the full two-reason rationale, and vector_store.py's docstring for why
-- the ticker-scoped exact-scan workaround is STILL required regardless of
-- which ANN index type sits underneath.
CREATE INDEX IF NOT EXISTS idx_disclosure_chunks_embedding
    ON disclosure_chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- causal_panel_snapshot - added 2026-07-29 (see
-- infra/postgres/migrations/004_causal_panel_snapshot.sql for the full
-- rationale: a FROZEN, citable copy of the company panel used for a
-- specific RQ6a estimate_ate()/estimate_ate_dml() run, since
-- src/causal/panel_builder.py's build_panel() is a live view that changes
-- as more data loads - kept in this fresh-install file too, matching the
-- convention for every table added by an incremental migration so far.
CREATE TABLE IF NOT EXISTS causal_panel_snapshot (
    id                      BIGSERIAL PRIMARY KEY,
    snapshot_version        VARCHAR(100) NOT NULL,
    ticker                  VARCHAR(16)  NOT NULL,
    fiscal_year             SMALLINT     NOT NULL,
    period                  VARCHAR(16)  NOT NULL,
    consolidation           VARCHAR(32),
    sector                  VARCHAR(64),
    exchange                VARCHAR(16),
    total_assets            NUMERIC,
    total_liabilities       NUMERIC,
    total_equity            NUMERIC,
    current_assets          NUMERIC,
    current_liabilities     NUMERIC,
    inventory               NUMERIC,
    revenue                 NUMERIC,
    cogs                    NUMERIC,
    gross_profit            NUMERIC,
    profit_before_tax       NUMERIC,
    net_profit              NUMERIC,
    operating_cash_flow     NUMERIC,
    leverage                NUMERIC,
    current_ratio           NUMERIC,
    gross_margin            NUMERIC,
    net_margin              NUMERIC,
    asset_turnover           NUMERIC,
    ocf_to_revenue          NUMERIC,
    cpi_annual_avg          NUMERIC,
    gdp_growth_annual_avg   NUMERIC,
    gold_price_annual_avg   NUMERIC,
    usdvnd_annual_avg       NUMERIC,
    oil_price_annual_avg    NUMERIC,
    eligible_sample         BOOLEAN      NOT NULL,
    exclusion_reason        TEXT,
    created_at              TIMESTAMPTZ  NOT NULL DEFAULT now(),
    CONSTRAINT uq_causal_panel_snapshot
        UNIQUE (snapshot_version, ticker, fiscal_year, period, consolidation)
);

CREATE INDEX IF NOT EXISTS idx_causal_panel_snapshot_version
    ON causal_panel_snapshot (snapshot_version);
CREATE INDEX IF NOT EXISTS idx_causal_panel_snapshot_ticker
    ON causal_panel_snapshot (ticker);

-- predictive_features - added 2026-07-29 (see
-- infra/postgres/migrations/005_predictive_features.sql for the full
-- rationale: wide-format, leakage-controlled feature table for the future
-- PREDICTIVE branch - data layer only, no model training code yet). Kept
-- in this fresh-install file too, matching the convention for every table
-- added by an incremental migration so far.
CREATE TABLE IF NOT EXISTS predictive_features (
    id                          BIGSERIAL PRIMARY KEY,
    feature_version             VARCHAR(100) NOT NULL,
    ticker                      VARCHAR(16)  NOT NULL,
    fiscal_year                 SMALLINT     NOT NULL,
    period                      VARCHAR(16)  NOT NULL,
    consolidation               VARCHAR(32),
    sector                      VARCHAR(64),
    exchange                    VARCHAR(16),
    total_assets                NUMERIC,
    total_liabilities           NUMERIC,
    total_equity                NUMERIC,
    current_assets              NUMERIC,
    current_liabilities        NUMERIC,
    inventory                   NUMERIC,
    revenue                     NUMERIC,
    cogs                        NUMERIC,
    gross_profit                NUMERIC,
    profit_before_tax           NUMERIC,
    net_profit                  NUMERIC,
    operating_cash_flow         NUMERIC,
    leverage                    NUMERIC,
    current_ratio               NUMERIC,
    gross_margin                NUMERIC,
    net_margin                  NUMERIC,
    asset_turnover              NUMERIC,
    ocf_to_revenue              NUMERIC,
    total_assets_lag1           NUMERIC,
    total_liabilities_lag1      NUMERIC,
    total_equity_lag1           NUMERIC,
    current_assets_lag1         NUMERIC,
    current_liabilities_lag1    NUMERIC,
    inventory_lag1               NUMERIC,
    revenue_lag1                 NUMERIC,
    cogs_lag1                    NUMERIC,
    gross_profit_lag1            NUMERIC,
    profit_before_tax_lag1       NUMERIC,
    net_profit_lag1              NUMERIC,
    operating_cash_flow_lag1     NUMERIC,
    leverage_lag1                NUMERIC,
    current_ratio_lag1           NUMERIC,
    gross_margin_lag1            NUMERIC,
    net_margin_lag1              NUMERIC,
    asset_turnover_lag1          NUMERIC,
    ocf_to_revenue_lag1          NUMERIC,
    revenue_growth_yoy           NUMERIC,
    net_profit_growth_yoy        NUMERIC,
    total_assets_growth_yoy      NUMERIC,
    cpi_annual_avg               NUMERIC,
    gdp_growth_annual_avg        NUMERIC,
    gold_price_annual_avg        NUMERIC,
    usdvnd_annual_avg            NUMERIC,
    oil_price_annual_avg         NUMERIC,
    information_cutoff          DATE,
    target_col                  VARCHAR(100) NOT NULL,
    horizon                     SMALLINT     NOT NULL,
    target_value                NUMERIC,
    target_observed_at          DATE,          -- added 2026-07-29 evening - information_cutoff of the FUTURE row that supplied target_value
    target_release_date         DATE,          -- reserved for a future macro-sourced target_col - NULL for every row the current builder produces, see migration 008's comment
    eligible_for_training        BOOLEAN      NOT NULL,
    feature_created_at          TIMESTAMPTZ,   -- added 2026-07-29 evening - when this row was actually computed (Python-side, not the DB write time)
    feature_hash                CHAR(64),      -- SHA-256 over this row's feature values - detects silent drift vs a fresh build without needing to overwrite (see src/predictive/feature_builder.py's _feature_hash())
    source_snapshot_version     VARCHAR(100),  -- free-text label, not a hard FK - which causal_panel_snapshot (if any) was built from the same underlying panel state
    created_at                  TIMESTAMPTZ  NOT NULL DEFAULT now(),
    updated_at                  TIMESTAMPTZ  NOT NULL DEFAULT now(),
    CONSTRAINT uq_predictive_features
        UNIQUE (feature_version, ticker, fiscal_year, period, consolidation, target_col, horizon)
);

CREATE INDEX IF NOT EXISTS idx_predictive_features_version
    ON predictive_features (feature_version);
CREATE INDEX IF NOT EXISTS idx_predictive_features_ticker
    ON predictive_features (ticker);
CREATE INDEX IF NOT EXISTS idx_predictive_features_cutoff
    ON predictive_features (information_cutoff);

-- causal_experiment_runs - added 2026-07-29 evening (see
-- infra/postgres/migrations/009_causal_experiment_runs.sql for the full
-- rationale: an append-only log of every estimate_ate()/estimate_ate_dml()
-- invocation via scripts/run_causal_experiment.py, recording both inputs
-- (treatment/outcome/covariates/estimator/nuisance model/seed/bootstrap
-- iterations/code commit) and outputs (n, point estimate, CI, confidence
-- label), including failed/refused runs - kept in this fresh-install file
-- too, matching the convention for every table added by an incremental
-- migration so far.
CREATE TABLE IF NOT EXISTS causal_experiment_runs (
    id                       BIGSERIAL PRIMARY KEY,
    snapshot_version         VARCHAR(100),
    source_dataset_version   VARCHAR(100) NOT NULL,
    treatment_name           VARCHAR(100) NOT NULL,
    outcome_name             VARCHAR(100) NOT NULL,
    covariates               TEXT         NOT NULL,
    estimator                VARCHAR(50)  NOT NULL,
    nuisance_model           VARCHAR(100),
    random_seed              INT,
    bootstrap_iterations     INT,
    min_sample_threshold     INT,
    code_commit              VARCHAR(64),
    n                        INT,
    n_treated                INT,
    n_control                INT,
    point_estimate           DOUBLE PRECISION,
    ci_low                   DOUBLE PRECISION,
    ci_high                  DOUBLE PRECISION,
    confidence_label         TEXT,
    ok                       BOOLEAN      NOT NULL,
    error_message            TEXT,
    created_at                TIMESTAMPTZ  NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_causal_experiment_runs_snapshot
    ON causal_experiment_runs (snapshot_version);
CREATE INDEX IF NOT EXISTS idx_causal_experiment_runs_treatment_outcome
    ON causal_experiment_runs (treatment_name, outcome_name);

-- predictive_split_manifest - added 2026-07-30 (see
-- infra/postgres/migrations/011_predictive_split_manifest.sql for the full
-- rationale: freezes the time-based train/validation/test assignment for
-- predictive_features rows in its OWN table, separate from
-- predictive_features itself, so a routine feature rebuild never silently
-- reshuffles which rows are in test). Kept in this fresh-install file too,
-- matching the convention for every table added by an incremental
-- migration so far.
CREATE TABLE IF NOT EXISTS predictive_split_manifest (
    id                          BIGSERIAL PRIMARY KEY,
    split_manifest_version      VARCHAR(100) NOT NULL,
    feature_version             VARCHAR(100) NOT NULL,
    ticker                      VARCHAR(16)  NOT NULL,
    fiscal_year                 SMALLINT     NOT NULL,
    period                      VARCHAR(16)  NOT NULL,
    consolidation                VARCHAR(32),
    target_col                  VARCHAR(100) NOT NULL,
    horizon                     SMALLINT     NOT NULL,
    information_cutoff          DATE         NOT NULL,
    split                       VARCHAR(16)  NOT NULL CHECK (split IN ('train', 'validation', 'test')),
    frozen_at                   TIMESTAMPTZ  NOT NULL DEFAULT now(),
    CONSTRAINT uq_predictive_split_manifest
        UNIQUE (split_manifest_version, ticker, fiscal_year, period, consolidation, target_col, horizon)
);

CREATE INDEX IF NOT EXISTS idx_predictive_split_manifest_version
    ON predictive_split_manifest (split_manifest_version);
CREATE INDEX IF NOT EXISTS idx_predictive_split_manifest_split
    ON predictive_split_manifest (split);


-- =====================================================================
-- model_registry_lite + ml_prediction_outputs - added 2026-08-03 (see
-- infra/postgres/migrations/012_prediction_store.sql for the full
-- rationale: official prediction store replacing the notebook's pandas
-- to_sql auto-created table; real FKs to companies/model registry;
-- append-only prediction history + one-time scoring columns). Kept in
-- this fresh-install file too, matching the convention for every table
-- added by an incremental migration so far.
-- =====================================================================

CREATE TABLE IF NOT EXISTS model_registry_lite (
    id                        BIGSERIAL PRIMARY KEY,
    task                      VARCHAR(100) NOT NULL,   -- e.g. 'net_margin_next_q', 'margin_negative_next_q'
    model_name                VARCHAR(100) NOT NULL,   -- e.g. 'persistence', 'random_forest'
    model_version             VARCHAR(100) NOT NULL,   -- freeze label, e.g. 'v1_frozen_20260803'
    feature_list              JSONB,                   -- exact feature names used (frozen at selection time)
    trained_through_quarter   INT,                     -- last quarter_index whose LABELS were used in training
    validation_metrics        JSONB,                   -- walk-forward validation metrics at freeze time (MAE/PR-AUC/...)
    calibration_method        VARCHAR(50),             -- NULL until calibration done ('platt'/'isotonic'), classifiers only
    is_champion               BOOLEAN      NOT NULL DEFAULT FALSE,
    frozen_at                 TIMESTAMPTZ,             -- when selection was frozen (promotion-gate exit)
    notes                     TEXT,
    created_at                TIMESTAMPTZ  NOT NULL DEFAULT now(),
    CONSTRAINT uq_model_registry_lite UNIQUE (task, model_name, model_version)
);

-- At most ONE champion per task, enforced at the DB level (partial unique
-- index) - promoting a new champion means UPDATE old row is_champion=FALSE
-- then new row TRUE, in one transaction. The registry rows themselves are
-- never deleted (history of past champions stays queryable).
CREATE UNIQUE INDEX IF NOT EXISTS uq_model_registry_lite_champion
    ON model_registry_lite (task) WHERE is_champion;

CREATE TABLE IF NOT EXISTS ml_prediction_outputs (
    id                   BIGSERIAL PRIMARY KEY,
    model_id             BIGINT       NOT NULL REFERENCES model_registry_lite(id),
    ticker               VARCHAR(10)  NOT NULL REFERENCES companies(ticker),
    task                 VARCHAR(100) NOT NULL,
    as_of_quarter        INT          NOT NULL,   -- quarter_index of the features used (information set)
    target_quarter       INT          NOT NULL,   -- quarter_index being predicted
    prediction           NUMERIC,                 -- point prediction (regression) / class label as 0-1 (classification)
    probability          NUMERIC CHECK (probability IS NULL OR (probability >= 0 AND probability <= 1)),
    interval_low         NUMERIC,                 -- reserved for conformal/quantile interval (Layer-3); NULL now
    interval_high        NUMERIC,
    information_cutoff   DATE,                    -- latest disclosure_date feeding the features - leakage audit field
    claim_scope          VARCHAR(40)  NOT NULL DEFAULT 'PREDICTIVE_NOT_CAUSAL'
                         CHECK (claim_scope IN ('PREDICTIVE_NOT_CAUSAL','EXPLORATORY_NOT_SERVED')),
    scored_error         NUMERIC,                 -- |y_true - prediction|, filled ONCE by thesis_target_scoring DAG
    scored_at            TIMESTAMPTZ,             -- when scoring happened (NULL = target not yet observed)
    created_at           TIMESTAMPTZ  NOT NULL DEFAULT now(),
    CONSTRAINT uq_ml_prediction_outputs
        UNIQUE (model_id, ticker, task, as_of_quarter, target_quarter)
);

CREATE INDEX IF NOT EXISTS idx_ml_prediction_outputs_ticker_task
    ON ml_prediction_outputs (ticker, task);
CREATE INDEX IF NOT EXISTS idx_ml_prediction_outputs_target_quarter
    ON ml_prediction_outputs (target_quarter);
CREATE INDEX IF NOT EXISTS idx_ml_prediction_outputs_unscored
    ON ml_prediction_outputs (target_quarter) WHERE scored_at IS NULL;


-- =====================================================================
-- feedback schema (HITL) - added 2026-08-03 (see
-- infra/postgres/migrations/013_feedback_hitl.sql for the full rationale
-- incl. the research basis: Wu et al. 2022 HITL survey; Settles 2009
-- active learning / uncertainty sampling; Christiano et al. 2017 batch
-- human-preference discipline). Separate SCHEMA so the whole feedback
-- loop can later move to its own database with one pg_dump. Kept in this
-- fresh-install file too, matching the convention.
-- =====================================================================

-- CREATE SCHEMA feedback gives that boundary TODAY at zero operational cost:
-- separate GRANTs possible, separate pg_dump possible, and the eventual move
-- to its own instance is one dump/restore with no renames. This is the
-- "chua cho cho backend" (reserve room for the backend) decision from DSA v5.
--
-- RESEARCH BASIS for the loop design (full citations to add to the thesis
-- reference list, Australian Harvard style like Master FINAL Part III):
--
--   * Wu, X., Xiao, L., Sun, Y., Zhang, J., Ma, T. & He, L. 2022, 'A survey
--     of human-in-the-loop for machine learning', Future Generation Computer
--     Systems, vol. 135, pp. 364-381. Their taxonomy splits HITL into
--     (1) data-processing improvement, (2) interventional training,
--     (3) system-independent HITL design. THIS schema implements (1)+(3):
--     ratings are collected primarily as an EVALUATION signal and a
--     review-prioritization signal - they only become TRAINING data after
--     explicit human review, never automatically.
--
--   * Settles, B. 2009, 'Active learning literature survey', Computer
--     Sciences Technical Report 1648, University of Wisconsin-Madison.
--     Uncertainty sampling: humans should spend review effort on the items
--     the system is most uncertain/wrong about. review_queue_lite encodes
--     exactly that: low rating, router fallback events, and low retrieval
--     similarity are the uncertainty proxies that auto-enqueue items.
--
--   * Christiano, P.F., Leike, J., Brown, T., Martic, M., Legg, S. &
--     Amodei, D. 2017, 'Deep reinforcement learning from human preferences',
--     in Advances in Neural Information Processing Systems 30. Human
--     preference signals are aggregated and used in BATCHES to fit a
--     quality/reward model - not applied per-label to the policy. Our
--     retrain_gate_log implements the same batch discipline: individual
--     ratings NEVER trigger training; a weekly aggregate either passes the
--     gate or keeps accumulating (also matches Master FINAL muc 19: "Khong
--     fine-tune/deploy sau tung nhan moi" - oscillation / catastrophic
--     forgetting / no-validation risks).
--
-- APPEND-ONLY DISCIPLINE: answer_runs, answer_feedback and retrain_gate_log
-- have NO update trigger on purpose - matching causal_experiment_runs
-- (migration 009). A rating is an observation at a point in time; corrections
-- happen by inserting a new row, never editing history. review_queue_lite is
-- the ONE mutable table (status transitions pending->reviewed->resolved),
-- because it is a WORK QUEUE, not a log - same distinction Dataflow FINAL
-- draws between review_items (mutable workflow) and review_decisions
-- (append-only judgments).
--
-- Run by hand against the running container:
--   docker exec -i thesis_postgres psql -U thesis_user -d financial_db \
--     < infra/postgres/migrations/013_feedback_hitl.sql

CREATE SCHEMA IF NOT EXISTS feedback;

-- One row per question the demo answered - the audit log of every chat turn.
-- This is Dataflow FINAL's answer_runs (trang 31) in mini form: question/
-- answer hashes for dedup+integrity, evidence JSONB standing in for the
-- future answer_evidence_links table (its keys: chunk_ids[], prediction_id,
-- experiment_run_id, sql_text - so migrating to real link rows later is a
-- mechanical JSONB unnest, no information loss).
CREATE TABLE IF NOT EXISTS feedback.answer_runs (
    id                  BIGSERIAL PRIMARY KEY,
    session_id          VARCHAR(64),              -- Streamlit session; groups a conversation
    asked_at            TIMESTAMPTZ  NOT NULL DEFAULT now(),
    question_text       TEXT         NOT NULL,
    question_hash       CHAR(64)     NOT NULL,    -- SHA-256 of normalized question (project-wide hash convention)
    resolved_question   TEXT,                     -- after RQ7 context resolver, NULL if unchanged
    branch              VARCHAR(20)  NOT NULL
                        CHECK (branch IN ('dag','rag','whatif','causal','compare','absa',
                                          'predictive','unsupported')),
    fallback_from       VARCHAR(20),              -- e.g. 'dag' when DAG->RAG fallback fired (router S3 signal)
    latency_ms          INT,
    answer_text         TEXT,
    answer_hash         CHAR(64),
    evidence            JSONB,                    -- {chunk_ids:[], prediction_id:, experiment_run_id:, sql_text:}
    router_version      VARCHAR(100),             -- rules.py version label at answer time
    created_at          TIMESTAMPTZ  NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_answer_runs_asked_at  ON feedback.answer_runs (asked_at);
CREATE INDEX IF NOT EXISTS idx_answer_runs_branch    ON feedback.answer_runs (branch);
CREATE INDEX IF NOT EXISTS idx_answer_runs_session   ON feedback.answer_runs (session_id);

-- One row per rating action - INCLUDING skips. A skip ("bo qua") is data:
-- skip rate per branch is itself an engagement metric, and treating skip as
-- NULL-rating-with-flag (instead of just not inserting) means "user saw the
-- widget and declined" is distinguishable from "widget never shown" -
-- without this, the denominator for response-rate is unknowable.
CREATE TABLE IF NOT EXISTS feedback.answer_feedback (
    id             BIGSERIAL PRIMARY KEY,
    answer_run_id  BIGINT      NOT NULL REFERENCES feedback.answer_runs(id),
    rating         SMALLINT    CHECK (rating BETWEEN 1 AND 5),
    is_skip        BOOLEAN     NOT NULL DEFAULT FALSE,
    reason_code    VARCHAR(40)
                   CHECK (reason_code IS NULL OR reason_code IN
                          ('good','wrong_number','wrong_branch','missing_evidence',
                           'hallucination','outdated','unclear','too_slow','other')),
    free_text      TEXT,
    rated_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    -- a row must be either a real rating or an explicit skip:
    CONSTRAINT chk_rating_or_skip CHECK (is_skip OR rating IS NOT NULL)
);

CREATE INDEX IF NOT EXISTS idx_answer_feedback_run ON feedback.answer_feedback (answer_run_id);

-- Uncertainty-sampling work queue (Settles 2009): what should the human look
-- at next. Rows are auto-enqueued by triggers-in-code (NOT db triggers - the
-- Streamlit app / Airflow DAG inserts them) when: rating <= 2, router
-- fallback fired, or retrieval similarity below threshold. Manual adds
-- allowed (source='manual'). THE ONE MUTABLE TABLE in this schema - it is a
-- queue, not a log. Maps 1:1 onto review_items when migration 019 lands.
CREATE TABLE IF NOT EXISTS feedback.review_queue_lite (
    id               BIGSERIAL PRIMARY KEY,
    answer_run_id    BIGINT REFERENCES feedback.answer_runs(id),   -- nullable: manual items may have no run
    source           VARCHAR(20) NOT NULL
                     CHECK (source IN ('low_rating','fallback','low_similarity','manual')),
    priority         SMALLINT    NOT NULL DEFAULT 3,               -- 1 = highest
    status           VARCHAR(20) NOT NULL DEFAULT 'pending'
                     CHECK (status IN ('pending','reviewed','resolved','dismissed')),
    resolution_note  TEXT,                                          -- what the human decided/did
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    reviewed_at      TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_review_queue_pending
    ON feedback.review_queue_lite (priority, created_at) WHERE status = 'pending';

-- Human-reviewed expected answers that can become a fixed regression test
-- set. They are deliberately separate from answer_feedback: a rating says
-- something was wrong, whereas a regression case supplies the expected
-- observable behaviour needed to prevent recurrence. Never auto-train from
-- this table; curation/acceptance remains a later explicit action.
CREATE TABLE IF NOT EXISTS feedback.regression_candidates (
    id                    BIGSERIAL PRIMARY KEY,
    review_id             BIGINT NOT NULL UNIQUE REFERENCES feedback.review_queue_lite(id),
    answer_run_id         BIGINT NOT NULL REFERENCES feedback.answer_runs(id),
    question_text         TEXT NOT NULL,
    expected_answer       TEXT NOT NULL,
    expected_sources      TEXT,
    reviewer_note         TEXT,
    status                VARCHAR(20) NOT NULL DEFAULT 'candidate'
                          CHECK (status IN ('candidate', 'accepted', 'rejected')),
    created_at            TIMESTAMPTZ NOT NULL DEFAULT now(),
    decided_at            TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_regression_candidates_status
    ON feedback.regression_candidates (status, created_at DESC);

-- Append-only log of every retrain-gate evaluation (weekly, by the
-- thesis_feedback_aggregate Airflow DAG). The gate rule lives HERE as data,
-- not hard-coded in the DAG, so the thesis can show the full decision trail:
-- "week W: branch B had n=42 mean=3.1 -> gate not triggered (n < 50)".
-- Batch discipline per Christiano et al. 2017 / Master FINAL muc 19:
-- gate_triggered=TRUE creates a review_queue_lite item (source='manual',
-- priority=1) for a HUMAN to decide - it never starts training by itself.
CREATE TABLE IF NOT EXISTS feedback.retrain_gate_log (
    id              BIGSERIAL PRIMARY KEY,
    window_start    DATE        NOT NULL,
    window_end      DATE        NOT NULL,
    branch          VARCHAR(20) NOT NULL,
    n_feedback      INT         NOT NULL,
    n_skip          INT         NOT NULL DEFAULT 0,
    mean_rating     NUMERIC,
    skip_rate       NUMERIC,
    gate_rule       VARCHAR(200) NOT NULL,   -- rule text evaluated, e.g. 'n>=50 AND mean<3.5'
    gate_triggered  BOOLEAN     NOT NULL,
    note            TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_retrain_gate_log_branch
    ON feedback.retrain_gate_log (branch, window_end);

-- =====================================================================
-- web_chunks - added 2026-09-07 (see
-- infra/postgres/migrations/014_web_chunks.sql for the full rationale:
-- a SEPARATE, source_url-keyed table for scraped press/web content, kept
-- structurally apart from disclosure_chunks so it can never be
-- accidentally mixed into the RQ1-RQ7 official-disclosure benchmark -
-- src/retrieval/rag_pipeline.py only reads it as an explicit fallback when
-- disclosure_chunks has nothing). Kept in this fresh-install file too,
-- matching the convention for every table added by an incremental
-- migration so far.
-- =====================================================================

CREATE TABLE IF NOT EXISTS web_chunks (
    id                BIGSERIAL PRIMARY KEY,
    ticker            VARCHAR(16),
    source_type       VARCHAR(32)  NOT NULL DEFAULT 'news_web',
    domain            VARCHAR(255) NOT NULL,
    source_url        TEXT         NOT NULL,
    title             TEXT,
    published_date    DATE,
    scraped_at        TIMESTAMPTZ  NOT NULL DEFAULT now(),
    chunk_index       INT          NOT NULL,
    chunk_text        TEXT         NOT NULL,
    char_count        INT          NOT NULL,
    embedding         VECTOR(1024),
    chunk_hash        CHAR(64),
    embedding_version VARCHAR(255),
    created_at        TIMESTAMPTZ  NOT NULL DEFAULT now(),
    CONSTRAINT uq_web_chunk UNIQUE (source_url, chunk_index)
);

CREATE INDEX IF NOT EXISTS idx_web_chunks_ticker  ON web_chunks (ticker);
CREATE INDEX IF NOT EXISTS idx_web_chunks_domain  ON web_chunks (domain);
CREATE INDEX IF NOT EXISTS idx_web_chunks_published_date ON web_chunks (published_date);

CREATE INDEX IF NOT EXISTS idx_web_chunks_embedding
    ON web_chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS idx_web_chunks_fts
    ON web_chunks USING gin (to_tsvector('simple', chunk_text));
