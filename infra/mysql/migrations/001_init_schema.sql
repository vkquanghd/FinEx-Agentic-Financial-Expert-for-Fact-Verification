-- DEPRECATED 2026-07-26 - superseded by infra/postgres/migrations/001_init_schema.sql
-- (single consolidated Postgres database, replacing the separate
-- MySQL + pgvector services - see docs/PROGRESS.md "Gộp MySQL + pgvector
-- thành một PostgreSQL duy nhất"). Kept on disk for historical reference
-- only - docker-compose.yml no longer mounts this file, it is NOT applied
-- to new deployments.
--
-- Initial schema for the thesis's structured (DAG) data layer.
-- Auto-applied by MySQL on first container startup (mounted into
-- /docker-entrypoint-initdb.d). Only runs once, on an EMPTY data volume.
--
-- Duplicate-safety design: every table that gets loaded repeatedly by the
-- ETL has a UNIQUE KEY on its natural key. The ETL must INSERT using
-- "ON DUPLICATE KEY UPDATE" (upsert) so re-running it, or loading exports
-- with overlapping periods, never creates duplicate rows - it just
-- overwrites the old value with the newer one.

CREATE TABLE IF NOT EXISTS companies (
    ticker          VARCHAR(10)  NOT NULL PRIMARY KEY,
    company_name    VARCHAR(255) NOT NULL,
    exchange        VARCHAR(10)  NOT NULL,   -- HOSE / HNX / UPCOM
    sector          VARCHAR(100) NOT NULL,   -- thesis sample sector label
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS financial_line_items (
    id              BIGINT AUTO_INCREMENT PRIMARY KEY,
    ticker          VARCHAR(10)  NOT NULL,
    fiscal_year     SMALLINT     NOT NULL,
    period          VARCHAR(10)  NOT NULL,   -- 'Q1','Q2','Q3','Q4','YEAR','H1','9M'
    ms_code         VARCHAR(10)  NOT NULL,   -- account code per Circular 200 (e.g. '100').
                                              -- NOTE: MS codes reset per statement (e.g. '01'
                                              -- exists in income_statement AND cash_flow_indirect
                                              -- with completely different meanings) - statement_type
                                              -- MUST stay part of the unique key below, never drop it.
    item_name       VARCHAR(255) NOT NULL,
    statement_type  ENUM('balance_sheet','income_statement','cash_flow_indirect','cash_flow_direct') NOT NULL,
    consolidation   VARCHAR(20)  NOT NULL DEFAULT 'consolidated',
    value           DECIMAL(24,0),
    unit            VARCHAR(10)  NOT NULL DEFAULT 'VND',
    updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (ticker) REFERENCES companies(ticker),
    UNIQUE KEY uq_line_item (ticker, fiscal_year, period, statement_type, ms_code, consolidation)
);

CREATE TABLE IF NOT EXISTS financial_ratios (
    id              BIGINT AUTO_INCREMENT PRIMARY KEY,
    ticker          VARCHAR(10)  NOT NULL,
    fiscal_year     SMALLINT     NOT NULL,
    period          VARCHAR(10)  NOT NULL,
    ratio_group     VARCHAR(100) NOT NULL,   -- 'Định giá','Sinh lợi','Tăng trưởng','Thanh khoản',
                                              -- 'Hiệu quả hoạt động','Đòn bẩy tài chính','Dòng tiền'
    ratio_name      VARCHAR(255) NOT NULL,   -- e.g. 'P/E', 'ROEA', 'Beta'
    value           DECIMAL(18,4),
    unit            VARCHAR(10),             -- 'Lần','%','VNĐ'
    updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (ticker) REFERENCES companies(ticker),
    UNIQUE KEY uq_ratio (ticker, fiscal_year, period, ratio_name)
);

CREATE TABLE IF NOT EXISTS disclosures (
    id                BIGINT AUTO_INCREMENT PRIMARY KEY,
    ticker            VARCHAR(10)  NOT NULL,
    fiscal_year       SMALLINT     NOT NULL,
    period            VARCHAR(10)  NOT NULL,
    consolidation     VARCHAR(20)  NOT NULL DEFAULT 'consolidated',
    report_type       ENUM('nam','ban_nien','quy') NOT NULL,
    audit_status      VARCHAR(50),            -- 'Kiểm toán' / 'Soát xét' / 'Chưa kiểm toán'
    auditor           VARCHAR(255),
    audit_opinion     VARCHAR(255),
    disclosure_date   DATE,
    signed_date       DATE,
    source_pdf_path   VARCHAR(500),           -- path under data/raw/pdf_disclosures
    structure_valid   BOOLEAN,                -- output of report_structure_validator.py
    updated_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (ticker) REFERENCES companies(ticker),
    UNIQUE KEY uq_disclosure (ticker, fiscal_year, period, consolidation)
);

-- Audit trail of what's been loaded - not required for correctness (the
-- UNIQUE keys above already make loading idempotent), but useful for the
-- course's "research conduct / reproducibility" assessment: shows exactly
-- which source file produced which rows, and when.
CREATE TABLE IF NOT EXISTS ingestion_log (
    id              BIGINT AUTO_INCREMENT PRIMARY KEY,
    file_name       VARCHAR(255) NOT NULL,
    file_type       ENUM('bao_cao_tai_chinh','chi_so_tai_chinh','du_lieu_doanh_nghiep','pdf_disclosure','du_lieu_vi_mo') NOT NULL,
    ticker          VARCHAR(10),
    row_count       INT,
    ingested_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    notes           VARCHAR(500)
);
