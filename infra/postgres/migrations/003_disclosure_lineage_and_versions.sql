-- Schema hardening from a 2026-07-29 architecture review (docs/PROGRESS.md
-- has the full writeup). Three additive, backward-compatible changes:
--
-- 1. disclosure_chunks.disclosure_id -- links a chunk back to the
--    disclosures row it came from (disclosures.id already IS a stable
--    "disclosure_id" in spirit; this just wires disclosure_chunks to
--    reference it instead of only trusting source_file, which can be
--    renamed, get an iCloud " 2" suffix, or be reused across corrected
--    versions of a filing). NULLABLE because the PDF-ingestion pipeline
--    (pdf_ingest_router.py) and the structured ETL (vietstock_etl.py) are
--    two independent pipelines with no shared identifier today -- see
--    src/retrieval/vector_store.py's resolve_disclosure_id() for the
--    best-effort join (ticker + fiscal year extracted from filename +
--    doctype heuristic). Old rows and rows the resolver can't confidently
--    match stay NULL rather than being wrongly linked.
--
-- 2. disclosure_chunks.chunking_version / normalization_version -- so a
--    stored chunk can answer "which normalize_page_text() logic and which
--    CHUNK_SIZE/OVERLAP config produced this row", not just chunk_hash/
--    embedding_version (added in migration 002). Same reasoning: an
--    algorithm change that doesn't touch the hashed TEXT (e.g. a
--    normalization bugfix that happens to produce byte-identical output
--    for most pages) should still be traceable per row.
--
-- 3. macro_indicators.release_date -- the date a macro figure was actually
--    published/available, as opposed to fiscal_year/period (the period the
--    figure DESCRIBES). Needed so any future forecasting/PREDICTIVE-branch
--    model can filter "only use macro data that was ALREADY PUBLIC as of
--    the prediction date" and avoid look-ahead/leakage. Nullable + no
--    backfill attempted here (Vietstock/GSO release-date isn't in the
--    current source files) -- populate it going forward as source data
--    allows; a NULL release_date should be treated as "unknown, exclude
--    from leakage-sensitive joins" by any future consumer, not "safe to use".
--
-- Deliberately NOT widening uq_price/uq_macro to include source/source_file
-- (a 2026-07-29 review suggested this) -- checked src/causal/macro_lookup.py
-- first: its queries assume exactly one row per (category, indicator_name,
-- period_type, fiscal_year, period) and read a single scalar value. Adding
-- source to the key would let two source files legitimately coexist as TWO
-- rows for the same period, which breaks that scalar-read assumption instead
-- of fixing anything - the actual risk (last-loaded source silently wins)
-- is already visible via the existing source/source_file column + updated_at
-- trigger, just not enforced at the constraint level. Left as a documented,
-- deliberate tradeoff rather than a silent gap.
--
-- Run by hand against the running container:
--   docker exec -i thesis_postgres psql -U thesis_user -d financial_db \
--     < infra/postgres/migrations/003_disclosure_lineage_and_versions.sql

ALTER TABLE disclosure_chunks
    ADD COLUMN IF NOT EXISTS disclosure_id BIGINT REFERENCES disclosures(id),
    ADD COLUMN IF NOT EXISTS chunking_version VARCHAR(100),
    ADD COLUMN IF NOT EXISTS normalization_version VARCHAR(100);

CREATE INDEX IF NOT EXISTS idx_disclosure_chunks_disclosure_id ON disclosure_chunks (disclosure_id);

ALTER TABLE macro_indicators
    ADD COLUMN IF NOT EXISTS release_date DATE;
