-- Task #7 from the 2026-07-29 (evening) architecture review round 2 -
-- docs/PROGRESS.md has the full writeup. Creates `pdf_documents`: one row
-- per PHYSICAL PDF file in the pdf_ingest_router.py corpus.
--
-- WHY a NEW table, not just adding file_hash/document_version/
-- extraction_version onto the existing `disclosures` table (as literally
-- suggested in the review): `disclosures` is structured-filing metadata
-- from src/ingestion/vietstock_etl.py - one row per
-- (ticker, fiscal_year, period, consolidation), with NO file identity at
-- all (no source_file/file_hash column, never has had one). A ticker can
-- legitimately have SEVERAL real PDFs for the same fiscal year - an annual
-- report, an audited BCTC, a later correction, a governance report - and
-- the OLD resolve_disclosure_id() (matching only on ticker + fiscal_year +
-- doctype against `disclosures`) had no way to tell these apart; it could
-- only ever pick ONE candidate `disclosures` row per (ticker, fiscal_year),
-- silently wrong whenever more than one PDF existed for that year. Giving
-- every PDF its own row here, keyed by the UNIQUE source_file, removes the
-- ambiguity structurally instead of trying to disambiguate it with a
-- smarter heuristic.
--
-- Populated by the NEW scripts/register_pdf_documents.py, which runs AFTER
-- pdf_ingest_router.py (reads its ingest_manifest.csv) and BEFORE
-- text_splitter.py - so disclosure_chunks.pdf_document_id (see migration
-- 007) is assigned once, deterministically, at ingest/registration time,
-- NOT guessed later per-chunk the way disclosure_id used to be. The lookup
-- vector_store.py does at upsert time (resolve_pdf_document_id()) is now a
-- plain UNIQUE-key SELECT on source_file, not a fuzzy multi-candidate match.
--
-- linked_disclosure_id keeps a SEPARATE, still best-effort (ticker +
-- fiscal_year + doctype) link from a PDF document to the structured-filing
-- table, for callers that want audit_status/auditor/disclosure_date -
-- still nullable, still never a hard guarantee, but now computed ONCE per
-- PDF document at registration time instead of once per CHUNK (much
-- cheaper, and honest about being a secondary/optional link rather than
-- the chunk's primary identity).
--
-- Run by hand against the running container:
--   docker exec -i thesis_postgres psql -U thesis_user -d financial_db \
--     < infra/postgres/migrations/006_pdf_documents.sql

CREATE TABLE IF NOT EXISTS pdf_documents (
    id                   BIGSERIAL PRIMARY KEY,
    source_file          VARCHAR(255) NOT NULL UNIQUE,
    pdf_hash             CHAR(64)     NOT NULL,
    ticker               VARCHAR(16),
    doctype              VARCHAR(32),
    fiscal_year          SMALLINT,
    n_pages_total        INT,
    extraction_version   VARCHAR(100) NOT NULL,
    extracted_at         TIMESTAMPTZ  NOT NULL DEFAULT now(),
    document_version     INT          NOT NULL DEFAULT 1,
    is_active            BOOLEAN      NOT NULL DEFAULT TRUE,
    linked_disclosure_id BIGINT REFERENCES disclosures(id),
    updated_at           TIMESTAMPTZ  NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_pdf_documents_ticker ON pdf_documents (ticker);
CREATE INDEX IF NOT EXISTS idx_pdf_documents_linked_disclosure ON pdf_documents (linked_disclosure_id);

-- set_updated_at() is already defined by 001_init_schema.sql (used by
-- companies/disclosures/etc) - reused here, not redefined.
DROP TRIGGER IF EXISTS trg_pdf_documents_updated_at ON pdf_documents;
CREATE TRIGGER trg_pdf_documents_updated_at
    BEFORE UPDATE ON pdf_documents
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();
