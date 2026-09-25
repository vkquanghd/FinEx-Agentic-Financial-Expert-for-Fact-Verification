-- Task #2 ("Hoàn thiện khoá pdf_document_id") from Quang's 2026-07-30
-- post-v3 methodology review - docs/PROGRESS.md has the full writeup.
--
-- Migrations 006/007 introduced pdf_documents + disclosure_chunks.
-- pdf_document_id but deliberately left it NULLABLE ("honest NULL" if
-- scripts/register_pdf_documents.py hadn't been run yet for a file - see
-- src/retrieval/vector_store.py's _lineage_fields() 2026-07-29 comment).
-- That was the right call WHILE the registration step was brand new and
-- unverified against real data. Now that Tasks #7/#8 are done and the
-- pipeline order (pdf_ingest_router.py -> register_pdf_documents.py ->
-- text_splitter.py -> vector_store.py) is the ONLY supported order, an
-- unregistered document reaching disclosure_chunks is a BUG to catch
-- loudly (see vector_store.py's new UnregisteredDocumentError), not a
-- state to store silently. This migration locks that down at the DB level
-- too, not just in application code.
--
-- Changes:
--   1. disclosure_chunks.pdf_document_id -> NOT NULL.
--   2. Natural/unique key -> (pdf_document_id, page, chunk_index), REPLACING
--      the old (ticker, source_file, page, chunk_index). Once every chunk
--      has a real pdf_document_id, `ticker`/`source_file` are redundant as
--      KEY material (both are already denormalized onto pdf_documents,
--      reachable via the FK) - keeping them in the unique key was only ever
--      a workaround for not having pdf_document_id at all. This also fixes a
--      latent inconsistency: two DIFFERENT pdf_documents rows could in
--      theory share the same (ticker, source_file) if source_file ever
--      collided across a ticker rename/re-ingest - vanishingly unlikely in
--      practice, but the new key is the more honest one regardless, since
--      pdf_document_id IS the physical-file identity by construction.
--
-- PRE-FLIGHT CHECK - run this by hand FIRST and confirm it returns 0 before
-- applying the ALTER TABLE below. If it returns >0, you have chunks from
-- BEFORE scripts/register_pdf_documents.py existed (or a file that was
-- never registered) - re-run the full pipeline in order for those files
-- first (or delete the orphaned rows deliberately), do NOT force this
-- migration through with orphans still present:
--
--   SELECT count(*) FROM disclosure_chunks WHERE pdf_document_id IS NULL;
--
-- Run by hand against the running container (AFTER the pre-flight check
-- above returns 0):
--   docker exec -i thesis_postgres psql -U thesis_user -d financial_db \
--     < infra/postgres/migrations/010_lock_pdf_document_id.sql

DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM disclosure_chunks WHERE pdf_document_id IS NULL) THEN
        RAISE EXCEPTION 'Refusing to migrate: % row(s) in disclosure_chunks still have pdf_document_id IS NULL. '
            'Run scripts/register_pdf_documents.py for every source_file first (or delete/backfill the orphans '
            'deliberately), then re-run this migration. See this file''s header comment for the pre-flight check.',
            (SELECT count(*) FROM disclosure_chunks WHERE pdf_document_id IS NULL);
    END IF;
END $$;

ALTER TABLE disclosure_chunks
    ALTER COLUMN pdf_document_id SET NOT NULL;

ALTER TABLE disclosure_chunks
    DROP CONSTRAINT IF EXISTS uq_disclosure_chunk;

-- DROP-then-ADD (not a bare ADD), fixed 2026-07-30: found running this for
-- real that a partial prior run (psql continues past an error to the NEXT
-- statement by default, since this file isn't wrapped in an explicit
-- transaction / ON_ERROR_STOP) can leave uq_disclosure_chunk_v2 ALREADY
-- created even though the SET NOT NULL above keeps correctly failing (real
-- NULLs still present) - a UNIQUE constraint doesn't itself reject NULLs
-- (SQL treats each NULL as distinct for uniqueness purposes), so it can
-- succeed independently of the NOT NULL step. Without this fix, re-running
-- this migration (the normal, expected way to retry once the NULLs are
-- finally resolved) would keep failing on "relation already exists" even
-- after the real blocker was fixed - every other migration in this project
-- is idempotent by design, this one should be too.
ALTER TABLE disclosure_chunks
    DROP CONSTRAINT IF EXISTS uq_disclosure_chunk_v2;

ALTER TABLE disclosure_chunks
    ADD CONSTRAINT uq_disclosure_chunk_v2 UNIQUE (pdf_document_id, page, chunk_index);

-- idx_disclosure_chunks_pdf_document_id (migration 007) already covers the
-- new key's leading column for FK-join lookups; the UNIQUE constraint above
-- creates its own backing index for (pdf_document_id, page, chunk_index)
-- lookups/ON CONFLICT resolution - no additional index needed here.
