-- Companion to migration 006_pdf_documents.sql - repoints
-- disclosure_chunks' document-identity column from `disclosure_id`
-- (BIGINT REFERENCES disclosures(id), added migration 003) to the new
-- `pdf_document_id` (BIGINT REFERENCES pdf_documents(id)).
--
-- IMPORTANT - this is NOT a pure additive migration, unlike every other
-- migration in this project so far. Run this AFTER migration 006. Two
-- real things happen:
--
-- 1. The OLD `disclosure_id` column is DROPPED, not kept alongside the new
--    one. If you already ran migration 003 and had ANY disclosure_id
--    values populated (e.g. from an upsert run before this evening's fix),
--    those values point into `disclosures.id`'s id-space, which is a
--    DIFFERENT sequence/table than `pdf_documents.id` - keeping the column
--    and just changing what NEW values mean would leave old rows
--    pointing at the wrong table's ids with no way to tell old-meaning
--    rows apart from new-meaning rows. Dropping and starting `pdf_document_id`
--    fresh (all NULL) is the honest choice - see src/retrieval/
--    vector_store.py's resolve_pdf_document_id() for how it gets
--    (re)populated on the next upsert run (after you've also run the new
--    scripts/register_pdf_documents.py - see that script's docstring).
--
-- 2. No backfill is attempted here. Run, in this order, after this
--    migration:
--      python3 -m scripts.register_pdf_documents   (NEW - creates pdf_documents rows)
--      python3 -m scripts.backfill_pdf_document_id  (fills in pdf_document_id for every
--                                                     EXISTING row via a direct SQL
--                                                     UPDATE...FROM on source_file - cheap,
--                                                     no re-embedding, no Python model load)
--
--    CORRECTED 2026-07-30 (found running this for real against a 661,657-
--    chunk corpus): this comment used to say "re-run python3 -m
--    src.retrieval.vector_store ... - cheap, ON CONFLICT DO UPDATE just
--    refreshes pdf_document_id". That stopped being true the same day
--    (migration 010 / Task #2's re-keying of _needs_embedding() to key
--    existing rows by (pdf_document_id, page, chunk_index) read straight
--    from disclosure_chunks): right after THIS migration runs, every
--    existing row's pdf_document_id is NULL, so _existing_chunk_state()
--    would return keys like (None, page, chunk_index) - which can never
--    match the real, freshly-resolved pdf_document_id upsert_records()
--    computes per record. Re-running the Python upsert pipeline at that
--    point would treat every single chunk as "new" and FULLY RE-EMBED the
--    entire corpus (hours of GPU/CPU work, for zero actual change) instead
--    of cheaply backfilling one column. Use scripts/backfill_pdf_document_id.py
--    instead - see that script's own module docstring for the full story.
--
-- Run by hand against the running container:
--   docker exec -i thesis_postgres psql -U thesis_user -d financial_db \
--     < infra/postgres/migrations/007_disclosure_chunks_pdf_document_id.sql

ALTER TABLE disclosure_chunks DROP COLUMN IF EXISTS disclosure_id;

ALTER TABLE disclosure_chunks
    ADD COLUMN IF NOT EXISTS pdf_document_id BIGINT REFERENCES pdf_documents(id);

CREATE INDEX IF NOT EXISTS idx_disclosure_chunks_pdf_document_id
    ON disclosure_chunks (pdf_document_id);
