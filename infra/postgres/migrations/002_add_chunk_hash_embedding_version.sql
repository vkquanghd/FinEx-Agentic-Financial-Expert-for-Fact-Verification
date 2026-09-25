-- Adds chunk_hash + embedding_version to an ALREADY-RUNNING disclosure_chunks
-- table (001_init_schema.sql only auto-applies on a fresh/empty data volume,
-- so anyone who already has data loaded needs this run by hand instead).
--
-- Why (2026-07-29): src/retrieval/vector_store.py's skip_existing logic used
-- to check ONLY whether a (ticker, source_file, page, chunk_index) key
-- already had a non-null embedding - never whether the STORED TEXT still
-- matched the current chunks.jsonl. A corrected/restated disclosure page
-- would rechunk correctly upstream (src/retrieval/text_splitter.py's
-- content-hash incremental logic), but the DB would silently keep the OLD
-- chunk_text and OLD embedding forever under the same key, because the row
-- was never re-selected for embedding. These two columns let
-- _needs_embedding() (see vector_store.py) tell "genuinely unchanged, safe
-- to skip" apart from "same key, different content, must re-embed".
--
-- Run this by hand against your running Postgres container, e.g.:
--   docker exec -i thesis_postgres psql -U thesis_user -d thesis \
--     < infra/postgres/migrations/002_add_chunk_hash_embedding_version.sql
-- (adjust container name / -U / -d to match your .env if different)
--
-- Safe to run more than once (IF NOT EXISTS everywhere). Existing rows get
-- NULL chunk_hash/embedding_version until the next upsert_records()/
-- upsert_precomputed() run touches them - _needs_embedding() treats a NULL
-- stored chunk_hash as "doesn't match" (old_hash != new_hash, since old_hash
-- is None), so every existing row gets its chunk_hash/embedding_version
-- backfilled correctly the next time the pipeline runs, WITHOUT forcing a
-- wasteful full re-embed: the row's vector value itself does not change
-- unless the text actually changed (UPSERT_SQL's DO UPDATE only overwrites
-- embedding with what embed_texts() computes for the given chunk_text, which
-- for an unchanged chunk_text just recomputes the same vector bge-m3 would
-- have produced before) -- so the ONE-TIME cost after applying this
-- migration is: every row gets re-embedded once (since chunk_hash was NULL,
-- i.e. "unknown"), not a bug in itself, just a one-time full re-embed to
-- backfill the new columns. If avoiding that one-time full re-embed matters
-- (e.g. right before a defense deadline), backfill chunk_hash directly from
-- the already-stored chunk_text instead of going through the pipeline:
--
--   UPDATE disclosure_chunks
--   SET chunk_hash = encode(digest(regexp_replace(trim(chunk_text), '\s+', ' ', 'g'), 'sha256'), 'hex'),
--       embedding_version = 'BAAI/bge-m3_normalized_v1'
--   WHERE embedding IS NOT NULL AND chunk_hash IS NULL;
--
-- (requires the pgcrypto extension for digest(); CREATE EXTENSION IF NOT
-- EXISTS pgcrypto; first. The regexp_replace matches src/retrieval/
-- text_splitter.py's chunk_hash()'s " ".join(text.split()) normalization.)

ALTER TABLE disclosure_chunks
    ADD COLUMN IF NOT EXISTS chunk_hash CHAR(64),
    ADD COLUMN IF NOT EXISTS embedding_version VARCHAR(255);

CREATE INDEX IF NOT EXISTS idx_disclosure_chunks_chunk_hash ON disclosure_chunks (chunk_hash);
