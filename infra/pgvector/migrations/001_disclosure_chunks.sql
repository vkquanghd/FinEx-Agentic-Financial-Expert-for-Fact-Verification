-- DEPRECATED 2026-07-26 - superseded by infra/postgres/migrations/001_init_schema.sql,
-- which now contains this exact table (unchanged) plus every table
-- previously in infra/mysql/migrations/*, all in ONE Postgres database
-- instead of a separate "vectorstore" service (see docs/PROGRESS.md "Gộp
-- MySQL + pgvector thành một PostgreSQL duy nhất"). Kept for historical
-- reference only, no longer applied by docker-compose.yml.
--
-- Chunk-level table for the RAG path (src/retrieval/), backing the
-- "RAG semantic" / "Nhánh C" route in docs/Thesis_Direction_Redesign.pdf.
--
-- Vector dimension is 1024 to match BAAI/bge-m3 (chosen after checking
-- VN-MTEB, arXiv:2507.21500 - bge-m3 scored well on Vietnamese retrieval,
-- supports up to 8192-token input, and the vietnamese-bi-encoder model
-- originally floated in docs/data_plan.md scored notably lower on the same
-- benchmark's retrieval task, hence NOT 768 dims). If the embedding model
-- ever changes, this table has to be dropped and rebuilt - the vector
-- dimension is fixed at table-creation time in pgvector, it can't ALTER.
--
-- Source content: chunks come from data/processed/text_native/*.txt, which
-- itself comes from scripts/pdf_ingest_router.py (text-native pages only -
-- pages that needed OCR are NOT in this table until the OCR step, not yet
-- built, produces its own text output to feed through the same chunker).
--
-- Run this by hand against the vectorstore service (it is NOT
-- auto-applied by docker-entrypoint-initdb.d the way infra/mysql/migrations
-- is, to keep the two Postgres/MySQL init flows independent):
--   docker exec -i thesis_vectorstore psql -U thesis_user -d thesis_vectors < infra/pgvector/migrations/001_disclosure_chunks.sql
-- (adjust -U/-d if your .env DB_USER / PGVECTOR_DB differ from the defaults)

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS disclosure_chunks (
    id              BIGSERIAL PRIMARY KEY,
    ticker          VARCHAR(16)  NOT NULL,   -- best-effort from filename, see scripts/pdf_ingest_router.py guess_ticker() - cross-check against MySQL companies.ticker before trusting it for joins
    doctype         VARCHAR(32)  NOT NULL,   -- annual_report / financial_statement / kqkd_explanation / agm_resolution / governance_report / board_resolution / other
    source_file     VARCHAR(255) NOT NULL,   -- original PDF filename (path.stem from the router)
    page            INT          NOT NULL,
    chunk_index     INT          NOT NULL,   -- 0-based index of this chunk within its page
    chunk_text      TEXT         NOT NULL,
    char_count      INT          NOT NULL,
    embedding       VECTOR(1024),            -- NULL until src/retrieval/embedder.py fills it in - lets chunking and embedding run as separate, independently re-runnable steps
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT now(),
    UNIQUE (ticker, source_file, page, chunk_index)  -- idempotent upserts, same spirit as the MySQL uq_line_item constraint
);

CREATE INDEX IF NOT EXISTS idx_disclosure_chunks_ticker  ON disclosure_chunks (ticker);
CREATE INDEX IF NOT EXISTS idx_disclosure_chunks_doctype ON disclosure_chunks (doctype);

-- Approximate nearest-neighbor index for cosine similarity search.
--
-- HNSW, not IVFFlat (changed 2026-07-26 - see docs/PROGRESS.md for the
-- full writeup). Two concrete reasons, not just "generally better":
--   1. IVFFlat's index quality depends on centroids fit from existing data,
--      so it had to be built AFTER loading chunks (re-run DROP+CREATE once
--      a real corpus existed) - HNSW's graph is built incrementally as rows
--      are inserted, so this index can exist from the start, no
--      empty-table caveat, no rebuild step.
--   2. HNSW gives materially better recall than IVFFlat at the same query
--      speed (well-established in the pgvector benchmarks) - directly
--      relevant here since recall is exactly what similarity_search()'s
--      ticker-scoped exact-scan workaround (see vector_store.py) exists to
--      protect.
-- m/ef_construction below are pgvector's own documented defaults (16 / 64)
-- - not tuned against this corpus (no labelled ranking data to tune
-- against in a 10-week project, same honest caveat as every other
-- untuned constant in this codebase, e.g. RECENCY_PENALTY_PER_YEAR).
--
-- IMPORTANT - what this migration does NOT fix by itself: HNSW alone does
-- not make similarity_search()'s ticker-scoped `SET LOCAL enable_indexscan/
-- enable_bitmapscan = off` workaround unnecessary. Both IVFFlat and HNSW
-- (pre-iterative-scan) run the ANN search first and apply any WHERE filter
-- AFTER, so a `WHERE ticker = ...` filter can still be applied to a small
-- approximate candidate set instead of that ticker's real row count,
-- regardless of which ANN algorithm sits underneath - this was previously
-- mis-attributed to "IVFFlat's approximate nature" in this file's comments,
-- but the real cause is index-then-filter query execution, which HNSW
-- shares. pgvector >= 0.8.0 adds `hnsw.iterative_scan` (relaxed_order /
-- strict_order), which lets a filtered query keep walking the graph until
-- enough post-filter rows are found, WITHOUT falling back to a full
-- sequential scan - that is the actual fix for the ticker-scoped case, not
-- just swapping IVFFlat for HNSW. Confirm the deployed pgvector version
-- first (`SELECT extversion FROM pg_extension WHERE extname = 'vector';`
-- against the vectorstore container) before turning `hnsw.iterative_scan`
-- on in vector_store.py - the docker-compose.yml `pgvector/pgvector:pg16`
-- image tag is NOT pinned to a specific pgvector version, so this must be
-- checked per-deployment, not assumed. Until that's confirmed, the exact-
-- scan workaround stays in vector_store.py unchanged - it is still correct,
-- just no longer described as an IVFFlat-only quirk.
CREATE INDEX IF NOT EXISTS idx_disclosure_chunks_embedding
    ON disclosure_chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
