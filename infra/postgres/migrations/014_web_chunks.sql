-- Migration 014 - web_chunks (2026-09-07)
--
-- WHY A SEPARATE TABLE (not more doctype values on disclosure_chunks):
-- decided explicitly 2026-09-07 while diagnosing why "Rủi ro pháp lý của
-- VNM là gì?" got a thin RAG answer even after the missing Annual Report
-- PDFs were ingested (see docs/PROGRESS.md same date) - VNM's OWN official
-- risk-management disclosure genuinely is that thin (checked against the
-- real PDF: https://www.vinamilk.com.vn/bao-cao-thuong-nien/bao-cao/2024/doc/vi/bc-quan-tri-rui-ro.pdf),
-- so no amount of re-ingesting OFFICIAL disclosures fixes this class of
-- question - the answer has to come from press coverage instead. Press
-- content is a fundamentally different evidence tier than an audited BCTC
-- or an Annual Report filed with HOSE/SSC: it can be wrong, retracted, or
-- just opinion, and a scraped page's content is a snapshot that may no
-- longer match the live URL by the time anyone checks it. Mixing it into
-- disclosure_chunks (even tagged via doctype) would let a router bug or a
-- future teammate accidentally pull web content into a RQ1-RQ7 benchmark
-- answer with no code-level barrier stopping it. A separate table makes
-- that barrier structural, not a convention someone has to remember -
-- src/retrieval/rag_pipeline.py only queries web_chunks as an explicit,
-- clearly-labeled FALLBACK when disclosure_chunks has nothing (see that
-- file's web_fallback_answer()), never merged into the same result set.
--
-- Deliberately NOT tied to pdf_documents/disclosures (unlike
-- disclosure_chunks - see infra/postgres/migrations/006_pdf_documents.sql
-- and 010_lock_pdf_document_id.sql for why THAT table needs an exact
-- physical-file identity row): a scraped web article has no PDF, no
-- fiscal_year/period/consolidation grain, and no audit trail to join
-- against - its natural identity is just (source_url, chunk_index).
--
-- Content is a SNAPSHOT, not a live pointer: chunk_text is exactly what
-- scripts/web_news_ingest.py fetched at scraped_at, saved to
-- data/raw/web_articles/ first (see that script) THEN chunked/embedded
-- here - same "citation must be checkable later" principle
-- docs/PROGRESS.md already applies elsewhere, just extended to web content
-- that can genuinely change or disappear (scripts/web_fallback_lookup.py's
-- WARNING_BANNER made the same point for the live, non-persisted case this
-- table now has a persisted counterpart for).
--
-- Run by hand against the running container (same convention as every
-- other incremental migration in this folder):
--   docker exec -i thesis_postgres psql -U thesis_user -d financial_db \
--     < infra/postgres/migrations/014_web_chunks.sql

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS web_chunks (
    id                BIGSERIAL PRIMARY KEY,
    ticker            VARCHAR(16),                 -- best-effort, NULL allowed (a scraped article may cover no single ticker, or several)
    source_type       VARCHAR(32)  NOT NULL DEFAULT 'news_web',  -- fixed tag - deliberately NOT one of disclosure_chunks.doctype's values, so the two are never confusable even if a future query joins/unions them by mistake
    domain            VARCHAR(255) NOT NULL,        -- e.g. 'cafef.vn' - trust-tier / provenance filtering at query time
    source_url        TEXT         NOT NULL,
    title             TEXT,
    published_date    DATE,                         -- best-effort parsed from the page, NULL if unknown - never guessed
    scraped_at        TIMESTAMPTZ  NOT NULL DEFAULT now(),
    chunk_index       INT          NOT NULL,
    chunk_text        TEXT         NOT NULL,        -- SNAPSHOT of the page content at scraped_at, not a live reference - see header comment
    char_count        INT          NOT NULL,
    embedding         VECTOR(1024),                  -- same BAAI/bge-m3, same dimension as disclosure_chunks - NULL until embedded
    chunk_hash        CHAR(64),                      -- SHA-256 of chunk_text, same convention as disclosure_chunks.chunk_hash
    embedding_version VARCHAR(255),                  -- same EMBEDDING_VERSION constant as src/retrieval/vector_store.py
    created_at        TIMESTAMPTZ  NOT NULL DEFAULT now(),
    CONSTRAINT uq_web_chunk UNIQUE (source_url, chunk_index)
);

CREATE INDEX IF NOT EXISTS idx_web_chunks_ticker  ON web_chunks (ticker);
CREATE INDEX IF NOT EXISTS idx_web_chunks_domain  ON web_chunks (domain);
CREATE INDEX IF NOT EXISTS idx_web_chunks_published_date ON web_chunks (published_date);

-- Same HNSW choice/params as disclosure_chunks - see
-- infra/pgvector/migrations/001_disclosure_chunks.sql's comment for the
-- full IVFFlat->HNSW rationale, unchanged here.
CREATE INDEX IF NOT EXISTS idx_web_chunks_embedding
    ON web_chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Same lexical/FTS support as disclosure_chunks (migration 002) - web
-- articles need exact-string matching (company names, figures) just as
-- much as official disclosures do.
CREATE INDEX IF NOT EXISTS idx_web_chunks_fts
    ON web_chunks USING gin (to_tsvector('simple', chunk_text));
