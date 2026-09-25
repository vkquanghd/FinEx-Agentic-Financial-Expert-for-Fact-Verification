-- Migration 016 - Vietstock Premium ingestion audit model (2026-09-17)
--
-- This stores observations made through the account owner's authenticated
-- browser session. It intentionally does NOT store passwords, cookies or
-- bearer tokens. Those remain in data/private/vietstock_profile/ on the
-- host and are ignored by Git.
--
-- Apply to an existing database:
--   docker exec -i thesis_postgres psql -U thesis_user -d thesis_fin_qa \
--     < infra/postgres/migrations/016_vietstock_premium_ingestion.sql

CREATE TABLE IF NOT EXISTS vietstock_ingestion_runs (
    id              BIGSERIAL PRIMARY KEY,
    scope           VARCHAR(32) NOT NULL
                    CHECK (scope IN ('login_check', 'directory', 'company_sync', 'document_sync')),
    status          VARCHAR(20) NOT NULL
                    CHECK (status IN ('running', 'completed', 'partial', 'failed')),
    started_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at     TIMESTAMPTZ,
    requested_by    VARCHAR(100),
    config          JSONB NOT NULL DEFAULT '{}'::jsonb,
    counters        JSONB NOT NULL DEFAULT '{}'::jsonb,
    error_summary   TEXT
);
CREATE INDEX IF NOT EXISTS idx_vietstock_ingestion_runs_started
    ON vietstock_ingestion_runs (started_at DESC);

CREATE TABLE IF NOT EXISTS vietstock_ticker_sync (
    id              BIGSERIAL PRIMARY KEY,
    run_id          BIGINT NOT NULL REFERENCES vietstock_ingestion_runs(id) ON DELETE CASCADE,
    ticker          VARCHAR(16) NOT NULL,
    company_url     TEXT NOT NULL,
    status          VARCHAR(20) NOT NULL
                    CHECK (status IN ('completed', 'partial', 'failed', 'skipped')),
    fetched_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    http_status     SMALLINT,
    page_title      TEXT,
    profile         JSONB NOT NULL DEFAULT '{}'::jsonb,
    metrics         JSONB NOT NULL DEFAULT '{}'::jsonb,
    raw_snapshot    TEXT,
    document_count  INT NOT NULL DEFAULT 0,
    news_count      INT NOT NULL DEFAULT 0,
    error_message   TEXT,
    CONSTRAINT uq_vietstock_ticker_sync_run UNIQUE (run_id, ticker)
);
CREATE INDEX IF NOT EXISTS idx_vietstock_ticker_sync_ticker_fetched
    ON vietstock_ticker_sync (ticker, fetched_at DESC);

CREATE TABLE IF NOT EXISTS vietstock_document_manifest (
    id              BIGSERIAL PRIMARY KEY,
    ticker          VARCHAR(16) NOT NULL,
    source_url      TEXT NOT NULL,
    title           TEXT,
    document_type   VARCHAR(80),
    published_at    DATE,
    first_seen_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_seen_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    remote_hash     CHAR(64),
    local_path      TEXT,
    local_hash      CHAR(64),
    download_status VARCHAR(20) NOT NULL DEFAULT 'discovered'
                    CHECK (download_status IN ('discovered', 'downloaded', 'not_pdf', 'access_denied', 'failed')),
    http_status     SMALLINT,
    last_error      TEXT,
    CONSTRAINT uq_vietstock_document_source UNIQUE (ticker, source_url)
);
CREATE INDEX IF NOT EXISTS idx_vietstock_document_manifest_ticker
    ON vietstock_document_manifest (ticker, last_seen_at DESC);
CREATE INDEX IF NOT EXISTS idx_vietstock_document_manifest_status
    ON vietstock_document_manifest (download_status, last_seen_at DESC);

CREATE TABLE IF NOT EXISTS vietstock_news_events (
    id              BIGSERIAL PRIMARY KEY,
    ticker          VARCHAR(16) NOT NULL,
    source_url      TEXT NOT NULL,
    title           TEXT,
    published_at    DATE,
    category        VARCHAR(64),
    first_seen_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_seen_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT uq_vietstock_news_event_source UNIQUE (ticker, source_url)
);
CREATE INDEX IF NOT EXISTS idx_vietstock_news_events_ticker
    ON vietstock_news_events (ticker, published_at DESC NULLS LAST);
