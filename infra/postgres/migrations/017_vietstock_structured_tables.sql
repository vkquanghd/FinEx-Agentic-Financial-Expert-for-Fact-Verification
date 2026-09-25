-- Migration 017 - rendered Vietstock trading/financial tables.
-- Raw tables are retained first because statement templates differ by sector.

CREATE TABLE IF NOT EXISTS vietstock_structured_runs (
    id            BIGSERIAL PRIMARY KEY,
    scope         VARCHAR(32) NOT NULL,
    status        VARCHAR(20) NOT NULL CHECK (status IN ('running', 'completed', 'partial', 'failed')),
    started_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at   TIMESTAMPTZ,
    config        JSONB NOT NULL DEFAULT '{}'::jsonb,
    counters      JSONB NOT NULL DEFAULT '{}'::jsonb,
    error_summary TEXT
);

CREATE TABLE IF NOT EXISTS vietstock_structured_tables (
    id            BIGSERIAL PRIMARY KEY,
    ticker        VARCHAR(16) NOT NULL REFERENCES companies(ticker),
    section       VARCHAR(32) NOT NULL,
    subsection    TEXT NOT NULL DEFAULT '',
    source_url    TEXT NOT NULL,
    table_index   INT NOT NULL,
    table_title   TEXT NOT NULL DEFAULT '',
    headers       JSONB NOT NULL DEFAULT '[]'::jsonb,
    rows          JSONB NOT NULL DEFAULT '[]'::jsonb,
    table_hash    CHAR(64) NOT NULL,
    http_status   SMALLINT,
    fetched_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT uq_vietstock_structured_table UNIQUE (ticker, source_url, table_index, table_hash)
);
CREATE INDEX IF NOT EXISTS idx_vietstock_structured_tables_ticker
    ON vietstock_structured_tables (ticker, section, fetched_at DESC);

