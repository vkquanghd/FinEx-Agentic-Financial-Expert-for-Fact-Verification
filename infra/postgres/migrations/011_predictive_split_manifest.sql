-- Task #18 from the 2026-07-30 methodology roadmap (roadmap item 4 -
-- "freeze the split into a file or table ... every model must use the
-- identical split"). docs/PROGRESS.md has the full writeup.
--
-- Why a separate table instead of just adding a "split" column to
-- predictive_features directly: predictive_features is REBUILT (feature_
-- builder.py re-runs, bumping feature_version or refreshing target_value
-- columns per migration 008's restricted-update policy) more often than
-- the train/val/test assignment should ever change. If "split" lived on
-- predictive_features itself, a routine feature rebuild could silently
-- reshuffle which rows are in test - exactly the kind of methodological
-- drift the 2026-07-30 review is trying to eliminate. Freezing the split
-- in its own append-mostly table, keyed by the SAME natural key as
-- predictive_features, means a feature rebuild changes feature VALUES but
-- never changes which split a given (ticker, fiscal_year, ...) row
-- belongs to.
--
-- Split rule (time-based, per the review): information_cutoff < 2023 ->
-- train; information_cutoff in 2023 -> validation; information_cutoff >=
-- 2024 -> test. split_manifest_version is a string derived from these
-- three boundary years (see scripts/predictive/freeze_predictive_split.py's
-- SPLIT_MANIFEST_VERSION) - changing the boundaries is a genuinely new
-- split and gets a new version string, same auto-derived-version pattern
-- already used for CHUNKING_VERSION/INGEST_VERSION elsewhere in this
-- project, rather than silently reassigning rows under the old version.
--
-- FREEZE semantics: scripts/predictive/freeze_predictive_split.py only
-- INSERTs new (feature_version, ticker, fiscal_year, period,
-- consolidation, target_col, horizon) keys it hasn't seen before under a
-- given split_manifest_version - it never UPDATEs an existing row's split
-- assignment (see that script's docstring for the explicit --allow-
-- reassign escape hatch, which requires the user to say so out loud).
-- This is why there's no ON CONFLICT DO UPDATE clause at all in the
-- script's INSERT, unlike every other upsert in this codebase - a silent
-- update here would be exactly the bug this table exists to prevent.
--
-- Run by hand against the running container:
--   docker exec -i thesis_postgres psql -U thesis_user -d financial_db \
--     < infra/postgres/migrations/011_predictive_split_manifest.sql

CREATE TABLE IF NOT EXISTS predictive_split_manifest (
    id                          BIGSERIAL PRIMARY KEY,
    split_manifest_version      VARCHAR(100) NOT NULL,   -- e.g. "time_train_lt2023_val2023_test_gte2024_v1"
    feature_version             VARCHAR(100) NOT NULL,   -- which predictive_features build this key was seen in
    ticker                      VARCHAR(16)  NOT NULL,
    fiscal_year                 SMALLINT     NOT NULL,
    period                      VARCHAR(16)  NOT NULL,
    consolidation                VARCHAR(32),
    target_col                  VARCHAR(100) NOT NULL,
    horizon                     SMALLINT     NOT NULL,
    information_cutoff          DATE         NOT NULL,   -- frozen copy of the value the split decision was based on
    split                       VARCHAR(16)  NOT NULL CHECK (split IN ('train', 'validation', 'test')),
    frozen_at                   TIMESTAMPTZ  NOT NULL DEFAULT now(),
    CONSTRAINT uq_predictive_split_manifest
        UNIQUE (split_manifest_version, ticker, fiscal_year, period, consolidation, target_col, horizon)
);

CREATE INDEX IF NOT EXISTS idx_predictive_split_manifest_version
    ON predictive_split_manifest (split_manifest_version);
CREATE INDEX IF NOT EXISTS idx_predictive_split_manifest_split
    ON predictive_split_manifest (split);
