-- Tasks #9/#10 from the 2026-07-29 (evening) architecture review round 2 -
-- docs/PROGRESS.md has the full writeup. Two additive columns for
-- per-feature leakage auditing and target provenance - see
-- src/predictive/feature_builder.py's module docstring (points 4/5) for
-- the full reasoning behind each.
--
-- target_observed_at: the information_cutoff (disclosure_date) of the
-- FUTURE row that supplied target_value - i.e. when the target was
-- actually knowable, not just what fiscal_year it describes.
--
-- target_release_date: reserved, NOT populated by the current builder
-- (every target_col today is a fundamentals/ratio field with its own
-- disclosure_date, captured by target_observed_at already) - kept for a
-- future target_col that is itself macro-sourced, where "the date THIS
-- specific macro reading was released" would be a distinct concept from
-- "the date the surrounding report was disclosed". NULL is the correct,
-- expected value for every row the current builder produces.
--
-- feature_created_at / feature_hash / source_snapshot_version: lineage
-- fields. feature_hash lets a later reader detect whether a fresh
-- build_predictive_features() call would compute DIFFERENT feature values
-- than what's stored, without needing to actually overwrite the row to
-- find out - relevant because scripts/build_predictive_features.py
-- deliberately does NOT overwrite feature columns on ON CONFLICT (see that
-- script's own updated docstring) - only target_value/target_observed_at/
-- eligible_for_training/updated_at are refreshed on a re-run.
-- source_snapshot_version is a free-text label (not a hard FK to
-- causal_panel_snapshot) recording which snapshot, if any, was built from
-- the same underlying panel state around the same time.
--
-- Run by hand against the running container:
--   docker exec -i thesis_postgres psql -U thesis_user -d financial_db \
--     < infra/postgres/migrations/008_predictive_features_leakage_and_target_metadata.sql

ALTER TABLE predictive_features
    ADD COLUMN IF NOT EXISTS target_observed_at DATE,
    ADD COLUMN IF NOT EXISTS target_release_date DATE,
    ADD COLUMN IF NOT EXISTS feature_created_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS feature_hash CHAR(64),
    ADD COLUMN IF NOT EXISTS source_snapshot_version VARCHAR(100);
