-- Task #11 from the 2026-07-29 (evening) architecture review round 2 -
-- docs/PROGRESS.md has the full writeup.
--
-- causal_panel_snapshot (migration 004) freezes WHICH ROWS went into an
-- RQ6a estimate - it says nothing about HOW the estimate itself was
-- produced: which estimator, which nuisance model, which random seed,
-- how many bootstrap iterations, what code was running. Two people (or
-- the same person, months apart) pointing estimate_ate() at the exact
-- same snapshot_version can still get numbers a reader has no way to
-- compare, because none of THAT metadata was ever recorded anywhere -
-- only the panel was frozen, not the experiment run against it.
--
-- causal_experiment_runs is an APPEND-ONLY LOG (no ON CONFLICT / no
-- unique key on the metadata columns - every call is a new row, by
-- design) of every estimate_ate()/estimate_ate_dml() invocation made via
-- scripts/run_causal_experiment.py, storing both the INPUTS (what was
-- asked for) and the OUTPUT (what came back), including failed attempts
-- (ok=FALSE + error_message) - a "why did this run refuse to estimate"
-- question should be answerable from this table too, not just successes.
--
-- snapshot_version vs source_dataset_version: snapshot_version is set
-- ONLY when the run was pointed at a FROZEN causal_panel_snapshot (via
-- --snapshot-version); most runs during active development instead call
-- panel_builder.build_panel() live, in which case snapshot_version is
-- NULL and source_dataset_version records a synthetic
-- "live_YYYYMMDD" label instead (same free-text-label pattern as
-- predictive_features.source_snapshot_version, migration 008) - the two
-- columns are deliberately NOT merged into one, because a run against a
-- frozen snapshot should still be able to say so unambiguously later,
-- rather than relying on a string-prefix convention to distinguish
-- "frozen" from "live" after the fact.
--
-- covariates is stored as a single comma-joined TEXT column (e.g.
-- "sector,fiscal_year"), not a Postgres native array/JSONB - matching
-- this project's established preference for plain scalar columns over
-- semi-structured types (see 001_init_schema.sql's own comment on why
-- ENUM was avoided in favor of VARCHAR+CHECK) and keeping this table
-- trivially writable via a plain %s/```:param``` bind, same as everywhere
-- else in this codebase.
--
-- code_commit is the short git SHA of the working tree AT RUN TIME,
-- best-effort (NULL if git isn't available or this isn't a git checkout -
-- see scripts/run_causal_experiment.py's _git_commit(), same
-- don't-fabricate-just-say-what's-missing pattern used throughout this
-- project for optional metadata).
--
-- Run by hand against the running container:
--   docker exec -i thesis_postgres psql -U thesis_user -d financial_db \
--     < infra/postgres/migrations/009_causal_experiment_runs.sql

CREATE TABLE IF NOT EXISTS causal_experiment_runs (
    id                       BIGSERIAL PRIMARY KEY,
    snapshot_version         VARCHAR(100),                 -- set only if run against a frozen causal_panel_snapshot
    source_dataset_version   VARCHAR(100) NOT NULL,         -- snapshot_version echoed, or "live_YYYYMMDD" if build_panel() was called live
    treatment_name           VARCHAR(100) NOT NULL,
    outcome_name             VARCHAR(100) NOT NULL,
    covariates               TEXT         NOT NULL,         -- comma-joined, e.g. "sector,fiscal_year"
    estimator                VARCHAR(50)  NOT NULL,         -- 't_learner' | 'dml'
    nuisance_model           VARCHAR(100),                  -- e.g. 'Ridge(alpha=1.0)' or 'RidgeCV/RidgeCV'
    random_seed              INT,
    bootstrap_iterations     INT,                           -- NULL for 'dml' (analytic CI, no bootstrap)
    min_sample_threshold     INT,                           -- MIN_N_EXPLORATORY in effect for this run (treatment_effect.py)
    code_commit              VARCHAR(64),                   -- best-effort short git SHA, NULL if unavailable
    n                        INT,
    n_treated                INT,                           -- NULL for 'dml' (no treated/control split)
    n_control                INT,                           -- NULL for 'dml'
    point_estimate           DOUBLE PRECISION,
    ci_low                   DOUBLE PRECISION,
    ci_high                  DOUBLE PRECISION,
    confidence_label         TEXT,
    ok                       BOOLEAN      NOT NULL,         -- FALSE for a refused/failed run - see error_message
    error_message            TEXT,
    created_at                TIMESTAMPTZ  NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_causal_experiment_runs_snapshot
    ON causal_experiment_runs (snapshot_version);
CREATE INDEX IF NOT EXISTS idx_causal_experiment_runs_treatment_outcome
    ON causal_experiment_runs (treatment_name, outcome_name);
