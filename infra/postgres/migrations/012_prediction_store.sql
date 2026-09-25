-- Migration 012 - prediction store ("012-mini" from docs/Thesis_Project_DSA_v5_2026-08-03.pdf
-- section 4, itself the thesis-scope subset of Dataflow FINAL's migration 022_predictions).
--
-- WHY this exists: notebooks/03_ml_task_suite_rag_ready.ipynb cell 37 writes
-- RAG-facing predictions via pandas DataFrame.to_sql("ml_prediction_outputs",
-- if_exists="append") - which lets pandas AUTO-INFER column types with no
-- constraints, no FK, no review. Both Dataflow FINAL (trang 31, "Migration
-- chinh thuc thay pandas auto-create") and Master FINAL (muc 11, "Khong cho
-- pandas tu suy schema prediction output") forbid that. This migration is the
-- official schema that replaces it.
--
-- DESIGN DECISIONS (matching the DSA v5 boundary: thesis-now keeps it small,
-- but every column needed by the future MLOps registry is already here so
-- Layer-3 upgrades never need a breaking ALTER):
--
--   * model_registry_lite is deliberately "lite": it records WHICH
--     champion/challenger was frozen per task (persistence, random_forest...)
--     so every prediction row can point at a real, immutable model identity -
--     NOT the full model_runs/model_versions/model_artifacts/calibration
--     lifecycle of Dataflow FINAL trang 29 (that stays future work; this
--     table maps 1:1 onto its model_versions when migration 021 arrives).
--
--   * ml_prediction_outputs is APPEND-ONLY for prediction values: a re-run
--     with new data INSERTs new rows (new as_of_quarter or new model_id),
--     it never UPDATEs old predictions - forecast history must survive,
--     per Master FINAL muc 11 ("forecast revision khong overwrite lich su").
--     The ONLY columns filled in later are scored_error/scored_at, written
--     once by the thesis_target_scoring Airflow DAG when the target quarter's
--     real value finally arrives (Master FINAL muc 15.2 "Score khi target
--     xuat hien").
--
--   * REAL FKs, not soft text links: model_id -> model_registry_lite,
--     ticker -> companies. This is the direct answer to the "database chua co
--     quan he gi ca" problem - every NEW table from here on ships with its
--     relationships enforced, even while the 13 legacy tables keep their
--     soft links until migrations 013+ (full) land.
--
--   * quarter encoding: *_quarter columns use the notebook's established
--     quarter_index convention (fiscal_year*4 + quarter_num, e.g. 2026Q1 =
--     8105) - INT, monotonic, sortable, already used by
--     artifacts/ml_task_suite/experiment_manifest.json.
--
--   * claim_scope defaults to 'PREDICTIVE_NOT_CAUSAL' and is CHECK-locked to
--     the two values the serving layer understands. The RAG/router branch
--     must render this scope verbatim in answers (DSA v5 section 5).
--
-- Run by hand against the running container:
--   docker exec -i thesis_postgres psql -U thesis_user -d financial_db \
--     < infra/postgres/migrations/012_prediction_store.sql

CREATE TABLE IF NOT EXISTS model_registry_lite (
    id                        BIGSERIAL PRIMARY KEY,
    task                      VARCHAR(100) NOT NULL,   -- e.g. 'net_margin_next_q', 'margin_negative_next_q'
    model_name                VARCHAR(100) NOT NULL,   -- e.g. 'persistence', 'random_forest'
    model_version             VARCHAR(100) NOT NULL,   -- freeze label, e.g. 'v1_frozen_20260803'
    feature_list              JSONB,                   -- exact feature names used (frozen at selection time)
    trained_through_quarter   INT,                     -- last quarter_index whose LABELS were used in training
    validation_metrics        JSONB,                   -- walk-forward validation metrics at freeze time (MAE/PR-AUC/...)
    calibration_method        VARCHAR(50),             -- NULL until calibration done ('platt'/'isotonic'), classifiers only
    is_champion               BOOLEAN      NOT NULL DEFAULT FALSE,
    frozen_at                 TIMESTAMPTZ,             -- when selection was frozen (promotion-gate exit)
    notes                     TEXT,
    created_at                TIMESTAMPTZ  NOT NULL DEFAULT now(),
    CONSTRAINT uq_model_registry_lite UNIQUE (task, model_name, model_version)
);

-- At most ONE champion per task, enforced at the DB level (partial unique
-- index) - promoting a new champion means UPDATE old row is_champion=FALSE
-- then new row TRUE, in one transaction. The registry rows themselves are
-- never deleted (history of past champions stays queryable).
CREATE UNIQUE INDEX IF NOT EXISTS uq_model_registry_lite_champion
    ON model_registry_lite (task) WHERE is_champion;

CREATE TABLE IF NOT EXISTS ml_prediction_outputs (
    id                   BIGSERIAL PRIMARY KEY,
    model_id             BIGINT       NOT NULL REFERENCES model_registry_lite(id),
    ticker               VARCHAR(10)  NOT NULL REFERENCES companies(ticker),
    task                 VARCHAR(100) NOT NULL,
    as_of_quarter        INT          NOT NULL,   -- quarter_index of the features used (information set)
    target_quarter       INT          NOT NULL,   -- quarter_index being predicted
    prediction           NUMERIC,                 -- point prediction (regression) / class label as 0-1 (classification)
    probability          NUMERIC CHECK (probability IS NULL OR (probability >= 0 AND probability <= 1)),
    interval_low         NUMERIC,                 -- reserved for conformal/quantile interval (Layer-3); NULL now
    interval_high        NUMERIC,
    information_cutoff   DATE,                    -- latest disclosure_date feeding the features - leakage audit field
    claim_scope          VARCHAR(40)  NOT NULL DEFAULT 'PREDICTIVE_NOT_CAUSAL'
                         CHECK (claim_scope IN ('PREDICTIVE_NOT_CAUSAL','EXPLORATORY_NOT_SERVED')),
    scored_error         NUMERIC,                 -- |y_true - prediction|, filled ONCE by thesis_target_scoring DAG
    scored_at            TIMESTAMPTZ,             -- when scoring happened (NULL = target not yet observed)
    created_at           TIMESTAMPTZ  NOT NULL DEFAULT now(),
    CONSTRAINT uq_ml_prediction_outputs
        UNIQUE (model_id, ticker, task, as_of_quarter, target_quarter)
);

CREATE INDEX IF NOT EXISTS idx_ml_prediction_outputs_ticker_task
    ON ml_prediction_outputs (ticker, task);
CREATE INDEX IF NOT EXISTS idx_ml_prediction_outputs_target_quarter
    ON ml_prediction_outputs (target_quarter);
CREATE INDEX IF NOT EXISTS idx_ml_prediction_outputs_unscored
    ON ml_prediction_outputs (target_quarter) WHERE scored_at IS NULL;
