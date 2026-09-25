-- Task #5 from the 2026-07-29 architecture review (docs/PROGRESS.md has the
-- full writeup): `predictive_features` - the data layer for a future
-- PREDICTIVE query-routing branch (forecasting a financial metric N fiscal
-- years ahead). See src/predictive/feature_builder.py's module docstring
-- for the full design rationale (why it's built on top of
-- src/causal/panel_builder.py's build_panel_with_exclusions(), how lag
-- features/target/horizon/information_cutoff are computed). Deliberately
-- data-layer only at this stage - NO model training code exists yet, this
-- table is what a future training script would read FROM.
--
-- One row = one (feature_version, ticker, fiscal_year, period,
-- consolidation, target_col, horizon) observation. target_col/horizon are
-- part of the natural key (not just metadata) because the SAME
-- feature_version can hold rows for multiple targets/horizons (e.g. a
-- net_margin-1-year-ahead run and a revenue-2-years-ahead run against the
-- same underlying panel) without them colliding.
--
-- Leakage control, the main design goal here: information_cutoff is the
-- REAL disclosure_date (from `disclosures`, an exact shared key with
-- financial_line_items - both loaded by the same vietstock_etl.py run, so
-- no fuzzy resolver needed here unlike disclosure_chunks.disclosure_id) of
-- the report this row's CURRENT-period fields came from. A chronological
-- train/test split should filter on this DATE, not just fiscal_year -
-- two companies with the same fiscal_year can legitimately disclose months
-- apart, so fiscal_year alone cannot tell you when a row's features were
-- actually public. NULL means "disclosure_date not populated for this
-- filing yet" - treat as unknown/excluded from a cutoff-sensitive split,
-- not "safe to use", same convention as macro_indicators.release_date.
--
-- target_value is NULL for the most recent fiscal_year(s) per ticker,
-- where the future period `horizon` years ahead hasn't been loaded/
-- reported yet - a real "not yet observed", not a data error.
-- eligible_for_training = TRUE iff target_value IS NOT NULL - the only
-- hard gate; lag-feature NaNs (e.g. a ticker's first loaded fiscal year)
-- are left for a training script's own dropna, same graceful-NaN
-- philosophy as src/causal/treatment_effect.py's estimate_ate().
--
-- UNLIKE causal_panel_snapshot (which is written ONCE and never updated -
-- immutable history for a cited result), this table IS expected to be
-- refreshed under the SAME feature_version over time: target_value starts
-- NULL for a ticker's most recent fiscal_year and becomes a real number
-- once that future period is actually loaded - re-running the builder
-- should UPDATE that row in place, not require a new version string. See
-- scripts/build_predictive_features.py's ON CONFLICT DO UPDATE (not DO
-- NOTHING) for this reason - a deliberate, documented difference from the
-- causal snapshot's immutability, not an inconsistency.
--
-- Run by hand against the running container:
--   docker exec -i thesis_postgres psql -U thesis_user -d financial_db \
--     < infra/postgres/migrations/005_predictive_features.sql

CREATE TABLE IF NOT EXISTS predictive_features (
    id                          BIGSERIAL PRIMARY KEY,
    feature_version             VARCHAR(100) NOT NULL,
    ticker                      VARCHAR(16)  NOT NULL,
    fiscal_year                 SMALLINT     NOT NULL,
    period                      VARCHAR(16)  NOT NULL,
    consolidation               VARCHAR(32),
    sector                      VARCHAR(64),
    exchange                    VARCHAR(16),
    -- Current-period raw fields + derived ratios (same set/semantics as
    -- causal_panel_snapshot - see src/causal/panel_builder.py's _PANEL_FIELDS).
    total_assets                NUMERIC,
    total_liabilities           NUMERIC,
    total_equity                NUMERIC,
    current_assets              NUMERIC,
    current_liabilities         NUMERIC,
    inventory                   NUMERIC,
    revenue                     NUMERIC,
    cogs                        NUMERIC,
    gross_profit                NUMERIC,
    profit_before_tax           NUMERIC,
    net_profit                  NUMERIC,
    operating_cash_flow         NUMERIC,
    leverage                    NUMERIC,
    current_ratio               NUMERIC,
    gross_margin                NUMERIC,
    net_margin                  NUMERIC,
    asset_turnover               NUMERIC,
    ocf_to_revenue              NUMERIC,
    -- Lag-1 (prior fiscal year) counterpart of every field/ratio above -
    -- NULL if the ticker has no eligible prior-year row (e.g. its first
    -- loaded fiscal year).
    total_assets_lag1            NUMERIC,
    total_liabilities_lag1       NUMERIC,
    total_equity_lag1            NUMERIC,
    current_assets_lag1          NUMERIC,
    current_liabilities_lag1     NUMERIC,
    inventory_lag1                NUMERIC,
    revenue_lag1                  NUMERIC,
    cogs_lag1                     NUMERIC,
    gross_profit_lag1             NUMERIC,
    profit_before_tax_lag1        NUMERIC,
    net_profit_lag1                NUMERIC,
    operating_cash_flow_lag1       NUMERIC,
    leverage_lag1                  NUMERIC,
    current_ratio_lag1             NUMERIC,
    gross_margin_lag1              NUMERIC,
    net_margin_lag1                NUMERIC,
    asset_turnover_lag1             NUMERIC,
    ocf_to_revenue_lag1             NUMERIC,
    -- YoY growth rates for the 3 most commonly used absolute fields.
    revenue_growth_yoy           NUMERIC,
    net_profit_growth_yoy        NUMERIC,
    total_assets_growth_yoy      NUMERIC,
    -- Macro/market covariates, same set as causal_panel_snapshot.
    cpi_annual_avg               NUMERIC,
    gdp_growth_annual_avg        NUMERIC,
    gold_price_annual_avg        NUMERIC,
    usdvnd_annual_avg            NUMERIC,
    oil_price_annual_avg         NUMERIC,
    -- Leakage control + supervised-learning target (see comment above).
    information_cutoff          DATE,
    target_col                  VARCHAR(100) NOT NULL,
    horizon                     SMALLINT     NOT NULL,
    target_value                NUMERIC,
    eligible_for_training        BOOLEAN      NOT NULL,
    created_at                  TIMESTAMPTZ  NOT NULL DEFAULT now(),
    updated_at                  TIMESTAMPTZ  NOT NULL DEFAULT now(),
    CONSTRAINT uq_predictive_features
        UNIQUE (feature_version, ticker, fiscal_year, period, consolidation, target_col, horizon)
);

CREATE INDEX IF NOT EXISTS idx_predictive_features_version
    ON predictive_features (feature_version);
CREATE INDEX IF NOT EXISTS idx_predictive_features_ticker
    ON predictive_features (ticker);
CREATE INDEX IF NOT EXISTS idx_predictive_features_cutoff
    ON predictive_features (information_cutoff);
