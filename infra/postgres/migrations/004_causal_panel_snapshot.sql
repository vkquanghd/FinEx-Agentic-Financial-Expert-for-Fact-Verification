-- Task #4 from the 2026-07-29 architecture review (docs/PROGRESS.md has the
-- full writeup): `causal_panel_snapshot` - a FROZEN, citable copy of the
-- company panel used for a specific RQ6a estimate_ate()/estimate_ate_dml()
-- run, for reproducibility.
--
-- WHY this table needs to exist at all, given src/causal/panel_builder.py's
-- build_panel() already computes this exact panel on demand: build_panel()
-- is a LIVE VIEW over financial_line_items/macro_indicators/company_price_
-- snapshot - as more Vietstock exports get loaded (an ongoing process, see
-- docs/PROGRESS.md's many "loaded N more tickers" entries), and as the
-- eligibility rules in panel_builder.py themselves get tuned (e.g.
-- _RATIO_BOUNDS widened, a new finance-subsector keyword added), calling
-- build_panel() again next month will silently return a DIFFERENT n, a
-- different set of tickers, and therefore a different ATE point estimate
-- for the SAME treatment/outcome/covariates arguments. A thesis needs to be
-- able to say "this ATE number came from exactly THIS panel, n=X, these
-- tickers, this build" months after the fact, even after the live tables
-- have moved on - that is what this table freezes.
--
-- One row = one (snapshot_version, ticker, fiscal_year, period,
-- consolidation) observation - the exact wide-panel row
-- src/causal/panel_builder.py's build_panel_with_exclusions() produces,
-- including BOTH eligible rows (what estimate_ate() actually used) and
-- excluded rows (finance-subsector companies, out-of-range ratios), each
-- tagged via eligible_sample/exclusion_reason so "why wasn't ticker X in
-- the sample" is answerable straight from this table later, without
-- re-deriving it against financial_line_items as it exists TODAY (which may
-- have since changed).
--
-- snapshot_version is a free-form string chosen by the caller of
-- scripts/build_causal_panel_snapshot.py (default: "panel_v1_YYYYMMDD",
-- see that script) - NOT auto-derived from a hash of panel_builder.py's
-- source the way CHUNKING_VERSION/EMBEDDING_VERSION are, because unlike
-- those, this snapshot's job is a one-time freeze for a specific reported
-- result, not an ongoing "is this stale, should I recompute" cache check.
-- Multiple snapshot_versions can coexist for the same ticker/fiscal_year -
-- rows are NEVER updated in place once written (see the builder script's
-- ON CONFLICT DO NOTHING, not DO UPDATE - a snapshot is meant to be
-- immutable history, not a live-refreshed cache).
--
-- Run by hand against the running container:
--   docker exec -i thesis_postgres psql -U thesis_user -d financial_db \
--     < infra/postgres/migrations/004_causal_panel_snapshot.sql

CREATE TABLE IF NOT EXISTS causal_panel_snapshot (
    id                      BIGSERIAL PRIMARY KEY,
    snapshot_version        VARCHAR(100) NOT NULL,
    ticker                  VARCHAR(16)  NOT NULL,
    fiscal_year             SMALLINT     NOT NULL,
    period                  VARCHAR(16)  NOT NULL,
    consolidation           VARCHAR(32),
    sector                  VARCHAR(64),
    exchange                VARCHAR(16),
    -- Raw line items feeding the derived ratios below (see
    -- src/causal/panel_builder.py's _PANEL_FIELDS) - kept alongside the
    -- ratios themselves so a future reader can re-derive/sanity-check a
    -- ratio without needing to go back to financial_line_items.
    total_assets            NUMERIC,
    total_liabilities       NUMERIC,
    total_equity            NUMERIC,
    current_assets          NUMERIC,
    current_liabilities     NUMERIC,
    inventory               NUMERIC,
    revenue                 NUMERIC,
    cogs                    NUMERIC,
    gross_profit            NUMERIC,
    profit_before_tax       NUMERIC,
    net_profit              NUMERIC,
    operating_cash_flow     NUMERIC,
    -- Derived ratios (simple accounting identities - see panel_builder.py).
    leverage                NUMERIC,
    current_ratio           NUMERIC,
    gross_margin            NUMERIC,
    net_margin              NUMERIC,
    asset_turnover          NUMERIC,
    ocf_to_revenue          NUMERIC,
    -- Macro/market covariates, joined by fiscal_year (see
    -- src/causal/macro_lookup.py, market_factors.py) - same value for every
    -- company in a given year. gold/forex are "today only" per collection
    -- run and will be sparse (near-all-NULL) for most historical fiscal
    -- years until the market-snapshot DAG has run for a long time - a real
    -- data-availability limit, not a bug in this table.
    cpi_annual_avg          NUMERIC,
    gdp_growth_annual_avg   NUMERIC,
    gold_price_annual_avg   NUMERIC,
    usdvnd_annual_avg       NUMERIC,
    oil_price_annual_avg    NUMERIC,
    -- Eligibility, from build_panel_with_exclusions(): eligible_sample=TRUE
    -- rows are exactly the ones estimate_ate()/estimate_ate_dml() would use
    -- for this snapshot (modulo their own per-call dropna on the specific
    -- treatment/outcome/covariates requested). exclusion_reason is NULL iff
    -- eligible_sample=TRUE.
    eligible_sample         BOOLEAN      NOT NULL,
    exclusion_reason        TEXT,
    created_at              TIMESTAMPTZ  NOT NULL DEFAULT now(),
    CONSTRAINT uq_causal_panel_snapshot
        UNIQUE (snapshot_version, ticker, fiscal_year, period, consolidation)
);

CREATE INDEX IF NOT EXISTS idx_causal_panel_snapshot_version
    ON causal_panel_snapshot (snapshot_version);
CREATE INDEX IF NOT EXISTS idx_causal_panel_snapshot_ticker
    ON causal_panel_snapshot (ticker);
