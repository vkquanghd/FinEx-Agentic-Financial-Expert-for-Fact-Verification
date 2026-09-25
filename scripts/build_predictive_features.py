"""Builds and writes `predictive_features` (see
infra/postgres/migrations/005_predictive_features.sql,
008_predictive_features_leakage_and_target_metadata.sql) via
src/predictive/feature_builder.py's build_predictive_features().

ON CONFLICT policy (revised 2026-07-29 evening, Task #10 from the round-2
review): earlier this same day this script updated EVERY column on
conflict (sector/all feature columns/lag/growth/macro), reasoning that
"predictive_features rows are expected to change over time". A review
correctly pointed out the real risk in that: re-running the full ETL ->
panel -> feature pipeline could silently rewrite HISTORICAL feature values
too (e.g. if an upstream financial_line_items correction lands, or a bug
fix changes a ratio's computation) - meaning a model already trained on
this feature_version could no longer be matched against the exact inputs
it was trained on, with no record that anything changed.

Fixed: ON CONFLICT DO UPDATE now touches ONLY target_value/
target_observed_at/eligible_for_training/updated_at - the columns that are
SUPPOSED to change as more future data loads (a target_value legitimately
starts NULL and becomes real once that future fiscal_year is reported).
Every feature column (current + lag1 + growth + macro), information_cutoff,
feature_created_at, feature_hash, and source_snapshot_version are set ONLY
on the initial INSERT and never touched again under the same
(feature_version, ticker, fiscal_year, period, consolidation, target_col,
horizon) key - a feature row is treated as immutable once written. This is
the "phương án đơn giản" (simple option) the review offered, not the fuller
predictive_feature_snapshot/predictive_targets table split - accepted
trade-off: a genuine upstream correction to feature values requires a NEW
--feature-version to be picked up, it will NOT silently update existing
rows. feature_hash exists precisely so you can tell, later, whether a fresh
build would now compute different values than what's stored (compare the
freshly-built DataFrame's feature_hash against the stored one) without that
comparison ever being automatic/silent.

Usage:
    python3 -m scripts.build_predictive_features
    python3 -m scripts.build_predictive_features --target net_margin --horizon 1
    python3 -m scripts.build_predictive_features --target revenue --horizon 2 --feature-version predfeat_v1
"""
import argparse
import math

from sqlalchemy import text

from src.core.db import get_session
from src.predictive.feature_builder import FEATURE_COLUMNS, build_predictive_features

UPSERT_FEATURE_ROW = text("""
    INSERT INTO predictive_features
        (feature_version, ticker, fiscal_year, period, consolidation, sector, exchange,
         total_assets, total_liabilities, total_equity, current_assets, current_liabilities,
         inventory, revenue, cogs, gross_profit, profit_before_tax, net_profit, operating_cash_flow,
         leverage, current_ratio, gross_margin, net_margin, asset_turnover, ocf_to_revenue,
         total_assets_lag1, total_liabilities_lag1, total_equity_lag1, current_assets_lag1,
         current_liabilities_lag1, inventory_lag1, revenue_lag1, cogs_lag1, gross_profit_lag1,
         profit_before_tax_lag1, net_profit_lag1, operating_cash_flow_lag1, leverage_lag1,
         current_ratio_lag1, gross_margin_lag1, net_margin_lag1, asset_turnover_lag1, ocf_to_revenue_lag1,
         revenue_growth_yoy, net_profit_growth_yoy, total_assets_growth_yoy,
         cpi_annual_avg, gdp_growth_annual_avg, gold_price_annual_avg, usdvnd_annual_avg, oil_price_annual_avg,
         information_cutoff, target_col, horizon, target_value, target_observed_at, target_release_date,
         eligible_for_training, feature_created_at, feature_hash, source_snapshot_version)
    VALUES
        (:feature_version, :ticker, :fiscal_year, :period, :consolidation, :sector, :exchange,
         :total_assets, :total_liabilities, :total_equity, :current_assets, :current_liabilities,
         :inventory, :revenue, :cogs, :gross_profit, :profit_before_tax, :net_profit, :operating_cash_flow,
         :leverage, :current_ratio, :gross_margin, :net_margin, :asset_turnover, :ocf_to_revenue,
         :total_assets_lag1, :total_liabilities_lag1, :total_equity_lag1, :current_assets_lag1,
         :current_liabilities_lag1, :inventory_lag1, :revenue_lag1, :cogs_lag1, :gross_profit_lag1,
         :profit_before_tax_lag1, :net_profit_lag1, :operating_cash_flow_lag1, :leverage_lag1,
         :current_ratio_lag1, :gross_margin_lag1, :net_margin_lag1, :asset_turnover_lag1, :ocf_to_revenue_lag1,
         :revenue_growth_yoy, :net_profit_growth_yoy, :total_assets_growth_yoy,
         :cpi_annual_avg, :gdp_growth_annual_avg, :gold_price_annual_avg, :usdvnd_annual_avg, :oil_price_annual_avg,
         :information_cutoff, :target_col, :horizon, :target_value, :target_observed_at, :target_release_date,
         :eligible_for_training, :feature_created_at, :feature_hash, :source_snapshot_version)
    ON CONFLICT (feature_version, ticker, fiscal_year, period, consolidation, target_col, horizon)
    DO UPDATE SET
        target_value = EXCLUDED.target_value,
        target_observed_at = EXCLUDED.target_observed_at,
        eligible_for_training = EXCLUDED.eligible_for_training,
        updated_at = now()
""")

# Every non-key column that comes straight from a build_predictive_features()
# column of the same name - explicit list (not **row.to_dict()), same
# fail-loudly-on-rename reasoning as build_causal_panel_snapshot.py's
# _VALUE_COLUMNS.
_VALUE_COLUMNS = (
    ["sector", "exchange"]
    + FEATURE_COLUMNS
    + [f"{c}_lag1" for c in FEATURE_COLUMNS]
    + ["revenue_growth_yoy", "net_profit_growth_yoy", "total_assets_growth_yoy"]
    + ["cpi_annual_avg", "gdp_growth_annual_avg", "gold_price_annual_avg", "usdvnd_annual_avg", "oil_price_annual_avg"]
)


def _clean(v):
    """NaN -> None, pandas Timestamp/date -> plain date-compatible value
    passed through as-is (psycopg2 handles datetime.date/Timestamp natively) -
    same NaN-guard as build_causal_panel_snapshot.py's _clean()."""
    if isinstance(v, float) and math.isnan(v):
        return None
    try:
        import pandas as pd
        if v is pd.NaT:
            return None
    except ImportError:
        pass
    return v


def write_features(feature_version: str, target_col: str = "net_margin", horizon: int = 1,
                    min_period_type: str = "YEAR", source_snapshot_version: str = None) -> dict:
    """Builds predictive_features for one (target_col, horizon) and writes
    every row. Returns {"n_total", "n_eligible_for_training"} for callers to
    assert on."""
    feats = build_predictive_features(target_col=target_col, horizon=horizon, min_period_type=min_period_type,
                                       source_snapshot_version=source_snapshot_version)
    if feats.empty:
        print("build_predictive_features() returned 0 rows - nothing to write. "
              "Check financial_line_items has data loaded.")
        return {"n_total": 0, "n_eligible_for_training": 0}

    n_total = len(feats)
    n_eligible = int(feats["eligible_for_training"].sum())

    with get_session() as session:
        for _, row in feats.iterrows():
            params = {
                "feature_version": feature_version,
                "ticker": row["ticker"],
                "fiscal_year": int(row["fiscal_year"]),
                "period": row["period"],
                "consolidation": row["consolidation"],
                "information_cutoff": _clean(row["information_cutoff"]),
                "target_col": row["target_col"],
                "horizon": int(row["horizon"]),
                "target_value": _clean(row["target_value"]),
                "target_observed_at": _clean(row["target_observed_at"]),
                "target_release_date": _clean(row["target_release_date"]),
                "eligible_for_training": bool(row["eligible_for_training"]),
                "feature_created_at": _clean(row["feature_created_at"]),
                "feature_hash": row["feature_hash"],
                "source_snapshot_version": row["source_snapshot_version"],
            }
            for col in _VALUE_COLUMNS:
                params[col] = _clean(row[col])
            session.execute(UPSERT_FEATURE_ROW, params)

    print(f"Feature set '{feature_version}' (target={target_col}, horizon={horizon}): "
          f"wrote {n_total} row(s) ({n_eligible} eligible_for_training=True) into predictive_features. "
          f"Re-running later will only UPDATE target_value/target_observed_at/eligible_for_training for "
          f"existing rows (ON CONFLICT DO UPDATE, restricted set - see this script's module docstring) - "
          f"feature columns are immutable once inserted under this feature_version.")
    return {"n_total": n_total, "n_eligible_for_training": n_eligible}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-version", default="predfeat_v1",
                     help="Label for this feature set (default: 'predfeat_v1'). Unlike "
                          "causal_panel_snapshot's snapshot_version, re-using the SAME value across runs is the "
                          "NORMAL/expected usage here - see this script's module docstring for why.")
    ap.add_argument("--target", default="net_margin",
                     help="Column to forecast (default: net_margin). Must be one of the raw fields or derived "
                          "ratios in src/predictive/feature_builder.py's FEATURE_COLUMNS.")
    ap.add_argument("--horizon", type=int, default=1,
                     help="Fiscal years ahead to forecast --target (default: 1).")
    ap.add_argument("--min-period-type", default="YEAR")
    ap.add_argument("--source-snapshot-version", default=None,
                     help="Optional label recording which causal_panel_snapshot (if any) was built from the "
                          "same underlying panel state - see feature_builder.py's build_predictive_features() "
                          "docstring. Defaults to 'live_YYYYMMDD' if not given.")
    args = ap.parse_args()

    write_features(args.feature_version, target_col=args.target, horizon=args.horizon,
                    min_period_type=args.min_period_type, source_snapshot_version=args.source_snapshot_version)


if __name__ == "__main__":
    main()
