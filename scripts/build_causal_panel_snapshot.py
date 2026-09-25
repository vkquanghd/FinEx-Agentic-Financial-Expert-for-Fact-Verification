"""Builds and writes a `causal_panel_snapshot` (see
infra/postgres/migrations/004_causal_panel_snapshot.sql) - a FROZEN, citable
copy of the company panel behind a specific RQ6a
estimate_ate()/estimate_ate_dml() result (src/causal/treatment_effect.py).

WHY this script exists given src/causal/panel_builder.py's build_panel()
already computes this panel on demand: build_panel() is a LIVE view over
financial_line_items/macro_indicators/company_price_snapshot. Re-running it
next month, after more Vietstock exports are loaded or after a bound in
_RATIO_BOUNDS is tuned, can silently change n and the resulting ATE for the
exact same treatment/outcome/covariates arguments - see the migration
file's comment for the full rationale. This script freezes ONE call to
build_panel_with_exclusions() (added 2026-07-29 alongside this script) under
a caller-chosen snapshot_version, so a thesis result can point at "exactly
this frozen table, this version string" rather than "whatever build_panel()
happens to return whenever someone re-runs the query".

Writes EVERY row build_panel_with_exclusions() returns - eligible AND
excluded (finance-subsector, out-of-range-ratio) - each tagged via
eligible_sample/exclusion_reason, so "why isn't ticker X in the RQ6a sample"
is answerable from the snapshot table itself, months later, without
re-deriving it against live data.

Snapshots are IMMUTABLE once written: `ON CONFLICT ... DO NOTHING`, not DO
UPDATE - re-running this script with the SAME --snapshot-version is a
harmless no-op for rows already written (not an error), but it will never
silently overwrite a historical snapshot row with new data. To capture a
NEW state of the panel, use a NEW --snapshot-version.

Usage:
    python3 -m scripts.build_causal_panel_snapshot
    python3 -m scripts.build_causal_panel_snapshot --snapshot-version panel_v1_pre_rq6a_writeup
    python3 -m scripts.build_causal_panel_snapshot --min-period-type YEAR
"""
import argparse
import math
from datetime import date

from sqlalchemy import text

from src.causal.panel_builder import build_panel_with_exclusions
from src.core.db import get_session

INSERT_SNAPSHOT_ROW = text("""
    INSERT INTO causal_panel_snapshot
        (snapshot_version, ticker, fiscal_year, period, consolidation, sector, exchange,
         total_assets, total_liabilities, total_equity, current_assets, current_liabilities,
         inventory, revenue, cogs, gross_profit, profit_before_tax, net_profit, operating_cash_flow,
         leverage, current_ratio, gross_margin, net_margin, asset_turnover, ocf_to_revenue,
         cpi_annual_avg, gdp_growth_annual_avg, gold_price_annual_avg, usdvnd_annual_avg, oil_price_annual_avg,
         eligible_sample, exclusion_reason)
    VALUES
        (:snapshot_version, :ticker, :fiscal_year, :period, :consolidation, :sector, :exchange,
         :total_assets, :total_liabilities, :total_equity, :current_assets, :current_liabilities,
         :inventory, :revenue, :cogs, :gross_profit, :profit_before_tax, :net_profit, :operating_cash_flow,
         :leverage, :current_ratio, :gross_margin, :net_margin, :asset_turnover, :ocf_to_revenue,
         :cpi_annual_avg, :gdp_growth_annual_avg, :gold_price_annual_avg, :usdvnd_annual_avg, :oil_price_annual_avg,
         :eligible_sample, :exclusion_reason)
    ON CONFLICT (snapshot_version, ticker, fiscal_year, period, consolidation) DO NOTHING
""")

# Every non-key column in causal_panel_snapshot that comes straight from a
# build_panel_with_exclusions() column of the same name - kept as an
# explicit list (rather than **row.to_dict()) so a future column rename in
# panel_builder.py fails loudly here (KeyError) instead of silently writing
# NULLs into the snapshot.
_VALUE_COLUMNS = [
    "consolidation", "sector", "exchange",
    "total_assets", "total_liabilities", "total_equity", "current_assets", "current_liabilities",
    "inventory", "revenue", "cogs", "gross_profit", "profit_before_tax", "net_profit", "operating_cash_flow",
    "leverage", "current_ratio", "gross_margin", "net_margin", "asset_turnover", "ocf_to_revenue",
    "cpi_annual_avg", "gdp_growth_annual_avg", "gold_price_annual_avg", "usdvnd_annual_avg", "oil_price_annual_avg",
]


def _clean(v):
    """NaN -> None (psycopg2/SQLAlchemy would otherwise try to bind a float
    NaN, which Postgres NUMERIC columns reject) - everything else passed
    through as-is."""
    if isinstance(v, float) and math.isnan(v):
        return None
    return v


def write_snapshot(snapshot_version: str, min_period_type: str = "YEAR") -> dict:
    """Builds the panel (with exclusions) and writes every row into
    causal_panel_snapshot under `snapshot_version`. Returns a small summary
    dict (n_total, n_eligible, n_excluded) rather than just printing, so
    callers (e.g. a future notebook cell) can assert on it."""
    panel = build_panel_with_exclusions(min_period_type=min_period_type)
    if panel.empty:
        print("build_panel_with_exclusions() returned 0 rows - nothing to snapshot. "
              "Check financial_line_items has data loaded (see src/ingestion/vietstock_etl.py).")
        return {"n_total": 0, "n_eligible": 0, "n_excluded": 0}

    n_total = len(panel)
    n_eligible = int(panel["eligible_sample"].sum())
    n_excluded = n_total - n_eligible

    with get_session() as session:
        for _, row in panel.iterrows():
            params = {
                "snapshot_version": snapshot_version,
                "ticker": row["ticker"],
                "fiscal_year": int(row["fiscal_year"]),
                "period": row["period"],
                "eligible_sample": bool(row["eligible_sample"]),
                "exclusion_reason": row["exclusion_reason"],
            }
            for col in _VALUE_COLUMNS:
                params[col] = _clean(row[col])
            session.execute(INSERT_SNAPSHOT_ROW, params)

    print(f"Snapshot '{snapshot_version}': wrote {n_total} row(s) "
          f"({n_eligible} eligible_sample=True, {n_excluded} excluded) into causal_panel_snapshot. "
          f"Re-running with the SAME --snapshot-version is a safe no-op (ON CONFLICT DO NOTHING) - "
          f"use a NEW --snapshot-version to capture a different panel state.")
    return {"n_total": n_total, "n_eligible": n_eligible, "n_excluded": n_excluded}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot-version", default=None,
                     help="Free-form label for this snapshot (default: 'panel_v1_YYYYMMDD', today's date). "
                          "Use a NEW value each time you want to freeze a distinct panel state - re-using an "
                          "existing value is a safe no-op for rows already written, not an overwrite.")
    ap.add_argument("--min-period-type", default="YEAR",
                     help="Passed through to build_panel_with_exclusions() - same period-granularity filter "
                          "as build_panel() (default 'YEAR').")
    args = ap.parse_args()

    snapshot_version = args.snapshot_version or f"panel_v1_{date.today():%Y%m%d}"
    write_snapshot(snapshot_version, min_period_type=args.min_period_type)


if __name__ == "__main__":
    main()
