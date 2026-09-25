"""Builds `predictive_features` (infra/postgres/migrations/005_predictive_features.sql,
extended by 008_predictive_features_leakage_and_target_metadata.sql) - a
wide-format, leakage-controlled feature table for the future PREDICTIVE
branch (forecasting a financial metric N fiscal years ahead). See
src/predictive/__init__.py for why this is data-layer-only for now.

Deliberately built ON TOP OF src/causal/panel_builder.py's
build_panel_with_exclusions() rather than re-querying financial_line_items
from scratch - the panel it produces (one row per eligible ticker-fiscal_year,
non-finance-subsector, plausible-range ratios, macro covariates already
joined) is EXACTLY the right cross-sectional base for feature engineering
too; duplicating that pivot/ratio-derivation/macro-join logic here would
risk the two silently drifting apart (e.g. a _RATIO_BOUNDS fix landing in
one copy but not the other). Rows build_panel_with_exclusions() marks
ineligible (finance-subsector, out-of-range ratios - a genuine schema
mismatch, not real signal) are excluded from predictive_features entirely
for the same reason they're excluded from the causal panel: a ratio
computed on bad/incomparable data is not a valid feature either.

Things this module adds on top of the causal panel:
1. Lag-1 features (`<field>_lag1`) - each ticker's PRIOR fiscal year's raw
   fields + ratios, via a self-join shifted by +1 year. Plus 3 YoY growth
   rates (revenue/net_profit/total_assets) computed from current vs lag1.
   Lag features are NOT separately leakage-checked against
   information_cutoff: a ticker's prior-year annual report is, by
   construction, disclosed before its CURRENT year's report (chronological
   filing order), so lag1 fields are safe-by-construction relative to this
   row's own information_cutoff - not re-verified per row, which would be
   redundant DB work for a fact that's already true by how fiscal years and
   filing dates relate to each other.
2. Explicit target_col/horizon/target_value - target_value is the SAME
   ticker's `target_col` measured `horizon` fiscal years in the FUTURE
   relative to each row, via a self-join shifted by -horizon years. NULL
   when that future period hasn't been loaded yet (a real, not-yet-
   observed future - not a data error). `eligible_for_training` is the
   ONLY hard gate here (True iff target_value is not null) - lag-feature
   NaNs (e.g. a ticker's first loaded fiscal year, with no prior year to
   look back to) are left for the model/caller to drop per its own needs,
   same graceful-NaN philosophy as src/causal/treatment_effect.py's
   estimate_ate(). target_observed_at carries the FUTURE row's own
   information_cutoff (disclosure_date) alongside target_value - i.e. the
   date the target was actually known, for the same leakage-audit reason
   information_cutoff exists for features.
3. information_cutoff - the ACTUAL disclosure_date (from the `disclosures`
   table, exact-matched on (ticker, fiscal_year, period, consolidation) -
   a real shared key, unlike disclosure_chunks' pdf_document_id) of the
   report this row's CURRENT-period figures came from. NULL when
   disclosure_date hasn't been populated for that filing - treat as
   "unknown, exclude from a cutoff-sensitive split".
4. (Added 2026-07-29 evening, round-2 review) PER-FEATURE macro leakage
   enforcement, not just a row-level split hint: having information_cutoff
   is necessary but NOT sufficient - src/causal/panel_builder.py joins each
   macro covariate (cpi_annual_avg etc.) onto EVERY row for that fiscal_year
   regardless of when the underlying macro_indicators readings were
   actually published (macro_indicators.release_date, added migration 003).
   An annual CPI average covering all of 2024 is not fully knowable until
   the LAST contributing monthly/quarterly reading for 2024 is released -
   which can be well AFTER a company's Q1 2024 report was disclosed. This
   module now checks, per macro covariate per fiscal_year (memoized, not
   re-queried per row): is the LATEST release_date among all
   macro_indicators rows feeding that year's aggregate <= THIS row's own
   information_cutoff? If not (or if any contributing row's release_date is
   unknown/NULL), the covariate is NULLED OUT for that row rather than kept
   - a conservative default matching this project's other "unknown means
   excluded, not safe" leakage conventions (macro_indicators.release_date's
   own doc comment, disclosure_id/pdf_document_id before it).
5. feature_hash/feature_created_at/source_snapshot_version - lineage/audit
   fields (see migration 008) so a later reader can tell WHEN a feature row
   was computed and detect (via feature_hash) if re-running the builder
   under the SAME feature_version would have produced different feature
   values - relevant because scripts/build_predictive_features.py
   deliberately does NOT overwrite feature columns on conflict (see that
   script's docstring) - feature_hash lets you notice silent drift between
   what's stored and what a fresh build would compute, without needing to
   actually overwrite anything to find out.
"""
import hashlib
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
from sqlalchemy import text

from src.causal.panel_builder import _MACRO_COVARIATES, _PANEL_FIELDS, build_panel_with_exclusions
from src.core.db import get_session

RATIO_COLUMNS = ["leverage", "current_ratio", "gross_margin", "net_margin", "asset_turnover", "ocf_to_revenue"]
RAW_COLUMNS = list(dict.fromkeys(_PANEL_FIELDS.values()))  # preserves definition order, de-duplicated
FEATURE_COLUMNS = RAW_COLUMNS + RATIO_COLUMNS

MACRO_COLUMNS = [
    "cpi_annual_avg", "gdp_growth_annual_avg", "gold_price_annual_avg",
    "usdvnd_annual_avg", "oil_price_annual_avg",
]

# (category, indicator_name_or_pattern, use_like) for each macro covariate -
# mirrors the exact query shapes src/causal/panel_builder.py's
# _MACRO_COVARIATES / src/causal/market_factors.py's gold/forex/oil
# aggregate functions use to COMPUTE the value, so the release-date check
# below looks at the SAME set of contributing rows as the value itself came
# from. cpi/gdp reuse panel_builder._MACRO_COVARIATES directly (single
# source of truth for those 2); gold/forex/oil are duplicated here from
# market_factors.py's SQL constants since those don't expose a
# (category, pattern) tuple as a reusable object, only baked into each
# function's own query string.
_MACRO_RELEASE_SOURCES = {
    "cpi_annual_avg": (*_MACRO_COVARIATES["cpi_annual_avg"], False),
    "gdp_growth_annual_avg": (*_MACRO_COVARIATES["gdp_growth_annual_avg"], False),
    "gold_price_annual_avg": ("MKT_GOLD_SJC", "%sell_price", True),
    "usdvnd_annual_avg": ("MKT_FOREX_VCB", "%USD%sell%", True),
    "oil_price_annual_avg": ("MKT_OIL_WTI", "WTI crude (Cushing, OK)", False),
}

# The 3 absolute (non-ratio) fields worth a YoY growth rate as a feature in
# their own right - revenue/profit/asset-base growth are common predictors
# in the accounting-forecasting literature. Ratios already express a form of
# normalized change year-to-year via their own _lag1 counterpart, so aren't
# duplicated here.
_GROWTH_BASE_COLUMNS = ["revenue", "net_profit", "total_assets"]

FETCH_DISCLOSURE_DATES_SQL = text(
    "SELECT ticker, fiscal_year, period, consolidation, disclosure_date FROM disclosures"
)

_EXACT_RELEASE_DATE_SQL = text(
    "SELECT release_date FROM macro_indicators WHERE category = :cat AND indicator_name = :name AND fiscal_year = :fy"
)
_LIKE_RELEASE_DATE_SQL = text(
    "SELECT release_date FROM macro_indicators WHERE category = :cat AND indicator_name LIKE :pat AND fiscal_year = :fy"
)


def _fetch_disclosure_dates() -> pd.DataFrame:
    """Every (ticker, fiscal_year, period, consolidation) -> disclosure_date
    in `disclosures` - an EXACT shared key with financial_line_items (both
    ETL'd by src/ingestion/vietstock_etl.py from the same source file).
    Fetches the whole table rather than one query per row - `disclosures`
    is small (one row per ticker x fiscal_year x period x consolidation,
    not per line item), so this is cheap and lets pandas do the join in one
    vectorized merge instead of N round-trips."""
    with get_session() as session:
        rows = session.execute(FETCH_DISCLOSURE_DATES_SQL).fetchall()
    return pd.DataFrame(rows, columns=["ticker", "fiscal_year", "period", "consolidation", "disclosure_date"])


def _max_release_date(session, category: str, name_or_pattern: str, fiscal_year: int, use_like: bool):
    """The LATEST release_date among every macro_indicators row feeding
    (category, name_or_pattern, fiscal_year)'s aggregate - or None if there
    are no contributing rows, OR if ANY contributing row has a NULL
    release_date (conservative: can't certify the aggregate was fully known
    by any cutoff if even one contributing reading's publish date is
    unknown). This is a per-(covariate, fiscal_year) computation, safe to
    memoize across all rows sharing that fiscal_year (see the cache in
    build_predictive_features()) - it does NOT depend on any individual
    row's information_cutoff."""
    sql = _LIKE_RELEASE_DATE_SQL if use_like else _EXACT_RELEASE_DATE_SQL
    param_key = "pat" if use_like else "name"
    rows = session.execute(sql, {"cat": category, param_key: name_or_pattern, "fy": fiscal_year}).fetchall()
    if not rows:
        return None
    release_dates = [r[0] for r in rows]
    if any(rd is None for rd in release_dates):
        return None
    return max(release_dates)


def _feature_hash(row: pd.Series, value_columns: list) -> str:
    """SHA-256 over the row's actual feature values (current + lag1 +
    growth + macro, in a fixed column order) - lets a later reader detect
    if a fresh build_predictive_features() call would produce DIFFERENT
    values than what's stored, without needing to overwrite the stored row
    to find out (see module docstring point 5 and
    scripts/build_predictive_features.py's no-overwrite-features policy)."""
    parts = [f"{col}={row[col]!r}" for col in value_columns]
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def build_predictive_features(target_col: str = "net_margin", horizon: int = 1,
                               min_period_type: str = "YEAR", source_snapshot_version: str = None) -> pd.DataFrame:
    """Returns the wide predictive-features DataFrame for one
    (target_col, horizon) combination. Call this once per target/horizon
    you want a training set for - scripts/build_predictive_features.py's
    --target/--horizon flags map straight to these arguments, and the
    resulting rows in `predictive_features` are additionally keyed by
    (target_col, horizon) so multiple targets/horizons can coexist under
    the same feature_version without colliding (see the migration file).

    source_snapshot_version: an optional label recording which
    causal_panel_snapshot (if any) was built from the same underlying panel
    state around the same time - purely a documentation/lineage string, not
    a hard dependency on causal_panel_snapshot existing. Defaults to
    f"live_{today:%Y%m%d}" if not given, marking this as a live pull from
    panel_builder rather than tied to a specific frozen snapshot.
    """
    if target_col not in FEATURE_COLUMNS:
        raise ValueError(
            f"target_col={target_col!r} is not a known feature column. "
            f"Valid options: {FEATURE_COLUMNS}"
        )
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1 fiscal year ahead, got {horizon}")

    if source_snapshot_version is None:
        source_snapshot_version = f"live_{date.today():%Y%m%d}"

    panel = build_panel_with_exclusions(min_period_type=min_period_type)
    if panel.empty:
        return pd.DataFrame()

    base = panel[panel["eligible_sample"]].drop(columns=["eligible_sample", "exclusion_reason"]).reset_index(drop=True)
    if base.empty:
        return pd.DataFrame()

    key_cols = ["ticker", "fiscal_year"]

    # information_cutoff computed FIRST (before lag/target joins) so both
    # the target self-join (for target_observed_at) and the per-feature
    # macro leakage check below can use it.
    disclosure_dates = _fetch_disclosure_dates()
    base = base.merge(
        disclosure_dates, on=["ticker", "fiscal_year", "period", "consolidation"], how="left"
    ).rename(columns={"disclosure_date": "information_cutoff"})

    # Lag-1 features: shift EACH ticker's row fiscal_year forward by +1, so
    # merging back on (ticker, fiscal_year) lands last year's values onto
    # THIS year's row. Safe-by-construction re: leakage (see module docstring).
    lag_source = base[key_cols + FEATURE_COLUMNS].copy()
    lag_source["fiscal_year"] = lag_source["fiscal_year"] + 1
    lag_source = lag_source.rename(columns={c: f"{c}_lag1" for c in FEATURE_COLUMNS})
    merged = base.merge(lag_source, on=key_cols, how="left")

    # Target: shift the target ticker's row fiscal_year BACKWARD by
    # `horizon`, so merging back on (ticker, fiscal_year) lands the FUTURE
    # value (+ the date it was actually disclosed) onto the row that should
    # predict it. NaN (via the left merge) when that future fiscal_year
    # hasn't been loaded/reported yet.
    target_source = base[key_cols + [target_col, "information_cutoff"]].copy()
    target_source["fiscal_year"] = target_source["fiscal_year"] - horizon
    target_source = target_source.rename(columns={target_col: "target_value", "information_cutoff": "target_observed_at"})
    merged = merged.merge(target_source, on=key_cols, how="left")

    for col in _GROWTH_BASE_COLUMNS:
        lag_col = f"{col}_lag1"
        with np.errstate(divide="ignore", invalid="ignore"):
            merged[f"{col}_growth_yoy"] = (merged[col] - merged[lag_col]) / merged[lag_col].replace(0, np.nan)

    # --- Per-feature macro leakage enforcement (point 4 in module docstring) ---
    # Memoized per (covariate, fiscal_year) - NOT per row - since the "latest
    # contributing release_date" is a fact about the YEAR's aggregate, not
    # about any individual ticker's row.
    distinct_years = [int(y) for y in merged["fiscal_year"].dropna().unique()]
    release_cache = {}  # (col, fiscal_year) -> date or None
    with get_session() as session:
        for col, (category, pattern, use_like) in _MACRO_RELEASE_SOURCES.items():
            for y in distinct_years:
                release_cache[(col, y)] = _max_release_date(session, category, pattern, y, use_like)

    def _lookup_release(y, c):
        return release_cache.get((c, int(y))) if pd.notna(y) else None

    for col in MACRO_COLUMNS:
        max_release = merged["fiscal_year"].map(lambda y, c=col: _lookup_release(y, c))
        cutoff_known = merged["information_cutoff"].notna()
        release_known = max_release.notna()
        safe = cutoff_known & release_known & (max_release <= merged["information_cutoff"])
        merged.loc[~safe, col] = np.nan

    merged["target_col"] = target_col
    merged["horizon"] = int(horizon)
    merged["eligible_for_training"] = merged["target_value"].notna()
    merged["feature_created_at"] = datetime.now(timezone.utc)
    merged["source_snapshot_version"] = source_snapshot_version
    merged["target_release_date"] = pd.NaT  # not populated by this builder - see migration 008's comment; reserved for a future target_col that is itself macro-sourced

    hash_columns = FEATURE_COLUMNS + [f"{c}_lag1" for c in FEATURE_COLUMNS] + \
        [f"{c}_growth_yoy" for c in _GROWTH_BASE_COLUMNS] + MACRO_COLUMNS
    merged["feature_hash"] = merged.apply(lambda row: _feature_hash(row, hash_columns), axis=1)

    return merged


if __name__ == "__main__":
    import sys
    target = sys.argv[1] if len(sys.argv) > 1 else "net_margin"
    horizon_arg = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    feats = build_predictive_features(target_col=target, horizon=horizon_arg)
    print(f"predictive_features preview: target={target}, horizon={horizon_arg}, {len(feats)} row(s)")
    if not feats.empty:
        n_trainable = int(feats["eligible_for_training"].sum())
        print(f"  {n_trainable}/{len(feats)} row(s) have a known target_value (eligible_for_training=True)")
        cols = ["ticker", "fiscal_year", target, f"{target}_lag1", "target_value", "information_cutoff"]
        print(feats[cols].head(20).to_string(index=False))
