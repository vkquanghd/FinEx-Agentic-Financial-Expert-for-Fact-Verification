"""Build a wide company-period panel from financial_line_items, for the
cross-sectional causal analysis in treatment_effect.py.

One row = one (ticker, fiscal_year, period, consolidation) observation,
with core balance-sheet/income-statement figures as columns, plus derived
ratios (leverage, margins, turnover) computed from them. This is the
"many units, few periods each" panel shape - not a single-company time
series - which is what makes cross-sectional causal methods applicable
here (see src/causal/__init__.py for the full rationale).
"""
import pandas as pd
from sqlalchemy import text

from src.causal.macro_lookup import annual_aggregate
from src.core.constants import FINANCE_SUBSECTOR_KEYWORDS
from src.core.db import get_session

# Macro covariates joined onto the panel by fiscal_year only - same value
# for every company in a given year, exactly what a shared macro
# confounder should look like (see macro_lookup.annual_aggregate).
_MACRO_COVARIATES = {
    "cpi_annual_avg": ("CPI", "CHỈ SỐ GIÁ TIÊU DÙNG"),
    "gdp_growth_annual_avg": ("GDP_REAL_GROWTH", "TỔNG SỐ"),
}

# Plausible range for each ratio - values outside this are treated as a
# schema mismatch (e.g. banks/securities/insurance report "revenue" and
# "COGS" completely differently under Circular 200's MS codes, producing
# nonsense ratios like a 2000%+ net margin) rather than a real outlier,
# and are dropped rather than fed into the T-learner. Bounds are generous
# on purpose - real distressed companies can have large negative margins.
_RATIO_BOUNDS = {
    "leverage": (0, 3),
    "current_ratio": (0, 50),
    "gross_margin": (-5, 5),
    "net_margin": (-5, 5),
    "asset_turnover": (0, 10),
    "ocf_to_revenue": (-5, 5),
}

# (statement_type, ms_code) -> short column name, for the fields the
# derived ratios below need. Kept small and explicit rather than pivoting
# all ~48 CORE_MS_CODES, since only these feed the current ratio set.
_PANEL_FIELDS = {
    ("balance_sheet", "280"): "total_assets",
    ("balance_sheet", "300"): "total_liabilities",
    ("balance_sheet", "400"): "total_equity",
    ("balance_sheet", "100"): "current_assets",
    ("balance_sheet", "310"): "current_liabilities",
    ("balance_sheet", "140"): "inventory",
    ("income_statement", "10"): "revenue",
    ("income_statement", "11"): "cogs",
    ("income_statement", "20"): "gross_profit",
    ("income_statement", "50"): "profit_before_tax",
    ("income_statement", "60"): "net_profit",
    ("cash_flow_indirect", "20"): "operating_cash_flow",
}

def _fetch_raw() -> pd.DataFrame:
    keys = list(_PANEL_FIELDS.keys())
    with get_session() as session:
        # Query one statement_type group at a time rather than a single
        # composite "(statement_type, ms_code) IN (...)" clause - this was
        # originally chosen because MySQL+SQLAlchemy's handling of a bound
        # tuple-of-tuples IN clause via text() was unreliable; kept unchanged
        # after the 2026-07-26 Postgres migration (same safe per-group query
        # shape works correctly under psycopg2 too, and this was not
        # re-tested against a real Postgres server - see docs/PROGRESS.md
        # verification-gap notes - so there was no reason to risk changing
        # working, if slightly more verbose, code during the migration).
        rows = []
        for stype in {k[0] for k in keys}:
            codes = [k[1] for k in keys if k[0] == stype]
            q = text("""
                SELECT f.ticker, f.fiscal_year, f.period, f.consolidation,
                       f.statement_type, f.ms_code, f.value,
                       c.sector, c.exchange, c.company_name
                FROM financial_line_items f
                JOIN companies c ON c.ticker = f.ticker
                WHERE f.statement_type = :stype AND f.ms_code IN :codes
            """).bindparams(codes=tuple(codes))
            rows.extend(session.execute(q, {"stype": stype, "codes": tuple(codes)}).fetchall())
    return pd.DataFrame(rows, columns=[
        "ticker", "fiscal_year", "period", "consolidation",
        "statement_type", "ms_code", "value", "sector", "exchange", "company_name",
    ])


def _join_macro_covariates(wide: pd.DataFrame) -> pd.DataFrame:
    """Adds the macro/market covariate columns (cpi, gdp growth, gold,
    forex, oil) to `wide` in place (by fiscal_year) - factored out
    2026-07-29 so build_panel() and build_panel_with_exclusions() (added
    for Task #4's causal_panel_snapshot, see docs/PROGRESS.md) share
    exactly one join implementation instead of two copies that could drift.
    """
    # Join macro covariates by fiscal_year (one lookup per distinct year,
    # not per row). Missing years (macro export not loaded / doesn't cover
    # that year yet) become NaN - estimate_ate() already drops NaN rows for
    # whichever covariates it's asked to use, so this degrades gracefully.
    for col, (category, indicator_name) in _MACRO_COVARIATES.items():
        year_values = {y: annual_aggregate(category, indicator_name, int(y)) for y in wide["fiscal_year"].unique()}
        wide[col] = wide["fiscal_year"].map(year_values)

    # Gold price covariate (2026-07-19) - joined separately from the loop
    # above since gold has no single indicator_name to plug into that
    # (category, indicator_name) pattern (see market_factors.py's module
    # docstring for why - fetch_gold() writes one row per branch/type, not
    # one canonical series). Coverage will be VERY sparse (near-all-NaN)
    # until the thesis_market_snapshot Airflow DAG has been running for a
    # full year, since fetch_gold() can only ever capture "today" - this is
    # a real, documented data-availability limit, not a bug. NaN rows are
    # already handled gracefully: estimate_ate() drops them for whichever
    # covariates it's asked to use, so this is harmless to include now and
    # will fill in on its own as more daily runs accumulate.
    from src.causal.market_factors import forex_annual_aggregate, gold_annual_aggregate, oil_annual_aggregate
    gold_year_values = {y: gold_annual_aggregate(int(y)) for y in wide["fiscal_year"].unique()}
    wide["gold_price_annual_avg"] = wide["fiscal_year"].map(gold_year_values)

    # Forex (USD/VND) and oil (WTI) covariates (2026-07-19) - same
    # data-availability caveat as gold: forex is "today only" per run
    # (very sparse); oil DOES backfill a real range from FRED but only if
    # FRED_API_KEY is set (see market_factors.py's module docstring) -
    # both harmless to include now (NaN-dropped gracefully by
    # estimate_ate()) even if currently near-empty for most fiscal years.
    forex_year_values = {y: forex_annual_aggregate(int(y)) for y in wide["fiscal_year"].unique()}
    wide["usdvnd_annual_avg"] = wide["fiscal_year"].map(forex_year_values)

    oil_year_values = {y: oil_annual_aggregate(int(y)) for y in wide["fiscal_year"].unique()}
    wide["oil_price_annual_avg"] = wide["fiscal_year"].map(oil_year_values)

    return wide


def _build_wide_all(min_period_type: str = "YEAR") -> pd.DataFrame:
    """Pivot + derive-ratios logic shared by build_panel() and
    build_panel_with_exclusions() (added 2026-07-29 for Task #4's
    causal_panel_snapshot - see docs/PROGRESS.md). Returns EVERY
    ticker-period row unconditionally, including finance-subsector
    companies and rows with out-of-range ratio values, tagged with
    `is_finance_subsector` (bool) and `_ratio_bound_violations` (list of
    ratio column names that failed their _RATIO_BOUNDS check) rather than
    silently dropping them - build_panel() then filters this down to the
    eligible subset (unchanged public behavior/columns), while
    build_panel_with_exclusions() keeps everything so a snapshot can record
    WHY a row was excluded, not just that it was."""
    raw = _fetch_raw()
    if raw.empty:
        return pd.DataFrame()

    raw = raw[raw["period"] == min_period_type].copy()
    if raw.empty:
        return pd.DataFrame()

    raw["field"] = list(zip(raw["statement_type"], raw["ms_code"]))
    raw["field"] = raw["field"].map(_PANEL_FIELDS)
    raw["value"] = pd.to_numeric(raw["value"], errors="coerce")

    # company_name included here (unlike the pre-2026-07-29 idx_cols) only
    # to compute is_finance_subsector per ticker-period below - it's a
    # per-ticker constant (via the companies join in _fetch_raw), so
    # including it cannot split/duplicate any ticker-period group. Dropped
    # again before build_panel() returns, to keep its output columns
    # byte-for-byte identical to before this refactor.
    idx_cols = ["ticker", "fiscal_year", "period", "consolidation", "sector", "exchange", "company_name"]
    wide = raw.pivot_table(
        index=idx_cols, columns="field", values="value", aggfunc="first"
    ).reset_index()
    wide.columns.name = None

    for col in _PANEL_FIELDS.values():
        if col not in wide.columns:
            wide[col] = pd.NA

    # Derived ratios (all simple accounting identities, no estimation).
    #
    # Neo literature (thêm 2026-09-12, vá khoảng trống "không công thức nào
    # được neo vào literature" - docs/RQ_audit_truoc_khi_training_2026-08-14.md
    # dòng 39-43): đây là các tỷ số kế toán CHUẨN (DuPont-style), không phải
    # công thức tự đặt - nhưng để đúng quy ước "reference phải từ 2023 tới
    # nay" của project (xem docs/PROGRESS.md mục Myers 1984/Abdeljawad &
    # Jaradat 2025), neo bằng 2 bài 2023-2025 CÙNG dùng đúng các tỷ số này
    # trên panel doanh nghiệp niêm yết Việt Nam (không phải "phát minh ra
    # công thức", mà là bằng chứng các tỷ số này vẫn đang được literature
    # gần đây dùng cho đúng thị trường/bối cảnh của project này):
    #   - leverage = total_liabilities/total_assets, và nhóm tỷ số thanh
    #     khoản/khả năng sinh lời cùng cấu trúc: Tran, T., Nguyen, N. H., Le,
    #     B. T., Thanh Vu, N., & Vo, D. H. (2023). "Examining financial
    #     distress of the Vietnamese listed firms using accounting-based
    #     models." PLOS ONE, 18(5), e0284451.
    #     https://doi.org/10.1371/journal.pone.0284451 - 500 DN niêm yết
    #     HOSE/HNX, dùng đúng 3 nhóm chỉ số (liquidity/profitability/leverage)
    #     panel_builder.py đang tính.
    #   - asset_turnover, ocf_to_revenue (hiệu quả sử dụng tài sản/dòng tiền
    #     trên doanh thu): Huynh, T. X. T., Nguyen, T. T. H., & Nguyen, C. V.
    #     (2025). "The impact of working capital management on the financial
    #     performance of listed enterprises: an empirical evidence from
    #     Vietnam." Cogent Business & Management, 12(1), 2473033.
    #     https://doi.org/10.1080/23311975.2025.2473033 - 441 DN niêm yết
    #     Việt Nam 2012-2022, đo hiệu quả vòng quay/dòng tiền vận hành lên
    #     hiệu quả tài chính.
    # DuPont/khung phân rã gốc (Ross-Westerfield-Jaffe-Jordan, "Corporate
    # Finance") vẫn được NHẮC TÊN như nguồn gốc thuật ngữ (như Myers 1984 với
    # pecking-order) - không dùng làm trích dẫn bằng chứng chính vì ngoài
    # khung 2023+.
    wide["leverage"] = wide["total_liabilities"] / wide["total_assets"]
    wide["current_ratio"] = wide["current_assets"] / wide["current_liabilities"]
    wide["gross_margin"] = wide["gross_profit"] / wide["revenue"]
    wide["net_margin"] = wide["net_profit"] / wide["revenue"]
    wide["asset_turnover"] = wide["revenue"] / wide["total_assets"]
    wide["ocf_to_revenue"] = wide["operating_cash_flow"] / wide["revenue"]

    # Banks/securities/insurance don't follow the same Circular 200
    # line-item semantics ("revenue"/"COGS" mean something different), so
    # ratios computed on them are not comparable to non-financial companies
    # (see FINANCE_SUBSECTOR_KEYWORDS in src/core/constants.py).
    wide["is_finance_subsector"] = wide["company_name"].str.contains(
        "|".join(FINANCE_SUBSECTOR_KEYWORDS), case=False, na=False
    )

    # Per-row list of which _RATIO_BOUNDS columns are implausible (schema
    # mismatch, not a real outlier - see _RATIO_BOUNDS's own comment). NaN
    # values pass (not a violation) - only genuinely out-of-range numbers do.
    def _violations(row) -> list:
        return [c for c, (lo, hi) in _RATIO_BOUNDS.items() if pd.notna(row[c]) and not (lo <= row[c] <= hi)]

    wide["_ratio_bound_violations"] = wide.apply(_violations, axis=1)

    return wide


def build_panel(min_period_type: str = "YEAR") -> pd.DataFrame:
    """Return the wide panel DataFrame, filtered to eligible rows only
    (finance-subsector companies and out-of-range-ratio rows dropped) -
    same public contract/columns as before the 2026-07-29 refactor that
    factored the shared pivot/ratio logic out into _build_wide_all().
    min_period_type filters to a single period granularity (default 'YEAR')
    so rows are comparable across companies that report quarterly vs
    annually only - mixing period types in the same cross-section would
    bias comparisons.
    """
    wide = _build_wide_all(min_period_type)
    if wide.empty:
        return wide

    n_all = len(wide)
    eligible_mask = ~wide["is_finance_subsector"] & (wide["_ratio_bound_violations"].map(len) == 0)
    n_finance = int(wide["is_finance_subsector"].sum())
    n_ratio_only = int((~wide["is_finance_subsector"] & (wide["_ratio_bound_violations"].map(len) > 0)).sum())

    wide = wide[eligible_mask].drop(columns=["is_finance_subsector", "_ratio_bound_violations", "company_name"]).reset_index(drop=True)

    if n_finance or n_ratio_only:
        print(
            f"[panel_builder] excluded {n_finance} finance-subsector ticker-period row(s) "
            f"and {n_ratio_only}/{n_all} additional ticker-period row(s) with out-of-range ratios "
            f"(likely schema mismatch, not real data)."
        )

    return _join_macro_covariates(wide)


def build_panel_with_exclusions(min_period_type: str = "YEAR") -> pd.DataFrame:
    """Like build_panel(), but returns EVERY ticker-period row (eligible AND
    excluded), with two extra columns: `eligible_sample` (bool - would this
    row be in build_panel()'s output?) and `exclusion_reason` (None if
    eligible, else a short machine-readable string e.g.
    'finance_subsector', 'ratio_out_of_bounds:leverage', or
    'ratio_out_of_bounds:leverage,net_margin' for multiple violations).

    Added 2026-07-29 (Task #4, `causal_panel_snapshot` - see
    docs/PROGRESS.md) specifically so a persisted snapshot of the panel used
    for an RQ6a estimate_ate() run can be audited later: "why wasn't ticker
    X, year Y in the sample" should be answerable from the snapshot table
    itself, not require re-deriving it from live financial_line_items
    (which may have since changed) or re-reading this file's source."""
    wide = _build_wide_all(min_period_type)
    if wide.empty:
        return wide

    def _reason(row) -> str:
        if row["is_finance_subsector"]:
            return "finance_subsector"
        if row["_ratio_bound_violations"]:
            return "ratio_out_of_bounds:" + ",".join(row["_ratio_bound_violations"])
        return None

    wide["exclusion_reason"] = wide.apply(_reason, axis=1)
    wide["eligible_sample"] = wide["exclusion_reason"].isna()
    wide = wide.drop(columns=["is_finance_subsector", "_ratio_bound_violations", "company_name"]).reset_index(drop=True)

    return _join_macro_covariates(wide)


def load_panel_from_snapshot(snapshot_version: str, eligible_only: bool = True) -> pd.DataFrame:
    """Reads a FROZEN panel back out of causal_panel_snapshot (written by
    scripts/build_causal_panel_snapshot.py) instead of recomputing it live
    from financial_line_items/macro_indicators - added 2026-07-29 evening
    for Task #11 (causal_experiment_runs, see docs/PROGRESS.md), so
    scripts/run_causal_experiment.py can point estimate_ate()/
    estimate_ate_dml() at an exact historical panel state via
    --snapshot-version instead of always calling build_panel() live.

    Returns the same column shape build_panel() does (drops
    eligible_sample/exclusion_reason when eligible_only=True, matching
    build_panel()'s public contract) so it's a drop-in `panel=` argument
    for either estimator function. eligible_only=False returns every row
    (including excluded ones, each still tagged) for callers that want to
    audit the frozen sample itself rather than run an estimate on it."""
    with get_session() as session:
        result = session.execute(
            text("SELECT * FROM causal_panel_snapshot WHERE snapshot_version = :v"),
            {"v": snapshot_version},
        )
        columns = list(result.keys())
        rows = result.fetchall()
    df = pd.DataFrame(rows, columns=columns)
    if df.empty:
        return df
    if eligible_only:
        df = df[df["eligible_sample"]].drop(columns=["eligible_sample", "exclusion_reason"]).reset_index(drop=True)
    return df


if __name__ == "__main__":
    panel = build_panel()
    print(f"Panel shape: {panel.shape[0]} ticker-period rows, {panel.shape[1]} columns")
    if not panel.empty:
        print(panel[["ticker", "fiscal_year", "sector", "leverage", "net_margin"]].head(20).to_string(index=False))
