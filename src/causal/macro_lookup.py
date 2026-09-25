"""Query helpers over macro_indicators, plus a ticker -> industry-index
mapping used to ground calculator.py's what-if scenarios in real numbers
instead of user-typed hypothetical percentages.

Mapping is deliberately conservative: only mapped where an industry index
genuinely corresponds to the company's actual output (e.g. HPG/HSG make
steel -> "Sản xuất kim loại"). Real-estate developers (VIC/VHM/NVL/KDH)
are left unmapped on purpose - IIP only covers mining/manufacturing/
utilities, not real estate, so forcing a mapping there would be exactly
the kind of unjustified stretch this project has been trying to avoid.
"""
import pandas as pd
from sqlalchemy import text

from src.core.db import get_session

# ticker -> (category, indicator_name) in macro_indicators. Only covers
# CORE_16 tickers with a genuine IIP/RETAIL industry match - extend as
# more companies are added, but don't force a mapping that isn't real.
TICKER_INDUSTRY_MAP: dict[str, tuple[str, str]] = {
    "HPG": ("IIP_BY_INDUSTRY", "Sản xuất kim loại"),
    "HSG": ("IIP_BY_INDUSTRY", "Sản xuất kim loại"),
    "VNM": ("IIP_BY_INDUSTRY", "Sản xuất, chế biến thực phẩm"),
    "MSN": ("IIP_BY_INDUSTRY", "Sản xuất, chế biến thực phẩm"),
    "KDC": ("IIP_BY_INDUSTRY", "Sản xuất, chế biến thực phẩm"),
    "SAB": ("IIP_BY_INDUSTRY", "Sản xuất đồ uống"),
    "DPM": ("IIP_BY_INDUSTRY", "Sản xuất hoá chất và sản phẩm hoá chất"),
    "DCM": ("IIP_BY_INDUSTRY", "Sản xuất hoá chất và sản phẩm hoá chất"),
    "GAS": ("IIP_BY_INDUSTRY", "Sản xuất than cốc, sản phẩm dầu mỏ tinh chế"),
    "MWG": ("RETAIL", "Bán lẻ hàng hóa"),
    "PNJ": ("RETAIL", "Bán lẻ hàng hóa"),
    "DGW": ("RETAIL", "Bán lẻ hàng hóa"),
    "FRT": ("RETAIL", "Bán lẻ hàng hóa"),
    # VIC, VHM, NVL, KDH (real estate): intentionally unmapped - not in IIP scope.
}

# 2026-09-15: ticker -> (category, commodity indicator_name, "export"|
# "import") - the EXPORTS_BY_COMMODITY/IMPORTS_BY_COMMODITY counterpart of
# TICKER_INDUSTRY_MAP above, for calculator.sensitivity_what_if_trade().
# Same conservative philosophy: only mapped where the commodity genuinely
# IS (most of) the company's real output/input, verified by reading the
# actual commodity row names in the ingested Vietstock export files
# (data/raw/vietstock_exports/macro/*XUẤT KHẨU THEO HÀNG HÓA*.xlsx and
# *NHẬP KHẨU THEO HÀNG HÓA*.xlsx) rather than assumed from the category
# name alone - "Sắt thép" and "Phân bón" are both real row labels in those
# files, not invented.
#   HPG/HSG (steel) -> export value of "Sắt thép" as a REVENUE-side
#     macro proxy: both are net exporters of steel products, so a rise in
#     the national steel export figure is directionally a demand/price
#     tailwind for their own revenue - same "revenue driver, not a COGS
#     driver" mechanism as OIL_SENSITIVE_TICKERS' GAS mapping below.
#   DPM/DCM (fertilizer) -> import value of "Phân bón" as a REVENUE-side
#     proxy, but via the OPPOSITE mechanism: rising fertilizer IMPORTS
#     signal import competition against DPM/DCM's own domestic sales, a
#     headwind, not a tailwind - the sign is handled at the calculator
#     level (see sensitivity_what_if_trade()'s docstring), not by flipping
#     the mapping itself. Flagged here as the LESS certain of the two
#     mappings (directionally plausible, not verified against DPM/DCM's
#     actual market-share or contract data) - same honesty standard as
#     OIL_SENSITIVE_TICKERS' own caveat.
TRADE_INDUSTRY_MAP: dict[str, tuple[str, str, str]] = {
    "HPG": ("EXPORTS_BY_COMMODITY", "Sắt thép", "export"),
    "HSG": ("EXPORTS_BY_COMMODITY", "Sắt thép", "export"),
    "DPM": ("IMPORTS_BY_COMMODITY", "Phân bón", "import"),
    "DCM": ("IMPORTS_BY_COMMODITY", "Phân bón", "import"),
}


def trade_index_for_ticker(ticker: str):
    """Return (category, indicator_name, direction) for ticker's mapped
    trade commodity, or None if not mapped (see TRADE_INDUSTRY_MAP)."""
    return TRADE_INDUSTRY_MAP.get(ticker)


_GET_MACRO_VALUE = text("""
    SELECT value FROM macro_indicators
    WHERE category = :category AND indicator_name = :indicator_name
      AND period_type = :period_type AND fiscal_year = :fiscal_year AND period = :period
    LIMIT 1
""")


def get_macro_value(category: str, indicator_name: str, period_type: str, fiscal_year: int, period: str):
    with get_session() as session:
        row = session.execute(_GET_MACRO_VALUE, {
            "category": category, "indicator_name": indicator_name, "period_type": period_type,
            "fiscal_year": fiscal_year, "period": period,
        }).first()
    return float(row[0]) if row and row[0] is not None else None


def macro_pct_change(category: str, indicator_name: str, period_type: str,
                      y1: int, p1: str, y2: int, p2: str) -> dict:
    """Exact % change of a macro indicator between two periods of the SAME
    period_type (don't mix quarterly and annual points of the same series)."""
    v1 = get_macro_value(category, indicator_name, period_type, y1, p1)
    v2 = get_macro_value(category, indicator_name, period_type, y2, p2)
    if v1 is None or v2 is None:
        return {"ok": False, "error": f"Thiếu dữ liệu macro ({category}/{indicator_name}): kỳ1={v1}, kỳ2={v2}"}
    if v1 == 0:
        return {"ok": False, "error": "Giá trị kỳ gốc bằng 0."}
    return {
        "ok": True, "category": category, "indicator_name": indicator_name,
        "v1": v1, "v2": v2, "pct_change": (v2 - v1) / abs(v1) * 100,
    }


def industry_index_for_ticker(ticker: str):
    """Return (category, indicator_name) for ticker's mapped industry
    index, or None if not mapped (see TICKER_INDUSTRY_MAP docstring)."""
    return TICKER_INDUSTRY_MAP.get(ticker)


_GET_YEAR_ROWS = text("""
    SELECT value FROM macro_indicators
    WHERE category = :category AND indicator_name = :indicator_name AND fiscal_year = :fiscal_year
""")


def annual_aggregate(category: str, indicator_name: str, fiscal_year: int):
    """Average all sub-annual data points (quarters or months) reported for
    this indicator in a given year into one annual scalar - used as a
    year-level covariate for the cross-sectional causal panel (every
    company in that year gets the same macro value, which is exactly what
    a shared macro confounder should look like). Returns None if nothing
    is loaded for that year."""
    with get_session() as session:
        rows = session.execute(_GET_YEAR_ROWS, {
            "category": category, "indicator_name": indicator_name, "fiscal_year": fiscal_year,
        }).fetchall()
    values = [float(r[0]) for r in rows if r[0] is not None]
    return sum(values) / len(values) if values else None


_GET_SERIES = text("""
    SELECT fiscal_year, period, value FROM macro_indicators
    WHERE category = :category AND indicator_name = :indicator_name AND period_type = :period_type
    ORDER BY fiscal_year, CAST(SUBSTRING(period, 2) AS INTEGER)
""")
# NOTE: plain "ORDER BY period" would sort 'M10' before 'M2' (string sort) -
# CAST(SUBSTRING(period,2) AS INTEGER) extracts the numeric part ('Q3'->3,
# 'M11'->11) so chronological order is correct. Only valid for MONTH/QUARTER
# period_type, where `period` is a coded string like 'M03'/'Q2' - see
# _GET_SERIES_YEAR below for why 'YEAR' can't use this same query.
# (UNSIGNED -> INTEGER 2026-07-26: UNSIGNED is MySQL-only, not a valid
# Postgres cast target - see docs/PROGRESS.md for the MySQL->Postgres
# migration. INTEGER is signed but every value CAST here comes from a
# period-suffix digit string, always >= 1, so the sign never matters.)

# 2026-09-15 (bugfix, found via Quang's live run of final_benchmark_macro_en.py's
# IIP_BY_INDUSTRY questions - a REAL psycopg2.errors.InvalidTextRepresentation,
# not a sandbox-only limitation): _GET_SERIES's ORDER BY unconditionally runs
# CAST(SUBSTRING(period, 2) AS INTEGER) on EVERY matched row before Postgres
# even gets to decide whether the final order needs it. The old comment above
# said "'YEAR' has no numeric suffix... doesn't matter" - true semantically
# (only 1 row/year, so within-year order is irrelevant) but wrong
# operationally: period_type='YEAR' rows store the literal string 'YEAR' (see
# src/ingestion/macro_etl.py), so SUBSTRING('YEAR', 2) = 'EAR', and
# CAST('EAR' AS INTEGER) raises immediately - the query never returns any
# rows for a YEAR-type series, not even the ones that would sort fine.
# Never caught before because nothing had ever run get_macro_series()/
# latest_two_periods() against real YEAR-type data end-to-end -
# sensitivity_what_if_macro()'s IIP_BY_INDUSTRY path was only reachable via
# app_streamlit.py's manual dropdown before this session's NL-router wiring,
# and classify_intent()'s sandbox regression (this session) only tests
# routing, never touches the DB - so this bug was invisible until Quang
# actually ran a live question. Fixed with a SEPARATE query for
# period_type=='YEAR' that skips the numeric-suffix CAST entirely -
# `ORDER BY fiscal_year` alone is a complete, correct order when there's
# only 1 row per year (the exact case the removed comment already assumed).
_GET_SERIES_YEAR = text("""
    SELECT fiscal_year, period, value FROM macro_indicators
    WHERE category = :category AND indicator_name = :indicator_name AND period_type = :period_type
    ORDER BY fiscal_year
""")


def get_macro_series(category: str, indicator_name: str, period_type: str) -> pd.DataFrame:
    query = _GET_SERIES_YEAR if period_type == "YEAR" else _GET_SERIES
    with get_session() as session:
        rows = session.execute(query, {
            "category": category, "indicator_name": indicator_name, "period_type": period_type,
        }).fetchall()
    return pd.DataFrame(rows, columns=["fiscal_year", "period", "value"])


def latest_two_periods(category: str, indicator_name: str, period_type: str):
    """Return ((y1, p1), (y2, p2)) for the most recent 2 loaded data points
    of this series, or None if fewer than 2 are loaded. Used so callers
    (calculator.sensitivity_what_if_macro) don't have to guess which years/
    months actually have data in the current export."""
    series = get_macro_series(category, indicator_name, period_type)
    if len(series) < 2:
        return None
    last_two = series.tail(2)
    (y1, p1), (y2, p2) = last_two[["fiscal_year", "period"]].to_numpy()
    return (int(y1), str(p1)), (int(y2), str(p2))


_LATEST_DEPOSIT_PERIOD = text("""
    SELECT fiscal_year, period FROM macro_indicators
    WHERE category = 'DEPOSIT_RATES'
    ORDER BY fiscal_year DESC, period DESC
    LIMIT 1
""")

_DEPOSIT_RATES_FOR_PERIOD = text("""
    SELECT indicator_name, value, unit FROM macro_indicators
    WHERE category = 'DEPOSIT_RATES' AND fiscal_year = :fiscal_year AND period = :period
    ORDER BY value DESC
    LIMIT :top_n
""")


def best_deposit_rate_banks(top_n: int = 10) -> dict:
    """Ranks banks in macro_indicators' DEPOSIT_RATES category (Vietstock
    "LÃI SUẤT TIỀN GỬI NGÂN HÀNG" export, src.ingestion.macro_etl - one
    `indicator_name` per bank code, e.g. TCB/VIB/VCB/CTG) by their most
    recently reported deposit rate, descending.

    2026-09-15, added in response to Quang's request for a bank-interest
    recommendation "based on the bank's own characteristics" - this
    answers the part that's actually backed by real data (which bank
    CURRENTLY reports the best deposit rate, market-wide) but explicitly
    does NOT answer "which bank should ticker X use" - no data anywhere in
    this project records which bank any given company actually borrows
    from or deposits with, so that half of the request cannot be answered
    without inventing a relationship that doesn't exist in the source
    data. Kept as its own ticker-independent lookup (branch
    "bank_rate_lookup", src/router/rules.py) rather than folded into any
    per-ticker WHATIF, for the same reason auditor_lookup.py's reverse
    lookup is its own branch: an aggregate-across-the-whole-table query
    is a structurally different shape from a per-ticker one.
    """
    with get_session() as session:
        latest = session.execute(_LATEST_DEPOSIT_PERIOD).first()
        if latest is None:
            return {"ok": False, "error": "Chưa có dữ liệu DEPOSIT_RATES - chạy src.ingestion.macro_etl trước."}
        fiscal_year, period = latest
        rows = session.execute(_DEPOSIT_RATES_FOR_PERIOD, {
            "fiscal_year": fiscal_year, "period": period, "top_n": top_n,
        }).fetchall()
    return {
        "ok": True, "fiscal_year": int(fiscal_year), "period": str(period),
        "banks": [{"bank": r[0], "rate": float(r[1]), "unit": r[2]} for r in rows],
        "note": "Xếp hạng theo lãi suất tiền gửi CÔNG BỐ (Vietstock), không phải khuyến nghị "
                "vay/gửi cho một công ty cụ thể - không có dữ liệu công ty nào đang giao dịch "
                "với ngân hàng nào trong project này.",
    }


if __name__ == "__main__":
    import json
    import sys

    ticker = sys.argv[1] if len(sys.argv) > 1 else "HSG"
    mapped = industry_index_for_ticker(ticker)
    print(f"{ticker} -> {mapped}")
    if mapped:
        category, indicator_name = mapped
        # IIP_BY_INDUSTRY / CONSUMPTION_BY_INDUSTRY / INVENTORY_BY_INDUSTRY are
        # annual-only (2021-2024 in the current export); RETAIL is monthly.
        period_type = "YEAR" if category == "IIP_BY_INDUSTRY" else "MONTH"
        series = get_macro_series(category, indicator_name, period_type)
        print(series.to_string(index=False))
