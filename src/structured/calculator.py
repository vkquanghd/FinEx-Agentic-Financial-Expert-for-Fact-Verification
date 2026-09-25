"""Deterministic what-if / decomposition calculator extending the DAG path.

Answers "intervention-style" questions (growth rate, bridge/variance
decomposition, ceteris-paribus sensitivity) using exact accounting
identities on data already in financial_line_items - NOT a statistical or
ML estimate. No training, no uncertainty, fully traceable to the formula
used. This is the right tool for questions like "how much of the margin
change came from revenue growth vs cost efficiency?" or "if COGS rose 5%,
what would gross profit be?" - they don't need causal inference, they need
arithmetic on numbers the company already reported.

For genuinely causal questions (does leverage affect margin ACROSS
companies, controlling for confounders) see src/causal/treatment_effect.py
instead - that needs statistical estimation because it's not a bridge
between two known numbers, it's an inference about an unobserved
counterfactual across the company panel.
"""
from dataclasses import dataclass
from typing import Optional

from sqlalchemy import text

from src.causal.macro_lookup import (
    industry_index_for_ticker,
    latest_two_periods,
    macro_pct_change,
    trade_index_for_ticker,
)
from src.core.db import get_session

# Tickers whose COGS is genuinely gold-driven (jewelry/bullion retailers -
# gold is their literal raw material), used by sensitivity_what_if_gold()
# below. Deliberately small and explicit, same conservative philosophy as
# TICKER_INDUSTRY_MAP in src/causal/macro_lookup.py - PNJ is the one CORE_16
# company where this is genuinely true; don't add a ticker here just
# because it happens to be affected by gold sentiment indirectly.
GOLD_SENSITIVE_TICKERS = {"PNJ"}

# Tickers whose REVENUE (not COGS) is genuinely oil-price-linked, used by
# sensitivity_what_if_oil() below. GAS (Tổng Công ty Khí Việt Nam - PV
# Gas) sells natural gas under contracts commonly benchmarked to oil-
# equivalent pricing formulas in Vietnam - a rise in oil price is a
# revenue tailwind for GAS, the OPPOSITE mechanism from PNJ/gold (where
# gold is a cost input, not a revenue driver). This is a directionally
# well-established relationship (GAS's own business is literally
# oil/gas), but the exact pricing-formula terms of GAS's actual contracts
# were not verified against a primary source here - treat the resulting
# scenario as illustrative of the mechanism, not a precise contractual
# pass-through rate.
OIL_SENSITIVE_TICKERS = {"GAS"}

_GET_VALUE = text("""
    SELECT value FROM financial_line_items
    WHERE ticker = :ticker AND fiscal_year = :fiscal_year AND period = :period
      AND statement_type = :statement_type AND ms_code = :ms_code
    LIMIT 1
""")


def _get_value(ticker: str, statement_type: str, ms_code: str, fiscal_year: int, period: str) -> Optional[float]:
    with get_session() as session:
        row = session.execute(_GET_VALUE, {
            "ticker": ticker, "fiscal_year": fiscal_year, "period": period,
            "statement_type": statement_type, "ms_code": ms_code,
        }).first()
    return float(row[0]) if row and row[0] is not None else None


@dataclass
class Period:
    fiscal_year: int
    period: str


def growth_rate(ticker: str, statement_type: str, ms_code: str, p1: Period, p2: Period) -> dict:
    """Exact % change of one indicator between two periods."""
    v1 = _get_value(ticker, statement_type, ms_code, p1.fiscal_year, p1.period)
    v2 = _get_value(ticker, statement_type, ms_code, p2.fiscal_year, p2.period)
    if v1 is None or v2 is None:
        return {"ok": False, "error": f"Thiếu dữ liệu ({ticker}, {statement_type}/{ms_code}): "
                                       f"kỳ1={v1}, kỳ2={v2}"}
    if v1 == 0:
        return {"ok": False, "error": "Giá trị kỳ gốc bằng 0, không tính được % tăng trưởng."}
    pct = (v2 - v1) / abs(v1) * 100
    return {
        "ok": True, "ticker": ticker, "v1": v1, "v2": v2, "pct_change": pct,
        "formula": "(v2 - v1) / |v1| * 100",
    }


def decompose_gross_margin_change(ticker: str, p1: Period, p2: Period) -> dict:
    """Bridge/variance decomposition of the change in gross profit between
    two periods into a "revenue growth effect" and a "margin-rate
    (cost efficiency) effect" - exact algebraic identity:

        delta_GP = delta_Rev * GM1  +  Rev2 * delta_GM

    (revenue-growth effect)   (margin-rate effect, i.e. cost-side driver)

    This is what a "70% from cost reduction, 30% from revenue growth"
    style statement actually is in financial analysis - no ML/causal
    model involved, exact given the two periods' reported numbers.
    """
    rev1 = _get_value(ticker, "income_statement", "10", p1.fiscal_year, p1.period)
    rev2 = _get_value(ticker, "income_statement", "10", p2.fiscal_year, p2.period)
    gp1 = _get_value(ticker, "income_statement", "20", p1.fiscal_year, p1.period)
    gp2 = _get_value(ticker, "income_statement", "20", p2.fiscal_year, p2.period)

    missing = [n for n, v in [("revenue kỳ1", rev1), ("revenue kỳ2", rev2),
                               ("gross profit kỳ1", gp1), ("gross profit kỳ2", gp2)] if v is None]
    if missing:
        return {"ok": False, "error": f"Thiếu dữ liệu: {missing}"}
    if rev1 == 0 or rev2 == 0:
        return {"ok": False, "error": "Doanh thu bằng 0 ở 1 trong 2 kỳ, không decompose được."}

    gm1, gm2 = gp1 / rev1, gp2 / rev2
    delta_rev, delta_gm = rev2 - rev1, gm2 - gm1
    revenue_effect = delta_rev * gm1
    margin_rate_effect = rev2 * delta_gm
    delta_gp = gp2 - gp1
    total_effect = revenue_effect + margin_rate_effect  # should equal delta_gp exactly

    def _share(x):
        return None if delta_gp == 0 else round(x / delta_gp * 100, 1)

    return {
        "ok": True, "ticker": ticker,
        "gross_profit_p1": gp1, "gross_profit_p2": gp2, "delta_gross_profit": delta_gp,
        "gross_margin_p1_pct": round(gm1 * 100, 2), "gross_margin_p2_pct": round(gm2 * 100, 2),
        "revenue_growth_effect": revenue_effect, "revenue_growth_effect_share_pct": _share(revenue_effect),
        "margin_rate_effect": margin_rate_effect, "margin_rate_effect_share_pct": _share(margin_rate_effect),
        "check_sums_to_delta_gp": round(total_effect - delta_gp, 6) == 0,
        "formula": "delta_GP = delta_Rev * GM1 + Rev2 * delta_GM (exact bridge/variance decomposition)",
    }


def sensitivity_what_if(ticker: str, p: Period, cogs_pct_change: float = 0.0, revenue_pct_change: float = 0.0) -> dict:
    """Ceteris-paribus projection: "if COGS changes by X% and revenue by
    Y%, all else equal, what would gross profit/margin be?" This is a
    what-if SCENARIO calculation, not a causal forecast - it explicitly
    assumes everything else in the period stays fixed, and says so.
    """
    rev = _get_value(ticker, "income_statement", "10", p.fiscal_year, p.period)
    cogs = _get_value(ticker, "income_statement", "11", p.fiscal_year, p.period)
    if rev is None or cogs is None:
        return {"ok": False, "error": f"Thiếu dữ liệu revenue/cogs cho {ticker} {p.period}/{p.fiscal_year}"}

    new_rev = rev * (1 + revenue_pct_change / 100)
    new_cogs = cogs * (1 + cogs_pct_change / 100)
    new_gp = new_rev - new_cogs
    old_gp = rev - cogs

    return {
        "ok": True, "ticker": ticker, "assumption": "ceteris paribus - tất cả yếu tố khác giữ nguyên",
        "revenue_before": rev, "revenue_after": new_rev, "revenue_pct_change": revenue_pct_change,
        "cogs_before": cogs, "cogs_after": new_cogs, "cogs_pct_change": cogs_pct_change,
        "gross_profit_before": old_gp, "gross_profit_after": new_gp,
        "gross_profit_change": new_gp - old_gp,
        "gross_margin_before_pct": round(old_gp / rev * 100, 2) if rev else None,
        "gross_margin_after_pct": round(new_gp / new_rev * 100, 2) if new_rev else None,
    }


def sensitivity_what_if_macro(ticker: str, p: Period) -> dict:
    """Same ceteris-paribus gross-profit scenario as sensitivity_what_if,
    but cogs_pct_change comes from a REAL industry production/price index
    (src/causal/macro_lookup.py) instead of a user-typed hypothetical -
    e.g. for HSG (steel), uses the actual most-recent change in
    "Sản xuất kim loại" (IIP_BY_INDUSTRY). Only works for tickers with a
    genuine industry mapping (see TICKER_INDUSTRY_MAP) - returns ok=False
    with an explanation otherwise, rather than guessing.
    """
    mapped = industry_index_for_ticker(ticker)
    if mapped is None:
        return {"ok": False, "error": f"{ticker} chưa có ánh xạ ngành trong TICKER_INDUSTRY_MAP "
                                       f"(src/causal/macro_lookup.py) - dùng sensitivity_what_if() "
                                       f"với % tự nhập thay thế."}
    category, indicator_name = mapped
    period_type = "YEAR" if category == "IIP_BY_INDUSTRY" else "MONTH"

    periods = latest_two_periods(category, indicator_name, period_type)
    if periods is None:
        return {"ok": False, "error": f"Chưa đủ dữ liệu macro cho {category}/{indicator_name} "
                                       f"(cần >= 2 kỳ) - chạy src.ingestion.macro_etl trước."}
    (y1, p1), (y2, p2) = periods

    macro_change = macro_pct_change(category, indicator_name, period_type, y1, p1, y2, p2)
    if not macro_change["ok"]:
        return macro_change

    result = sensitivity_what_if(ticker, p, cogs_pct_change=macro_change["pct_change"])
    if result["ok"]:
        result["macro_source"] = {
            "category": category, "indicator_name": indicator_name,
            "period_from": f"{p1}/{y1}", "period_to": f"{p2}/{y2}",
            "pct_change_used_as_cogs_shock": macro_change["pct_change"],
            "note": "cogs_pct_change lấy từ biến động chỉ số ngành thực tế, KHÔNG phải giả định người dùng nhập.",
        }
    return result


def sensitivity_what_if_gold(ticker: str, p: Period) -> dict:
    """Same ceteris-paribus gross-profit scenario as sensitivity_what_if,
    but cogs_pct_change comes from the REAL domestic gold price's most
    recent day-over-day move (src/causal/market_factors.py) instead of a
    user-typed hypothetical - for GOLD_SENSITIVE_TICKERS (currently just
    PNJ, whose raw material literally IS gold), a rise in gold price is a
    direct COGS shock. Only works for tickers in GOLD_SENSITIVE_TICKERS -
    returns ok=False with an explanation otherwise.

    NOTE on data sparsity: scripts/market_snapshot.py's fetch_gold() can
    only ever write TODAY's snapshot (SJC's API has no historical range
    endpoint - see that function's docstring), so gold_price_series() only
    has as many days as the Airflow DAG has actually run so far - the
    "most recent 2 readings" used here may be widely spaced (or, right
    after deployment, may not exist at all) rather than a clean
    day-over-day comparison. Returns ok=False rather than a misleading
    result if fewer than 2 readings are loaded yet.
    """
    if ticker not in GOLD_SENSITIVE_TICKERS:
        return {"ok": False, "error": f"{ticker} chưa được xác định là nhạy cảm với giá vàng "
                                       f"(xem GOLD_SENSITIVE_TICKERS trong calculator.py) - dùng "
                                       f"sensitivity_what_if() với % tự nhập thay thế."}

    from src.causal.market_factors import gold_price_series
    gold = gold_price_series()
    if len(gold) < 2:
        return {"ok": False, "error": f"Chưa đủ dữ liệu giá vàng thực tế (hiện có {len(gold)} ngày, cần >= 2) - "
                                       f"chạy scripts/market_snapshot.py (hoặc chờ DAG thesis_market_snapshot "
                                       f"chạy thêm vài ngày) để tích luỹ thêm."}

    v1, v2 = float(gold["gold_price"].iloc[-2]), float(gold["gold_price"].iloc[-1])
    d1, d2 = str(gold["trade_date"].iloc[-2]), str(gold["trade_date"].iloc[-1])
    if v1 == 0:
        return {"ok": False, "error": "Giá vàng kỳ gốc bằng 0, không tính được % thay đổi."}
    pct_change = (v2 - v1) / v1 * 100

    result = sensitivity_what_if(ticker, p, cogs_pct_change=pct_change)
    if result["ok"]:
        result["gold_source"] = {
            "gold_price_before": v1, "gold_price_after": v2,
            "date_before": d1, "date_after": d2,
            "pct_change_used_as_cogs_shock": round(pct_change, 4),
            "note": "cogs_pct_change lấy từ biến động giá vàng SJC thực tế (2 lần ghi nhận gần nhất trong "
                    "macro_indicators), KHÔNG phải giả định người dùng nhập. Giả định: vàng nguyên liệu là "
                    "driver chính của COGS cho ticker này - chỉ hợp lý cho công ty kinh doanh vàng/trang sức.",
        }
    return result


def sensitivity_what_if_oil(ticker: str, p: Period) -> dict:
    """Ceteris-paribus scenario using the REAL WTI oil price's most recent
    move as a REVENUE shock (not a COGS shock like sensitivity_what_if_gold)
    - for OIL_SENSITIVE_TICKERS (currently just GAS), oil price is a
    revenue driver, since GAS sells gas under contracts commonly
    benchmarked to oil-equivalent pricing (see OIL_SENSITIVE_TICKERS'
    docstring for the caveat on precision). Only works for tickers in
    OIL_SENSITIVE_TICKERS - returns ok=False with an explanation otherwise.

    NOTE on data availability: UNLIKE gold, fetch_oil() in
    scripts/market_snapshot.py DOES backfill a real date range from FRED
    per run - but ONLY if FRED_API_KEY is set in .env (skipped entirely,
    0 rows, otherwise). Check which situation you're in before assuming an
    ok=False result here means "oil data will never be available" versus
    "just needs a free FRED key registered."
    """
    if ticker not in OIL_SENSITIVE_TICKERS:
        return {"ok": False, "error": f"{ticker} chưa được xác định là nhạy cảm với giá dầu "
                                       f"(xem OIL_SENSITIVE_TICKERS trong calculator.py) - dùng "
                                       f"sensitivity_what_if() với % tự nhập thay thế."}

    from src.causal.market_factors import oil_price_series
    oil = oil_price_series()
    if len(oil) < 2:
        return {"ok": False, "error": f"Chưa đủ dữ liệu giá dầu thực tế (hiện có {len(oil)} ngày, cần >= 2) - "
                                       f"kiểm tra FRED_API_KEY đã đặt trong .env chưa (xem README Step 11), "
                                       f"rồi chạy scripts/market_snapshot.py hoặc chờ DAG chạy thêm."}

    v1, v2 = float(oil["oil_price"].iloc[-2]), float(oil["oil_price"].iloc[-1])
    d1, d2 = str(oil["trade_date"].iloc[-2]), str(oil["trade_date"].iloc[-1])
    if v1 == 0:
        return {"ok": False, "error": "Giá dầu kỳ gốc bằng 0, không tính được % thay đổi."}
    pct_change = (v2 - v1) / v1 * 100

    result = sensitivity_what_if(ticker, p, revenue_pct_change=pct_change)
    if result["ok"]:
        result["oil_source"] = {
            "oil_price_before": v1, "oil_price_after": v2,
            "date_before": d1, "date_after": d2,
            "pct_change_used_as_revenue_shock": round(pct_change, 4),
            "note": "revenue_pct_change lấy từ biến động giá dầu WTI thực tế (2 lần ghi nhận gần nhất trong "
                    "macro_indicators), KHÔNG phải giả định người dùng nhập. Giả định: doanh thu bán khí của "
                    "GAS gắn với công thức giá tham chiếu dầu - xem OIL_SENSITIVE_TICKERS để biết giới hạn "
                    "của giả định này (chưa xác minh chi tiết hợp đồng thực tế).",
        }
    return result


def sensitivity_what_if_trade(ticker: str, p: Period) -> dict:
    """Ceteris-paribus REVENUE-shock scenario using the REAL national
    export/import value of the ticker's mapped trade commodity (see
    src/causal/macro_lookup.TRADE_INDUSTRY_MAP - "Sắt thép" for HPG/HSG,
    "Phân bón" for DPM/DCM) as the shock - the EXPORTS_BY_COMMODITY/
    IMPORTS_BY_COMMODITY counterpart of sensitivity_what_if_macro()
    (which uses IIP_BY_INDUSTRY/RETAIL instead). Only works for tickers in
    TRADE_INDUSTRY_MAP - returns ok=False with an explanation otherwise.

    Sign handling: for an "export"-direction mapping (HPG/HSG), a RISE in
    the national export figure is applied as-is (a tailwind - more demand
    for the commodity the company sells). For an "import"-direction
    mapping (DPM/DCM), the mechanism is the OPPOSITE - a rise in the
    national IMPORT figure signals import competition against the
    company's own domestic sales, a headwind - so the pct_change sign is
    FLIPPED before being applied as a revenue shock. This flip is the
    reason the two ticker groups cannot share one undifferentiated helper
    despite querying structurally identical macro data.
    """
    mapped = trade_index_for_ticker(ticker)
    if mapped is None:
        return {"ok": False, "error": f"{ticker} chưa có ánh xạ mặt hàng xuất/nhập khẩu trong "
                                       f"TRADE_INDUSTRY_MAP (src/causal/macro_lookup.py) - dùng "
                                       f"sensitivity_what_if() với % tự nhập thay thế."}
    category, indicator_name, direction = mapped

    periods = latest_two_periods(category, indicator_name, "MONTH")
    if periods is None:
        return {"ok": False, "error": f"Chưa đủ dữ liệu xuất/nhập khẩu cho {category}/{indicator_name} "
                                       f"(cần >= 2 kỳ) - chạy src.ingestion.macro_etl trước."}
    (y1, p1), (y2, p2) = periods

    macro_change = macro_pct_change(category, indicator_name, "MONTH", y1, p1, y2, p2)
    if not macro_change["ok"]:
        return macro_change
    pct_change = macro_change["pct_change"]
    if direction == "import":
        pct_change = -pct_change  # import competition headwind, see docstring

    result = sensitivity_what_if(ticker, p, revenue_pct_change=pct_change)
    if result["ok"]:
        result["trade_source"] = {
            "category": category, "indicator_name": indicator_name, "direction": direction,
            "period_from": f"{p1}/{y1}", "period_to": f"{p2}/{y2}",
            "raw_pct_change": round(macro_change["pct_change"], 4),
            "pct_change_used_as_revenue_shock": round(pct_change, 4),
            "note": (
                f"revenue_pct_change lấy từ biến động {'xuất' if direction == 'export' else 'nhập'} "
                f"khẩu {indicator_name} thực tế (macro_indicators/{category}), KHÔNG phải giả định "
                f"người dùng nhập." + (
                    " Chiều NHẬP KHẨU được diễn giải là áp lực cạnh tranh (dấu bị đảo ngược so với "
                    "biến động gốc) - cơ chế hợp lý về mặt hướng tác động nhưng CHƯA xác minh với dữ "
                    "liệu thị phần/hợp đồng thực tế của công ty." if direction == "import" else ""
                )
            ),
        }
    return result


_LOOKUP_INCOME_STATEMENT_LINE = text("""
    SELECT value FROM financial_line_items
    WHERE ticker = :ticker AND fiscal_year = :fiscal_year AND period = :period
      AND statement_type = 'income_statement' AND ms_code = :ms_code
    LIMIT 1
""")


def sensitivity_what_if_interest_rate(ticker: str, p: Period) -> dict:
    """Ceteris-paribus scenario using the REAL interbank overnight rate's
    (VNIBOR, macro_indicators category "VNIBOR", indicator "Qua đêm" -
    ingested from the 2026-09-15 macro batch) most recent move as a shock
    to "Chi phí tài chính" (income_statement ms_code 23), which flows
    straight to "Tổng lợi nhuận kế toán trước thuế" (ms_code 50).

    Deliberately a STANDALONE function, not a call into
    sensitivity_what_if() with COGS repurposed as financial_expense: an
    interest-rate shock hits a genuinely different income-statement line
    (financial expense, not cost of goods sold) and flows to a different
    downstream total (pretax profit, not gross profit) - reusing
    sensitivity_what_if() would either mislabel the output or require it
    to become a different function anyway.

    Applies an ECONOMY-WIDE rate (VNIBOR is the interbank rate every
    company effectively borrows against, directly or indirectly) rather
    than any company-specific rate: no data anywhere in this project
    records which bank a given ticker actually borrows from (Quang asked
    for this 2026-09-15; confirmed no such table/export exists - see
    docs/PROGRESS.md's 2026-09-15 entry). Stating this explicitly in the
    result rather than quietly pretending otherwise.
    """
    from src.causal.macro_lookup import get_macro_value, latest_two_periods

    periods = latest_two_periods("VNIBOR", "Qua đêm", "DAILY")
    if periods is None:
        return {"ok": False, "error": "Chưa đủ dữ liệu lãi suất VNIBOR thực tế (cần >= 2 kỳ) - "
                                       "chạy src.ingestion.macro_etl trước."}
    (y1, p1), (y2, p2) = periods
    v1 = get_macro_value("VNIBOR", "Qua đêm", "DAILY", y1, p1)
    v2 = get_macro_value("VNIBOR", "Qua đêm", "DAILY", y2, p2)
    if v1 is None or v2 is None or v1 == 0:
        return {"ok": False, "error": "Không lấy được đủ 2 giá trị lãi suất VNIBOR để tính % thay đổi."}
    pct_change = (v2 - v1) / v1 * 100

    with get_session() as session:
        fin_cost_row = session.execute(_LOOKUP_INCOME_STATEMENT_LINE, {
            "ticker": ticker, "fiscal_year": p.fiscal_year, "period": p.period, "ms_code": "23",
        }).first()
        pretax_row = session.execute(_LOOKUP_INCOME_STATEMENT_LINE, {
            "ticker": ticker, "fiscal_year": p.fiscal_year, "period": p.period, "ms_code": "50",
        }).first()
    if fin_cost_row is None or pretax_row is None:
        return {"ok": False, "error": f"Thiếu dữ liệu Chi phí tài chính/Lợi nhuận trước thuế cho "
                                       f"{ticker} {p.period}/{p.fiscal_year}"}

    fin_cost, pretax = float(fin_cost_row[0]), float(pretax_row[0])
    new_fin_cost = fin_cost * (1 + pct_change / 100)
    delta = new_fin_cost - fin_cost
    new_pretax = pretax - delta  # a financial-expense INCREASE reduces pretax profit

    return {
        "ok": True, "ticker": ticker,
        "assumption": "ceteris paribus - tất cả yếu tố khác giữ nguyên",
        "financial_expense_before": fin_cost, "financial_expense_after": new_fin_cost,
        "financial_expense_pct_change": round(pct_change, 4),
        "pretax_profit_before": pretax, "pretax_profit_after": new_pretax,
        "pretax_profit_change": new_pretax - pretax,
        "interest_rate_source": {
            "category": "VNIBOR", "indicator_name": "Qua đêm",
            "period_from": f"{p1}/{y1}", "period_to": f"{p2}/{y2}",
            "pct_change_used_as_financial_expense_shock": round(pct_change, 4),
            "note": "Lãi suất liên ngân hàng qua đêm (VNIBOR) - lãi suất THỊ TRƯỜNG CHUNG, "
                    "KHÔNG phải lãi suất riêng của ngân hàng mà công ty này đang vay (project "
                    "không có dữ liệu quan hệ tín dụng công ty-ngân hàng).",
        },
    }


if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) < 6:
        print("Usage: python -m src.structured.calculator TICKER year1 period1 year2 period2")
        print('Example: python -m src.structured.calculator HSG 2024 YEAR 2025 YEAR')
        sys.exit(1)

    ticker, y1, per1, y2, per2 = sys.argv[1:6]
    result = decompose_gross_margin_change(
        ticker, Period(int(y1), per1), Period(int(y2), per2)
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
