"""Top-level query router - System S3 ("Router") in the thesis's 3-system
comparison (S1 RAG-only / S2 DAG-only / S3 Router). Classifies a question
via src/router/rules.py, dispatches to the right branch, and normalizes
every branch's very different native return shape (a plain string from
dag_pipeline, a dict from rag_pipeline, raw calculator/treatment_effect
dicts with no NL formatting at all) into one consistent result dict.

The "Adaptive" part of "Adaptive Retrieval Routing" is this: DAG is a
narrow, bounded lookup (src/structured/schema.py's alias table) that can
recognize a question's shape correctly but still find no matching row
(period not loaded yet, wrong consolidation, etc.). Rather than surface a
bare SQL-shaped error, the router automatically falls back to RAG in that
case - "TOTAL ASSETS OF X IN 2026" might not be in financial_line_items
yet, but could still be answerable from the raw disclosure text. This
fallback is the one piece of real "routing between branches at runtime"
behavior; classification into WHATIF/CAUSAL is otherwise a single
deterministic choice (see rules.py's docstring for why: those two branches
answer fundamentally different question shapes than RAG, so falling back
to RAG on failure wouldn't help - RAG can't compute an ATE or a bridge
decomposition either).

Usage:
    python -m src.router.query_router "Tổng tài sản của VNM quý 1 năm 2026 là bao nhiêu?"
    python -m src.router.query_router "Rủi ro pháp lý của VNM là gì?"
    python -m src.router.query_router "Nếu COGS của CTD tăng 5% trong 2025, lợi nhuận gộp sẽ ra sao?"
    python -m src.router.query_router "So sánh doanh thu của VNM giữa 2024 và 2025"
    python -m src.router.query_router "Đòn bẩy có ảnh hưởng đến biên lợi nhuận ròng không?"
    python -m src.router.query_router "So sánh PNJ và CTD, có kèm giá vàng"
    python -m src.router.query_router "Nếu giá vàng tăng thì PNJ sẽ ra sao?"

NOT run end-to-end against a real Postgres/Ollama stack by the
environment that wrote it (same situation as every other DB/LLM-backed
module in this project - see docs/PROGRESS.md). Each individual branch
(RAG, DAG, calculator, treatment_effect) IS already independently verified
on the real project owner's machine; what's new and unverified here is
purely the classification + dispatch + fallback logic on top of them.
"""
import sys
from datetime import date

from src.absa.absa_pipeline import ASPECTS as ABSA_ASPECTS
from src.absa.absa_pipeline import analyze_ticker as absa_analyze_ticker
from src.causal.macro_lookup import best_deposit_rate_banks
from src.causal.market_factors import (
    compare_tickers,
    correlate_ticker_with_gold,
    latest_price_snapshot,
    price_range_series,
    price_window_series,
)
from src.causal.treatment_effect import estimate_ate
from src.models.rationale import generate_rationale
from src.retrieval.rag_pipeline import answer as rag_answer
from src.router.rules import QueryIntent, classify_intent
from src.core.lang_detect import resolve_answer_language
from src.router.display_labels import (
    ASSUMPTION_EN, BANK_RATE_NOTE_EN, INTEREST_RATE_NOTE_EN, confidence_label, data_error, error_en,
    join_list, line_item_label, ratio_group_label, ratio_label, series_label, trade_note, unit_label,
)
from src.router.scope_notes import polish_not_found
from src.structured.auditor_lookup import lookup_companies_by_auditor, lookup_companies_by_auditor_fts
from src.structured.calculator import (
    Period,
    decompose_gross_margin_change,
    growth_rate,
    sensitivity_what_if,
    sensitivity_what_if_gold,
    sensitivity_what_if_interest_rate,
    sensitivity_what_if_macro,
    sensitivity_what_if_oil,
    sensitivity_what_if_trade,
)
from src.structured import trend
from src.structured.query_builder import execute as dag_execute
from src.structured.query_builder import parse_question


# 2026-09-22 - BILINGUAL ANSWER FORMATTING.
#
# Every _format_* helper below now takes `lang` ("vi" | "en"), resolved ONCE
# per request in route() via lang_detect.resolve_answer_language(). Why this
# was needed: the RAG branch had been bilingual since 2026-09-08 (see
# src/models/generation.py's English system prompt), but every STRUCTURED
# branch built its answer from hard-coded Vietnamese f-strings here - so an
# English question routed to "dag" came back fully in Vietnamese, which is
# exactly what a thesis defense delivered in English cannot do.
#
# Deliberately NOT done by asking the LLM to translate the finished answer:
# these branches answer in 30-40ms with zero model calls, and an 8B model
# re-emitting "11,352,254,039,576" is a real corruption risk. Translating
# the SCAFFOLDING (labels, notes, units of explanation) while never touching
# a computed number keeps this thesis's central claim intact - the language
# model stays out of every step whose output has to be exactly correct.
#
# The pattern throughout: `en = lang == "en"`, then branch on it. Numbers,
# tickers, MS codes, and period labels are formatted identically in both.


def _format_dag(result: dict, lang: str = "vi") -> str:
    """Mirrors src/structured/dag_pipeline.py's answer() formatting."""
    if lang == "en":
        return (
            f"{result['value']:,} {result['unit']} "
            f"(source: {result['ticker']}, "
            f"{line_item_label(result['statement_type'], result['ms_code'], result['item_name'])} "
            f"[{result['statement_type']} MS={result['ms_code']}], period "
            f"{result['period']}/{result['fiscal_year']}, {result['consolidation']})"
        )
    return (
        f"{result['value']:,} {result['unit']} "
        f"(nguồn: {result['ticker']}, {result['item_name']} [{result['statement_type']} "
        f"MS={result['ms_code']}], kỳ {result['period']}/{result['fiscal_year']}, {result['consolidation']})"
    )


def _format_ratio_lookup(result: dict, lang: str = "vi") -> str:
    """Mirrors _format_dag() above - same 'value, unit, source' shape, just
    against financial_ratios (ratio_group/ratio_name) instead of
    financial_line_items (statement_type/ms_code/item_name)."""
    if lang == "en":
        return (
            f"{result['value']:,} {unit_label(result['unit'])} "
            f"(source: {result['ticker']}, {ratio_label(result['ratio_name'])} "
            f"[{ratio_group_label(result['ratio_group'])} ratios], "
            f"period {result['period']}/{result['fiscal_year']})"
        )
    return (
        f"{result['value']:,} {result['unit']} "
        f"(nguồn: {result['ticker']}, {result['ratio_name']} [{result['ratio_group']}], "
        f"kỳ {result['period']}/{result['fiscal_year']})"
    )


def _format_whatif(kind: str, result: dict, lang: str = "vi") -> str:
    en = lang == "en"
    if not result["ok"]:
        return (f"Could not compute: {error_en(result['error'])}" if en
                else f"Không tính được: {result['error']}")

    if kind == "sensitivity":
        if en:
            return (
                f"Ceteris-paribus scenario for {result['ticker']}: revenue "
                f"{result['revenue_before']:,.0f} -> {result['revenue_after']:,.0f} "
                f"({result['revenue_pct_change']:+.1f}%), COGS {result['cogs_before']:,.0f} -> "
                f"{result['cogs_after']:,.0f} ({result['cogs_pct_change']:+.1f}%). "
                f"Gross profit: {result['gross_profit_before']:,.0f} -> {result['gross_profit_after']:,.0f} "
                f"(change {result['gross_profit_change']:,.0f}). Gross margin: "
                f"{result['gross_margin_before_pct']}% -> {result['gross_margin_after_pct']}%. "
                f"Assumption: {ASSUMPTION_EN}."
            )
        return (
            f"Kịch bản ceteris-paribus cho {result['ticker']}: doanh thu "
            f"{result['revenue_before']:,.0f} -> {result['revenue_after']:,.0f} "
            f"({result['revenue_pct_change']:+.1f}%), COGS {result['cogs_before']:,.0f} -> "
            f"{result['cogs_after']:,.0f} ({result['cogs_pct_change']:+.1f}%). "
            f"Lợi nhuận gộp: {result['gross_profit_before']:,.0f} -> {result['gross_profit_after']:,.0f} "
            f"(thay đổi {result['gross_profit_change']:,.0f}). Biên LN gộp: "
            f"{result['gross_margin_before_pct']}% -> {result['gross_margin_after_pct']}%. "
            f"Giả định: {result['assumption']}."
        )
    if kind == "decompose_margin":
        if en:
            return (
                f"Gross-profit change decomposition for {result['ticker']}: {result['gross_profit_p1']:,.0f} -> "
                f"{result['gross_profit_p2']:,.0f} (delta={result['delta_gross_profit']:,.0f}). "
                f"Revenue-growth effect: {result['revenue_growth_effect']:,.0f} "
                f"({result['revenue_growth_effect_share_pct']}%). Margin-rate effect (cost efficiency): "
                f"{result['margin_rate_effect']:,.0f} ({result['margin_rate_effect_share_pct']}%). "
                f"Gross margin: {result['gross_margin_p1_pct']}% -> {result['gross_margin_p2_pct']}%. "
                f"({result['formula']})"
            )
        return (
            f"Phân rã thay đổi lợi nhuận gộp {result['ticker']}: {result['gross_profit_p1']:,.0f} -> "
            f"{result['gross_profit_p2']:,.0f} (delta={result['delta_gross_profit']:,.0f}). "
            f"Hiệu ứng tăng trưởng doanh thu: {result['revenue_growth_effect']:,.0f} "
            f"({result['revenue_growth_effect_share_pct']}%). Hiệu ứng biên lợi nhuận (hiệu quả chi phí): "
            f"{result['margin_rate_effect']:,.0f} ({result['margin_rate_effect_share_pct']}%). "
            f"Biên LN gộp: {result['gross_margin_p1_pct']}% -> {result['gross_margin_p2_pct']}%. "
            f"({result['formula']})"
        )
    if kind == "growth_rate":
        if en:
            return (
                f"{result['ticker']}: {result['v1']:,.0f} -> {result['v2']:,.0f} "
                f"(growth {result['pct_change']:.2f}%). Formula: {result['formula']}."
            )
        return (
            f"{result['ticker']}: {result['v1']:,.0f} -> {result['v2']:,.0f} "
            f"(tăng trưởng {result['pct_change']:.2f}%). Công thức: {result['formula']}."
        )
    if kind == "gold_sensitivity":
        if en:
            base = (
                f"Actual gold-price scenario for {result['ticker']}: revenue "
                f"{result['revenue_before']:,.0f} -> {result['revenue_after']:,.0f}, COGS "
                f"{result['cogs_before']:,.0f} -> {result['cogs_after']:,.0f} "
                f"({result['cogs_pct_change']:+.2f}%). Gross profit: {result['gross_profit_before']:,.0f} -> "
                f"{result['gross_profit_after']:,.0f} (change {result['gross_profit_change']:,.0f}). "
                f"Gross margin: {result['gross_margin_before_pct']}% -> {result['gross_margin_after_pct']}%."
            )
        else:
            base = (
                f"Kịch bản giá vàng thực tế cho {result['ticker']}: doanh thu "
                f"{result['revenue_before']:,.0f} -> {result['revenue_after']:,.0f}, COGS "
                f"{result['cogs_before']:,.0f} -> {result['cogs_after']:,.0f} "
                f"({result['cogs_pct_change']:+.2f}%). Lợi nhuận gộp: {result['gross_profit_before']:,.0f} -> "
                f"{result['gross_profit_after']:,.0f} (thay đổi {result['gross_profit_change']:,.0f}). "
                f"Biên LN gộp: {result['gross_margin_before_pct']}% -> {result['gross_margin_after_pct']}%."
            )
        gs = result.get("gold_source")
        if gs:
            base += (
                f" Based on actual SJC gold prices: {gs['gold_price_before']:,.0f} ({gs['date_before']}) -> "
                f"{gs['gold_price_after']:,.0f} ({gs['date_after']})."
                if en else
                f" Dựa trên giá vàng SJC thực tế: {gs['gold_price_before']:,.0f} ({gs['date_before']}) -> "
                f"{gs['gold_price_after']:,.0f} ({gs['date_after']})."
            )
        return base
    if kind == "oil_sensitivity":
        if en:
            base = (
                f"Actual oil-price scenario for {result['ticker']}: revenue "
                f"{result['revenue_before']:,.0f} -> {result['revenue_after']:,.0f} "
                f"({result['revenue_pct_change']:+.2f}%), COGS {result['cogs_before']:,.0f} -> "
                f"{result['cogs_after']:,.0f}. Gross profit: {result['gross_profit_before']:,.0f} -> "
                f"{result['gross_profit_after']:,.0f} (change {result['gross_profit_change']:,.0f}). "
                f"Gross margin: {result['gross_margin_before_pct']}% -> {result['gross_margin_after_pct']}%."
            )
        else:
            base = (
                f"Kịch bản giá dầu thực tế cho {result['ticker']}: doanh thu "
                f"{result['revenue_before']:,.0f} -> {result['revenue_after']:,.0f} "
                f"({result['revenue_pct_change']:+.2f}%), COGS {result['cogs_before']:,.0f} -> "
                f"{result['cogs_after']:,.0f}. Lợi nhuận gộp: {result['gross_profit_before']:,.0f} -> "
                f"{result['gross_profit_after']:,.0f} (thay đổi {result['gross_profit_change']:,.0f}). "
                f"Biên LN gộp: {result['gross_margin_before_pct']}% -> {result['gross_margin_after_pct']}%."
            )
        os_ = result.get("oil_source")
        if os_:
            base += (
                f" Based on actual WTI oil prices: {os_['oil_price_before']:,.2f} USD/bbl ({os_['date_before']}) -> "
                f"{os_['oil_price_after']:,.2f} USD/bbl ({os_['date_after']})."
                if en else
                f" Dựa trên giá dầu WTI thực tế: {os_['oil_price_before']:,.2f} USD/bbl ({os_['date_before']}) -> "
                f"{os_['oil_price_after']:,.2f} USD/bbl ({os_['date_after']})."
            )
        return base
    if kind == "macro_sensitivity":
        # 2026-09-15: mirrors gold_sensitivity/oil_sensitivity above -
        # calculator.sensitivity_what_if_macro()'s result carries the same
        # ceteris-paribus revenue/COGS/gross-profit fields, plus a
        # "macro_source" dict (not "gold_source"/"oil_source") naming the
        # real IIP_BY_INDUSTRY/RETAIL indicator and period pair used as the
        # COGS shock.
        if en:
            base = (
                f"Actual industry-index scenario for {result['ticker']}: revenue "
                f"{result['revenue_before']:,.0f} -> {result['revenue_after']:,.0f}, COGS "
                f"{result['cogs_before']:,.0f} -> {result['cogs_after']:,.0f} "
                f"({result['cogs_pct_change']:+.2f}%). Gross profit: {result['gross_profit_before']:,.0f} -> "
                f"{result['gross_profit_after']:,.0f} (change {result['gross_profit_change']:,.0f}). "
                f"Gross margin: {result['gross_margin_before_pct']}% -> {result['gross_margin_after_pct']}%."
            )
        else:
            base = (
                f"Kịch bản chỉ số ngành thực tế cho {result['ticker']}: doanh thu "
                f"{result['revenue_before']:,.0f} -> {result['revenue_after']:,.0f}, COGS "
                f"{result['cogs_before']:,.0f} -> {result['cogs_after']:,.0f} "
                f"({result['cogs_pct_change']:+.2f}%). Lợi nhuận gộp: {result['gross_profit_before']:,.0f} -> "
                f"{result['gross_profit_after']:,.0f} (thay đổi {result['gross_profit_change']:,.0f}). "
                f"Biên LN gộp: {result['gross_margin_before_pct']}% -> {result['gross_margin_after_pct']}%."
            )
        ms = result.get("macro_source")
        if ms:
            base += (
                f" Based on the actual industry-index move ({ms['category']}/{series_label(ms['indicator_name'])}): "
                f"{ms['period_from']} -> {ms['period_to']}, change "
                f"{ms['pct_change_used_as_cogs_shock']:+.2f}%."
                if en else
                f" Dựa trên biến động chỉ số ngành thực tế ({ms['category']}/{ms['indicator_name']}): "
                f"{ms['period_from']} -> {ms['period_to']}, thay đổi "
                f"{ms['pct_change_used_as_cogs_shock']:+.2f}%."
            )
        return base
    if kind == "trade_sensitivity":
        # 2026-09-15: mirrors macro_sensitivity above, but the shock hits
        # REVENUE (not COGS) - sensitivity_what_if_trade()'s docstring
        # explains why (a real export/import commodity move is a demand-
        # side signal, not a cost-side one). Result carries a "trade_source"
        # dict (category/indicator_name/direction/period pair/pct_change),
        # mirroring "macro_source"/"oil_source" above.
        if en:
            base = (
                f"Actual export/import commodity scenario for {result['ticker']}: revenue "
                f"{result['revenue_before']:,.0f} -> {result['revenue_after']:,.0f} "
                f"({result['revenue_pct_change']:+.2f}%). Gross profit: {result['gross_profit_before']:,.0f} -> "
                f"{result['gross_profit_after']:,.0f} (change {result['gross_profit_change']:,.0f}). "
                f"Gross margin: {result['gross_margin_before_pct']}% -> {result['gross_margin_after_pct']}%."
            )
        else:
            base = (
                f"Kịch bản xuất/nhập khẩu hàng hóa thực tế cho {result['ticker']}: doanh thu "
                f"{result['revenue_before']:,.0f} -> {result['revenue_after']:,.0f} "
                f"({result['revenue_pct_change']:+.2f}%). Lợi nhuận gộp: {result['gross_profit_before']:,.0f} -> "
                f"{result['gross_profit_after']:,.0f} (thay đổi {result['gross_profit_change']:,.0f}). "
                f"Biên LN gộp: {result['gross_margin_before_pct']}% -> {result['gross_margin_after_pct']}%."
            )
        ts_ = result.get("trade_source")
        if ts_:
            is_export = ts_["direction"] == "export"
            if en:
                direction_label = "export" if is_export else "import"
                base += (
                    f" Based on the actual {direction_label} move ({ts_['category']}/{series_label(ts_['indicator_name'])}): "
                    f"{ts_['period_from']} -> {ts_['period_to']}, raw change "
                    f"{ts_['raw_pct_change']:+.2f}%, applied as a revenue shock of "
                    f"{ts_['pct_change_used_as_revenue_shock']:+.2f}%. "
                    f"{trade_note(ts_['direction'], ts_['indicator_name'], ts_['category'])}"
                )
            else:
                direction_vi = "xuất khẩu" if is_export else "nhập khẩu"
                base += (
                    f" Dựa trên biến động {direction_vi} thực tế ({ts_['category']}/{ts_['indicator_name']}): "
                    f"{ts_['period_from']} -> {ts_['period_to']}, thay đổi gốc "
                    f"{ts_['raw_pct_change']:+.2f}%, áp dụng làm cú sốc doanh thu "
                    f"{ts_['pct_change_used_as_revenue_shock']:+.2f}%. {ts_['note']}"
                )
        return base
    if kind == "interest_rate_sensitivity":
        # 2026-09-15: different shape from every other WHATIF kind above -
        # sensitivity_what_if_interest_rate() shocks "Chi phí tài chính"
        # and reports the impact on PRETAX PROFIT, not gross profit (see
        # that function's docstring for why interest rate isn't a
        # COGS/revenue shock).
        if en:
            base = (
                f"Actual interest-rate scenario for {result['ticker']}: financial expense "
                f"{result['financial_expense_before']:,.0f} -> {result['financial_expense_after']:,.0f} "
                f"({result['financial_expense_pct_change']:+.2f}%). Pre-tax profit: "
                f"{result['pretax_profit_before']:,.0f} -> {result['pretax_profit_after']:,.0f} "
                f"(change {result['pretax_profit_change']:,.0f})."
            )
        else:
            base = (
                f"Kịch bản lãi suất thực tế cho {result['ticker']}: Chi phí tài chính "
                f"{result['financial_expense_before']:,.0f} -> {result['financial_expense_after']:,.0f} "
                f"({result['financial_expense_pct_change']:+.2f}%). Lợi nhuận trước thuế: "
                f"{result['pretax_profit_before']:,.0f} -> {result['pretax_profit_after']:,.0f} "
                f"(thay đổi {result['pretax_profit_change']:,.0f})."
            )
        irs = result.get("interest_rate_source")
        if irs:
            base += (
                f" Based on the actual overnight VNIBOR rate: {irs['period_from']} -> "
                f"{irs['period_to']}, change {irs['pct_change_used_as_financial_expense_shock']:+.2f}%. "
                f"{INTEREST_RATE_NOTE_EN}"
                if en else
                f" Dựa trên lãi suất VNIBOR qua đêm thực tế: {irs['period_from']} -> "
                f"{irs['period_to']}, thay đổi {irs['pct_change_used_as_financial_expense_shock']:+.2f}%. "
                f"{irs['note']}"
            )
        return base
    return str(result)


def _format_bank_rate_lookup(result: dict, lang: str = "vi") -> str:
    en = lang == "en"
    if not result["ok"]:
        return (f"Could not rank: {error_en(result['error'])}" if en
                else f"Không xếp hạng được: {result['error']}")
    header = (
        f"Deposit rates by bank (period {result['period']}/{result['fiscal_year']}, highest -> lowest):"
        if en else
        f"Lãi suất tiền gửi theo ngân hàng (kỳ {result['period']}/{result['fiscal_year']}, cao -> thấp):"
    )
    lines = [header]
    for b in result["banks"]:
        lines.append(f"  {b['bank']}: {b['rate']} {b['unit']}")
    lines.append(f"\n({BANK_RATE_NOTE_EN if en else result['note']})")
    return "\n".join(lines)


def _format_price_snapshot(result: dict, lang: str = "vi") -> str:
    en = lang == "en"
    if not result["ok"]:
        return (f"No price data: {error_en(result['error'])}" if en
                else f"Không có dữ liệu giá: {result['error']}")
    r = result
    lines = [
        f"{r['ticker']} price - most recent session with data ({r['trade_date']}):" if en
        else f"Giá {r['ticker']} - phiên giao dịch gần nhất có dữ liệu ({r['trade_date']}):"
    ]
    has_ohlc = None not in (r["open_price"], r["high_price"], r["low_price"], r["close_price"])
    if en:
        lines.append(
            f"  Open: {r['open_price']:,.0f}  ·  High: {r['high_price']:,.0f}  ·  "
            f"Low: {r['low_price']:,.0f}  ·  Close: {r['close_price']:,.0f}"
            if has_ohlc else f"  Close: {r['close_price']}"
        )
    else:
        lines.append(
            f"  Mở: {r['open_price']:,.0f}  ·  Cao: {r['high_price']:,.0f}  ·  "
            f"Thấp: {r['low_price']:,.0f}  ·  Đóng: {r['close_price']:,.0f}"
            if has_ohlc else f"  Đóng cửa: {r['close_price']}"
        )
    if r["volume"] is not None:
        lines.append(f"  {'Volume' if en else 'Khối lượng'}: {r['volume']:,}")
    if r["pct_change"] is not None:
        sign = "+" if r["pct_change"] >= 0 else ""
        lines.append(
            f"  {'vs previous session' if en else 'So với phiên trước'} ({r['prev_trade_date']}): "
            f"{sign}{r['abs_change']:,.0f} ({sign}{r['pct_change']:.2f}%)"
        )
    else:
        lines.append(
            "  No earlier session in the data to compute a % change." if en
            else "  Không có phiên trước đó trong dữ liệu để tính % thay đổi."
        )
    # Cảnh báo trung thực về độ trễ - KHÔNG được ngụ ý đây là giá thời gian
    # thực. company_price_snapshot nạp 1 lần/ngày qua Airflow, và "hôm nay"
    # trong câu hỏi của người dùng có thể sớm hơn lần nạp gần nhất.
    if r["days_stale"] == 0:
        lines.append(
            "  (This is today's own end-of-day data.)" if en
            else "  (Đây là dữ liệu cuối ngày của đúng hôm nay.)"
        )
    elif r["is_stale"]:
        lines.append(
            f"  NOTE: this is NOT today's data - the most recent row in the system is "
            f"{r['days_stale']} days old. Price data is loaded once a day (end of day), "
            f"not in real time/intraday."
            if en else
            f"  LƯU Ý: đây KHÔNG phải dữ liệu hôm nay - dòng gần nhất trong hệ thống cách đây "
            f"{r['days_stale']} ngày. Dữ liệu giá chỉ nạp 1 lần/ngày (cuối ngày), không phải "
            f"thời gian thực/trong ngày."
        )
    else:
        lines.append(
            "  (The latest row is the most recent session with data - it may be yesterday if "
            "today is a weekend/holiday, or if today's data has not been loaded yet.)"
            if en else
            "  (Dòng gần nhất là phiên giao dịch gần nhất có dữ liệu, có thể là hôm qua nếu "
            "hôm nay là cuối tuần/nghỉ lễ hoặc dữ liệu hôm nay chưa được nạp.)"
        )
    return "\n".join(lines)


def _format_price_series(result: dict, lang: str = "vi") -> str:
    """Bảng giá N phiên gần nhất + tổng kết cả khoảng.

    2026-09-22, đi cùng market_factors.price_window_series(). Khác
    _format_price_snapshot() ở trên: cái đó trả lời "bây giờ bao nhiêu",
    cái này trả lời "đã biến động thế nào" - câu hỏi khác nhau, cách trình
    bày khác nhau (một bảng theo phiên + tổng kết cao/thấp/thay đổi ròng).
    Vẫn KHÔNG gọi model nào; mọi con số do SQL và Python tính.
    """
    en = lang == "en"
    if not result["ok"]:
        return (f"No price data: {error_en(result['error'])}" if en
                else f"Không có dữ liệu giá: {result['error']}")
    r = result
    head = (f"{r['ticker']} price over the last {r['n_sessions']} trading sessions "
            f"({r['first_trade_date']} → {r['last_trade_date']}):"
            if en else
            f"Giá {r['ticker']} trong {r['n_sessions']} phiên giao dịch gần nhất "
            f"({r['first_trade_date']} → {r['last_trade_date']}):")
    lines = [head]
    for s in r["sessions"]:                       # đã sắp mới -> cũ
        close = f"{s['close_price']:,.0f}" if s["close_price"] is not None else "n/a"
        chg = ""
        if s["pct_change"] is not None:
            sign = "+" if s["pct_change"] >= 0 else ""
            chg = f"  ({sign}{s['pct_change']:.2f}%)"
        vol = ""
        if s["volume"] is not None:
            vol = (f"  ·  vol {s['volume']:,}" if en else f"  ·  KL {s['volume']:,}")
        lines.append(f"  {s['trade_date']}: {close}{chg}{vol}")

    if r["net_pct_change"] is not None:
        sign = "+" if r["net_pct_change"] >= 0 else ""
        lines.append(
            f"  => Net change across the window: {sign}{r['net_abs_change']:,.0f} "
            f"({sign}{r['net_pct_change']:.2f}%)"
            if en else
            f"  => Thay đổi ròng cả khoảng: {sign}{r['net_abs_change']:,.0f} "
            f"({sign}{r['net_pct_change']:.2f}%)"
        )
    if r["window_high"] is not None and r["window_low"] is not None:
        # 2026-09-24: nói RÕ đây là đỉnh/đáy TRONG PHIÊN (cột high_price/
        # low_price), không phải giá đóng cửa. Bảng ở trên chỉ liệt kê giá
        # ĐÓNG CỬA, nên một dòng "cao nhất 604 · thấp nhất 530" đứng trơ
        # trọng dưới một cột toàn 550-592 trông như số liệu mâu thuẫn -
        # trong khi thực ra là hai đại lượng khác nhau. Ghi rõ nguồn của
        # từng con số quan trọng hơn là ngắn gọn.
        lines.append(
            f"  => Intraday high {r['window_high']:,.0f} · intraday low {r['window_low']:,.0f} "
            f"within this window (from each session's high/low, not the closes listed above)"
            if en else
            f"  => Đỉnh trong phiên {r['window_high']:,.0f} · đáy trong phiên {r['window_low']:,.0f} "
            f"trong khoảng này (lấy từ giá cao/thấp mỗi phiên, không phải giá đóng cửa ở trên)"
        )
    # Cùng cảnh báo trung thực như _format_price_snapshot: dữ liệu EOD, nạp
    # 1 lần/ngày, KHÔNG phải giá thời gian thực.
    if r["is_stale"]:
        lines.append(
            f"  NOTE: the most recent row is {r['days_stale']} days old. Price data is loaded "
            f"once a day (end of day), not in real time/intraday."
            if en else
            f"  LƯU Ý: dòng gần nhất cách đây {r['days_stale']} ngày. Dữ liệu giá chỉ nạp "
            f"1 lần/ngày (cuối ngày), không phải thời gian thực/trong ngày."
        )
    else:
        lines.append(
            "  (End-of-day data, loaded once a day - not intraday.)" if en
            else "  (Dữ liệu cuối ngày, nạp 1 lần/ngày - không phải trong ngày.)"
        )
    return "\n".join(lines)


# Trên ngưỡng này thì liệt kê từng phiên là không đọc nổi -> gộp theo tháng.
# 15 phiên ~ 3 tuần: vẫn vừa một màn hình chat, mà một khoảng 4 tháng (~80
# phiên) thì chắc chắn phải gộp.
_PRICE_RANGE_DETAIL_MAX_SESSIONS = 15

_MONTH_ABBR_VI = ["", "Th1", "Th2", "Th3", "Th4", "Th5", "Th6",
                  "Th7", "Th8", "Th9", "Th10", "Th11", "Th12"]
_MONTH_ABBR_EN = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
                  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _format_price_range(result: dict, lang: str = "vi") -> str:
    """Giá theo một khoảng ngày TƯỜNG MINH (2026-09-24).

    Khoảng ngắn (<= _PRICE_RANGE_DETAIL_MAX_SESSIONS phiên) liệt kê từng
    phiên; khoảng dài gộp theo tháng. Cùng một câu hỏi, hai cách trình bày -
    vì "giá VNZ tháng 10/2025" và "giá VNZ từ 10/2025 đến 1/2026" là cùng
    một loại truy vấn nhưng lượng dữ liệu chênh nhau cả chục lần.
    """
    en = lang == "en"
    if not result["ok"]:
        return (f"No price data: {error_en(result['error'])}" if en
                else f"Không có dữ liệu giá: {result['error']}")
    r = result
    abbr = _MONTH_ABBR_EN if en else _MONTH_ABBR_VI
    lines = [
        f"{r['ticker']} price from {r['requested_start']} to {r['requested_end']} "
        f"({r['n_sessions']} trading sessions with data):"
        if en else
        f"Giá {r['ticker']} từ {r['requested_start']} đến {r['requested_end']} "
        f"({r['n_sessions']} phiên giao dịch có dữ liệu):"
    ]

    if r["n_sessions"] <= _PRICE_RANGE_DETAIL_MAX_SESSIONS:
        for s in r["sessions"]:                      # đã mới -> cũ
            close = f"{s['close_price']:,.0f}" if s["close_price"] is not None else "n/a"
            chg = ""
            if s["pct_change"] is not None:
                sign = "+" if s["pct_change"] >= 0 else ""
                chg = f"  ({sign}{s['pct_change']:.2f}%)"
            lines.append(f"  {s['trade_date']}: {close}{chg}")
    else:
        lines.append("  " + ("By month (first close → last close):" if en
                             else "Theo tháng (giá đóng đầu → cuối):"))
        for m in r["months"]:
            first = f"{m['first_close']:,.0f}" if m["first_close"] is not None else "n/a"
            last = f"{m['last_close']:,.0f}" if m["last_close"] is not None else "n/a"
            chg = ""
            if m["pct_change"] is not None:
                sign = "+" if m["pct_change"] >= 0 else ""
                chg = f"  ({sign}{m['pct_change']:.2f}%)"
            hl = ""
            if m["high"] is not None and m["low"] is not None:
                hl = (f"  ·  hi {m['high']:,.0f} / lo {m['low']:,.0f}")
            n = f"  ·  {m['n_sessions']} " + ("sessions" if en else "phiên")
            lines.append(f"    {abbr[m['month']]}/{m['year']}: {first} → {last}{chg}{hl}{n}")

    if r["net_pct_change"] is not None:
        sign = "+" if r["net_pct_change"] >= 0 else ""
        lines.append(
            f"  => Net change over the whole range: {sign}{r['net_abs_change']:,.0f} "
            f"({sign}{r['net_pct_change']:.2f}%), from {r['first_trade_date']} to {r['last_trade_date']}"
            if en else
            f"  => Thay đổi ròng cả khoảng: {sign}{r['net_abs_change']:,.0f} "
            f"({sign}{r['net_pct_change']:.2f}%), từ {r['first_trade_date']} đến {r['last_trade_date']}"
        )
    if r["window_high"] is not None and r["window_low"] is not None:
        lines.append(
            f"  => Intraday high {r['window_high']:,.0f} · intraday low {r['window_low']:,.0f} "
            f"(from each session's high/low, not the closes above)"
            if en else
            f"  => Đỉnh trong phiên {r['window_high']:,.0f} · đáy trong phiên {r['window_low']:,.0f} "
            f"(lấy từ giá cao/thấp mỗi phiên, không phải giá đóng cửa ở trên)"
        )
    # Nói THẲNG khi dữ liệu không phủ hết khoảng được hỏi, thay vì lặng lẽ
    # trả về ít hơn rồi để người đọc tự tưởng đó là toàn bộ khoảng.
    if not r["covers_start"] or not r["covers_end"]:
        lines.append(
            f"  LƯU Ý: dữ liệu chỉ có từ {r['first_trade_date']} đến {r['last_trade_date']}, "
            f"chưa phủ hết khoảng bạn hỏi ({r['requested_start']} - {r['requested_end']})."
            if not en else
            f"  NOTE: data only covers {r['first_trade_date']} to {r['last_trade_date']}, "
            f"which does not span the full range you asked for "
            f"({r['requested_start']} - {r['requested_end']})."
        )
    lines.append(
        "  (End-of-day data, loaded once a day - not intraday.)" if en
        else "  (Dữ liệu cuối ngày, nạp 1 lần/ngày - không phải trong ngày.)"
    )
    return "\n".join(lines)


def _format_compare(result: dict, lang: str = "vi") -> str:
    en = lang == "en"
    if not result["ok"]:
        return (f"Could not compare: {error_en(result['error'])}" if en
                else f"Không so sánh được: {result['error']}")
    wide = result["wide"]
    last_row = wide.iloc[-1]
    parts = []
    for col in wide.columns:
        if col == "trade_date":
            continue
        pct = last_row[col] - 100
        span = (f"(from {wide['trade_date'].iloc[0]} to {last_row['trade_date']})" if en
                else f"(từ {wide['trade_date'].iloc[0]} đến {last_row['trade_date']})")
        parts.append(f"{col}: {pct:+.2f}% {span}")
    header = ("Price performance comparison (rebased to 100 on the first day):" if en
              else "So sánh hiệu suất giá (rebase=100 tại ngày đầu):")
    lines = [header] + [f"  {p}" for p in parts]
    if result["missing"]:
        lines.append(
            f"  (no data: {result['missing']})" if en
            else f"  (không có dữ liệu: {result['missing']})"
        )
    return "\n".join(lines)


def _format_auditor_lookup(structured: dict, fts: dict | None, lang: str = "vi") -> str:
    """Hai mức độ tin cậy, KHÔNG gộp chung một danh sách (xem
    auditor_lookup.lookup_companies_by_auditor_fts()'s docstring cho lý do):
    (1) đã xác nhận - disclosures.auditor, cột có cấu trúc, nạp từ Vietstock;
    (2) chỉ nhắc tới trong văn bản đã nạp (kể cả TiniX) - ỨNG VIÊN cần đối
    chiếu thêm với trang xác nhận kiểm toán thật, KHÔNG khẳng định là auditor
    chính thức chỉ vì tên hãng xuất hiện trong chunk."""
    en = lang == "en"
    firm = structured["firm_query"]
    by_year = structured["by_year"]
    lines = []

    if by_year:
        lines.append(
            f"Confirmed (disclosures.auditor, structured, ANNUAL period) - '{firm}' is the auditor of:"
            if en else
            f"Đã xác nhận (disclosures.auditor, có cấu trúc, kỳ NĂM) - '{firm}' là auditor của:"
        )
        for year, tickers in by_year.items():
            suffix = (f"({len(tickers)} companies)" if en else f"({len(tickers)} công ty)")
            lines.append(f"  {year}: {', '.join(tickers)} {suffix}")
    else:
        lines.append(
            f"No company records '{firm}' as its auditor in the loaded disclosures.auditor "
            f"(ANNUAL period)."
            if en else
            f"Không có công ty nào ghi nhận '{firm}' là auditor trong disclosures.auditor "
            f"(kỳ NĂM) đã nạp."
        )

    if fts:
        fts_tickers_by_year = {y: ts for y, ts in fts.get("by_year", {}).items()}
        already = {t for ts in by_year.values() for t in ts}
        new_by_year = {
            y: [t for t in ts if t not in already]
            for y, ts in fts_tickers_by_year.items()
        }
        new_by_year = {y: ts for y, ts in new_by_year.items() if ts}
        if new_by_year:
            lines.append(
                f"\nMentions '{firm}' in loaded financial-statement text but NOT yet present in "
                f"disclosures.auditor (including TiniX sources - cross-check the actual audit "
                f"confirmation page before treating this as certain, since the firm name may "
                f"appear for other reasons in the notes):"
                if en else
                f"\nNhắc tới '{firm}' trong văn bản BCTC đã nạp nhưng CHƯA có trong "
                f"disclosures.auditor (kể cả nguồn TiniX - cần đối chiếu thêm trang xác "
                f"nhận kiểm toán thật trước khi coi là chắc chắn, tên hãng có thể xuất "
                f"hiện vì lý do khác trong thuyết minh):"
            )
            for year, tickers in new_by_year.items():
                lines.append(f"  {year}: {', '.join(tickers)}")

    names = structured.get("raw_auditor_names") or []
    if names:
        label = ("Full names in disclosures.auditor" if en
                 else "Tên đầy đủ trong disclosures.auditor")
        lines.append(f"\n{label}: {'; '.join(names)}")
    lines.append(
        "\n(Confirmed = disclosures.auditor, no RAG/LLM involved, no fabrication risk. "
        "The 'mentions' section = full-text match over disclosure_chunks, a lower "
        "confidence tier.)"
        if en else
        "\n(Đã xác nhận = disclosures.auditor, không qua RAG/LLM, không có rủi ro bịa. "
        "Phần 'nhắc tới' = khớp toàn văn trên disclosure_chunks, mức tin cậy thấp hơn.)"
    )
    return "\n".join(lines)


def _format_absa(result: dict, lang: str = "vi") -> str:
    """RQ5 (Week 7) - mirrors src/absa/absa_pipeline.py's own format_result(),
    duplicated here (not imported) so this module's every _format_* helper
    stays self-contained and independently readable, same convention as
    _format_dag/_format_whatif/_format_causal/_format_compare above.

    2026-09-25: English answers now show English aspect labels. The earlier
    note here argued the Vietnamese labels had to stay to match the RQ5
    taxonomy - but the taxonomy is the KEYS (profitability,
    liquidity_leverage, ...), which are unchanged; only their display text
    differs. Keeping Vietnamese labels just produced a half-Vietnamese
    English answer."""
    if not result["ok"]:
        return (f"Could not run sentiment analysis: {result['problems']}" if lang == "en"
                else f"Không phân tích được cảm xúc: {result['problems']}")
    _en_aspects = {
        "profitability": "Profitability / business results",
        "liquidity_leverage": "Liquidity & financial leverage",
        "growth_outlook": "Growth & outlook",
        "governance": "Corporate governance & management",
        "risk_legal": "Operational & legal risk",
    }
    lines = []
    for key, label in ABSA_ASPECTS.items():
        if lang == "en":
            label = _en_aspects.get(key, label)
        entry = result["aspects"].get(key, {})
        sentiment = entry.get("sentiment", "?")
        evidence = entry.get("evidence")
        line = f"- {label}: {sentiment}"
        if evidence:
            line += f' ("{evidence}")'
        lines.append(line)
    return "\n".join(lines)


# Nhãn tiếng Việt dễ đọc cho 6 khái niệm nhân quả đã biết (đồng bộ với
# _CAUSAL_CONCEPTS trong rules.py - lấy cụm từ tiếng Việt đầu tiên/tự nhiên
# nhất cho mỗi cột panel). Chỉ dùng để hiển thị câu trả lời - KHÔNG dùng để
# định tuyến (routing vẫn dựa vào rules.py).
_CAUSAL_DISPLAY_NAMES = {
    "leverage": "đòn bẩy tài chính",
    "gross_margin": "biên lợi nhuận gộp",
    "net_margin": "biên lợi nhuận ròng",
    "asset_turnover": "vòng quay tài sản",
    "current_ratio": "khả năng thanh toán hiện hành",
    "ocf_to_revenue": "dòng tiền hoạt động trên doanh thu",
}

# English counterparts of _CAUSAL_DISPLAY_NAMES (2026-09-22). Same keys, same
# ordering - kept as a separate dict rather than nested {"vi":..,"en":..} so a
# missing key still falls back to the raw column name via .get(x, x) exactly
# as before, in both languages.
_CAUSAL_DISPLAY_NAMES_EN = {
    "leverage": "financial leverage",
    "gross_margin": "gross profit margin",
    "net_margin": "net profit margin",
    "asset_turnover": "asset turnover",
    "current_ratio": "current ratio",
    "ocf_to_revenue": "operating cash flow to revenue",
}


def _format_causal(result: dict, lang: str = "vi") -> str:
    """Thêm câu trả lời ngôn ngữ tự nhiên TRƯỚC phần số liệu kỹ thuật
    (2026-09-12, sau khi Quang phản hồi bản cũ "đây đâu phải là 1 câu trả
    lời" - bản cũ chỉ in thẳng ATE/CI/n, không có câu nào trả lời trực tiếp
    câu hỏi "X ảnh hưởng đến Y như thế nào?"). Đúng tinh thần "giải thích
    được" đã thảo luận cho P1 (Eldifrawi et al. 2024) - một con số không kèm
    diễn giải không phải là một câu trả lời, dù con số đó đúng.

    Số liệu kỹ thuật (ATE/CI/n/confidence/naive) GIỮ NGUYÊN y hệt bản cũ,
    chỉ chuyển xuống dưới làm phần "Chi tiết kỹ thuật" - không đổi bất kỳ
    con số nào, chỉ thêm diễn giải phía trên."""
    en = lang == "en"
    if not result["ok"]:
        return (f"Could not estimate: {error_en(result['error'])}" if en
                else f"Không ước lượng được: {result['error']}")
    ate = result["ate_t_learner"]
    ci_lo, ci_hi = result["ate_95ci"]
    treatment, outcome = result["treatment"], result["outcome"]
    names = _CAUSAL_DISPLAY_NAMES_EN if en else _CAUSAL_DISPLAY_NAMES
    treatment_name = names.get(treatment, treatment)
    outcome_name = names.get(outcome, outcome)

    is_negative = ate < 0
    magnitude_points = abs(ate) * 100  # net_margin/gross_margin/... đều là tỷ lệ 0-1
    ci_points = sorted([ci_lo * 100, ci_hi * 100])  # KTC có thể lệch dấu quanh 0 - sort để in đúng thứ tự thấp->cao

    if en:
        direction = "NEGATIVE (decreases)" if is_negative else "POSITIVE (increases)"
        comparative = "lower" if is_negative else "higher"
        summary = (
            f"There is a {direction} relationship: companies with HIGH {treatment_name} "
            f"(above the sample median) have {outcome_name} about {magnitude_points:.1f} "
            f"percentage points {comparative} than companies with LOW {treatment_name} "
            f"(below the median), after controlling for {join_list(result['covariates'])} "
            f"(95% confidence interval: {ci_points[0]:.1f} to {ci_points[1]:.1f} percentage points)."
        )
    else:
        direction = "NGHỊCH CHIỀU (làm giảm)" if is_negative else "THUẬN CHIỀU (làm tăng)"
        comparative = "thấp hơn" if is_negative else "cao hơn"
        summary = (
            f"Có quan hệ {direction}: nhóm công ty có {treatment_name} CAO (trên trung vị mẫu) "
            f"có {outcome_name} {comparative} khoảng {magnitude_points:.1f} điểm phần trăm so với "
            f"nhóm {treatment_name} THẤP (dưới trung vị), sau khi đã kiểm soát {result['covariates']} "
            f"(khoảng tin cậy 95%: {ci_points[0]:.1f} đến {ci_points[1]:.1f} điểm phần trăm)."
        )
    # Neo literature CHỈ áp dụng đúng cặp leverage->net_margin đã có trích
    # dẫn thật (xem docs/PROGRESS.md 2026-09-04/2026-09-12) - không áp dụng
    # cho các cặp khác để tránh gán sai trích dẫn.
    if treatment == "leverage" and outcome == "net_margin":
        summary += (
            " The result matches the DIRECTION predicted by pecking-order theory (Myers 1984; "
            "emerging-market empirical evidence: Abdeljawad & Jaradat 2025) - but this is a "
            "GROUP COMPARISON (high/low versus the median), NOT the marginal effect of "
            "'+1 unit of leverage', and it rests on only 2 treatment/outcome pairs, so it is "
            "NOT sufficient evidence to confirm any theory."
            if en else
            " Kết quả khớp HƯỚNG với lý thuyết trật tự ưu tiên vốn (pecking-order, "
            "Myers 1984; bằng chứng thực nghiệm thị trường mới nổi: Abdeljawad & Jaradat "
            "2025) - nhưng đây là SO SÁNH NHÓM (cao/thấp so với trung vị), KHÔNG phải hiệu "
            "ứng biên của '+1 đơn vị đòn bẩy', và chỉ dựa trên 2 cặp treatment/outcome nên "
            "CHƯA đủ bằng chứng để khẳng định lý thuyết nào đúng."
        )
    else:
        summary += (
            " This is an OBSERVATIONAL analysis (not a randomised experiment) and a GROUP "
            "COMPARISON (high/low versus the median), not a linear marginal effect - so read "
            "it as 'an association with known confounders controlled for', not as absolute "
            "causal evidence."
            if en else
            " Đây là phân tích QUAN SÁT (không phải thử nghiệm ngẫu nhiên hoá) và là SO "
            "SÁNH NHÓM (cao/thấp so với trung vị), không phải hiệu ứng biên tuyến tính - nên "
            "diễn giải là 'liên hệ có kiểm soát nhiễu đã biết', không phải bằng chứng nhân "
            "quả tuyệt đối."
        )

    if en:
        detail = (
            f"Technical detail: ATE estimate (T-learner) of '{treatment}' on '{outcome}' "
            f"(controlling for {join_list(result['covariates'])}): {ate:.4f} (95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]), "
            f"n={result['n']} (treated={result['n_treated']}, control={result['n_control']}). "
            f"Confidence level: {confidence_label(result['confidence'])}. "
            f"Naive comparison (no confounder control): {result['naive_mean_diff']:.4f}."
        )
    else:
        detail = (
            f"Chi tiết kỹ thuật: Ước lượng ATE (T-learner) của '{treatment}' lên '{outcome}' "
            f"(kiểm soát {result['covariates']}): {ate:.4f} (KTC 95%: [{ci_lo:.4f}, {ci_hi:.4f}]), "
            f"n={result['n']} (treated={result['n_treated']}, control={result['n_control']}). "
            f"Mức độ tin cậy: {result['confidence']}. "
            f"So sánh naive (không kiểm soát biến gây nhiễu): {result['naive_mean_diff']:.4f}."
        )
    return f"{summary}\n\n{detail}"


# 2026-09-14 - a TINY, allow-listed set of well-known public figures who get
# an extra, factually-safe one-line bio in the third_party_opinion message
# (added specifically for Quang's audit feedback on id=51 "Warren Buffett
# nghĩ gì về HPG?" - "phải trả lời sao cho nó hợp lý chứ ví dụ như warren
# buffett là người ở đâu tại sao phải nghỉ và liên quan"). Deliberately NOT
# a lookup the LLM fills in at request time (that's exactly the "bịa một ý
# kiến" risk this branch exists to avoid) - each bio here is a well-known,
# easily independently verifiable public fact (nationality/role), never an
# opinion attributed to the person. Any name NOT in this dict still gets
# the honest generic explanation below - the dict only makes the message
# richer for names common enough to be worth hardcoding, it is never
# required for the branch to work correctly.
_KNOWN_PUBLIC_FIGURES = {
    "warren buffett": {
        "vi": "Warren Buffett là nhà đầu tư người Mỹ, Chủ tịch kiêm CEO tập đoàn "
              "Berkshire Hathaway, chủ yếu đầu tư vào các công ty niêm yết tại Mỹ.",
        "en": "Warren Buffett is an American investor, Chairman and CEO of "
              "Berkshire Hathaway, whose investments are overwhelmingly in "
              "US-listed companies.",
    },
}


def _format_out_of_scope(intent: QueryIntent, lang: str = "vi") -> str:
    """Deterministic, zero-LLM-call explanation for the 3 OUT_OF_SCOPE
    question types (see rules.py's markers/docstring) - replaces the old
    behavior of falling through to RAG and getting scope_notes.py's generic
    "câu hỏi không khớp, thử nêu rõ kỳ/chỉ tiêu" message, which is actively
    misleading for these 3 types (no rephrasing makes them answerable).
    Bilingual (`lang` resolved once per request in route() via
    src.core.lang_detect.resolve_answer_language, same value every other
    formatter in this module receives) - EN wording is a plain translation,
    not a separate policy."""
    from src.router.scope_notes import _SUPPORTED_KINDS, _SUPPORTED_KINDS_EN
    supported = _SUPPORTED_KINDS_EN if lang == "en" else _SUPPORTED_KINDS

    if intent.oos_kind == "third_party_opinion":
        person = intent.oos_person or ("this person" if lang == "en" else "người này")
        bio = _KNOWN_PUBLIC_FIGURES.get(person.lower(), {}).get(lang)
        if lang == "en":
            head = (f"This asks for {person}'s personal opinion, not a fact published in "
                    f"a company's financial statements or disclosures. "
                    + (bio + " " if bio else "")
                    + f"This system verifies facts already published in a company's own "
                    f"documents - it does not fabricate an opinion and attribute it to "
                    f"{person} when there is no verified public statement on file.")
        else:
            head = (f"Đây là câu hỏi xin quan điểm cá nhân của {person}, không phải một sự "
                    f"kiện đã công bố trong BCTC/tài liệu công ty. "
                    + (bio + " " if bio else "")
                    + f"Hệ thống này xác minh sự kiện đã công bố thật, không tự bịa một ý "
                    f"kiến rồi gán cho {person} khi không có phát biểu công khai đã xác minh.")
        return head + "\n" + supported

    if intent.oos_kind == "investment_advice":
        if lang == "en":
            head = ("This asks for personal investment advice (buy/sell a specific stock). "
                    "This system provides verified financial data and analysis, not "
                    "buy/sell recommendations - that is a deliberate compliance boundary, "
                    "not a data gap. You can ask about a specific financial figure/ratio "
                    "instead (e.g. \"What was HPG's gross margin in 2025?\") to inform your "
                    "own judgment.")
        else:
            head = ("Đây là yêu cầu tư vấn đầu tư cá nhân (nên mua/bán một cổ phiếu cụ "
                    "thể). Hệ thống này cung cấp dữ liệu/phân tích tài chính đã kiểm chứng, "
                    "không đưa ra khuyến nghị mua/bán - đây là ranh giới tuân thủ cố ý, "
                    "không phải khoảng trống dữ liệu. Bạn có thể hỏi một chỉ tiêu/tỷ lệ tài "
                    "chính cụ thể (vd: \"Biên lợi nhuận gộp của HPG năm 2025 là bao nhiêu?\") "
                    "để tự đánh giá.")
        return head + "\n" + supported

    # future_prediction (default/fallback oos_kind)
    if lang == "en":
        head = ("This asks for a future stock price prediction. This system verifies "
                "financial facts already published in real financial statements/"
                "disclosures - it is not a price-forecasting model. A specific number "
                "here would be a guess, not a verified fact, so this system does not "
                "provide one.")
    else:
        head = ("Đây là câu hỏi dự đoán giá cổ phiếu trong tương lai. Hệ thống này xác "
                "minh sự kiện tài chính ĐÃ công bố thật (fact verification), không phải "
                "một mô hình dự báo giá - trả lời bằng một con số cụ thể ở đây sẽ là suy "
                "đoán, không phải sự thật đã kiểm chứng, nên hệ thống không đưa ra con số.")
    return head + "\n" + supported


def _dispatch_whatif(intent: QueryIntent) -> dict:
    if intent.whatif_kind == "sensitivity":
        year, period = intent.periods[0]
        raw = sensitivity_what_if(
            intent.ticker, Period(year, period),
            cogs_pct_change=intent.cogs_pct_change,
            revenue_pct_change=intent.revenue_pct_change,
        )
    elif intent.whatif_kind == "decompose_margin":
        (y1, p1), (y2, p2) = intent.periods
        raw = decompose_gross_margin_change(intent.ticker, Period(y1, p1), Period(y2, p2))
    elif intent.whatif_kind == "growth_rate":
        (y1, p1), (y2, p2) = intent.periods
        raw = growth_rate(
            intent.ticker, intent.indicator_stype, intent.indicator_ms_code,
            Period(y1, p1), Period(y2, p2),
        )
    elif intent.whatif_kind == "gold_sensitivity":
        # year - 1, not the current year - see the matching comment in
        # rules.py's pct_match branch (2026-07-19 fix): the current fiscal
        # year's annual report doesn't exist yet mid-year, so defaulting to
        # "today" almost always produced "thiếu dữ liệu revenue/cogs".
        year = intent.periods[0][0] if intent.periods else date.today().year - 1
        raw = sensitivity_what_if_gold(intent.ticker, Period(year, "YEAR"))
    elif intent.whatif_kind == "oil_sensitivity":
        year = intent.periods[0][0] if intent.periods else date.today().year - 1
        raw = sensitivity_what_if_oil(intent.ticker, Period(year, "YEAR"))
    elif intent.whatif_kind == "macro_sensitivity":
        # 2026-09-15: same year-default rationale as gold/oil above.
        year = intent.periods[0][0] if intent.periods else date.today().year - 1
        raw = sensitivity_what_if_macro(intent.ticker, Period(year, "YEAR"))
    elif intent.whatif_kind == "interest_rate_sensitivity":
        year = intent.periods[0][0] if intent.periods else date.today().year - 1
        raw = sensitivity_what_if_interest_rate(intent.ticker, Period(year, "YEAR"))
    elif intent.whatif_kind == "trade_sensitivity":
        # 2026-09-15: same year-default rationale as gold/oil/macro above.
        year = intent.periods[0][0] if intent.periods else date.today().year - 1
        raw = sensitivity_what_if_trade(intent.ticker, Period(year, "YEAR"))
    else:
        raise ValueError(f"Unknown whatif_kind: {intent.whatif_kind}")
    return raw


def route(question: str, top_k: int = 5, with_rationale: bool = False,
          use_regime_signal: bool = True, ui_lang: str | None = None,
          answer_lang: str | None = None) -> dict:
    """Classify `question` and dispatch to the right branch. Returns:
    {"branch", "intent_reason", "question", "answer", "raw", "fallback_from"}
    - "branch" is the branch that actually produced the answer (may differ
    from the classifier's first choice if a DAG->RAG fallback happened;
    "fallback_from" records the original choice in that case).

    `with_rationale=True` adds a "rationale" key (plain-language narration
    of the already-computed numbers, via src/models/rationale.py) for the
    whatif/causal branches only - the ones whose native output is a
    formula-shaped string, not already-natural-language RAG/DAG prose.
    Off by default: it's an extra LLM call (extra latency, requires Ollama
    running, which whatif/causal/dag/compare otherwise don't need) - opt in
    explicitly rather than silently changing existing callers' behavior.

    `use_regime_signal` (RQ3, Trend-Aligned Dynamic Reranking, default
    True) is forwarded into every rag_answer() call below - set to False
    to reproduce the "Router tĩnh" ablation baseline for RQ3 (see
    src/retrieval/rag_pipeline.answer()'s docstring). Only affects the RAG
    branch; DAG/whatif/causal/compare don't use RAG chunk ranking at all.

    `ui_lang` (2026-09-22) is the interface language toggle, used ONLY as a
    tie-breaker: the answer language is resolved once here, from the question
    itself first (see lang_detect.resolve_answer_language()), and every
    branch below - formatter, trend enrichment, fallback note and classifier
    reason - is rendered in that single resolved language. Resolving it once
    at the top (rather than calling detect_language() ad hoc per branch, as
    this function used to) is what guarantees an answer can never come back
    half Vietnamese and half English.

    `answer_lang` ("vi"|"en") overrides that resolution outright. The QA
    service passes it because of a multi-turn hazard: it hands route() the
    CONTEXT-RESOLVED question (api/services/qa/main.py's _resolve_query()),
    which on a follow-up turn is an LLM rewrite, not what the user typed.
    That rewrite is produced by a Vietnamese-oriented prompt, so an English
    follow-up ("and for 2024?") can come back rewritten in Vietnamese - and
    resolving the language from the rewrite would silently flip the answer
    language mid-conversation, exactly where a demo is most visible. The
    service therefore resolves the language from the ORIGINAL question once
    and pins it here. Left None (CLI, benchmarks, tests), behaviour is
    unchanged: resolve from `question` as described above.
    """
    lang = answer_lang or resolve_answer_language(question, ui_lang)
    en = lang == "en"
    intent = classify_intent(question, lang)
    result = {
        "question": question, "branch": intent.branch, "intent_reason": intent.reason,
        "raw": None, "fallback_from": None,
    }

    if intent.branch == "dag":
        parsed = parse_question(question)
        raw = dag_execute(parsed)
        if raw["ok"]:
            result["raw"] = raw
            answer = _format_dag(raw, lang)
            # 2026-08-06 (yêu cầu từ phiên test Chat): chỉ tiêu biến động
            # theo kỳ thì không trả lời "chung chung" 1 con số - nối thêm
            # diễn biến quý + %QoQ/%YoY. Fail-soft: enrich() tự trả None
            # khi thiếu chuỗi so sánh, không bao giờ phá câu trả lời gốc.
            trend_part = trend.enrich(parsed, raw, lang)
            if trend_part:
                answer += "\n" + trend_part
            result["answer"] = answer
            return result
        # 2026-08-06: câu hỏi theo NĂM nhưng DB chỉ có các dòng QUÝ (export
        # quý-only) - dựng câu trả lời từ chuỗi quý (cộng dồn/Q4 tùy loại
        # báo cáo) trước khi nghĩ tới RAG. Đây chính là ca "Tổng tài sản
        # của HPG năm 2025" rơi fallback oan trong phiên test 2026-08-06.
        from_quarters = trend.answer_from_quarters(parsed, lang)
        if from_quarters:
            result["raw"] = {"ok": True, "source": "quarter_series", "parsed": vars(parsed)}
            result["intent_reason"] += (
                f" | No YEAR row -> {from_quarters['note']}." if en
                else f" | Không có dòng YEAR -> {from_quarters['note']}."
            )
            result["answer"] = from_quarters["answer"]
            return result
        # Adaptive fallback: DAG recognized the question's shape but found
        # no matching row - try RAG before giving up (see module docstring).
        rag_raw = rag_answer(question, top_k=top_k, ticker=intent.ticker, use_regime_signal=use_regime_signal)
        result["branch"] = "rag"
        result["fallback_from"] = "dag"
        result["intent_reason"] += (
            f" | DAG has no data ({error_en(raw['error'])}) -> falling back to RAG." if en
            else f" | DAG không có dữ liệu ({raw['error']}) -> chuyển sang RAG."
        )
        result["raw"] = rag_raw
        # 2026-08-06: "không tìm thấy" trần trụi -> nối thêm hướng dẫn phạm
        # vi (scope_notes.py) - deterministic, không thêm lượt gọi LLM.
        result["answer"] = polish_not_found(rag_raw["answer"], intent.ticker, lang=lang)
        return result

    if intent.branch == "ratio_lookup":
        # 2026-09-15: same adaptive DAG->RAG fallback philosophy as the
        # "dag" branch above (module docstring) - a ratio question can
        # recognize its own shape correctly (ticker+period+ratio_name all
        # resolved) but still find no row if that ticker/period wasn't in
        # the ingested Chi-so-tai-chinh batch - fall back to RAG rather
        # than surface a bare "no row" error.
        from src.structured.ratio_lookup import ParsedRatioQuery, execute as ratio_execute
        year, period = intent.periods[0]
        parsed_ratio = ParsedRatioQuery(
            ticker=intent.ticker, fiscal_year=year, period=period,
            ratio_group=intent.indicator_stype, ratio_name=intent.indicator_ms_code,
            matched_alias=None,
        )
        raw = ratio_execute(parsed_ratio)
        if raw["ok"]:
            result["raw"] = raw
            result["answer"] = _format_ratio_lookup(raw, lang)
            return result
        rag_raw = rag_answer(question, top_k=top_k, ticker=intent.ticker, use_regime_signal=use_regime_signal)
        result["branch"] = "rag"
        result["fallback_from"] = "ratio_lookup"
        result["intent_reason"] += (
            f" | RATIO_LOOKUP has no data ({error_en(raw['error'])}) -> falling back to RAG." if en
            else f" | RATIO_LOOKUP không có dữ liệu ({raw['error']}) -> chuyển sang RAG."
        )
        result["raw"] = rag_raw
        result["answer"] = polish_not_found(rag_raw["answer"], intent.ticker, lang=lang)
        return result

    if intent.branch == "price_snapshot":
        # 2026-09-21: "giá/cổ phiếu X biến động hôm nay thế nào" - xem
        # rules.py's PRICE_TODAY_MARKERS cho điều kiện khớp, và
        # market_factors.latest_price_snapshot()'s docstring cho lý do
        # KHÔNG dùng compare_tickers() (cần >=2 mã) hay Branch B free-form
        # SQL (0% tin cậy theo Chapter 4) cho câu hỏi này.
        # 2026-09-22: cùng một nhánh phục vụ CẢ hai khung thời gian - một
        # phiên ("giá X hôm nay") hay một khoảng ("giá X vài ngày gần đây").
        # rules.py's _price_window_days() quyết định; xem QueryIntent.
        n_days = getattr(intent, "price_window_days", 1) or 1
        if not intent.ticker:
            raw = {
                "ok": False,
                "error": ("Could not identify a ticker symbol in the question." if en
                          else "Không xác định được mã cổ phiếu trong câu hỏi."),
            }
            result["raw"] = raw
            result["answer"] = _format_price_snapshot(raw, lang)
            return result
        d0 = getattr(intent, "price_start_date", None)
        d1 = getattr(intent, "price_end_date", None)
        if d0 and d1:
            raw = price_range_series(intent.ticker, d0, d1)
            result["raw"] = raw
            result["answer"] = _format_price_range(raw, lang)
            return result
        if n_days > 1:
            raw = price_window_series(intent.ticker, n_days)
            result["raw"] = raw
            result["answer"] = _format_price_series(raw, lang)
            return result
        raw = latest_price_snapshot(intent.ticker)
        result["raw"] = raw
        result["answer"] = _format_price_snapshot(raw, lang)
        return result

    if intent.branch == "bank_rate_lookup":
        # 2026-09-15: aggregate query, no ticker/DB-fallback needed (same
        # zero-LLM-call, zero-hallucination-risk shape as auditor_lookup).
        raw = best_deposit_rate_banks()
        result["raw"] = raw
        result["answer"] = _format_bank_rate_lookup(raw, lang)
        return result

    if intent.branch == "dag_multi":
        # 2026-09-14 - fixes P3 audit bug id=42 ("Tổng tài sản của cả VNM,
        # HPG và PNJ năm 2024 là bao nhiêu?"): the single-ticker "dag"
        # branch above resolves exactly ONE ParsedQuery (built around
        # query_builder._extract_ticker()'s single match) and would have
        # silently answered ONLY the first named ticker, dropping the rest
        # with no indication anything was skipped. Here we parse the
        # question ONCE for its indicator/period/year (shared across every
        # ticker - see rules.py's DAG_MULTI check, which only fires when
        # these already resolved), then substitute EACH ticker in turn via
        # dataclasses.replace() and run the SAME dag -> quarter-series ->
        # RAG fallback chain as the single-ticker path above, per ticker -
        # so a mix of tiers (e.g. one PANEL ticker with real DAG data next
        # to two CORE_16 tickers with none) is reported HONESTLY per ticker
        # instead of averaged into one misleading answer.
        from dataclasses import replace as _dc_replace

        probe = parse_question(question)
        per_ticker_lines = []
        any_ok = False
        per_ticker_raw = {}
        for t in intent.tickers:
            parsed_t = _dc_replace(probe, ticker=t)
            raw_t = dag_execute(parsed_t)
            if raw_t["ok"]:
                any_ok = True
                line = _format_dag(raw_t, lang)
                trend_part = trend.enrich(parsed_t, raw_t, lang)
                if trend_part:
                    line += " " + trend_part.replace("\n", " ")
                per_ticker_lines.append(f"- {t}: {line}")
                per_ticker_raw[t] = raw_t
                continue
            from_quarters_t = trend.answer_from_quarters(parsed_t, lang)
            if from_quarters_t:
                any_ok = True
                per_ticker_lines.append(f"- {t}: {from_quarters_t['answer']}")
                per_ticker_raw[t] = {"ok": True, "source": "quarter_series"}
                continue
            # 2026-09-25: câu hỏi gửi sang RAG cho TỪNG mã phải nói rõ là chỉ
            # hỏi mã đó. Trước đây nó nhận nguyên câu nhiều mã, nên lượt test
            # "What were the total assets of both HPG and VNM in 2024?" cho ra
            # dòng VNM bắt đầu bằng "No information found ... about the total
            # assets of HPG" - model trả lời lẫn sang mã kia. Retrieval vẫn
            # lọc theo `ticker=t`; hậu tố này chỉ nói cho phần SINH CHỮ biết
            # đang trả lời cho ai. Giữ nguyên câu gốc (không dựng câu mới) để
            # detect_language trong rag_answer vẫn nhận đúng ngôn ngữ.
            q_t = (f"{question} (Answer only for {t}.)" if en
                   else f"{question} (Chỉ trả lời cho mã {t}.)")
            rag_raw_t = rag_answer(q_t, top_k=top_k, ticker=t, use_regime_signal=use_regime_signal)
            rag_ans_t = polish_not_found(rag_raw_t["answer"], t, lang=lang)
            per_ticker_lines.append(f"- {t}: {rag_ans_t}")
            per_ticker_raw[t] = rag_raw_t

        result["raw"] = {"ok": any_ok, "per_ticker": per_ticker_raw}
        result["answer"] = "\n".join(per_ticker_lines)
        return result

    if intent.branch == "dag_trend":
        # 2026-08-07: câu hỏi xu hướng nhiều năm không nêu năm cụ thể - xem
        # rules.py's TREND_LANGUAGE_MARKERS. Dựng chuỗi năm có cấu trúc từ
        # financial_line_items trước khi nghĩ tới RAG - tránh để LLM tự suy
        # diễn xu hướng từ đoạn trích rời rạc.
        parsed = parse_question(question)
        trend_result = trend.answer_recent_years(parsed, lang)
        if trend_result:
            result["raw"] = {"ok": True, "source": "recent_years_trend", "parsed": vars(parsed)}
            result["answer"] = trend_result["answer"]
            return result
        # Không đủ chuỗi năm có cấu trúc (< 2 năm) - RAG vẫn có thể tìm
        # được nội dung định tính liên quan, cùng tinh thần fallback
        # DAG->RAG ở nhánh "dag" phía trên.
        rag_raw = rag_answer(question, top_k=top_k, ticker=intent.ticker, use_regime_signal=use_regime_signal)
        result["branch"] = "rag"
        result["fallback_from"] = "dag_trend"
        result["intent_reason"] += (
            " | Not enough structured yearly data (< 2 years) -> falling back to RAG." if en
            else " | Không đủ chuỗi năm có cấu trúc (< 2 năm) -> chuyển sang RAG."
        )
        result["raw"] = rag_raw
        result["answer"] = polish_not_found(rag_raw["answer"], intent.ticker, lang=lang)
        return result

    if intent.branch == "rag":
        raw = rag_answer(question, top_k=top_k, ticker=intent.ticker, use_regime_signal=use_regime_signal)
        result["raw"] = raw
        result["answer"] = polish_not_found(raw["answer"], intent.ticker, lang=lang)
        return result

    if intent.branch == "out_of_scope":
        # No RAG/LLM call at all - see rules.py's OUT_OF_SCOPE markers and
        # _format_out_of_scope()'s docstring for why (deterministic, ~0s,
        # zero hallucination risk, same principle as the whatif/dag
        # branches' own "no data -> honest error, not a guess" behavior).
        result["raw"] = {"ok": True, "oos_kind": intent.oos_kind, "oos_person": intent.oos_person}
        result["answer"] = _format_out_of_scope(intent, lang=lang)
        return result

    if intent.branch == "whatif":
        raw = _dispatch_whatif(intent)
        result["raw"] = raw
        result["answer"] = _format_whatif(intent.whatif_kind, raw, lang)
        if with_rationale and raw.get("ok"):
            result["rationale"] = generate_rationale(question, result["answer"])["rationale"]
        return result

    if intent.branch == "auditor_lookup":
        if intent.ticker:
            from src.structured.auditor_history import answer_auditor_history
            years = [year for year, _ in (intent.periods or [])]
            raw = answer_auditor_history(question, intent.ticker, years)
            result.update(raw=raw, answer=raw['answer'])
            return result
        raw = lookup_companies_by_auditor(intent.auditor_query)
        # Best-effort: quét toàn văn TiniX/PDF bổ sung không được để hỏng
        # câu trả lời có cấu trúc đã có nếu FTS lỗi (index thiếu, DB lag...)
        # - fail-soft, giống mọi lớp bổ sung khác trong router này.
        try:
            fts = lookup_companies_by_auditor_fts(intent.auditor_query)
        except Exception:
            fts = None
        result["raw"] = raw
        result["raw"]["fts"] = fts
        result["answer"] = _format_auditor_lookup(raw, fts, lang)
        return result

    if intent.branch == "absa":
        raw = absa_analyze_ticker(intent.ticker)
        result["raw"] = raw
        result["answer"] = _format_absa(raw, lang)
        return result

    if intent.branch == "causal":
        raw = estimate_ate(treatment_col=intent.treatment_col, outcome_col=intent.outcome_col)
        result["raw"] = raw
        result["answer"] = _format_causal(raw, lang)
        if with_rationale and raw.get("ok"):
            result["rationale"] = generate_rationale(question, result["answer"])["rationale"]
        return result

    if intent.branch == "compare":
        raw = compare_tickers(intent.tickers)
        if raw["ok"] and intent.compare_with_gold:
            # Attach each ticker's return-correlation with gold as extra
            # context (not part of compare_tickers() itself, which is pure
            # price-series retrieval) - only computed when the question
            # actually mentioned gold, to avoid an unnecessary extra query.
            raw["gold_correlations"] = {
                t: correlate_ticker_with_gold(t) for t in intent.tickers if t not in raw["missing"]
            }
        result["raw"] = raw
        result["answer"] = _format_compare(raw, lang)
        return result

    # whatif_unsupported / causal_unsupported / compare_unsupported -
    # classifier recognized the question's intent but couldn't extract
    # enough to dispatch. Explain what's missing rather than silently
    # falling back to RAG, since RAG genuinely cannot answer a
    # what-if/causal/compare question either.
    result["answer"] = (f"Cannot answer automatically yet: {intent.reason}" if en
                        else f"Chưa thể trả lời tự động: {intent.reason}")
    return result


def format_route_result(result: dict) -> str:
    """CLI/demo-friendly rendering - matches rag_pipeline.py / dag_pipeline.py's style."""
    lines = [f"[branch={result['branch']} | reason={result['intent_reason']}]", "", result["answer"]]
    is_web_fallback = result["branch"] == "rag" and result["raw"] and result["raw"].get("source_tier") == "web_fallback"
    if is_web_fallback:
        lines.append(
            "\n" + "=" * 78 + "\n"
            "CẢNH BÁO: NGUỒN BÁO CHÍ/WEB - KHÔNG PHẢI TÀI LIỆU CÔNG BỐ CHÍNH THỨC.\n"
            + "=" * 78
        )
    if result["branch"] in ("rag", "absa") and result["raw"] and result["raw"].get("sources"):
        lines.append("\nNguồn:")
        for i, s in enumerate(result["raw"]["sources"], start=1):
            if is_web_fallback:
                date_part = f" - ngày {s['published_date']}" if s.get("published_date") else ""
                lines.append(
                    f"  [{i}] {s['domain']}{date_part} - {s['source_url']} "
                    f"(similarity={s['similarity']})"
                )
                continue
            lines.append(
                f"  [{i}] {s['ticker']} - {s['doctype']} - {s['source_file']} "
                f"trang {s['page']} (similarity={s['similarity']})"
            )
    if result["branch"] == "rag" and result["raw"] and result["raw"].get("doctype_filter"):
        lines.append(f"\n[FinAgentBench - lọc theo loại tài liệu: {result['raw']['doctype_filter']}]")
    if result["branch"] == "rag" and result["raw"] and result["raw"].get("market_regime"):
        tag = f"\n[RQ3 - chu kỳ thị trường: {result['raw']['market_regime']}]"
        if result["raw"].get("regime_rerank_note"):
            tag += f" ({result['raw']['regime_rerank_note']})"
        lines.append(tag)
    if result.get("rationale"):
        lines.append(f"\nGiải thích:\n{result['rationale']}")
    return "\n".join(lines)


if __name__ == "__main__":
    args = sys.argv[1:]
    want_rationale = "--rationale" in args
    if want_rationale:
        args = [a for a in args if a != "--rationale"]

    q = " ".join(args) or "Tổng tài sản của VNM quý 1 năm 2026 là bao nhiêu?"
    print(f"Q: {q}" + (" (+rationale)" if want_rationale else ""))
    res = route(q, with_rationale=want_rationale)
    print(format_route_result(res))
