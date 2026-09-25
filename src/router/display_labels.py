"""English display labels for text that reaches an answer from the DATA,
not from a formatter (2026-09-25).

Why this module exists. The 2026-09-22 bilingual pass translated every
formatter's own wording, but an English answer still came out half
Vietnamese, because formatters also print strings that arrive ready-made:

  * database labels - financial_line_items.item_name ("TỔNG CỘNG TÀI SẢN",
    "II. Tài sản cố định"), financial_ratios.ratio_name / ratio_group / unit
    ("Tỷ số thanh toán hiện hành (ngắn hạn)", "NHÓM CHỈ SỐ THANH KHOẢN",
    "Lần"), macro_indicators series names ("Sắt thép", "Sản xuất kim loại");
  * prose written by lower modules into their result dicts - calculator's
    "assumption"/"note", macro_lookup's ranking "note", treatment_effect's
    confidence label, query_builder's error strings.

The test used to sign off that pass fed the formatters hand-written result
dicts, so none of this real data ever went through it. It passed because it
could not see the problem.

Design rules:
  1. Nothing is translated ad hoc. Line-item and ratio labels are DERIVED from
     the English aliases the project already maintains for question parsing
     (schema._ENGLISH_ALIASES / _UNALIASED_MS_CODE_ALIASES_EN,
     ratio_schema._RATIOS[3]) - the same vocabulary the router already
     accepts from English users, so what the answer says is what a user can
     type back. All 45 core line items and all 71 ratios have one.
  2. Anything without a known label falls back to the ORIGINAL text rather
     than a guess. A Vietnamese word in an English answer is a visible,
     honest gap; a wrong English word is not.
  3. Numbers, tickers, MS codes and period labels never pass through here.
  4. The Vietnamese path never calls this module, so every Vietnamese
     answer is byte-for-byte what it was.
"""
from __future__ import annotations

import re
from functools import lru_cache


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:] if s else s


@lru_cache(maxsize=1)
def _line_item_labels() -> dict:
    from src.structured import schema as S

    labels: dict = {}
    # first alias listed for a code wins - the lists are ordered with the
    # plainest wording first ("total assets" before "total asset")
    for alias, stype, code in list(S._ENGLISH_ALIASES) + list(S._UNALIASED_MS_CODE_ALIASES_EN):
        labels.setdefault((stype, str(code)), alias)
    return labels


def line_item_label(statement_type: str, ms_code, fallback: str) -> str:
    """'TỔNG CỘNG TÀI SẢN' (balance_sheet, 280) -> 'Total assets'."""
    label = _line_item_labels().get((statement_type, str(ms_code)))
    return _cap(label) if label else fallback


_ABBR_RE = re.compile(r"\(([A-Z][A-Z/]{1,7})\)")


@lru_cache(maxsize=1)
def _ratio_labels() -> dict:
    from src.structured import ratio_schema as R

    labels = {}
    for entry in R._RATIOS:
        name, _group, _vi, en = entry[0], entry[1], entry[2], entry[3]
        if not en:
            continue
        label = _cap(max(en, key=len))
        # keep the market abbreviation the Vietnamese name carries - "(P/E)",
        # "(ROEA)" - it is what analysts actually say, in either language
        abbr = _ABBR_RE.search(name)
        if abbr and abbr.group(1).lower() not in label.lower():
            label = f"{label} ({abbr.group(1)})"
        labels[name] = label
    return labels


def ratio_label(ratio_name: str) -> str:
    return _ratio_labels().get(ratio_name, ratio_name)


_RATIO_GROUPS = {
    "NHÓM CHỈ SỐ ĐỊNH GIÁ": "valuation",
    "NHÓM CHỈ SỐ SINH LỢI": "profitability",
    "NHÓM CHỈ SỐ TĂNG TRƯỞNG": "growth",
    "NHÓM CHỈ SỐ THANH KHOẢN": "liquidity",
    "NHÓM CHỈ SỐ HIỆU QUẢ HOẠT ĐỘNG": "efficiency",
    "NHÓM CHỈ SỐ ĐÒN BẨY TÀI CHÍNH": "leverage",
    "NHÓM CHỈ SỐ DÒNG TIỀN": "cash flow",
}


def ratio_group_label(group: str) -> str:
    return _RATIO_GROUPS.get(group, group)


_UNITS = {
    "lần": "x", "vòng": "turns", "ngày": "days", "đồng": "VND",
    "vnd": "VND", "vnđ": "VND", "%": "%",
}


def unit_label(unit) -> str:
    if unit is None:
        return ""
    return _UNITS.get(str(unit).strip().lower(), str(unit))


# macro_indicators series names used by the what-if branches. Small and
# closed on purpose: calculator/macro_lookup only ever map tickers to these
# (see TICKER_INDUSTRY_MAP / TRADE_INDUSTRY_MAP in src/causal/macro_lookup.py).
_SERIES = {
    "Sắt thép": "steel",
    "Phân bón": "fertiliser",
    "Sản xuất kim loại": "metal manufacturing",
    "Sản xuất, chế biến thực phẩm": "food production and processing",
    "Sản xuất đồ uống": "beverage production",
    "Sản xuất hoá chất và sản phẩm hoá chất": "chemicals and chemical products",
    "Sản xuất than cốc, sản phẩm dầu mỏ tinh chế": "coke and refined petroleum products",
    "Bán lẻ hàng hóa": "retail sales of goods",
    "Qua đêm": "overnight",
}


def series_label(name: str) -> str:
    return _SERIES.get(name, name)


# --- prose that lower modules write into result dicts ----------------------

ASSUMPTION_EN = "ceteris paribus - all other factors held constant"


def confidence_label(text: str) -> str:
    """treatment_effect._confidence_label() output -> English. The leading
    token and the (n ...) part are already English; only the gloss is not."""
    if not text:
        return text
    head = text.split(" - ", 1)[0]
    gloss = {
        "exploratory": "illustrative only, NOT strong enough for a causal conclusion",
        "preliminary": "a preliminary tendency, interpret with caution",
        "standard power": "sample size adequate for a standard estimate",
    }
    for key, en in gloss.items():
        if head.startswith(key):
            return f"{head} - {en}"
    return text


def data_error(text: str) -> str:
    """query_builder / ratio_lookup error strings carry a Vietnamese gloss in
    parentheses; the English part before it is already the full message."""
    if not text:
        return text
    return (text
            .replace(" (chưa nạp dữ liệu cho kỳ/mã này)", " (no data loaded for this ticker/period)")
            .replace("(chưa nạp dữ liệu cho kỳ/mã này)", "(no data loaded for this ticker/period)"))


# Every Vietnamese error string the router's branches can surface, as
# (regex, English template) pairs. Collected 2026-09-25 by walking the AST of
# calculator.py, market_factors.py, macro_lookup.py and treatment_effect.py
# for every "error" value containing Vietnamese - 39 distinct messages, all
# covered below. Translated HERE rather than in those modules because those
# modules are called directly by the benchmark scripts; changing their
# output would change recorded results. Order matters only where one
# pattern is a prefix of another (the more specific one comes first).
# Anything that matches nothing is returned unchanged (rule 2 above).
_ERRORS = [
    (r"Thiếu dữ liệu revenue/cogs cho (\S+) (\S+)", r"Missing revenue/COGS data for \1 \2"),
    (r"Thiếu dữ liệu Chi phí tài chính/Lợi nhuận trước thuế cho (\S+) (\S+)",
     r"Missing financial expense / pre-tax profit data for \1 \2"),
    (r"Thiếu dữ liệu macro \((.*?)\): kỳ1=(.*?), kỳ2=(.*)", r"Missing macro data (\1): period1=\2, period2=\3"),
    (r"Thiếu dữ liệu \((.*?)\): kỳ1=(.*?), kỳ2=(.*)", r"Missing data (\1): period1=\2, period2=\3"),
    (r"Thiếu dữ liệu trước hoặc sau ngày sự kiện trong khung đã chọn\.",
     "Missing data before or after the event date in the selected window."),
    (r"Thiếu dữ liệu: (.*)", r"Missing data: \1"),
    (r"Giá trị kỳ gốc bằng 0, không tính được % tăng trưởng\.",
     "The base-period value is 0, so a growth % cannot be computed."),
    (r"Giá trị kỳ gốc bằng 0\.", "The base-period value is 0."),
    (r"Doanh thu bằng 0 ở 1 trong 2 kỳ, không decompose được\.",
     "Revenue is 0 in one of the two periods, so it cannot be decomposed."),
    (r"(\S+) chưa có ánh xạ ngành trong TICKER_INDUSTRY_MAP.*",
     r"\1 has no industry mapping in TICKER_INDUSTRY_MAP - use a scenario with an explicit % instead."),
    (r"(\S+) chưa có ánh xạ mặt hàng xuất/nhập khẩu trong TRADE_INDUSTRY_MAP.*",
     r"\1 has no export/import commodity mapping in TRADE_INDUSTRY_MAP - use a scenario with an explicit % instead."),
    (r"(\S+) chưa được xác định là nhạy cảm với giá vàng.*",
     r"\1 is not registered as gold-price sensitive (see GOLD_SENSITIVE_TICKERS) - use a scenario with an explicit % instead."),
    (r"(\S+) chưa được xác định là nhạy cảm với giá dầu.*",
     r"\1 is not registered as oil-price sensitive (see OIL_SENSITIVE_TICKERS) - use a scenario with an explicit % instead."),
    (r"Chưa đủ dữ liệu macro cho (.*?) \(cần >= 2 kỳ\).*",
     r"Not enough macro data for \1 (need >= 2 periods)."),
    (r"Chưa đủ dữ liệu xuất/nhập khẩu cho (.*?) \(cần >= 2 kỳ\).*",
     r"Not enough export/import data for \1 (need >= 2 periods)."),
    (r"Chưa đủ dữ liệu giá vàng thực tế \(hiện có (\d+) ngày, cần >= 2\).*",
     r"Not enough actual gold-price data (\1 days available, need >= 2)."),
    (r"Chưa đủ dữ liệu giá dầu thực tế \(hiện có (\d+) ngày, cần >= 2\).*",
     r"Not enough actual oil-price data (\1 days available, need >= 2)."),
    (r"Chưa đủ dữ liệu lãi suất VNIBOR thực tế \(cần >= 2 kỳ\).*",
     "Not enough actual VNIBOR interest-rate data (need >= 2 periods)."),
    (r"Không lấy được đủ 2 giá trị lãi suất VNIBOR để tính % thay đổi\.",
     "Could not get two VNIBOR values to compute a % change."),
    (r"Giá vàng kỳ gốc bằng 0, không tính được % thay đổi\.",
     "The base-period gold price is 0, so a % change cannot be computed."),
    (r"Giá dầu kỳ gốc bằng 0, không tính được % thay đổi\.",
     "The base-period oil price is 0, so a % change cannot be computed."),
    (r"Không có dòng nào trong company_price_snapshot cho mã (\S+) - .*",
     r"No rows in company_price_snapshot for \1 - it may not have been loaded by the thesis_market_snapshot job yet."),
    (r"Không có phiên giao dịch nào của (\S+) trong khoảng (.*?) - (.*?) trong company_price_snapshot\.",
     r"No trading sessions for \1 between \2 and \3 in company_price_snapshot."),
    (r"Không có dữ liệu giá cho bất kỳ mã nào trong (.*)\.", r"No price data for any of \1."),
    (r"Không có dữ liệu giá cho (\S+)\.", r"No price data for \1."),
    (r"Không có dữ liệu giá trong khung thời gian này\.", "No price data in this time window."),
    (r"Chỉ có (\d+) ngày trùng khớp giữa 2 chuỗi.*",
     r"Only \1 overlapping days between the two series - need >= 6 for a meaningful correlation."),
    (r"Chỉ có (\d+) cặp lợi suất ngày trùng khớp.*",
     r"Only \1 overlapping daily-return pairs - not enough to compute a correlation."),
    (r"Chưa có dữ liệu giá vàng \(MKT_GOLD_SJC\).*", "No gold-price data (MKT_GOLD_SJC) in macro_indicators yet."),
    (r"Chưa có dữ liệu VN-Index \(MKT_VNINDEX\).*", "No VN-Index data (MKT_VNINDEX) in macro_indicators yet."),
    (r"Chưa có dữ liệu giá dầu \(MKT_OIL_WTI\).*", "No oil-price data (MKT_OIL_WTI) in macro_indicators yet."),
    (r"Chưa có dữ liệu tỷ giá \(MKT_FOREX_VCB\).*", "No exchange-rate data (MKT_FOREX_VCB) in macro_indicators yet."),
    (r"Chưa có dữ liệu DEPOSIT_RATES.*", "No DEPOSIT_RATES data yet - run src.ingestion.macro_etl first."),
    (r"n=(\d+) < (\d+) - không đủ dữ liệu để ước lượng.*",
     r"n=\1 < \2 - not enough data to estimate, even at an exploratory level."),
    (r"Nhóm treated/control quá lệch \((.*?)\).*",
     r"Treated/control groups are too unbalanced (\1) - more data is needed."),
    (r"n=(\d+) quá nhỏ để chia (\d+)-fold cross-fitting cho DML\.",
     r"n=\1 is too small for \2-fold cross-fitting in DML."),
    (r"Lỗi nội bộ khi ước lượng T-learner \((.*?)\): (.*)", r"Internal error while estimating the T-learner (\1): \2"),
    (r"Lỗi nội bộ khi ước lượng DML \((.*?)\): (.*)", r"Internal error while estimating DML (\1): \2"),
    (r"Chưa cài đặt `econml`.*", "`econml` is not installed - the DML estimator cannot run (the T-learner does not need it)."),
]
_ERRORS = [(re.compile(p, re.S), r) for p, r in _ERRORS]


def error_en(text):
    """English version of a lower-module error string, for EN answers only."""
    if not text or not isinstance(text, str):
        return text
    text = data_error(text)
    for pattern, repl in _ERRORS:
        if pattern.fullmatch(text.strip()):
            text = pattern.sub(repl, text.strip())
            break
    # series names embedded inside a message ("EXPORTS_BY_COMMODITY/Sắt thép")
    # - longest first so "Sản xuất, chế biến thực phẩm" is not cut by a
    # shorter name it contains
    for vi in sorted(_SERIES, key=len, reverse=True):
        if vi in text:
            text = text.replace(vi, _SERIES[vi])
    return text


BANK_RATE_NOTE_EN = (
    "Ranked by PUBLISHED deposit rates (Vietstock) - not a borrowing or deposit "
    "recommendation for any specific company; this project has no data on which "
    "company banks with which bank."
)

INTEREST_RATE_NOTE_EN = (
    "Overnight interbank rate (VNIBOR) - a GENERAL MARKET rate, NOT the specific rate "
    "this company borrows at (the project has no company-bank credit data)."
)


def trade_note(direction: str, indicator_name: str, category: str) -> str:
    """English version of calculator.sensitivity_what_if_trade()'s note."""
    flow = "export" if direction == "export" else "import"
    base = (f"The revenue change is taken from the actual {flow} move in "
            f"{series_label(indicator_name)} (macro_indicators/{category}), NOT from a "
            f"figure the user supplied.")
    if direction == "import":
        base += (" An IMPORT increase is read as competitive pressure (sign reversed "
                 "relative to the raw move) - directionally reasonable, but NOT verified "
                 "against the company's market-share or contract data.")
    return base


def join_list(value) -> str:
    """['sector', 'fiscal_year'] -> 'sector and fiscal_year' (was printed as a
    raw Python list)."""
    if isinstance(value, (list, tuple)):
        items = [str(v) for v in value]
        if len(items) <= 1:
            return "".join(items)
        return ", ".join(items[:-1]) + " and " + items[-1]
    return str(value)
