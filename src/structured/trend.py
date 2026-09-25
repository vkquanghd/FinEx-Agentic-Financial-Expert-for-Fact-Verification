"""Trend/series answers for time-varying indicators (2026-08-06).

Yêu cầu trực tiếp từ phiên test Chat 2026-08-06: "giá cả nó sẽ biến động
theo từng năm từng quý... hỏi trong 1 năm thì không thể trả lời chung chung
được phải liệt kê ra. Có tăng hay giảm gì so với năm trước hoặc quý trước."

Ba nhiệm vụ, đều đọc financial_line_items (không thêm bảng mới):

1. enrich(parsed, raw): câu hỏi DAG đã trả lời được 1 giá trị đơn -> nối
   thêm phần diễn biến: câu hỏi theo NĂM được liệt kê đủ các quý trong năm
   (kèm %QoQ, %YoY từng quý); câu hỏi theo QUÝ được kèm so sánh với quý
   liền trước và cùng kỳ năm trước.
2. answer_from_quarters(parsed): câu hỏi theo NĂM nhưng financial_line_items
   KHÔNG có dòng YEAR (export quý-only) -> trả lời từ chính chuỗi quý thay
   vì rơi fallback sang RAG (với balance_sheet, giá trị cuối năm = Q4).
3. Mọi hàm đều fail-soft (trả None) - caller (query_router) chỉ nối thêm
   khi có, không bao giờ vỡ câu trả lời gốc vì phần enrich lỗi.

Không đụng tới giá cổ phiếu ở đây - price series nằm ở
company_price_snapshot và đã có compare_tickers() (nhánh COMPARE) xử lý.
"""
from typing import Optional

from sqlalchemy import text

from src.core.db import get_session

_SERIES_SQL = text("""
    SELECT fiscal_year, period, value, unit, item_name, consolidation
    FROM financial_line_items
    WHERE ticker = :ticker AND statement_type = :statement_type
      AND ms_code = :ms_code AND period IN ('Q1', 'Q2', 'Q3', 'Q4', 'YEAR')
      AND fiscal_year BETWEEN :y0 AND :y1
    ORDER BY fiscal_year, period
""")

_Q_ORDER = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4}


def _fetch(ticker: str, statement_type: str, ms_code: str,
           y0: int, y1: int) -> dict:
    """-> {(fiscal_year, period): row_dict}; ưu tiên consolidated khi một kỳ
    có cả 2 bản (consolidated/standalone)."""
    with get_session() as session:
        rows = session.execute(_SERIES_SQL, {
            "ticker": ticker, "statement_type": statement_type,
            "ms_code": ms_code, "y0": y0, "y1": y1,
        }).mappings().all()
    out: dict = {}
    for r in rows:
        key = (r["fiscal_year"], r["period"])
        if key not in out or r["consolidation"] == "consolidated":
            out[key] = dict(r)
    return out


def _pct(cur, prev) -> Optional[float]:
    if cur is None or prev in (None, 0):
        return None
    try:
        return (float(cur) - float(prev)) / abs(float(prev)) * 100.0
    except (TypeError, ZeroDivisionError):
        return None


def _fmt_pct(p: Optional[float]) -> str:
    return f"{p:+.1f}%" if p is not None else "n/a"


def _prev_quarter(year: int, q: str) -> tuple[int, str]:
    n = _Q_ORDER[q]
    return (year - 1, "Q4") if n == 1 else (year, f"Q{n - 1}")


def _quarter_lines(series: dict, year: int, lang: str = "vi") -> list[str]:
    """Liệt kê các quý của `year` có dữ liệu, kèm %QoQ và %YoY."""
    en = lang == "en"
    lines = []
    for q in ("Q1", "Q2", "Q3", "Q4"):
        row = series.get((year, q))
        if not row:
            continue
        py, pq = _prev_quarter(year, q)
        qoq = _pct(row["value"], (series.get((py, pq)) or {}).get("value"))
        yoy = _pct(row["value"], (series.get((year - 1, q)) or {}).get("value"))
        comparison = (
            f"(vs previous quarter: {_fmt_pct(qoq)}; vs same quarter last year: {_fmt_pct(yoy)})"
            if en else
            f"(so quý trước: {_fmt_pct(qoq)}; so cùng kỳ năm trước: {_fmt_pct(yoy)})"
        )
        lines.append(f"  {q}/{year}: {row['value']:,.0f} {row['unit']} {comparison}")
    return lines


# Cơ cấu đi kèm một chỉ tiêu tổng (2026-08-07, học từ phép so sánh với câu
# trả lời Gemini Pro cho "TTS của VNZ": phần "cơ cấu" của nó tính được từ
# chính financial_line_items của ta - không cần kiến thức web). Chỉ khai
# báo cho các TỔNG có phân rã 2 vế rõ ràng + 1 dòng highlight; các chỉ
# tiêu khác không có cơ cấu chuẩn thì thôi, không suy diễn.
# ms_code tổng -> (các vế [(ms_code, nhãn)], highlight (ms_code, nhãn) | None)
_COMPOSITION: dict[tuple[str, str], tuple[list, Optional[tuple]]] = {
    ("balance_sheet", "280"): (  # TỔNG CỘNG TÀI SẢN
        [("100", "Tài sản ngắn hạn"), ("200", "Tài sản dài hạn")],
        ("110", "tiền & tương đương tiền"),
    ),
    ("balance_sheet", "440"): (  # TỔNG NGUỒN VỐN
        [("300", "Nợ phải trả"), ("400", "Vốn chủ sở hữu")],
        None,
    ),
}


# English labels for the _COMPOSITION line items above (2026-09-22). Keyed by
# MS code, the statutory Circular 200/2014/TT-BTC identifier, which is what
# makes the mapping unambiguous - the Vietnamese label in _COMPOSITION and the
# English label here name the SAME statutory line, so no translation judgment
# happens at request time. A code missing here falls back to its Vietnamese
# label rather than guessing.
_MS_LABEL_EN = {
    "100": "Current assets",
    "200": "Non-current assets",
    "110": "cash & cash equivalents",
    "300": "Liabilities",
    "400": "Owners' equity",
}


def _composition_lines(parsed, raw: dict, lang: str = "vi") -> Optional[str]:
    """1 dòng cơ cấu cho chỉ tiêu tổng (tỷ trọng từng vế + %YoY của dòng
    highlight). Trả None nếu chỉ tiêu không có trong _COMPOSITION hoặc
    thiếu dữ liệu vế."""
    en = lang == "en"
    conf = _COMPOSITION.get((parsed.statement_type, parsed.ms_code))
    if not conf:
        return None
    components, highlight = conf
    y, per = parsed.fiscal_year, parsed.period
    total = float(raw["value"])
    if not total:
        return None
    parts = []
    for code, label in components:
        series = _fetch(parsed.ticker, parsed.statement_type, code, y0=y - 1, y1=y)
        row = series.get((y, per))
        if not row:
            continue
        share = float(row["value"]) / total * 100.0
        shown = _MS_LABEL_EN.get(code, label) if en else label
        parts.append(f"{shown} {row['value']:,.0f} {row['unit']} ({share:.1f}%)")
    if highlight:
        code, label = highlight
        series = _fetch(parsed.ticker, parsed.statement_type, code, y0=y - 1, y1=y)
        row = series.get((y, per))
        if row:
            yoy = _pct(row["value"], (series.get((y - 1, per)) or {}).get("value"))
            shown = _MS_LABEL_EN.get(code, label) if en else label
            lead = "of which" if en else "trong đó"
            yoy_note = ""
            if yoy is not None:
                yoy_note = (f" ({_fmt_pct(yoy)} YoY)" if en
                            else f" ({_fmt_pct(yoy)} so cùng kỳ)")
            parts.append(f"{lead} {shown} {row['value']:,.0f} {row['unit']}{yoy_note}")
    if not parts:
        return None
    header = f"Composition at {per}/{y}: " if en else f"Cơ cấu tại {per}/{y}: "
    return header + "; ".join(parts) + "."


def enrich(parsed, raw: dict, lang: str = "vi") -> Optional[str]:
    """Phần diễn biến nối thêm SAU câu trả lời giá trị đơn đã thành công.
    `parsed` là ParsedQuery (query_builder), `raw` là dict kết quả execute().
    Trả None nếu không có gì đáng nói (thiếu chuỗi so sánh).

    `lang` ("vi"|"en", 2026-09-22) chỉ đổi phần CHỮ; mọi con số, mã MS, nhãn
    kỳ và phép tính giữ nguyên hệt nhau ở cả hai ngôn ngữ."""
    en = lang == "en"
    try:
        y, per = parsed.fiscal_year, parsed.period
        series = _fetch(parsed.ticker, parsed.statement_type, parsed.ms_code,
                        y0=y - 1, y1=y)
        if per == "YEAR":
            lines = _quarter_lines(series, y, lang)
            yoy = _pct(raw["value"], (series.get((y - 1, "YEAR")) or {}).get("value"))
            parts = []
            if yoy is not None:
                parts.append(f"vs {y - 1}: {_fmt_pct(yoy)}." if en
                             else f"So với năm {y - 1}: {_fmt_pct(yoy)}.")
            comp = _composition_lines(parsed, raw, lang)
            if comp:
                parts.append(comp)
            # 2026-08-07 (phát hiện qua Chat với TTS/VNZ): trên bảng cân
            # đối, dòng YEAR (BCTC năm - thường ĐÃ kiểm toán) có thể lệch
            # Q4 (BCTC quý - chưa kiểm toán) do điều chỉnh kiểm toán/
            # restatement. Hai vintage cùng tồn tại trong DB là ĐÚNG thiết
            # kế (xem DSA v6 §5 - bitemporal limitation), nhưng trả lời mà
            # không chú thích thì người đọc thấy số "đá nhau". Note tự
            # động khi lệch > 0.1%.
            if parsed.statement_type == "balance_sheet":
                q4 = (series.get((y, "Q4")) or {}).get("value")
                if q4 is not None and raw["value"] not in (None, 0):
                    gap = _pct(q4, raw["value"])
                    if gap is not None and abs(gap) > 0.1:
                        parts.append(
                            f"Note: the year-end figure per the ANNUAL report ({raw['value']:,.0f}) "
                            f"differs by {_fmt_pct(gap)} from the Q4/{y} quarterly report ({q4:,.0f}) - "
                            f"usually due to post-audit adjustments; the annual-report figure is "
                            f"the official one."
                            if en else
                            f"Lưu ý: số cuối năm theo BCTC năm ({raw['value']:,.0f}) "
                            f"lệch {_fmt_pct(gap)} so với BCTC quý Q4/{y} ({q4:,.0f}) - "
                            f"thường do điều chỉnh sau kiểm toán; số BCTC năm là số "
                            f"chính thức."
                        )
            if lines:
                parts.append(f"Quarterly movement during {y}:" if en
                             else f"Diễn biến theo quý trong năm {y}:")
                parts.extend(lines)
            return "\n".join(parts) if parts else None
        if per in _Q_ORDER:
            py, pq = _prev_quarter(y, per)
            qoq = _pct(raw["value"], (series.get((py, pq)) or {}).get("value"))
            yoy = _pct(raw["value"], (series.get((y - 1, per)) or {}).get("value"))
            parts = []
            if qoq is not None or yoy is not None:
                parts.append(
                    f"vs previous quarter ({pq}/{py}): {_fmt_pct(qoq)}; "
                    f"vs same quarter last year ({per}/{y - 1}): {_fmt_pct(yoy)}."
                    if en else
                    f"So với quý liền trước ({pq}/{py}): {_fmt_pct(qoq)}; "
                    f"so với cùng kỳ năm trước ({per}/{y - 1}): {_fmt_pct(yoy)}."
                )
            comp = _composition_lines(parsed, raw, lang)
            if comp:
                parts.append(comp)
            return "\n".join(parts) if parts else None
        return None  # H1/9M: chưa làm chuỗi so sánh riêng
    except Exception:
        return None  # fail-soft: enrich không bao giờ phá câu trả lời gốc


def answer_recent_years(parsed, lang: str = "vi", n_years: int = 5) -> Optional[dict]:
    """Câu hỏi hỏi xu hướng NHIỀU năm gần đây, KHÔNG nêu năm cụ thể - ví dụ
    "Doanh thu VNM biến động qua các năm gần đây thế nào?" (xem
    src/router/rules.py's TREND_LANGUAGE_MARKERS - đây là nơi router điều
    hướng những câu đó tới). Dựng chuỗi năm trực tiếp từ
    financial_line_items (period='YEAR', có cấu trúc, chính xác) THAY VÌ để
    rơi vào RAG và phó mặc LLM tự suy diễn xu hướng từ các đoạn trích rời
    rạc - rủi ro bịa một xu hướng không có thật hoặc trộn dữ liệu nhiều năm
    sai cách. Trả None nếu có ít hơn 2 năm dữ liệu (không đủ để nói "xu
    hướng"), fail-soft như mọi hàm khác trong module này - caller
    (query_router) tự rơi tiếp sang RAG khi None."""
    try:
        from datetime import date

        y1 = date.today().year
        y0 = y1 - n_years
        series = _fetch(parsed.ticker, parsed.statement_type, parsed.ms_code, y0=y0, y1=y1)
        years = sorted({y for (y, per) in series if per == "YEAR"})
        if len(years) < 2:
            return None
        en = lang == "en"
        any_row = series[(years[-1], "YEAR")]
        item, unit = any_row["item_name"], any_row["unit"]
        if en:  # 2026-09-25: xem src/router/display_labels.py
            from src.router.display_labels import line_item_label
            item = line_item_label(parsed.statement_type, parsed.ms_code, item)
        lines = [
            f"{item} for {parsed.ticker} across the most recent years with data:" if en
            else f"{item} của {parsed.ticker} qua các năm gần nhất có dữ liệu:"
        ]
        prev_val = None
        for y in years:
            row = series[(y, "YEAR")]
            yoy = _pct(row["value"], prev_val)
            yoy_note = ""
            if yoy is not None:
                yoy_note = (f" (vs previous year: {_fmt_pct(yoy)})" if en
                            else f" (so năm trước: {_fmt_pct(yoy)})")
            lines.append(f"  {y}: {row['value']:,.0f} {unit}{yoy_note}")
            prev_val = row["value"]
        return {"answer": "\n".join(lines), "years": years, "item_name": item, "unit": unit}
    except Exception:
        return None  # fail-soft: caller rơi tiếp sang RAG khi None


def answer_from_quarters(parsed, lang: str = "vi") -> Optional[dict]:
    """Câu hỏi theo NĂM nhưng không có dòng YEAR trong DB (export quý-only).
    Trả {"answer": str, "note": str} dựng từ chuỗi quý, hoặc None nếu chuỗi
    quý cũng không có. Với balance_sheet (số dư thời điểm) giá trị "cả năm"
    = Q4; với income_statement/cash_flow (số phát sinh) chỉ cộng dồn khi có
    ĐỦ 4 quý - thiếu quý nào thì liệt kê phần có và nói rõ còn thiếu."""
    try:
        if parsed.period != "YEAR":
            return None
        y = parsed.fiscal_year
        series = _fetch(parsed.ticker, parsed.statement_type, parsed.ms_code,
                        y0=y - 1, y1=y)
        quarters = [q for q in ("Q1", "Q2", "Q3", "Q4") if (y, q) in series]
        if not quarters:
            return None
        en = lang == "en"
        any_row = series[(y, quarters[0])]
        item, unit = any_row["item_name"], any_row["unit"]
        if en:  # 2026-09-25: xem src/router/display_labels.py
            from src.router.display_labels import line_item_label
            item = line_item_label(parsed.statement_type, parsed.ms_code, item)
        lines = [
            f"No full-year {y} row in the loaded data - answering from the quarterly "
            f"series ({item}, {parsed.ticker}):"
            if en else
            f"Không có dòng tổng cả năm {y} trong dữ liệu đã nạp - "
            f"trả lời từ chuỗi quý ({item}, {parsed.ticker}):"
        ]
        lines += _quarter_lines(series, y, lang)

        if parsed.statement_type == "balance_sheet":
            if "Q4" in quarters:
                v = series[(y, "Q4")]["value"]
                yoy = _pct(v, (series.get((y - 1, "Q4")) or {}).get("value"))
                yoy_note = ""
                if yoy is not None:
                    yoy_note = (f" (vs end of {y - 1}: {_fmt_pct(yoy)})" if en
                                else f" (so cuối năm {y - 1}: {_fmt_pct(yoy)})")
                label = (f"  => Closing balance for {y} (= Q4/{y})" if en
                         else f"  => Số dư cuối năm {y} (= Q4/{y})")
                lines.append(f"{label}: {v:,.0f} {unit}{yoy_note}")
            else:
                lines.append(
                    f"  (Q4/{y} is not available yet, so the year-end balance cannot be fixed.)"
                    if en else
                    f"  (Chưa có Q4/{y} nên chưa chốt được số dư cuối năm.)"
                )
        else:
            if len(quarters) == 4:
                total = sum(float(series[(y, q)]["value"]) for q in quarters)
                prev_qs = [q for q in ("Q1", "Q2", "Q3", "Q4") if (y - 1, q) in series]
                yoy = None
                if len(prev_qs) == 4:
                    prev_total = sum(float(series[(y - 1, q)]["value"]) for q in prev_qs)
                    yoy = _pct(total, prev_total)
                yoy_note = ""
                if yoy is not None:
                    yoy_note = (f" (vs {y - 1}: {_fmt_pct(yoy)})" if en
                                else f" (so với năm {y - 1}: {_fmt_pct(yoy)})")
                label = (f"  => Full-year {y} total" if en
                         else f"  => Cộng dồn cả năm {y}")
                lines.append(f"{label}: {total:,.0f} {unit}{yoy_note}")
            else:
                missing = [q for q in ("Q1", "Q2", "Q3", "Q4") if q not in quarters]
                lines.append(
                    f"  ({', '.join(missing)}/{y} missing, so no full-year total is computed.)"
                    if en else
                    f"  (Thiếu {', '.join(missing)}/{y} nên không cộng dồn cả năm.)"
                )

        note = (f"answered from the quarterly series ({len(quarters)}/4 quarters with data)" if en
                else f"trả lời từ chuỗi quý ({len(quarters)}/4 quý có dữ liệu)")
        return {"answer": "\n".join(lines), "note": note}
    except Exception:
        return None
