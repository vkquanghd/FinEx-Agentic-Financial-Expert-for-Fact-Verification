"""Rule-based query classifier - decides which of the branches (DAG /
RAG / What-if / Causal / Compare / ABSA) should answer a given question. This is the actual
"Adaptive Retrieval Routing" decision logic the thesis title refers to:
given no fine-tuning is in scope (see docs/related_work_competitive_check.md
for why that's a deliberate choice vs FinStat2SQL), routing is done with
transparent, auditable keyword/regex rules rather than a trained
classifier - every decision comes with a `reason` string explaining WHY,
which doubles as free debugging/evaluation output.

Branch semantics (see each module's own docstring for the full rationale):
- DAG     (src/structured/dag_pipeline.py)   - "what WAS indicator X for
  ticker T in period P" - a single known number, already reported.
- RAG     (src/retrieval/rag_pipeline.py)    - anything else about a
  specific company - qualitative, explanatory, not a line-item lookup.
- WHATIF  (src/structured/calculator.py)     - "what would X be IF Y
  changed" or "how much of the change in X came from Y vs Z" for ONE
  company - exact arithmetic on reported numbers, not an estimate. Also
  covers the gold-price scenario for GOLD_SENSITIVE_TICKERS (currently
  just PNJ) via src/causal/market_factors.py's real gold price data.
- CAUSAL  (src/causal/treatment_effect.py)   - "does X affect Y" ACROSS
  companies, controlling for confounders - a statistical estimate over
  the whole panel, not about any single ticker.
- COMPARE (src/causal/market_factors.py, added 2026-07-19) - "so sánh mã
  A với mã B" - comparing TWO OR MORE tickers' price performance against
  each other (optionally vs gold/VN-Index) over a period - a data/chart
  request, not a scenario calculation or a causal estimate. Distinct from
  WHATIF's "so sánh 2 kỳ của 1 mã" (one ticker, two periods, wants a %
  growth figure) - COMPARE is "so sánh N>=2 mã" (multiple tickers, wants
  relative performance).
- ABSA     (src/absa/absa_pipeline.py, added 2026-07-24, RQ5/Week 7) -
  zero-shot LLM aspect-based sentiment analysis over ONE company's own
  disclosure text, across 5 fixed financial aspects (profitability,
  liquidity/leverage, growth/outlook, governance, operational/legal risk).
  Distinct from every other branch: not a number lookup, not a scenario
  calculation, not a cross-sectional estimate - a qualitative sentiment
  read grounded in real retrieved text (reuses RAG's own retrieval).
- AUDITOR_LOOKUP (src/structured/auditor_lookup.py, added 2026-08-07) -
  the REVERSE of DAG's per-company auditor lookup: "which companies did
  auditor FIRM audit, and in which years" - ACROSS the whole panel, not
  one ticker. A GROUP BY over the structured `disclosures.auditor` column
  (loaded straight from Vietstock, not inferred from RAG text), because
  RAG's per-question top-k similarity search cannot aggregate a field
  across every company's documents the way a direct SQL query can.

Priority order when multiple signals could apply: CAUSAL > ABSA >
AUDITOR_LOOKUP > COMPARE > WHATIF > DAG > DAG_TREND > OUT_OF_SCOPE > RAG.
OUT_OF_SCOPE (future price prediction / named third-party opinion /
personal investment advice, added 2026-09-14) sits just above the RAG
catch-all: these 3 question types are never answerable from disclosure
text regardless of retrieval quality, so they get their own deterministic,
zero-LLM-call explanation instead of RAG's generic "try being more
specific" message (see OUT_OF_SCOPE markers below for the audit finding
that motivated this). Causal/ABSA/compare/what-if language is rarer and more
specific than DAG/RAG phrasing, so checking them first avoids e.g. "if
COGS had been 5% lower, what would gross profit be" being misrouted to
DAG just because it also mentions "gross profit" (a registered indicator
alias). ABSA is checked right after CAUSAL since "cảm xúc"/"sắc thái"/
"sentiment" never collide with any other branch's markers. COMPARE is
checked before WHATIF specifically because both share the "so sánh"/
"compare" marker - the tiebreaker is ticker COUNT (>=2 tickers -> COMPARE,
exactly 1 ticker -> WHATIF's existing 2-periods logic).
"""
import re
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

from src.structured.auditor_lookup import extract_auditor_firm
from src.structured.query_builder import (
    NON_TICKER_ABBREVIATIONS,
    TICKER_RE,
    YEAR_RE,
    _extract_ticker,
    parse_question,
)
from src.structured.schema import resolve_indicator

CAUSAL_MARKERS = [
    r"ảnh hưởng (đến|tới)", r"tác động (của|đến|tới)", r"có làm .* thay đổi",
    r"có khiến", r"nguyên nhân (của|khiến)", r"gây ra", r"quan hệ nhân quả",
    r"\bcauses?\b", r"\baffects?\b", r"\bimpact of\b", r"\bcausal\b",
    # 2026-09-14: passive-voice variants - the 4 patterns above only match
    # ACTIVE voice ("X ảnh hưởng đến Y" / "X tác động đến Y", treatment
    # named first). Audit id=28 ("Dòng tiền hoạt động trên doanh thu CÓ BỊ
    # ẢNH HƯỞNG BỞI đòn bẩy tài chính không?") phrases it PASSIVELY (outcome
    # named first, "Y có bị ảnh hưởng bởi X") - "ảnh hưởng bởi" never
    # matches "ảnh hưởng (đến|tới)", so this fell through to RAG and got the
    # generic "không tìm thấy trong tài liệu" refusal instead of a real
    # causal estimate, even though ocf_to_revenue/leverage are both valid,
    # already-supported panel concepts (see _CAUSAL_CONCEPTS below - the gap
    # was purely in marker-phrase matching, not in panel/data coverage).
    # Also covers the EN passive equivalent ("is X affected by Y").
    #
    # Deliberately narrow to "ảnh hưởng bởi" (not the broader "bị .*ảnh
    # hưởng", tried first then reverted 2026-09-14 after it caused a real
    # regression: it also matched id=24's WHATIF gold-scenario question
    # ("biên lợi nhuận của PNJ BỊ ẢNH HƯỞNG thế nào?" - no "bởi", pure
    # coincidental substring overlap with an unrelated "nếu giá vàng tăng
    # 10%..." scenario question), incorrectly intercepting it into
    # causal_unsupported before it ever reached the WHATIF check below -
    # CAUSAL is checked before WHATIF (priority order above), so a marker
    # that's too eager here silently steals questions from WHATIF instead
    # of just failing safe. "ảnh hưởng bởi"/"tác động (của|bởi)" are
    # precise enough to not collide with WHATIF's "nếu/giả sử" scenario
    # phrasing in practice.
    r"ảnh hưởng bởi", r"chịu (sự )?tác động (của|bởi)",
    r"bị tác động (của|bởi)", r"\baffected by\b",
]

WHATIF_MARKERS = [
    r"\bnếu\b", r"\bgiả sử\b", r"what\s*-?\s*if", r"\bsuppose\b",
    r"đóng góp bao nhiêu", r"bao nhiêu (phần trăm|%) (là )?do",
    r"decomposition", r"phân rã", r"\bbridge\b", r"kịch bản",
    # "so sánh X giữa 2 kỳ" naturally wants a % change, not two bare DAG
    # lookups - added 2026-07-19 after testing showed "So sánh doanh thu
    # của VNM giữa 2024 và 2025" fell through to DAG (which only returns
    # ONE period's value, silently ignoring the comparison intent) since
    # "so sánh" wasn't a recognized marker.
    r"\bso sánh\b", r"\bcompare\b", r"\btăng trưởng\b", r"\bgrowth rate\b",
    # 2026-09-24 - BẤT ĐỐI XỨNG VI/EN, phát hiện khi soạn kịch bản demo
    # tiếng Anh. Tiếng Việt có r"\bnếu\b" bắt được mọi câu điều kiện; tiếng
    # Anh chỉ có "what if"/"suppose", KHÔNG có "if" đứng đầu câu. Hậu quả:
    #   "What if CTD's COGS rose 5% in 2025?"  -> whatif  (đúng)
    #   "If CTD's COGS rose 5% in 2025, what would gross profit be?" -> dag (SAI)
    # Câu thứ hai chính là ví dụ in trên SLIDE 3 của bài thuyết trình - câu
    # dùng để mở bài cho toàn bộ luận điểm "một pipeline không làm được cả
    # ba việc". Nó rơi vào dag vì có đủ ticker + năm + alias "gross profit",
    # nên trả về một con số đã công bố thay vì tính kịch bản.
    #
    # KHÔNG thêm r"\bif\b" trơn: "tell me if...", "check if..." sẽ dính oan.
    # Thay vào đó đòi "if" đi kèm HÌNH DẠNG của một kịch bản - một tỷ lệ
    # phần trăm, hoặc một động từ biến thiên - đúng thứ phân biệt câu giả
    # định với câu hỏi thường.
    # 2026-09-25: động từ "decompose" - chỉ có danh từ "decomposition" ở
    # trên, nên "How did CTD's gross margin change between 2024 and 2025 -
    # decompose it" rơi xuống RAG. Tiếng Việt "phân rã" vốn là động từ lẫn
    # danh từ nên không bị lỗ này.
    r"\bdecompos(?:e|es|ed|ing)\b",
    r"\bif\b.{0,80}?\d+(?:\.\d+)?\s*%",
    r"\bif\b.{0,60}?\b(?:rose|rise|rises|risen|increase[sd]?|increasing|grew|grow|grows|"
    r"fall|falls|fell|fallen|drop|drops|dropped|declin(?:e|es|ed)|decrease[sd]?)\b",
]

# A historical comparison is not a scenario merely because it uses
# "compare".  The distinction matters for questions such as "VNM in 2024
# compared with the previous year": the user is asking about reported facts,
# often with an explanation and disclosure-page citations.  Keep this marker
# deliberately narrow so the existing two-explicit-year growth-rate path is
# unchanged.
_PRIOR_YEAR_MARKERS = (
    r"\bnăm trước\b", r"\bnăm liền trước\b", r"\bkỳ trước\b",
    r"\bprevious year\b", r"\bprior year\b", r"\bpreceding year\b",
)
_DOCUMENT_GROUNDING_MARKERS = (
    r"giải thích", r"lý do", r"nguyên nhân", r"trích dẫn", r"trang",
    r"tài liệu công bố", r"báo cáo", r"\bexplain\b", r"\bwhy\b",
    r"\bcite\b", r"\bcitation\b", r"\bdisclosure\b", r"\bpage(?:s)?\b",
)

_PERCENT_RE = re.compile(r"(-?\d+(?:[.,]\d+)?)\s*%")
_GOLD_WORDS = ("vàng", "gold")
_OIL_WORDS = ("dầu", "oil")
# 2026-09-15: same pattern as gold/oil above, for calculator.py's
# sensitivity_what_if_macro() (uses the ticker's own real industry
# production/price index - IIP_BY_INDUSTRY or RETAIL in macro_indicators,
# via src/causal/macro_lookup.TICKER_INDUSTRY_MAP - as the COGS shock
# instead of a user-typed %). Found while wiring in the 2026-09-15 macro
# data ingest: sensitivity_what_if_macro() already existed and was already
# called from app_streamlit.py's manual ticker/year dropdown, but had NO
# natural-language trigger anywhere in this classifier - a typed chat
# question could never reach it, only the Streamlit demo's manual picker
# could. Kept as a SEPARATE marker set from gold/oil (not merged) since the
# Vietnamese/English phrasing is genuinely different ("chỉ số ngành" vs
# "vàng"/"dầu").
_INDUSTRY_WORDS = (
    "chỉ số ngành", "chỉ số sản xuất công nghiệp", "chỉ số sản xuất ngành",
    "industry index", "industry production index",
)
# 2026-09-15: same pattern again, for calculator.sensitivity_what_if_
# interest_rate() (uses VNIBOR's real overnight-rate move as a shock to
# "Chi phí tài chính", flowing to pretax profit - see that function's
# docstring for why it's a standalone calculation, not a repurposed COGS
# shock). No ticker-eligibility set needed here (unlike gold/oil/industry
# above): any PANEL ticker has a real "Chi phí tài chính" line, and a
# CORE_16 ticker just gets the same honest "no financial_line_items data"
# message every other WHATIF path already gives it - no separate
# GOLD_SENSITIVE_TICKERS-style allowlist required.
_INTEREST_RATE_WORDS = ("lãi suất", "interest rate")

# 2026-09-15 - same pattern again, for calculator.sensitivity_what_if_
# trade() (uses the ticker's own real export/import commodity value -
# EXPORTS_BY_COMMODITY/IMPORTS_BY_COMMODITY in macro_indicators, via
# src/causal/macro_lookup.TRADE_INDUSTRY_MAP - as a REVENUE shock). Quang's
# 2026-09-15 request explicitly named "xuất nhập khẩu hàng hóa" as a second
# macro driver worth wiring in alongside the industry-production index
# above, since export/import figures map to the same
# TICKER_INDUSTRY_MAP-style per-ticker restriction (only HPG/HSG/DPM/DCM
# are mapped - see TRADE_INDUSTRY_MAP's docstring). Kept ticker-restricted
# like gold/oil/industry above (NOT unrestricted like interest_rate, since
# unlike VNIBOR - which is economy-wide and touches every company's "Chi
# phí tài chính" - a commodity's export/import volume is only a meaningful
# revenue proxy for a company that actually produces/competes in that exact
# commodity).
_TRADE_WORDS = (
    "xuất khẩu", "nhập khẩu", "export", "import",
)

# 2026-09-14 - REAL BUG FOUND while investigating the "compound_p3_bug"
# audit category (id=41): _PERCENT_RE only recognizes a LITERAL "-" written
# right before the digits (e.g. "-10%"). Vietnamese never phrases a decrease
# that way - it's the WORD "giảm" ("doanh thu giảm 10%"), with no minus sign
# anywhere in the text - so `float(pct_match.group(1))` was silently
# returning +10 for a question that means -10. This wasn't a compound-
# question-only bug: id=20/21 in this file's own QUESTIONS list are single-
# percent, non-compound WHATIF questions, and id=21 ("Giả sử doanh thu của
# HSG GIẢM 10%...") was ALREADY affected before this fix - it was silently
# computing a +10% INCREASE scenario, the opposite of what was asked, and
# nothing in the audit harness would have caught it (numeric_verification
# only applies to RAG-sourced numbers, not this calculator's own output).
_DECREASE_WORDS = ("giảm", "decrease", "sụt giảm", "giảm xuống")


def _extract_pct_scenarios(question: str) -> list:
    """Returns [(signed_pct, is_cogs_local, is_revenue_local), ...], one
    tuple per '%' occurrence in `question`, in order of appearance.

    - Sign: looks at the text strictly BETWEEN the previous '%' match (or
      the start of the question) and this one for a decrease word - correct
      for both id=21 (single "giảm 10%") and compound id=41 ("COGS tăng 5%
      VÀ doanh thu giảm 3%", where each % needs its OWN sign, not one
      global sign for the whole question).
    - is_cogs_local/is_revenue_local: same windowed approach, looking for
      the literal words "cogs"/"giá vốn" vs "doanh thu"/"revenue" in that
      window - lets a compound question like id=41 tag each % to the right
      target. A single-% question with neither word present (e.g. id=22's
      "giá thép đầu vào tăng 8%") gets both flags False here; the caller
      falls back to the pre-existing is_gold/is_oil-aware heuristic in that
      case, unchanged from before this fix."""
    low = question.lower()
    scenarios = []
    start = 0
    for m in _PERCENT_RE.finditer(question):
        window = low[start:m.start()]
        sign = -1.0 if any(w in window for w in _DECREASE_WORDS) else 1.0
        is_cogs_local = ("cogs" in window or "giá vốn" in window)
        is_revenue_local = ("doanh thu" in window or "revenue" in window)
        pct = abs(float(m.group(1).replace(",", "."))) * sign
        scenarios.append((pct, is_cogs_local, is_revenue_local))
        start = m.end()
    return scenarios

# DAG_TREND marker (2026-08-07) - câu hỏi mang ngôn ngữ "xu hướng nhiều
# năm" nhưng KHÔNG nêu năm cụ thể (ví dụ "Doanh thu VNM biến động qua các
# năm gần đây thế nào?"). Trước đây loại câu này rơi vào RAG - phó mặc LLM
# tự suy diễn xu hướng từ các đoạn trích rời rạc, rủi ro bịa một xu hướng
# không có thật hoặc trộn số liệu nhiều năm sai cách. Khoảng trống phát
# hiện khi rà soát 2026-08-07: trend.py's enrich() chỉ nối vào nhánh DAG
# đã có SẴN 1 năm/kỳ cụ thể, không có đường nào xử lý câu hỏi kiểu này.
TREND_LANGUAGE_MARKERS = [
    r"qua các năm", r"qua từng năm", r"biến động", r"xu hướng",
    r"mấy năm gần đây", r"những năm gần đây", r"gần đây thế nào",
    r"gần đây ra sao", r"thay đổi thế nào", r"thay đổi ra sao",
    r"diễn biến", r"\btrend\b",
]

# PRICE_SNAPSHOT marker (2026-09-21) - câu hỏi về GIÁ CỔ PHIẾU hôm nay/gần
# đây (ví dụ "VNM biến động hôm nay thế nào", "giá cổ phiếu HPG hôm nay ra
# sao") - trước đây KHÔNG có nhánh nào xử lý: dag_trend ở trên chỉ bắt được
# "biến động" khi CÓ ms_code khớp (một chỉ tiêu BCTC như doanh thu/lợi
# nhuận - xem điều kiện `parsed.ticker and parsed.ms_code` trước
# TREND_LANGUAGE_MARKERS bên dưới), nên một câu hỏi giá thuần túy (không
# nhắc chỉ tiêu BCTC nào) không khớp ms_code -> rơi thẳng xuống RAG, nơi
# không có tài liệu công bố nào nói về giá đóng cửa hôm nay cả (giá cổ
# phiếu KHÔNG nằm trong disclosure_chunks) -> trả lời "không tìm thấy" sai
# lý do (nghe như thiếu tài liệu, thật ra là RAG chưa bao giờ là nơi đúng
# để hỏi câu này). company_price_snapshot có đúng dữ liệu (nạp qua
# scripts/market_snapshot.py) nhưng CHỈ 1 lần/ngày (EOD, không real-time) -
# xem market_factors.latest_price_snapshot()'s docstring cho giới hạn đầy
# đủ. Yêu cầu CẢ 2: từ khóa giá + từ khóa "hôm nay/gần đây" - "biến động"
# đứng một mình vẫn ưu tiên dag_trend/RAG như cũ nếu không có ngữ cảnh giá,
# tránh cướp nhầm câu hỏi định tính khác dùng chữ "biến động" theo nghĩa
# khác (ví dụ "biến động nhân sự", "biến động thị trường" nói chung).
PRICE_WORD_MARKERS = [r"\bgiá\b", r"\bcổ phiếu\b", r"\bchứng khoán\b",
                       r"\bbiến động\b", r"\bdao động\b",
                       r"\bstock price\b", r"\bshare price\b"]
PRICE_TODAY_MARKERS = [r"\bhôm nay\b", r"\bngày hôm nay\b", r"\bphiên hôm nay\b",
                        r"\btrong ngày\b", r"\btoday\b", r"\bthis session\b"]

# 2026-09-22 - LỖ ĐỊNH TUYẾN THẬT, phát hiện khi chạy thử: "Cho tôi biết giá
# cả của VNZ trong vài ngày gần đây" KHỚP PRICE_WORD_MARKERS ("giá") nhưng
# KHÔNG khớp PRICE_TODAY_MARKERS nào cả - nên rơi thẳng xuống RAG catch-all,
# mất 26 giây, rồi trả về thông báo fallback của Ollama. Trong khi dữ liệu
# giá của đúng những phiên đó đã nằm sẵn trong company_price_snapshot.
#
# Nguyên nhân gốc: điều kiện nhánh này đòi hỏi câu hỏi phải nói "HÔM NAY".
# Nhưng người ta hỏi về giá theo RẤT nhiều khung thời gian - "vài ngày gần
# đây", "tuần qua", "1 tháng nay" - và tất cả đều trả lời được từ cùng một
# bảng, chỉ khác số phiên lấy ra.
#
# (số nhóm bắt, nếu có) -> số phiên mặc định
_PRICE_RECENT_MARKERS: list[tuple[str, int]] = [
    (r"\b(\d{1,2})\s*(?:phiên|ngày)\s*(?:gần đây|vừa qua|qua|trở lại đây)\b", 0),  # "5 ngày gần đây" -> lấy số
    (r"\blast\s+(\d{1,2})\s+(?:days?|sessions?)\b", 0),
    # 2026-09-25 - mốc QUÁ KHỨ GẦN theo kiểu "hôm qua" / "N ngày trước".
    # Lỗ định tuyến thật khi test demo: "What was VNM stock price yesterday?"
    # và "What is VNM's stock price 3 day ago?" có đủ mã + "stock price"
    # nhưng không khớp mốc thời gian nào ở đây -> rơi xuống RAG, trả "No
    # information found" trong khi company_price_snapshot có sẵn đúng phiên
    # đó. -1 = "N ngày trước" -> lấy N+1 phiên, để phiên N ngày trước chắc
    # chắn nằm trong cửa sổ (xem _price_window_days()). "Hôm qua" lấy 2
    # phiên chứ không phải 1: dữ liệu là EOD nạp mỗi ngày, nên tuỳ lúc hỏi,
    # phiên mới nhất có thể là hôm nay HOẶC đã là hôm qua - trả 2 phiên có
    # ghi ngày thì đúng trong cả hai trường hợp, không phải đoán.
    (r"\b(\d{1,2})\s*(?:trading\s+)?(?:days?|sessions?)\s+ago\b", -1),
    (r"\b(\d{1,2})\s*(?:ngày|phiên)\s*trước\b", -1),
    (r"\bday before yesterday\b", 3),
    (r"\bhôm kia\b", 3),
    (r"\byesterday\b", 2),
    (r"\bhôm qua\b", 2),
    (r"\bphiên (?:trước|hôm qua)\b", 2),
    (r"\b(?:previous|last|prior) (?:trading )?session\b", 2),
    (r"\b(\d{1,2})\s*(?:tuần)\s*(?:gần đây|vừa qua|qua)\b", -7),                   # n tuần -> n*7
    (r"\bvài (?:ngày|phiên)\b", 7),
    (r"\bmấy (?:ngày|phiên)\b", 7),
    (r"\bnhững ngày (?:gần đây|vừa qua)\b", 7),
    (r"\bgần đây\b", 7),
    (r"\bvừa qua\b", 7),
    (r"\btuần (?:này|qua|vừa rồi|trước)\b", 7),
    (r"\btháng (?:này|qua|vừa rồi|trước)\b", 30),
    (r"\brecent(?:ly)?\b", 7),
    (r"\blast few (?:days|sessions)\b", 7),
    (r"\bpast (?:week|few days)\b", 7),
    (r"\bthis week\b", 7),
    (r"\blast month\b", 30),
    (r"\bover time\b", 30),
]


# 2026-09-24 - khoảng ngày TƯỜNG MINH cho câu hỏi giá.
#
# _price_window_days() phía dưới chỉ hiểu khoảng TƯƠNG ĐỐI ("vài ngày gần
# đây", "10 phiên vừa qua"). Nhưng người dùng cũng hỏi bằng mốc TUYỆT ĐỐI:
#   "giá VNZ từ tháng 10/2025 đến tháng 1/2026"
#   "VNZ share price from october 2025 to january 2026"
# Cả hai đều rơi xuống RAG rồi chết ở Ollama, dù company_price_snapshot có
# đủ dữ liệu của đúng khoảng đó. Cùng HỌ lỗi với "vài ngày gần đây" hôm
# trước: luật chỉ biết một kiểu diễn đạt.
_MONTH_WORDS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
}
_MONTH_NAME_RE = re.compile(
    r"\b(" + "|".join(sorted(_MONTH_WORDS, key=len, reverse=True)) + r")\b[\s,]*(\d{4})\b",
    re.IGNORECASE,
)
# "tháng 10/2025", "tháng 10 năm 2025", "tháng 10-2025"
_MONTH_VI_RE = re.compile(r"\btháng\s*(\d{1,2})\s*(?:/|-|năm\s*)\s*(\d{4})\b", re.IGNORECASE)
# "10/2025" trần (không có ngày đứng trước)
_MONTH_NUM_RE = re.compile(r"(?<!\d)(?<!\d/)(\d{1,2})/(\d{4})\b")
# ngày đầy đủ: "15/10/2025" hoặc "2025-10-15"
_FULL_DATE_RE = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b|\b(\d{4})-(\d{1,2})-(\d{1,2})\b")


def _price_date_range(question: str) -> Optional[tuple]:
    """(start_date, end_date) nếu câu hỏi nêu mốc thời gian TƯỜNG MINH, else None.

    Quy tắc: gom mọi mốc tìm được theo thứ tự xuất hiện, lấy mốc ĐẦU làm
    điểm bắt đầu và mốc CUỐI làm điểm kết thúc. Một mốc duy nhất cũng hợp lệ
    ("giá VNZ tháng 10/2025") - khi đó khoảng là trọn tháng đó.

    Mốc dạng THÁNG được mở rộng thành trọn tháng (ngày 1 -> ngày cuối), vì
    "từ tháng 10/2025 đến tháng 1/2026" nghĩa là hết tháng 1, không phải
    ngày 1 tháng 1. Đây là chỗ rất dễ lệch một tháng nếu làm ẩu.
    """
    import calendar

    marks: list[tuple[int, date, bool]] = []   # (vị trí, ngày, có_phải_mốc_tháng)

    for m in _FULL_DATE_RE.finditer(question):
        try:
            if m.group(1):
                d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
            else:
                y, mo, d = int(m.group(4)), int(m.group(5)), int(m.group(6))
            marks.append((m.start(), date(y, mo, d), False))
        except ValueError:
            continue

    spans = [(m.start(), m.end()) for m in _FULL_DATE_RE.finditer(question)]

    def _inside_full_date(pos: int) -> bool:
        return any(a <= pos < b for a, b in spans)

    for rx, grp in ((_MONTH_VI_RE, (1, 2)), (_MONTH_NUM_RE, (1, 2))):
        for m in rx.finditer(question):
            if _inside_full_date(m.start()):
                continue          # "15/10/2025" đã được bắt ở trên, đừng bắt lại "10/2025"
            mo, y = int(m.group(grp[0])), int(m.group(grp[1]))
            if 1 <= mo <= 12:
                marks.append((m.start(), date(y, mo, 1), True))

    for m in _MONTH_NAME_RE.finditer(question):
        mo = _MONTH_WORDS[m.group(1).lower()]
        marks.append((m.start(), date(int(m.group(2)), mo, 1), True))

    if not marks:
        return None
    marks.sort(key=lambda x: x[0])              # theo vị trí trong câu
    first, last = marks[0], marks[-1]

    # Chốt thứ tự THỜI GIAN trước, rồi mới mở rộng mốc-tháng. Làm ngược lại
    # sẽ sai với câu viết ngược ("từ tháng 1/2026 VỀ tháng 10/2025"): mốc
    # đứng sau trong câu lại là mốc SỚM hơn về thời gian, mở rộng nó thành
    # cuối tháng rồi mới hoán vị sẽ cho ra khoảng 31/10/2025 -> 01/01/2026,
    # tức là lệch cả hai đầu.
    early, late = (first, last) if first[1] <= last[1] else (last, first)
    start = early[1]                            # mốc-tháng vốn đã là ngày 1
    end = late[1]
    if late[2]:                                 # mốc sau là THÁNG -> hết tháng đó
        end = date(end.year, end.month, calendar.monthrange(end.year, end.month)[1])
    return (start, end)


def _price_window_days(question: str) -> Optional[int]:
    """Số PHIÊN cần lấy nếu câu hỏi nói về giá theo một khoảng thời gian gần
    đây; None nếu câu hỏi không có dấu hiệu khoảng thời gian nào.

    Duyệt theo thứ tự _PRICE_RECENT_MARKERS: mẫu có CON SỐ tường minh đứng
    trước mẫu chung chung, nên "5 ngày gần đây" lấy đúng 5 chứ không rơi vào
    mẫu "gần đây" -> 7. Trả về số PHIÊN giao dịch, không phải ngày lịch -
    xem market_factors.price_window_series() cho lý do.
    """
    for pattern, default_days in _PRICE_RECENT_MARKERS:
        m = re.search(pattern, question, re.IGNORECASE)
        if not m:
            continue
        if default_days == 0 and m.groups():       # số ngày/phiên tường minh
            return max(2, min(int(m.group(1)), 60))
        if default_days == -7 and m.groups():      # n tuần -> n*7
            return max(2, min(int(m.group(1)) * 7, 60))
        if default_days == -1 and m.groups():      # "N ngày trước" -> N+1 phiên
            return max(2, min(int(m.group(1)) + 1, 60))
        return default_days
    return None
# Loại trừ tường minh: "giá" + "hôm nay" + có mã CŨNG khớp cho câu hỏi về
# giá HÀNG HÓA (vàng/dầu) tác động lên một mã, ví dụ "giá vàng hôm nay ảnh
# hưởng đến VNM không" - câu này đã được các nhánh WHATIF vàng/dầu ở TRÊN
# bắt trước (nếu đúng ngôn ngữ kịch bản); nhưng nếu diễn đạt thuần thông
# tin (không phải "kịch bản X% thì Y") thì không khớp WHATIF và sẽ rơi tới
# đây - loại trừ bằng tay để không tra NHẦM giá cổ phiếu của ticker thay vì
# nói về giá hàng hóa.
_PRICE_SNAPSHOT_EXCLUDE = [r"\bvàng\b", r"\bdầu\b", r"\bgold\b", r"\boil\b", r"\bwti\b", r"\bsjc\b"]


def _extract_all_tickers(question: str) -> list[str]:
    """Plural version of query_builder._extract_ticker() - returns every
    distinct valid-looking ticker found, in order of first appearance, for
    the COMPARE branch (which needs 2+ tickers, unlike every other branch
    that only ever needs the first one).

    Falls back to company-NAME matching (2026-07-24, see
    src/structured/company_lookup.py) when fewer than 2 bare tickers were
    found by regex - e.g. "so sánh Tập đoàn Hòa Phát và Vàng bạc Đá quý Phú
    Nhuận" has zero regex-matchable tickers but should still resolve to
    [HPG, PNJ]. Only triggers below the 2-ticker floor so a question that
    already named 2+ real tickers never pays the extra DB round-trip."""
    seen = []
    for match in TICKER_RE.finditer(question):
        candidate = match.group(1)
        if candidate in NON_TICKER_ABBREVIATIONS or candidate in seen:
            continue
        # 2026-08-05: same false-positive-ticker fix as
        # query_builder._extract_ticker() - a bare regex match ("VINA" out
        # of "VINA Game") isn't necessarily real; validating here matters
        # even more for COMPARE since an unvalidated false positive can
        # silently reach the 2-ticker floor and skip the name-resolution
        # fallback entirely, instead of just failing one lookup.
        try:
            from src.structured.company_lookup import is_known_ticker
            if is_known_ticker(candidate):
                seen.append(candidate)
        except Exception:
            seen.append(candidate)  # can't verify (DB down) - fail open, old behavior
    if len(seen) < 2:
        try:
            from src.structured.company_lookup import resolve_all_company_names
            for ticker in resolve_all_company_names(question):
                if ticker not in seen:
                    seen.append(ticker)
        except Exception:
            pass
    return seen

# Concept phrase -> panel column (src/causal/panel_builder.py's derived
# ratios). Longest/most specific phrase should be checked first so e.g.
# "biên lợi nhuận ròng" matches net_margin, not the shorter "biên lợi
# nhuận" alias also present - order matters, kept as a list not a dict.
_CAUSAL_CONCEPTS = [
    ("đòn bẩy tài chính", "leverage"),
    ("đòn bẩy", "leverage"),
    ("tỷ lệ nợ", "leverage"),
    ("leverage", "leverage"),
    ("biên lợi nhuận gộp", "gross_margin"),
    ("gross margin", "gross_margin"),
    ("biên lợi nhuận ròng", "net_margin"),
    ("biên lợi nhuận", "net_margin"),
    ("net margin", "net_margin"),
    ("net profit margin", "net_margin"),
    ("vòng quay tài sản", "asset_turnover"),
    ("asset turnover", "asset_turnover"),
    ("khả năng thanh toán hiện hành", "current_ratio"),
    ("current ratio", "current_ratio"),
    ("dòng tiền hoạt động trên doanh thu", "ocf_to_revenue"),
    ("ocf to revenue", "ocf_to_revenue"),
]

# RQ5 (Week 7) - ABSA zero-shot LLM branch marker. Deliberately specific
# words ("cảm xúc"/"sắc thái"/"sentiment") that don't overlap with any
# other branch's keyword set (CAUSAL/WHATIF/COMPARE markers are all about
# numeric scenarios/relationships, never sentiment language) - see
# src/absa/absa_pipeline.py for the analysis itself (5 khía cạnh tài chính).
ABSA_MARKERS = [
    r"\bcảm xúc\b", r"\bsắc thái\b", r"\bsentiment\b",
    r"phân tích cảm xúc", r"đánh giá cảm xúc",
    r"tích cực hay tiêu cực", r"khía cạnh (tài chính|cảm xúc)",
]

# OUT_OF_SCOPE (2026-09-14) - Quang's audit feedback on ids 50/51 ("Giá cổ
# phiếu VNM ngày mai sẽ là bao nhiêu?" / "Warren Buffett nghĩ gì về HPG?"):
# these fell through to the RAG branch, which (via
# src.router.scope_notes.polish_not_found) appended "Dữ liệu của X đã có
# trong hệ thống, nhưng câu hỏi này không khớp... Thử nêu rõ chỉ tiêu + kỳ"
# - advice that is actively WRONG for these two questions, since no amount
# of rephrasing makes "predict tomorrow's price" or "what does a named
# third party think" answerable from disclosure documents. That RAG message
# is correct only for the "ticker has data but the QUESTION phrasing is
# off" case; these questions are a structurally different failure mode
# (the question TYPE, not the ticker/period, is unanswerable from financial
# statements), so they need their own branch with their own honest
# explanation - not a generic "try being more specific" nudge.
#
# Deliberately 3 separate marker sets (not merged into a "misc" bucket) so
# each keeps a distinct, precise explanation - see _format_out_of_scope() in
# query_router.py. Deliberately checked with plain regex, no LLM call - a
# deterministic "no" here costs ~0s and carries zero hallucination risk,
# same "no invented ground truth" principle as every other branch in this
# router (see rules.py's module docstring).
FUTURE_PREDICTION_MARKERS = [
    r"ngày mai.*(giá|sẽ)", r"\btuần tới\b.*giá", r"\btháng tới\b.*giá",
    r"dự đoán giá", r"dự báo giá", r"giá .* (sẽ tăng|sẽ giảm)",
    r"sẽ (tăng|giảm) (bao nhiêu|thế nào|ra sao)",
    r"\btomorrow\b.*\bprice\b", r"\bpredict\b.*\bprice\b",
    r"\bforecast\b.*\bprice\b", r"\bwill .* price be\b",
    # 2026-09-22 - LỖ NGHIÊM TRỌNG NHẤT trong đợt quét cách diễn đạt, vì đây
    # là một RANH GIỚI TUÂN THỦ chứ không phải một tính năng: "Giá VNM tháng
    # sau sẽ là bao nhiêu?" lọt qua HẾT các mẫu trên và rơi xuống RAG, tức
    # là hệ thống đem một câu hỏi DỰ ĐOÁN GIÁ TƯƠNG LAI đi hỏi Qwen trên các
    # đoạn trích BCTC - đúng thứ nhánh out_of_scope sinh ra để chặn.
    # Hai chỗ trượt: (a) các mẫu cũ chỉ biết "tuần/tháng TỚI", không biết
    # "SAU"; (b) chúng đều đòi từ "giá" đứng SAU mốc thời gian, trong khi
    # tiếng Việt nói "Giá VNM tháng sau..." - giá đứng TRƯỚC.
    # Sửa: bắt cặp (mốc thời gian tương lai) + (từ về giá) theo CẢ HAI thứ tự.
    r"\b(?:ngày mai|tuần (?:tới|sau)|tháng (?:tới|sau)|năm (?:tới|sau)|sắp tới)\b.*\b(?:giá|cổ phiếu)\b",
    r"\b(?:giá|cổ phiếu)\b.*\b(?:ngày mai|tuần (?:tới|sau)|tháng (?:tới|sau)|năm (?:tới|sau)|sắp tới)\b",
    r"\b(?:next (?:week|month|year)|tomorrow)\b.*\b(?:price|stock)\b",
    r"\b(?:price|stock)\b.*\b(?:next (?:week|month|year)|tomorrow)\b",
]

# Captures the named person so the formatted answer can name them back
# ("đây là câu hỏi về quan điểm của {person}...") without needing a
# hardcoded name list - only a tiny allow-listed set of well-known public
# figures (see _KNOWN_PUBLIC_FIGURES in query_router.py) gets an extra,
# verified one-line bio; every other name still gets the honest generic
# explanation, never a fabricated opinion.
THIRD_PARTY_OPINION_PATTERNS = [
    r"^(.+?)\s+nghĩ gì về\b", r"quan điểm của (.+?) về\b",
    r"theo (.+?)(?:,| thì)\b",
    r"^what does (.+?) think\b", r"\bopinion of (.+?) on\b",
]

INVESTMENT_ADVICE_MARKERS = [
    r"có nên mua\b", r"có nên bán\b", r"nên đầu tư\b", r"có đáng đầu tư\b",
    r"\bshould i buy\b", r"\bshould i invest\b", r"\bis it worth buying\b",
    r"\bgood time to buy\b",
]


def _match_third_party_opinion(question: str) -> Optional[str]:
    """Returns the captured person/entity name if `question` asks for a
    specific named third party's opinion, else None. Checked before the
    generic INVESTMENT_ADVICE/FUTURE_PREDICTION markers since it's the most
    specific of the three (requires a capture, not just a keyword hit)."""
    for pat in THIRD_PARTY_OPINION_PATTERNS:
        m = re.search(pat, question, re.IGNORECASE)
        if m:
            name = m.group(1).strip(" ?.,")
            if name:
                return name
    return None


@dataclass
class QueryIntent:
    branch: str  # "dag" | "dag_multi" | "dag_trend" | "ratio_lookup" | "rag" | "whatif" | "causal" | "compare" | "absa" | "auditor_lookup" | "bank_rate_lookup" | "price_snapshot" | "out_of_scope" | "whatif_unsupported" | "causal_unsupported" | "compare_unsupported" | "absa_unsupported"
    reason: str
    ticker: Optional[str] = None
    # auditor_lookup-only:
    auditor_query: Optional[str] = None
    # whatif-only:
    whatif_kind: Optional[str] = None  # "growth_rate" | "decompose_margin" | "sensitivity" | "gold_sensitivity" | "oil_sensitivity" | "macro_sensitivity"
    periods: list = field(default_factory=list)  # [(fiscal_year, period_code), ...]
    cogs_pct_change: float = 0.0
    revenue_pct_change: float = 0.0
    indicator_stype: Optional[str] = None
    indicator_ms_code: Optional[str] = None
    # causal-only:
    treatment_col: Optional[str] = None
    outcome_col: Optional[str] = None
    # compare-only:
    tickers: list = field(default_factory=list)
    compare_with_gold: bool = False
    # out_of_scope-only:
    oos_kind: Optional[str] = None  # "future_prediction" | "third_party_opinion" | "investment_advice"
    oos_person: Optional[str] = None  # captured name, third_party_opinion only
    # price_snapshot-only (2026-09-22): số PHIÊN giao dịch cần lấy.
    # 1 = chỉ phiên gần nhất ("giá X hôm nay") -> latest_price_snapshot();
    # >1 = một khoảng ("giá X vài ngày gần đây") -> price_window_series().
    # Cùng MỘT nhánh, chỉ khác cửa sổ - vì cả hai đọc đúng một bảng
    # (company_price_snapshot) bằng đúng một cách, không phải hai năng lực
    # khác nhau. Giữ một nhánh cũng tránh phải thêm nhãn mới vào Branch
    # Literal + CHECK của migration 014 + baseBranch() ở frontend.
    price_window_days: int = 1
    # 2026-09-24: khoảng ngày TƯỜNG MINH ("từ tháng 10/2025 đến tháng 1/2026").
    # Khi có, nó THẮNG price_window_days - người đã nêu mốc cụ thể thì không
    # còn là câu hỏi "gần đây" nữa.
    price_start_date: Optional[date] = None
    price_end_date: Optional[date] = None


def _match_any(patterns: list[str], text: str) -> bool:
    low = text.lower()
    return any(re.search(p, low) for p in patterns)


def _find_concepts(question: str) -> list[str]:
    low = question.lower()
    found = []
    for phrase, col in _CAUSAL_CONCEPTS:
        if phrase in low and col not in found:
            found.append(col)
    return found


def _ratio_label_en(name: str) -> str:
    """English ratio name for EN reason strings (2026-09-25) - see
    src/router/display_labels.py. Lazy import keeps rules.py importable on
    its own, same as the other lazy imports in this module."""
    from src.router.display_labels import ratio_label
    return ratio_label(name)


def _ratio_group_en(group: str) -> str:
    from src.router.display_labels import ratio_group_label
    return ratio_group_label(group)


def _r(lang: str, vi: str, en: str) -> str:
    """Pick a QueryIntent.reason string for the resolved answer language.

    Added 2026-09-22 alongside the bilingual answer formatters in
    query_router.py. The reason string is USER-VISIBLE (the UI prints it
    under every answer as the routing explanation, and it is what the thesis
    means by "every routing decision carries an explicit, human-readable
    reason string"), so leaving it Vietnamese would have left a Vietnamese
    line under every English answer.

    Only the WORDING is translated. The branch label, the extracted ticker,
    the matched alias, the period and every function name quoted inside the
    reason are identical in both languages - so the routing decision being
    explained is provably the same one, and a benchmark that records
    intent_reason (scripts/final_benchmark_*.py) stays comparable across
    languages.
    """
    return en if lang == "en" else vi


def classify_intent(question: str, lang: str = "vi") -> QueryIntent:
    # 1. CAUSAL - explicit causal-language marker + >=2 distinct known
    # panel concepts (treatment, outcome). No ticker check - causal
    # estimation is deliberately cross-sectional (whole panel), not about
    # one company (see treatment_effect.py's docstring).
    # 2026-09-22: CAUSAL_MARKERS đòi "ảnh hưởng"/"tác động" DÍNH LIỀN giới
    # từ ("ảnh hưởng đến"), nên hai cách nói rất tự nhiên bị trượt hết:
    #   "Đòn bẩy ảnh hưởng THẾ NÀO đến biên lợi nhuận ròng?"
    #   "Đòn bẩy tài chính tác động THẾ NÀO tới biên lợi nhuận ròng?"
    # Cả hai rơi xuống RAG và nhận câu "không tìm thấy trong tài liệu", dù
    # leverage/net_margin đều là khái niệm panel hợp lệ. Cùng HỌ lỗi với lỗ
    # bị động ("ảnh hưởng bởi") đã ghi trong comment của CAUSAL_MARKERS, và
    # cùng họ với lỗ "giá ... vài ngày gần đây" ở nhánh price_snapshot: luật
    # bắt đúng một cách diễn đạt, người dùng nói cách khác.
    #
    # Nới CÓ ĐIỀU KIỆN, không nới thẳng vào CAUSAL_MARKERS: mẫu rộng chỉ
    # được tính khi đã tìm thấy >=2 khái niệm panel. Lý do - nới thẳng sẽ
    # kéo cả câu định tính ("Ảnh hưởng của dịch bệnh đến doanh thu VNM?")
    # vào causal_unsupported, tức là thay câu trả lời RAG hữu ích bằng một
    # lời từ chối. Cách này chỉ THÊM được định tuyến causal khi chắc chắn,
    # không bao giờ lấy mất đường RAG của câu hỏi định tính.
    _causal_loose = [r"(ảnh hưởng|tác động)\s+(?:\S+\s+){1,3}?(đến|tới|lên)\b"]
    concepts_probe = _find_concepts(question)
    if _match_any(CAUSAL_MARKERS, question) or (
            len(concepts_probe) >= 2 and _match_any(_causal_loose, question)):
        concepts = concepts_probe
        if len(concepts) >= 2:
            return QueryIntent(
                branch="causal",
                reason=_r(lang,
                    f"Từ khóa nhân quả + 2 khái niệm đã biết trong panel: {concepts[0]} -> {concepts[1]}.",
                    f"Causal keyword + 2 known panel concepts: {concepts[0]} -> {concepts[1]}."),
                treatment_col=concepts[0], outcome_col=concepts[1],
            )
        return QueryIntent(
            branch="causal_unsupported",
            reason=_r(lang,
                f"Có từ khóa nhân quả nhưng chỉ nhận diện được {len(concepts)}/2 khái niệm "
                f"đã biết trong panel (cần: biến can thiệp + biến kết quả, ví dụ 'đòn bẩy' và "
                f"'biên lợi nhuận ròng'). Khái niệm nhận được: {concepts}.",
                f"Causal keyword present but only {len(concepts)}/2 known panel concepts were "
                f"recognised (need: a treatment variable + an outcome variable, e.g. 'leverage' "
                f"and 'net margin'). Concepts found: {concepts}."),
        )

    # 2. ABSA (RQ5, Week 7) - explicit sentiment/aspect-analysis language +
    # a resolvable ticker/company name. Analyzes ONE company's own
    # disclosure text (like WHATIF), not cross-sectional like CAUSAL -
    # checked right after CAUSAL since its markers are specific enough
    # (cảm xúc/sắc thái/sentiment) to never collide with CAUSAL's.
    if _match_any(ABSA_MARKERS, question):
        ticker = _extract_ticker(question)
        if not ticker:
            return QueryIntent(
                branch="absa_unsupported",
                reason=_r(lang,
                    "Có từ khóa phân tích cảm xúc nhưng không nhận diện được mã/tên công ty trong câu hỏi.",
                    "Sentiment-analysis keyword present but no ticker/company name was recognised in the question."),
            )
        return QueryIntent(
            branch="absa", ticker=ticker,
            reason=_r(lang,
                f"Từ khóa phân tích cảm xúc/sắc thái + mã {ticker} -> ABSA zero-shot LLM "
                f"(5 khía cạnh tài chính, xem src/absa/absa_pipeline.py).",
                f"Sentiment/aspect keyword + ticker {ticker} -> ABSA zero-shot LLM "
                f"(5 financial aspects, see src/absa/absa_pipeline.py)."),
        )

    # 3. AUDITOR_LOOKUP (2026-08-07, phản hồi test "KPMG kiểm toán các công
    # ty nào qua từng năm?") - câu hỏi NGƯỢC: từ tên hãng kiểm toán ra danh
    # sách công ty, KHÔNG gắn với một mã cụ thể. Khác với "Công ty kiểm
    # toán của PNJ năm 2024 là ai?" (CÓ mã -> vẫn đi RAG như trước, đó là
    # tra cứu 1 chiều RAG làm tốt). Đây là tổng hợp (GROUP BY) trên toàn bộ
    # cột disclosures.auditor (có cấu trúc, nạp thẳng từ Vietstock - xem
    # src/structured/auditor_lookup.py) - RAG (top-k similarity theo 1 câu
    # hỏi) không tổng hợp xuyên nhiều công ty được, nên trước khi có nhánh
    # này câu hỏi loại đó luôn rơi vào RAG và báo "không tìm thấy" dù dữ
    # liệu có sẵn và chính xác trong DB. Điều kiện "không có mã cụ thể" là
    # ranh giới giữa 2 câu hỏi trên.
    # 2026-09-22 - BILINGUAL PARITY FIX. This gate used to read
    # `"auditor" in question.lower()`, a bare substring test that only
    # matched the NOUN. "Công ty kiểm toán của VNM là ai" therefore reached
    # this branch and got the structured disclosures.auditor answer, while
    # its direct English equivalent "Who audited VNZ in 2025?" did NOT -
    # "audited" does not contain "auditor" - so it fell all the way through
    # to the RAG catch-all and answered from document text instead of the
    # structured column. Found by running classify_intent() over matched
    # VI/EN question pairs while making the answer formatters bilingual; the
    # identity_request regex just below ALREADY had `who.*audit`, so only
    # this outer gate was blocking it. Word-boundary regex (not a substring)
    # so it covers audit/audits/audited/auditing/auditor/auditors without
    # matching unrelated words that merely start with "audit".
    if re.search(r"kiểm toán|\baudit(s|ed|ing|or|ors)?\b", question, re.IGNORECASE):
        firm = extract_auditor_firm(question)
        ticker_here = _extract_ticker(question)
        identity_request = re.search(
            r"công ty kiểm toán|đơn vị kiểm toán|hãng kiểm toán|auditor|who.*audit|audit.*firm",
            question, re.IGNORECASE,
        )
        # 2026-09-25 - Ý KIẾN kiểm toán khác DANH TÍNH kiểm toán. Chữ "auditor"
        # trơn trong regex trên khớp cả hai loại câu, nên "What did the
        # auditor SAY about VNZ's 2025 financial statements?" bị trả lời bằng
        # "EY" - đúng tên hãng, nhưng không trả lời câu được hỏi. Người hỏi
        # muốn biết NỘI DUNG báo cáo kiểm toán (chấp nhận toàn phần / ngoại
        # trừ / vấn đề cần nhấn mạnh), thứ chỉ nằm trong văn bản PDF, không
        # nằm trong cột disclosures.auditor. Câu loại này để rơi xuống RAG.
        opinion_request = re.search(
            r"\b(?:say|said|says|opinion|conclude[ds]?|conclusion|qualif(?:y|ied|ication)|"
            r"unqualified|modified|emphasis of matter|adverse|disclaimer)\b"
            r"|ý kiến kiểm toán|kết luận kiểm toán|nói gì|ngoại trừ|chấp nhận toàn phần|"
            r"vấn đề cần nhấn mạnh",
            question, re.IGNORECASE,
        )
        if ticker_here and identity_request and not firm and not opinion_request:
            years = sorted({int(y) for y in re.findall(r"\b(?:19|20)\d{2}\b", question)})
            return QueryIntent(
                branch="auditor_lookup", ticker=ticker_here,
                periods=[(year, "YEAR") for year in years],
                reason=_r(lang,
                    "Tra cứu hãng kiểm toán theo mã/năm trong metadata báo cáo; thiếu metadata thì tìm tài liệu.",
                    "Audit-firm lookup by ticker/year from report metadata; falls back to document search when metadata is missing."),
            )
        if firm and not ticker_here:
            return QueryIntent(
                branch="auditor_lookup", auditor_query=firm,
                reason=_r(lang,
                    f"Từ khóa kiểm toán + tên hãng '{firm}', không có mã cụ thể trong câu hỏi "
                    f"-> tra cứu tổng hợp trực tiếp trên disclosures.auditor (có cấu trúc, "
                    f"không qua RAG).",
                    f"Audit keyword + firm name '{firm}', no specific ticker in the question "
                    f"-> aggregate lookup straight over disclosures.auditor (structured, "
                    f"no RAG involved)."),
            )

    # 3b. BANK_RATE_LOOKUP (2026-09-15) - "ngân hàng nào lãi suất tốt nhất"
    # style question: an aggregate query ACROSS every bank in
    # macro_indicators' DEPOSIT_RATES category (src.causal.macro_lookup.
    # best_deposit_rate_banks()), not tied to any one ticker - same
    # "GROUP BY the whole table, no ticker needed" shape as AUDITOR_LOOKUP
    # just above, so checked right next to it. Deliberately requires BOTH
    # "ngân hàng" AND a superlative word (not just "lãi suất" alone),
    # otherwise this would swallow every ticker-specific interest-rate
    # WHATIF question below ("Nếu lãi suất tăng thì HPG ra sao?" has no
    # "ngân hàng nào"/"tốt nhất", so it correctly falls through to WHATIF's
    # own _INTEREST_RATE_WORDS check instead).
    if "ngân hàng" in question.lower() or "bank" in question.lower():
        has_rate_word = "lãi suất" in question.lower() or "deposit rate" in question.lower() or "interest rate" in question.lower()
        has_superlative = _match_any(
            [r"tốt nhất", r"cao nhất", r"\bbest\b", r"\bhighest\b"], question,
        )
        ticker_here = _extract_ticker(question)
        if has_rate_word and has_superlative and not ticker_here:
            return QueryIntent(
                branch="bank_rate_lookup",
                reason=_r(lang,
                    "Từ khóa 'ngân hàng' + 'lãi suất' + so sánh cấp cao nhất, không có mã cụ "
                    "thể trong câu hỏi -> xếp hạng lãi suất tiền gửi trực tiếp trên "
                    "macro_indicators (DEPOSIT_RATES, có cấu trúc, không qua RAG).",
                    "Keywords 'bank' + 'interest rate' + a superlative, with no specific ticker "
                    "in the question -> deposit-rate ranking straight over macro_indicators "
                    "(DEPOSIT_RATES, structured, no RAG involved)."),
            )

    # 4. COMPARE - "so sánh"/"compare" + 2 OR MORE distinct tickers ->
    # multi-ticker performance comparison (src/causal/market_factors.py),
    # NOT a scenario calculation. Checked before WHATIF specifically
    # because both share the "so sánh"/"compare" marker; the tiebreaker is
    # ticker count (COMPARE needs >=2, WHATIF's equivalent path needs
    # exactly 1 ticker + 2 periods - see WHATIF_MARKERS' comment).
    compare_markers = [r"\bso sánh\b", r"\bcompare\b", r"\bso với\b", r"\bversus\b", r"\bvs\.?\b"]
    if _match_any(compare_markers, question):
        tickers = _extract_all_tickers(question)
        if len(tickers) >= 2:
            with_gold = any(w in question.lower() for w in _GOLD_WORDS)
            return QueryIntent(
                branch="compare", tickers=tickers, compare_with_gold=with_gold,
                reason=_r(lang,
                    f"Từ khóa so sánh + {len(tickers)} mã: {tickers} -> so sánh hiệu suất giá "
                    f"(compare_tickers())" + (" kèm giá vàng." if with_gold else "."),
                    f"Comparison keyword + {len(tickers)} tickers: {tickers} -> price performance "
                    f"comparison (compare_tickers())" + (" including the gold price." if with_gold else ".")),
            )

    # 4b. Historical comparison against the prior year.  A phrase such as
    # "compare ... with the previous year" has only one explicit year, so it
    # used to enter WHATIF because of the broad ``compare`` marker and then
    # fail as if it lacked data.  Infer the adjacent completed fiscal year.
    # Compound/explanatory/citation requests need disclosure evidence across
    # multiple facts, so send those to RAG.  A single recognised line item
    # remains an exact structured growth calculation.
    ticker = _extract_ticker(question)
    years = sorted({int(y) for y in YEAR_RE.findall(question)})
    if ticker and len(years) == 1 and _match_any(_PRIOR_YEAR_MARKERS, question):
        low = question.lower()
        if _match_any(_DOCUMENT_GROUNDING_MARKERS, question):
            return QueryIntent(
                branch="rag", ticker=ticker,
                reason=_r(lang,
                    f"So sánh số liệu lịch sử {ticker} năm {years[0]} với năm trước, có yêu cầu "
                    "giải thích/trích dẫn tài liệu -> RAG trên disclosure chunks.",
                    f"Historical comparison of {ticker} in {years[0]} against the prior year, with a "
                    "request for explanation/citation -> RAG over disclosure chunks."),
            )
        resolved = resolve_indicator(question)
        if resolved:
            stype, ms_code, alias = resolved
            return QueryIntent(
                branch="whatif", ticker=ticker, whatif_kind="growth_rate",
                periods=[(years[0] - 1, "YEAR"), (years[0], "YEAR")],
                indicator_stype=stype, indicator_ms_code=ms_code,
                reason=_r(lang,
                    f"So sánh chỉ tiêu lịch sử '{alias}' của {ticker}: năm trước "
                    f"({years[0] - 1}) so với {years[0]} -> growth_rate().",
                    f"Historical comparison of line item '{alias}' for {ticker}: prior year "
                    f"({years[0] - 1}) versus {years[0]} -> growth_rate()."),
            )
        return QueryIntent(
            branch="rag", ticker=ticker,
            reason=_r(lang,
                f"So sánh số liệu lịch sử {ticker} năm {years[0]} với năm trước nhưng không "
                "xác định được một chỉ tiêu có cấu trúc duy nhất -> RAG trên disclosure chunks.",
                f"Historical comparison of {ticker} in {years[0]} against the prior year, but no "
                "single structured line item could be resolved -> RAG over disclosure chunks."),
        )

    # 5. WHATIF - scenario/decomposition language, always tied to ONE
    # ticker (calculator.py operates on a single company's own numbers).
    if _match_any(WHATIF_MARKERS, question):
        ticker = _extract_ticker(question)
        if not ticker:
            return QueryIntent(
                branch="whatif_unsupported",
                reason=_r(lang,
                    "Có từ khóa what-if/kịch bản nhưng không nhận diện được mã cổ phiếu trong câu hỏi.",
                    "What-if/scenario keyword present but no ticker symbol was recognised in the question."),
            )

        years = sorted({int(y) for y in YEAR_RE.findall(question)})
        pct_match = _PERCENT_RE.search(question)
        low = question.lower()
        is_gold = any(w in low for w in _GOLD_WORDS)
        is_oil = any(w in low for w in _OIL_WORDS)
        is_industry = any(w in low for w in _INDUSTRY_WORDS)
        is_interest_rate = any(w in low for w in _INTEREST_RATE_WORDS)
        is_trade = any(w in low for w in _TRADE_WORDS)

        # Gold-price scenario, no explicit % given - use the real latest
        # gold price move (src/causal/market_factors.py) instead of asking
        # the user to type a hypothetical. Only for calculator.py's
        # GOLD_SENSITIVE_TICKERS (currently just PNJ - gold is literally
        # their raw material, see calculator.py's GOLD_SENSITIVE_TICKERS
        # docstring for why this mapping is kept small/explicit).
        if is_gold and not pct_match:
            from src.structured.calculator import GOLD_SENSITIVE_TICKERS
            if ticker not in GOLD_SENSITIVE_TICKERS:
                return QueryIntent(
                    branch="whatif_unsupported",
                    reason=_r(lang,
                        f"Có từ khóa kịch bản giá vàng nhưng {ticker} chưa được xác định là "
                        f"nhạy cảm với giá vàng (xem GOLD_SENSITIVE_TICKERS trong calculator.py "
                        f"- hiện chỉ có PNJ).",
                        f"Gold-price scenario keyword present but {ticker} is not registered as "
                        f"gold-sensitive (see GOLD_SENSITIVE_TICKERS in calculator.py "
                        f"- currently only PNJ)."),
                )
            year = years[-1] if years else None
            return QueryIntent(
                branch="whatif", ticker=ticker, whatif_kind="gold_sensitivity",
                periods=[(year, "YEAR")] if year else [],
                reason=_r(lang,
                    f"Kịch bản giá vàng thực tế cho {ticker} (không có % cụ thể trong câu hỏi) "
                    f"-> sensitivity_what_if_gold(), dùng biến động giá vàng SJC gần nhất.",
                    f"Actual gold-price scenario for {ticker} (no explicit % in the question) "
                    f"-> sensitivity_what_if_gold(), using the latest SJC gold-price move."),
            )

        # Same pattern for oil, but as a REVENUE driver, not COGS - see
        # calculator.OIL_SENSITIVE_TICKERS's docstring for why (GAS sells
        # gas under oil-indexed pricing, unlike PNJ where gold is a cost).
        if is_oil and not pct_match:
            from src.structured.calculator import OIL_SENSITIVE_TICKERS
            if ticker not in OIL_SENSITIVE_TICKERS:
                return QueryIntent(
                    branch="whatif_unsupported",
                    reason=_r(lang,
                        f"Có từ khóa kịch bản giá dầu nhưng {ticker} chưa được xác định là "
                        f"nhạy cảm với giá dầu (xem OIL_SENSITIVE_TICKERS trong calculator.py "
                        f"- hiện chỉ có GAS).",
                        f"Oil-price scenario keyword present but {ticker} is not registered as "
                        f"oil-sensitive (see OIL_SENSITIVE_TICKERS in calculator.py "
                        f"- currently only GAS)."),
                )
            year = years[-1] if years else None
            return QueryIntent(
                branch="whatif", ticker=ticker, whatif_kind="oil_sensitivity",
                periods=[(year, "YEAR")] if year else [],
                reason=_r(lang,
                    f"Kịch bản giá dầu thực tế cho {ticker} (không có % cụ thể trong câu hỏi) "
                    f"-> sensitivity_what_if_oil(), dùng biến động giá dầu WTI gần nhất (tác động "
                    f"lên DOANH THU, không phải COGS).",
                    f"Actual oil-price scenario for {ticker} (no explicit % in the question) "
                    f"-> sensitivity_what_if_oil(), using the latest WTI oil-price move (it hits "
                    f"REVENUE, not COGS)."),
            )

        # Industry-index scenario (2026-09-15) - real IIP_BY_INDUSTRY/RETAIL
        # move for the ticker's own industry, via TICKER_INDUSTRY_MAP. See
        # _INDUSTRY_WORDS above for why this was missing until now.
        if is_industry and not pct_match:
            from src.causal.macro_lookup import TICKER_INDUSTRY_MAP
            if ticker not in TICKER_INDUSTRY_MAP:
                return QueryIntent(
                    branch="whatif_unsupported",
                    reason=_r(lang,
                        f"Có từ khóa kịch bản chỉ số ngành nhưng {ticker} chưa có ánh xạ "
                        f"ngành trong TICKER_INDUSTRY_MAP (src/causal/macro_lookup.py) - "
                        f"mapping cố ý nhỏ, chỉ gán khi IIP/RETAIL thật sự khớp hoạt động "
                        f"kinh doanh của công ty.",
                        f"Industry-index scenario keyword present but {ticker} has no industry "
                        f"mapping in TICKER_INDUSTRY_MAP (src/causal/macro_lookup.py) - the "
                        f"mapping is deliberately small, assigned only where IIP/RETAIL genuinely "
                        f"matches the company's business."),
                )
            year = years[-1] if years else None
            return QueryIntent(
                branch="whatif", ticker=ticker, whatif_kind="macro_sensitivity",
                periods=[(year, "YEAR")] if year else [],
                reason=_r(lang,
                    f"Kịch bản chỉ số ngành thực tế cho {ticker} (không có % cụ thể trong "
                    f"câu hỏi) -> sensitivity_what_if_macro(), dùng biến động chỉ số "
                    f"IIP_BY_INDUSTRY/RETAIL gần nhất (macro_indicators, nạp qua "
                    f"src.ingestion.macro_etl).",
                    f"Actual industry-index scenario for {ticker} (no explicit % in the question) "
                    f"-> sensitivity_what_if_macro(), using the latest IIP_BY_INDUSTRY/RETAIL "
                    f"index move (macro_indicators, loaded via src.ingestion.macro_etl)."),
            )

        # Interest-rate scenario (2026-09-15) - real VNIBOR overnight-rate
        # move as a shock to "Chi phí tài chính" -> pretax profit. See
        # _INTEREST_RATE_WORDS above for why no ticker-eligibility set is
        # needed (unlike gold/oil/industry).
        if is_interest_rate and not pct_match:
            year = years[-1] if years else None
            return QueryIntent(
                branch="whatif", ticker=ticker, whatif_kind="interest_rate_sensitivity",
                periods=[(year, "YEAR")] if year else [],
                reason=_r(lang,
                    f"Kịch bản lãi suất thực tế cho {ticker} (không có % cụ thể trong câu "
                    f"hỏi) -> sensitivity_what_if_interest_rate(), dùng biến động VNIBOR qua "
                    f"đêm gần nhất làm cú sốc Chi phí tài chính.",
                    f"Actual interest-rate scenario for {ticker} (no explicit % in the question) "
                    f"-> sensitivity_what_if_interest_rate(), using the latest overnight VNIBOR "
                    f"move as a shock to financial expense."),
            )

        # Export/import-commodity scenario (2026-09-15) - real
        # EXPORTS_BY_COMMODITY/IMPORTS_BY_COMMODITY move for the ticker's
        # own mapped commodity, via TRADE_INDUSTRY_MAP. See _TRADE_WORDS
        # above for why this was missing until now.
        if is_trade and not pct_match:
            from src.causal.macro_lookup import TRADE_INDUSTRY_MAP
            if ticker not in TRADE_INDUSTRY_MAP:
                return QueryIntent(
                    branch="whatif_unsupported",
                    reason=_r(lang,
                        f"Có từ khóa kịch bản xuất/nhập khẩu nhưng {ticker} chưa có ánh xạ "
                        f"mặt hàng trong TRADE_INDUSTRY_MAP (src/causal/macro_lookup.py) - "
                        f"mapping cố ý nhỏ, chỉ gán khi mặt hàng xuất/nhập khẩu thật sự khớp "
                        f"hoạt động kinh doanh của công ty (hiện chỉ có HPG/HSG - thép, "
                        f"DPM/DCM - phân bón).",
                        f"Export/import scenario keyword present but {ticker} has no commodity "
                        f"mapping in TRADE_INDUSTRY_MAP (src/causal/macro_lookup.py) - the mapping "
                        f"is deliberately small, assigned only where the traded commodity genuinely "
                        f"matches the company's business (currently only HPG/HSG - steel, "
                        f"DPM/DCM - fertiliser)."),
                )
            year = years[-1] if years else None
            return QueryIntent(
                branch="whatif", ticker=ticker, whatif_kind="trade_sensitivity",
                periods=[(year, "YEAR")] if year else [],
                reason=_r(lang,
                    f"Kịch bản xuất/nhập khẩu hàng hóa thực tế cho {ticker} (không có % cụ "
                    f"thể trong câu hỏi) -> sensitivity_what_if_trade(), dùng biến động "
                    f"EXPORTS_BY_COMMODITY/IMPORTS_BY_COMMODITY gần nhất (macro_indicators, "
                    f"nạp qua src.ingestion.macro_etl) làm cú sốc DOANH THU.",
                    f"Actual export/import commodity scenario for {ticker} (no explicit % in the "
                    f"question) -> sensitivity_what_if_trade(), using the latest "
                    f"EXPORTS_BY_COMMODITY/IMPORTS_BY_COMMODITY move (macro_indicators, loaded via "
                    f"src.ingestion.macro_etl) as a REVENUE shock."),
            )

        if pct_match:
            # Year is optional here - default to the LAST COMPLETED fiscal
            # year if the question doesn't name one (e.g. "Nếu giá dầu tăng
            # 10% thì GAS ra sao?" has no year at all). Found via testing
            # 2026-07-19: previously required `pct_match AND years`, so a
            # %-scenario with no explicit year fell all the way through to
            # the generic "thiếu 2 năm hoặc thiếu % kịch bản" message below
            # - confusing, since the % WAS present, just not a year. First
            # fix defaulted to the CURRENT calendar year, but that's wrong
            # in practice - annual financial statements for the current,
            # still-in-progress fiscal year don't exist yet (companies file
            # with a lag of a few months after year-end), so "năm hiện tại"
            # almost always produced "thiếu dữ liệu revenue/cogs" - a real
            # bug caught via a live test against GAS on 2026-07-19. Using
            # year - 1 assumes calendar-year fiscal years (true for the
            # companies in this project's corpus) and that the prior year's
            # annual report is already filed by the time this question is
            # asked - true for most of the year except the first ~1-2
            # quarters, when even year-1 might not be filed yet either.
            year = years[-1] if years else date.today().year - 1
            year_note = "" if years else " (không nêu rõ năm, mặc định năm tài chính gần nhất đã hoàn tất)"
            scenarios = _extract_pct_scenarios(question)

            # 2026-09-14 fix (see _extract_pct_scenarios()'s docstring for
            # the full "giảm" sign bug + compound-question rationale):
            # COMPOUND case first - id=41 ("COGS tăng 5% VÀ doanh thu giảm
            # 3% cùng lúc") - only fires when EACH % is independently
            # tagged to its own keyword, so a single-% question that
            # happens to match len(scenarios)>=2 for some other reason
            # (shouldn't happen given _PERCENT_RE, but stay conservative)
            # never mis-fires into this branch.
            cogs_tagged = [p for p, ic, ir in scenarios if ic]
            revenue_tagged = [p for p, ic, ir in scenarios if ir]
            if len(scenarios) >= 2 and cogs_tagged and revenue_tagged:
                cogs_pct, revenue_pct = cogs_tagged[0], revenue_tagged[0]
                return QueryIntent(
                    branch="whatif", ticker=ticker, whatif_kind="sensitivity",
                    periods=[(year, "YEAR")],
                    cogs_pct_change=cogs_pct, revenue_pct_change=revenue_pct,
                    reason=_r(lang,
                        f"Kịch bản ceteris-paribus ĐỒNG THỜI: {ticker}, năm {year}{year_note}, "
                        f"COGS {cogs_pct:+.1f}% VÀ doanh thu {revenue_pct:+.1f}% cùng lúc "
                        f"-> sensitivity_what_if() (2 tham số, sửa lỗi P3 'chỉ bắt % đầu tiên').",
                        f"SIMULTANEOUS ceteris-paribus scenario: {ticker}, year {year}{year_note}, "
                        f"COGS {cogs_pct:+.1f}% AND revenue {revenue_pct:+.1f}% at once "
                        f"-> sensitivity_what_if() (2 parameters, fixes the P3 'only the first % was read' bug)."),
                )

            # Single-scenario (existing behavior) - sign now comes from
            # scenarios[0] (the local tăng/giảm keyword), not a bare
            # float(pct_match...) that silently ignored "giảm" entirely.
            pct, is_cogs_local, is_revenue_local = scenarios[0]
            if is_cogs_local:
                is_cogs = True
            elif is_revenue_local:
                is_cogs = False
            else:
                is_cogs = ("cogs" in low or "giá vốn" in low or is_gold) and not is_oil
            return QueryIntent(
                branch="whatif", ticker=ticker, whatif_kind="sensitivity",
                periods=[(year, "YEAR")],
                cogs_pct_change=pct if is_cogs else 0.0,
                revenue_pct_change=pct if not is_cogs else 0.0,
                reason=_r(lang,
                    f"Kịch bản ceteris-paribus: {ticker}, năm {year}{year_note}, "
                    f"{'COGS' if is_cogs else 'doanh thu'} thay đổi {pct:+.1f}% "
                    f"-> sensitivity_what_if().",
                    f"Ceteris-paribus scenario: {ticker}, year {year}{year_note}, "
                    f"{'COGS' if is_cogs else 'revenue'} changes {pct:+.1f}% "
                    f"-> sensitivity_what_if()."),
            )

        if len(years) >= 2:
            is_margin = any(w in low for w in ("margin", "biên lợi nhuận", "phân rã", "bridge"))
            if is_margin:
                return QueryIntent(
                    branch="whatif", ticker=ticker, whatif_kind="decompose_margin",
                    periods=[(years[0], "YEAR"), (years[-1], "YEAR")],
                    reason=_r(lang,
                        f"So sánh biên lợi nhuận gộp {ticker} giữa {years[0]} và {years[-1]} "
                        f"-> decompose_gross_margin_change() (bridge revenue-effect vs "
                        f"margin-rate-effect).",
                        f"Gross-margin comparison for {ticker} between {years[0]} and {years[-1]} "
                        f"-> decompose_gross_margin_change() (bridge revenue-effect vs "
                        f"margin-rate-effect)."),
                )
            resolved = resolve_indicator(question)
            if resolved:
                stype, ms_code, alias = resolved
                return QueryIntent(
                    branch="whatif", ticker=ticker, whatif_kind="growth_rate",
                    periods=[(years[0], "YEAR"), (years[-1], "YEAR")],
                    indicator_stype=stype, indicator_ms_code=ms_code,
                    reason=_r(lang,
                        f"So sánh chỉ tiêu '{alias}' của {ticker} giữa {years[0]} và "
                        f"{years[-1]} -> growth_rate().",
                        f"Comparison of line item '{alias}' for {ticker} between {years[0]} and "
                        f"{years[-1]} -> growth_rate()."),
                )
            return QueryIntent(
                branch="whatif_unsupported",
                reason=_r(lang,
                    f"Có 2 năm ({years[0]}, {years[-1]}) và mã {ticker} nhưng không nhận diện "
                    f"được chỉ tiêu cần so sánh (không khớp danh sách alias trong "
                    f"src/structured/schema.py).",
                    f"Two years ({years[0]}, {years[-1]}) and ticker {ticker} were found, but the "
                    f"line item to compare was not recognised (no match in the alias list in "
                    f"src/structured/schema.py)."),
            )

        # 2026-09-15: last resort before giving up - "tăng trưởng"/"growth
        # rate" is a registered WHATIF_MARKERS phrase (so a growth
        # QUESTION is caught here, before RATIO_LOOKUP ever runs), but
        # Vietstock's financial_ratios ALSO has 9 pre-computed "Tăng
        # trưởng..." ratios (see ratio_schema.py's NHÓM CHỈ SỐ TĂNG TRƯỞNG
        # group) that need only ONE period, not two. Found via
        # classify_intent() regression: "Tăng trưởng doanh thu thuần của
        # CTD năm 2024 là bao nhiêu?" (a single-period ratio lookup)
        # incorrectly fell all the way to whatif_unsupported here, since
        # `years` only had 1 entry - never reaching the growth_rate path
        # above (which requires len(years) >= 2) or RATIO_LOOKUP (checked
        # later, step 6a, which this WHATIF block would already have
        # short-circuited past). Try the ratio table before failing.
        if ticker and len(years) == 1:
            from src.structured.ratio_schema import resolve_ratio
            ratio_resolved = resolve_ratio(question)
            if ratio_resolved:
                ratio_group, ratio_name, ratio_alias = ratio_resolved
                probe = parse_question(question)
                period = probe.period or "YEAR"
                return QueryIntent(
                    branch="ratio_lookup", ticker=ticker,
                    indicator_stype=ratio_group, indicator_ms_code=ratio_name,
                    periods=[(years[0], period)],
                    reason=_r(lang,
                        f"Có từ khóa what-if ('tăng trưởng'/'growth') nhưng câu hỏi thực ra "
                        f"khớp 1 chỉ số financial_ratios có sẵn ('{ratio_name}', nhóm "
                        f"{ratio_group}) cho 1 kỳ duy nhất, không phải kịch bản 2 kỳ -> "
                        f"financial_ratios thay vì growth_rate().",
                        f"What-if keyword ('tăng trưởng'/'growth') present, but the question "
                        f"actually matches one pre-computed financial_ratios metric "
                        f"('{_ratio_label_en(ratio_name)}', {_ratio_group_en(ratio_group)} ratios) "
                        f"for a SINGLE period, not a "
                        f"two-period scenario -> financial_ratios instead of growth_rate()."),
                )

        return QueryIntent(
            branch="whatif_unsupported",
            reason=_r(lang,
                f"Có từ khóa what-if và mã {ticker} nhưng thiếu 2 năm để so sánh, hoặc thiếu "
                f"% kịch bản cụ thể (ví dụ 'nếu COGS tăng 5%').",
                f"What-if keyword and ticker {ticker} were found, but the question lacks either "
                f"two years to compare or an explicit scenario % (e.g. 'if COGS rose 5%')."),
        )

    # 5b. DAG_MULTI (2026-09-14, fixes P3 audit bug id=42: "Tổng tài sản của
    # cả VNM, HPG và PNJ năm 2024 là bao nhiêu?"). BEFORE this fix, the
    # single-ticker DAG check below called query_builder._extract_ticker(),
    # which returns only the FIRST valid ticker it finds and silently
    # drops the rest - the question above would get answered as if it had
    # only asked about VNM, with HPG/PNJ dropped with NO indication
    # anything was skipped (worse than an explicit refusal - a confident,
    # silently-partial answer). Checked here (after WHATIF, before the
    # single-ticker DAG check) since a DAG-SHAPED question (indicator +
    # period both resolvable) naming >=2 REAL tickers is unambiguously a
    # multi-ticker lookup, never a single-ticker one. Does not collide with
    # COMPARE (checked much earlier, step 4) since COMPARE requires its own
    # "so sánh"/"compare" marker, which this kind of question doesn't need.
    all_tickers = _extract_all_tickers(question)
    if len(all_tickers) >= 2:
        probe = parse_question(question)
        if probe.period and probe.fiscal_year and probe.ms_code:
            return QueryIntent(
                branch="dag_multi", tickers=all_tickers,
                reason=_r(lang,
                    f"{len(all_tickers)} mã ({all_tickers}) + chỉ tiêu '{probe.matched_alias}' "
                    f"+ kỳ {probe.period}/{probe.fiscal_year} -> tra cứu DAG RIÊNG CHO TỪNG MÃ "
                    f"(trước đây chỉ trả lời mã đầu tiên, âm thầm bỏ qua các mã còn lại).",
                    f"{len(all_tickers)} tickers ({all_tickers}) + line item '{probe.matched_alias}' "
                    f"+ period {probe.period}/{probe.fiscal_year} -> a SEPARATE DAG lookup PER TICKER "
                    f"(previously only the first ticker was answered, the rest silently dropped)."),
            )

    # 6 / 6a. DAG vs RATIO_LOOKUP - a known indicator (financial_line_items)
    # OR ratio (financial_ratios, src/structured/ratio_schema.py) alias
    # resolves AND ticker/period/year are all extractable. Resolved
    # TOGETHER, not DAG-first-ratio-as-fallback, because a ratio_name often
    # CONTAINS a shorter financial_line_items alias as a plain substring -
    # e.g. "Tỷ số Nợ trên Vốn chủ sở hữu" contains "vốn chủ sở hữu", itself
    # a registered ms_code 400 alias. Found 2026-09-15 while verifying
    # RATIO_LOOKUP: checking DAG first (as an early version of this code
    # did) let "vốn chủ sở hữu" win over the longer, more specific ratio
    # phrase for EVERY leverage/structure ratio whose name embeds a
    # balance-sheet line item name (most of them - see ratio_schema.py's
    # NHÓM CHỈ SỐ ĐÒN BẨY TÀI CHÍNH / CƠ CẤU... groups) - not a one-off
    # edge case. Fix: resolve both, prefer whichever matched alias string
    # is LONGER (same "most specific substring wins" principle
    # resolve_indicator()/resolve_ratio() already each use internally,
    # extended here to compare ACROSS the two alias tables).
    parsed = parse_question(question)
    ratio_resolved = None
    if parsed.ticker and parsed.period and parsed.fiscal_year:
        from src.structured.ratio_schema import resolve_ratio
        ratio_resolved = resolve_ratio(question)

    dag_alias_len = len(parsed.matched_alias) if parsed.matched_alias else -1
    ratio_alias_len = len(ratio_resolved[2]) if ratio_resolved else -1

    if parsed.ticker and parsed.period and parsed.fiscal_year and parsed.ms_code and dag_alias_len >= ratio_alias_len:
        return QueryIntent(
            branch="dag", ticker=parsed.ticker,
            reason=_r(lang,
                f"Đủ 4 thành phần cho tra cứu có cấu trúc: mã={parsed.ticker}, "
                f"chỉ tiêu='{parsed.matched_alias}', kỳ={parsed.period}/{parsed.fiscal_year}.",
                f"All 4 components present for a structured lookup: ticker={parsed.ticker}, "
                f"line item='{parsed.matched_alias}', period={parsed.period}/{parsed.fiscal_year}."),
        )

    if ratio_resolved and parsed.ticker and parsed.period and parsed.fiscal_year:
        ratio_group, ratio_name, ratio_alias = ratio_resolved
        return QueryIntent(
            branch="ratio_lookup", ticker=parsed.ticker,
            indicator_stype=ratio_group, indicator_ms_code=ratio_name,
            periods=[(parsed.fiscal_year, parsed.period)],
            reason=_r(lang,
                f"Đủ 4 thành phần cho tra cứu chỉ số tài chính: mã={parsed.ticker}, "
                f"chỉ số='{ratio_name}' (nhóm {ratio_group}), kỳ={parsed.period}/"
                f"{parsed.fiscal_year} -> financial_ratios (KHÁC financial_line_items).",
                f"All 4 components present for a financial-ratio lookup: ticker={parsed.ticker}, "
                f"ratio='{_ratio_label_en(ratio_name)}' ({_ratio_group_en(ratio_group)} ratios), "
                f"period={parsed.period}/"
                f"{parsed.fiscal_year} -> financial_ratios (DIFFERENT from financial_line_items)."),
        )

    # 6b. DAG_TREND (2026-08-07) - ticker + chỉ tiêu nhận diện được nhưng
    # KHÔNG có năm/kỳ cụ thể, kèm ngôn ngữ xu hướng nhiều năm (xem
    # TREND_LANGUAGE_MARKERS ở trên) - dựng chuỗi NHIỀU năm gần nhất trực
    # tiếp từ financial_line_items (có cấu trúc, chính xác) thay vì rơi
    # vào RAG. Đặt SAU DAG (cần fiscal_year) nhưng TRƯỚC RAG fallback -
    # chỉ bắt đúng ca "có chỉ tiêu, thiếu năm, hỏi kiểu xu hướng", không
    # đụng tới câu hỏi định tính thật sự (rủi ro pháp lý, quản trị...).
    if parsed.ticker and parsed.ms_code and not parsed.fiscal_year:
        if _match_any(TREND_LANGUAGE_MARKERS, question):
            return QueryIntent(
                branch="dag_trend", ticker=parsed.ticker,
                reason=_r(lang,
                    f"Có mã {parsed.ticker} + chỉ tiêu '{parsed.matched_alias}' nhưng không "
                    f"nêu năm cụ thể, kèm ngôn ngữ xu hướng nhiều năm -> dựng chuỗi năm gần "
                    f"nhất từ financial_line_items thay vì RAG.",
                    f"Ticker {parsed.ticker} + line item '{parsed.matched_alias}' present but no "
                    f"specific year named, together with multi-year trend language -> build the "
                    f"recent-years series from financial_line_items instead of RAG."),
            )

    # 6c. OUT_OF_SCOPE (2026-09-14) - checked AFTER every real branch (so a
    # question that's genuinely answerable never gets short-circuited) but
    # BEFORE the RAG catch-all, since these 3 question types are never
    # answerable from disclosure text no matter how RAG's retrieval is
    # tuned - see the marker definitions above for why a dedicated branch
    # (vs RAG's generic "try being more specific" message) matters here.
    # Order below matters: third-party-opinion checked first since it's the
    # most specific (requires an actual name capture); investment-advice
    # and future-prediction are plain keyword hits and could theoretically
    # both match a contrived sentence, but real audit questions never do.
    person = _match_third_party_opinion(question)
    if person:
        return QueryIntent(
            branch="out_of_scope", oos_kind="third_party_opinion", oos_person=person,
            ticker=parsed.ticker,
            reason=_r(lang,
                f"Câu hỏi xin ý kiến/quan điểm cá nhân của '{person}' - đây không phải "
                f"một sự kiện đã công bố trong BCTC/tài liệu công ty, nên KHÔNG chuyển "
                f"sang RAG (rủi ro model bịa một ý kiến gán cho '{person}').",
                f"The question asks for the personal opinion/view of '{person}' - that is not a "
                f"fact published in any financial statement or company document, so it is NOT "
                f"routed to RAG (risk of the model fabricating an opinion attributed to '{person}')."),
        )
    if _match_any(INVESTMENT_ADVICE_MARKERS, question):
        return QueryIntent(
            branch="out_of_scope", oos_kind="investment_advice", ticker=parsed.ticker,
            reason=_r(lang,
                "Câu hỏi xin tư vấn đầu tư cá nhân (mua/bán) - ranh giới tuân thủ cố ý, "
                "không phải khoảng trống dữ liệu, nên không chuyển sang RAG/LLM.",
                "The question asks for personal investment advice (buy/sell) - a deliberate "
                "compliance boundary, not a data gap, so it is not routed to RAG/LLM."),
        )
    if _match_any(FUTURE_PREDICTION_MARKERS, question):
        return QueryIntent(
            branch="out_of_scope", oos_kind="future_prediction", ticker=parsed.ticker,
            reason=_r(lang,
                "Câu hỏi dự đoán giá cổ phiếu tương lai - hệ thống xác minh sự kiện đã "
                "công bố (fact verification), không phải mô hình dự báo giá, nên không "
                "chuyển sang RAG/LLM (một con số cụ thể ở đây sẽ là suy đoán).",
                "The question asks for a future stock-price prediction - this system verifies "
                "published facts (fact verification), it is not a price-forecasting model, so it "
                "is not routed to RAG/LLM (any specific number here would be speculation)."),
        )

    # 6d. PRICE_SNAPSHOT (2026-09-21) - xem PRICE_WORD_MARKERS/
    # PRICE_TODAY_MARKERS's comment ở trên cho lý do cần nhánh riêng thay vì
    # để rơi vào RAG (disclosure_chunks không có giá đóng cửa hôm nay) hay
    # dag_trend (cần ms_code, câu hỏi giá thuần túy không có). Đặt SAU
    # out_of_scope (một câu "giá X hôm nay có nên mua không" phải rơi vào
    # investment_advice ở trên, không phải price_snapshot) nhưng TRƯỚC RAG
    # fallback. Cần CẢ ticker lẫn cả 2 nhóm từ khóa - thiếu ticker thì
    # market_factors.latest_price_snapshot() không biết tra mã nào.
    price_ticker = parsed.ticker or _extract_ticker(question)
    # 2026-09-22: chấp nhận CẢ HAI kiểu khung thời gian, không chỉ "hôm nay".
    # Trước đây điều kiện là `_match_any(PRICE_TODAY_MARKERS, ...)` trơn, nên
    # "giá VNZ vài ngày gần đây" rơi xuống RAG dù dữ liệu có sẵn - xem
    # _PRICE_RECENT_MARKERS ở trên.
    window_days = _price_window_days(question)
    asks_today = _match_any(PRICE_TODAY_MARKERS, question)
    date_range = _price_date_range(question)
    if (price_ticker and _match_any(PRICE_WORD_MARKERS, question)
            and (asks_today or window_days or date_range)
            and not _match_any(_PRICE_SNAPSHOT_EXCLUDE, question)):
        # Mốc TƯỜNG MINH thắng mọi thứ khác: người đã nêu "từ tháng 10/2025
        # đến tháng 1/2026" thì không còn hỏi "gần đây" nữa.
        if date_range:
            d0, d1 = date_range
            return QueryIntent(
                branch="price_snapshot", ticker=price_ticker,
                price_start_date=d0, price_end_date=d1,
                reason=_r(lang,
                    f"Câu hỏi về giá cổ phiếu {price_ticker} trong khoảng {d0} - {d1} "
                    f"-> tra company_price_snapshot trực tiếp (EOD, KHÔNG phải RAG - không "
                    f"có tài liệu công bố nào ghi giá đóng cửa hàng ngày).",
                    f"Question about {price_ticker}'s stock price between {d0} and {d1} "
                    f"-> read company_price_snapshot directly (end-of-day, NOT RAG - no "
                    f"published disclosure records a daily closing price)."),
            )
        # "hôm nay" mà kèm cả cụm khoảng thời gian thì ưu tiên khoảng
        # (người hỏi muốn thấy diễn biến, không phải một con số).
        n_days = window_days or 1
        if n_days > 1:
            return QueryIntent(
                branch="price_snapshot", ticker=price_ticker, price_window_days=n_days,
                reason=_r(lang,
                    f"Câu hỏi về giá cổ phiếu {price_ticker} trong {n_days} phiên gần nhất "
                    f"-> tra company_price_snapshot trực tiếp (EOD, KHÔNG phải RAG - không "
                    f"có tài liệu công bố nào ghi giá đóng cửa hàng ngày).",
                    f"Question about {price_ticker}'s stock price over the last {n_days} "
                    f"sessions -> read company_price_snapshot directly (end-of-day, NOT RAG - "
                    f"no published disclosure records a daily closing price)."),
            )
        return QueryIntent(
            branch="price_snapshot", ticker=price_ticker, price_window_days=1,
            reason=_r(lang,
                f"Câu hỏi về giá cổ phiếu {price_ticker} hôm nay/gần đây -> tra "
                f"company_price_snapshot trực tiếp (EOD, KHÔNG phải RAG - không có tài "
                f"liệu công bố nào ghi giá đóng cửa hàng ngày).",
                f"Question about {price_ticker}'s stock price today/recently -> read "
                f"company_price_snapshot directly (end-of-day, NOT RAG - no published disclosure "
                f"records a daily closing price)."),
        )

    # 7. RAG - fallback. Covers qualitative/explanatory questions AND
    # DAG-shaped questions missing a piece (e.g. no year mentioned) - RAG
    # can still search the actual disclosure text for those.
    missing = parsed.missing_fields()
    return QueryIntent(
        branch="rag", ticker=parsed.ticker,
        reason=_r(lang,
            "Không khớp mẫu Causal/What-if, và tra cứu có cấu trúc thiếu: "
            f"{missing} -> chuyển sang tìm kiếm trong tài liệu công bố (RAG).",
            "No Causal/What-if pattern matched, and the structured lookup is missing: "
            f"{missing} -> falling back to search over published disclosures (RAG)."),
    )
