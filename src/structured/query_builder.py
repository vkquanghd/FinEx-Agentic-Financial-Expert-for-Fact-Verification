"""Turn a natural-language-ish question into a parameterized SQL lookup
against financial_line_items, and execute it. This is the "structured
lookup" implementation of the DAG path - bounded to CORE_MS_CODES via
src/structured/schema.py, not a general NL2SQL translator.
"""
import re
from dataclasses import dataclass
from typing import Optional

from sqlalchemy import text

from src.core.db import get_session
from src.structured.schema import resolve_indicator

TICKER_RE = re.compile(r"\b([A-Z]{3,4})\b")
YEAR_RE = re.compile(r"\b(20\d{2})\b")

# Vietnamese + English period phrasing, checked in this order (quarter wins
# over year-level words, since "quý 1 năm 2026" contains both "quý" and
# "năm" - the quarter is the intended period).
QUARTER_RE_VI = re.compile(r"[Qq]u[yý]\s*([1-4])")
QUARTER_RE_EN = re.compile(r"\bQ([1-4])\b|\bquarter\s*([1-4])\b", re.IGNORECASE)
HALF_RE = re.compile(r"6\s*th[aá]ng|\bH1\b|\bfirst half\b|\b1st half\b", re.IGNORECASE)
NINE_MONTH_RE = re.compile(r"9\s*th[aá]ng|\b9M\b|\bnine\s*months?\b", re.IGNORECASE)
YEAR_PERIOD_RE = re.compile(
    r"\bc[aả]\s*n[aă]m\b|\bn[aă]m\b|\bannual\b|\bfull\s*year\b|\bFY\b|\byear\b", re.IGNORECASE
)

# Financial abbreviations that look like tickers (3-4 uppercase letters) but
# are NOT tickers - without this, "EPS của PNJ..." would wrongly extract
# "EPS" as the ticker instead of "PNJ" (found via testing before shipping).
# "COGS"/"OCF" added 2026-07-19: found via testing src/router/rules.py's
# what-if classification - "Nếu COGS của CTD tăng 5%..." was extracting
# "COGS" as the ticker (it comes first in the sentence) instead of "CTD",
# since what-if questions naturally mention COGS/OCF a lot more than
# typical DAG questions do.
NON_TICKER_ABBREVIATIONS = {
    "EPS", "ROE", "ROA", "ROI", "ROS", "EBIT", "EBITDA", "CEO", "CFO", "CFA",
    "GTGT", "TNDN", "KQKD", "BCTC", "LNST", "LNTT", "VND", "USD", "TSCĐ",
    "HOSE", "HNX", "MS", "PE", "PB", "PS", "COGS", "OCF",
    # 2026-08-06: đồng bộ với src/structured/abbreviations.py (khai triển
    # viết tắt chỉ tiêu) - các viết tắt 3-4 chữ hoa này trông y hệt ticker
    # với TICKER_RE; is_known_ticker() thường chặn được, nhưng blocklist
    # vẫn cần cho nhánh fail-open khi DB không truy vấn được. "LNG" đặc
    # biệt: vừa là viết tắt lợi nhuận gộp vừa là khí hóa lỏng, và nếu sàn
    # có mã LNG thật thì is_known_ticker sẽ ưu tiên nhận nó là ticker.
    "DTT", "GVHB", "TTS", "TSNH", "TSDH", "VCSH", "NPT", "HTK", "LNG",
    "TDTT", "DTKD",
    # 2026-09-14: audit-firm names, found while verifying
    # scripts/final_benchmark_en.py's English auditor_lookup question
    # ("Which companies is KPMG the auditor for?"). KPMG is exactly 4
    # uppercase letters, matching TICKER_RE - in the sandbox verification
    # (DB unreachable, is_known_ticker() forced to error) this fell into
    # the SAME documented fail-open blind spot the comment above already
    # warns about, misread as a real ticker, which blocks
    # rules.py's AUDITOR_LOOKUP check ("firm and not ticker_here"). In a
    # DB-connected environment is_known_ticker("KPMG") correctly returns
    # False so this is a rare, DB-outage-only edge case in production -
    # but it costs nothing to close it here the same way every other
    # 3-4-letter abbreviation on this list already is. EY (2 letters) and
    # Deloitte/PwC (not 3-4 letters) never match TICKER_RE, so they don't
    # need an entry here.
    "KPMG",
    # 2026-09-15: same bug class, found while verifying
    # scripts/final_benchmark_100_en.py's id=72 ("AASC kiểm toán những công
    # ty nào?") - AASC (4 uppercase letters) hit the identical sandbox
    # fail-open blind spot as KPMG above, blocking AUDITOR_LOOKUP. UHY (3
    # uppercase letters, also a real firm name in
    # src/structured/auditor_lookup.py's _AUDITOR_FIRM_PATTERNS) is added
    # pre-emptively for the same reason even though no benchmark question
    # currently exercises it - closing every 3-4-letter audit-firm
    # abbreviation the same way rather than one at a time as each is
    # individually hit by a test question.
    "AASC", "UHY",
    # 2026-09-15: same bug class again, found while verifying the new
    # RATIO_LOOKUP branch (src/structured/ratio_schema.py) - "ROAA của KTS
    # năm 2024 là bao nhiêu?" extracted "ROAA" as the ticker (appears
    # before "KTS" in the sentence) instead of "KTS", since ROAA is 4
    # uppercase letters and ROE/ROA/ROI/ROS were already blocked but ROAA/
    # ROEA/ROCE (the actual Vietstock ratio-name abbreviations - see
    # ratio_schema.py's _RATIOS list) were not. Added every 3-4-letter
    # abbreviation from that ratio list not already covered above: BVPS,
    # ROEA, ROCE, ROAA, CPS.
    "BVPS", "ROEA", "ROCE", "ROAA", "CPS",
}


def _extract_ticker(question: str) -> Optional[str]:
    for match in TICKER_RE.finditer(question):
        candidate = match.group(1)
        if candidate in NON_TICKER_ABBREVIATIONS:
            continue
        # 2026-08-05 fix: a bare regex match that LOOKS like a ticker (3-4
        # uppercase letters) isn't necessarily a REAL one - "VINA Game"
        # matches as "VINA", which used to be returned as-is (real ticker
        # is VNZ) and silently answered "not found" instead of ever trying
        # the smarter name-resolution fallback below. Validate against
        # company_lookup.is_known_ticker() before trusting the match; if
        # verification itself fails (DB unreachable), fail OPEN and trust
        # the regex match rather than breaking extraction over a transient
        # DB hiccup - same resilience philosophy as the fallback below.
        try:
            from src.structured.company_lookup import is_known_ticker
            if is_known_ticker(candidate):
                return candidate
            continue  # verified NOT a real ticker - keep scanning / fall through
        except Exception:
            return candidate
    # Fallback (2026-07-24): no ACCEPTED bare-ticker regex match - try
    # resolving a company NAME instead (src/structured/company_lookup.py),
    # e.g. "Tổng tài sản của Tập đoàn Hòa Phát..." or a hand-picked brand
    # alias (BRAND_NAME_ALIASES, e.g. "VINA Game" -> VNZ). Reached when the
    # fast regex path found nothing ACCEPTED, since it costs a DB
    # round-trip (cached per process after the first call). Swallow any
    # DB-connectivity error rather than raising - ticker extraction failing
    # softly (falls through to RAG's "missing ticker" handling) is much
    # better than crashing the whole question over a lookup that was only
    # ever a best-effort convenience.
    try:
        from src.structured.company_lookup import resolve_company_name
        return resolve_company_name(question)
    except Exception:
        return None


@dataclass
class ParsedQuery:
    ticker: Optional[str]
    fiscal_year: Optional[int]
    period: Optional[str]
    statement_type: Optional[str]
    ms_code: Optional[str]
    matched_alias: Optional[str]

    def missing_fields(self) -> list[str]:
        return [f for f in ("ticker", "fiscal_year", "period", "ms_code") if getattr(self, f) is None]


def parse_question(question: str) -> ParsedQuery:
    ticker = _extract_ticker(question)
    year_match = YEAR_RE.search(question)
    quarter_match = QUARTER_RE_VI.search(question) or QUARTER_RE_EN.search(question)

    if quarter_match:
        q_num = next(g for g in quarter_match.groups() if g)
        period = f"Q{q_num}"
    elif HALF_RE.search(question):
        period = "H1"
    elif NINE_MONTH_RE.search(question):
        period = "9M"
    elif YEAR_PERIOD_RE.search(question):
        period = "YEAR"
    elif year_match:
        # A specific year was mentioned but no quarter/half/9-month keyword -
        # default to full year (e.g. "total assets of CTD in 2024").
        period = "YEAR"
    else:
        period = None

    resolved = resolve_indicator(question)
    stype, ms_code, alias = resolved if resolved else (None, None, None)

    return ParsedQuery(
        ticker=ticker,
        fiscal_year=int(year_match.group(1)) if year_match else None,
        period=period,
        statement_type=stype,
        ms_code=ms_code,
        matched_alias=alias,
    )


LOOKUP_LINE_ITEM = text("""
    SELECT value, unit, consolidation, item_name
    FROM financial_line_items
    WHERE ticker = :ticker AND fiscal_year = :fiscal_year AND period = :period
      AND statement_type = :statement_type AND ms_code = :ms_code
    LIMIT 1
""")


def execute(parsed: ParsedQuery) -> dict:
    missing = parsed.missing_fields()
    if missing:
        return {"ok": False, "error": f"Missing required fields: {missing}"}

    with get_session() as session:
        row = session.execute(LOOKUP_LINE_ITEM, {
            "ticker": parsed.ticker,
            "fiscal_year": parsed.fiscal_year,
            "period": parsed.period,
            "statement_type": parsed.statement_type,
            "ms_code": parsed.ms_code,
        }).first()

    if row is None:
        return {"ok": False, "error": "No matching row in financial_line_items (chưa nạp dữ liệu cho kỳ/mã này)"}

    value, unit, consolidation, item_name = row
    return {
        "ok": True,
        "value": value,
        "unit": unit,
        "consolidation": consolidation,
        "item_name": item_name,
        "ticker": parsed.ticker,
        "fiscal_year": parsed.fiscal_year,
        "period": parsed.period,
        "statement_type": parsed.statement_type,
        "ms_code": parsed.ms_code,
    }
