"""Resolve a Vietnamese company NAME (not just its 3-4 letter ticker) to a
ticker, so questions like "Tổng tài sản của Tập đoàn Hòa Phát...", "...của
Hòa Phát...", "...của TĐ Hòa Phát...", or even "...của Hoa Phat..."/typo'd
"...Hòaa pháttt..." all route the same way "...của HPG..." already does.
Every branch that needs a single ticker (DAG/RAG/WHATIF) goes through
query_builder._extract_ticker(); the multi-ticker COMPARE branch goes
through rules._extract_all_tickers(). Both fall back to this module's
resolvers when the fast regex-based ticker match fails - a DB round-trip
is much more expensive than a regex scan, so it's only ever tried second,
and only ever costs one extra query per process (see `_cache`).

Source of truth: the `companies` table (infra/postgres/migrations/
001_init_schema.sql, seeded by scripts/seed_companies.py) - deliberately
NOT data/reference/top200_candidate_companies.csv directly, since that CSV
is missing every CORE_16 large-cap ticker (VNM, VIC, HPG, GAS, MWG, PNJ,
VHM, NVL, KDH, DPM, DCM, DGW, FRT, SAB, MSN, KDC - see seed_companies.py's
CORE_16 constant) that gets seeded into `companies` separately.

Candidate generation per company (see _name_candidates()):
1. `_strip_legal_form()`: strip a known LEADING legal-form marker (CTCP /
   Công ty Cổ phần / Công ty TNHH / Ngân hàng TMCP / Ngân hàng) and a
   trailing " - CTCP" suffix, e.g. "Tập đoàn VINGROUP - CTCP" -> "Tập đoàn
   VINGROUP". Always kept as one candidate.
2. If what's left still starts with a secondary corporate-form phrase
   ("Tổng Công ty (Cổ phần)" / "Tập đoàn"), two more candidates are added:
   the phrase stripped again (e.g. "Hòa Phát", "VINGROUP" - what a
   Vietnamese speaker would actually say), and a common abbreviation
   ("TĐ Hòa Phát", "TCT Khí Việt Nam").

Three-tier matching (2026-07-24, extended twice from an original
exact-only design after live testing surfaced 2 real gaps - see git
history / docs/PROGRESS.md's "Ticker mở rộng" sections for the full
before/after):

  Tier 1 - EXACT (case-insensitive, diacritics matter): a candidate is a
  literal substring of the question. Always tried first and wins outright
  if it finds anything - the most reliable tier, nothing below it should
  ever override it.

  Tier 2 - DIACRITIC-FOLDED exact match: both the candidate and the
  question have Vietnamese diacritics/tone marks stripped (see
  _fold_diacritics() - handles đ/Đ specially since Unicode NFD doesn't
  decompose that letter), then compared as an exact substring again. This
  is still a DETERMINISTIC transform, not approximate - it exists because
  typing Vietnamese without diacritics ("Hoa Phat", "Khi Viet Nam") is
  extremely common and isn't a "typo" so much as a different, valid input
  convention. Only tried if tier 1 found nothing.

  Tier 3 - FUZZY (difflib.SequenceMatcher, sliding word-window): only
  tried if tiers 1-2 found nothing. Compares each candidate against every
  contiguous word-window of the question with the same word count as the
  candidate (a fuzzy SUBSTRING search - comparing a short candidate
  against the whole long question directly would unfairly tank the ratio
  even for a perfect match). A window must score >= _FUZZY_THRESHOLD
  (0.82, chosen empirically 2026-07-24 - real doubled-letter typos like
  "Hòaa pháttt" vs "Hòa Phát" score ~0.84; genuinely different real words
  like "Hòa Bình" vs "Hòa Phát" score ~0.63, comfortably below threshold).
  This is the ONLY genuinely approximate tier and the only one with real
  false-positive risk, which is why it's last-resort only.

Ambiguity handling: resolve_company_name() (singular - DAG/RAG/WHATIF)
takes each ticker's longest matching candidate at whichever tier fired,
then picks the overall longest across tickers; a genuine length-tie
between two DIFFERENT tickers returns None rather than guessing - a wrong
ticker is worse than no match, same philosophy as _extract_ticker()'s
NON_TICKER_ABBREVIATIONS blocklist and vector_store's doctype-filter
fallback. resolve_all_company_names() (plural - COMPARE) instead
accumulates ALL tiers' matches (exact + folded always; fuzzy only if still
short of 2 tickers) into one set, since COMPARE already wants 2+ tickers
and an extra match there is a much smaller risk than a wrong single answer.

Known limitations - documented rather than silently failing:
- Only matches the LEGAL company name (+ generated variants), NOT
  trading/brand names absent from it - "Vinamilk" is not a substring (folded
  or not) of VNM's "CTCP Sữa Việt Nam", so a question that only says
  "Vinamilk" will NOT resolve. Hand-curating a brand-alias table for ~216
  companies is out of scope for the remaining thesis weeks.
- Tier 2 (diacritic folding) reliably catches SHORT (2-3 word) fully
  ASCII-typed names ("Hoa Phat" -> 0.75-0.92 range tested) but longer
  multi-word official names lose too much information when fully
  stripped to pass even the fuzzy tier's threshold (e.g. "Vang bac da quy
  phu nhuan" for PNJ's full legal name scores only ~0.72) - tier 2 (exact
  after folding) still catches these regardless of length since it's
  exact, not similarity-scored, but a candidate that ALSO needs fuzzy
  tolerance (e.g. accent-stripped AND slightly typo'd at once) is a
  genuine, accepted gap.
- Tier 3 uses no dependency beyond the stdlib (`difflib`) rather than
  rapidfuzz/python-Levenshtein - deliberate: the closed ~216-name
  vocabulary doesn't need a heavier library's extra matching modes, and
  avoiding a new dependency keeps this testable without any install step.
  Swapping in rapidfuzz later (better/faster ratio functions) is a
  contained change if real usage ever shows difflib's quality is
  insufficient.
"""
import re
import unicodedata
from difflib import SequenceMatcher
from typing import Optional

from sqlalchemy import text

from src.core.db import get_session

_LEADING_LEGAL_FORMS = [
    "công ty cổ phần", "ctcp", "công ty tnhh",
    "ngân hàng thương mại cổ phần", "ngân hàng tmcp", "ngân hàng",
]
_TRAILING_SUFFIX = " - ctcp"

# Checked longest-phrase-first so "tổng công ty cổ phần" doesn't get
# mis-split as "tổng công ty" + leftover "cổ phần ...".
_SECONDARY_FORMS = [
    ("tổng công ty cổ phần", "TCT"),
    ("tổng công ty", "TCT"),
    ("tập đoàn", "TĐ"),
]

# Pragmatic, HAND-PICKED exceptions to the "brand/trading name not in legal
# name" limitation documented above - added case-by-case when a real query
# hits the gap (2026-08-05: "VINA Game" for VNZ/VNG Corporation - VNG was
# founded/known as "Vinagame"/"VNG" before its current legal name, which
# shares NO words with "Vina Game" so even tier-3 fuzzy matching can't
# bridge it). This is deliberately NOT the "hand-curate ~216 companies"
# effort the module docstring rules out - it's an opt-in list that grows
# only when a specific brand name actually gets asked about and fails,
# same philosophy as NON_TICKER_ABBREVIATIONS in query_builder.py (a
# blocklist grown from real failures, not a preemptive exhaustive one).
# Ticker does NOT need a `companies` table row for this to work - these
# candidates are injected into the lookup index directly (see
# _load_index()), since disclosure_chunks.ticker is populated by PDF
# ingestion independent of the companies seed list (see is_known_ticker()).
BRAND_NAME_ALIASES: dict[str, list[str]] = {
    "VNZ": ["Vina Game", "VinaGame", "Vinagame"],
}

_MIN_NAME_LEN = 6
_FUZZY_THRESHOLD = 0.82
# Every ALIGNED word pair (candidate word i vs matched window word i) must
# ALSO individually clear this floor - added 2026-07-24 after a real false
# positive found via live testing (see docs/PROGRESS.md): the question
# "Sắc thái chung của thị trường chứng khoán Việt Nam..." (no company
# mentioned at all) resolved to ticker AISEC ("CTCP Chứng khoán AIS")
# because the whole-phrase ratio of "Chứng khoán AIS" vs the window
# "chứng khoán Việt" was 0.839 - ABOVE the 0.82 threshold - purely because
# "chứng"/"khoán" (a common, generic Vietnamese finance phrase meaning
# "securities/stock market") matched perfectly, even though the actually
# DISTINGUISHING word ("AIS" vs "Việt") had a per-word ratio of only
# 0.286. A long shared generic prefix can inflate SequenceMatcher's
# aggregate ratio past threshold even when the one word that actually
# identifies the company is completely different - the per-word guard
# below blocks exactly this failure mode without touching the phrase-level
# threshold (which real typo cases like "Hòaa pháttt" vs "Hòa Phát" need
# to keep passing - per-word ratios there are 0.857/0.800, comfortably
# above this floor).
_FUZZY_MIN_WORD_RATIO = 0.55
_WORD_RE = re.compile(r"\S+")
_DIACRITIC_MAP = str.maketrans({"đ": "d", "Đ": "D"})

_COMPANIES_SQL = text("SELECT ticker, company_name FROM companies")

_cache: Optional[list] = None  # [(ticker, candidate_name), ...] - lazy, per-process; MAY have several rows per ticker.


def _strip_legal_form(name: str) -> str:
    stripped = name.strip()
    low = stripped.lower()
    for prefix in _LEADING_LEGAL_FORMS:
        if low.startswith(prefix):
            stripped = stripped[len(prefix):].strip()
            low = stripped.lower()
            break  # only one leading marker can ever apply
    if low.endswith(_TRAILING_SUFFIX):
        stripped = stripped[: -len(_TRAILING_SUFFIX)].strip()
    return stripped


def _name_candidates(raw_name: str) -> list:
    """All matchable name variants for one company - see module docstring
    step 1-2 for what these are and why."""
    base = _strip_legal_form(raw_name)
    candidates = {base}
    low = base.lower()
    for phrase, abbrev in _SECONDARY_FORMS:
        if low.startswith(phrase):
            core = base[len(phrase):].strip()
            if core:
                candidates.add(core)
                candidates.add(f"{abbrev} {core}")
            break  # only one secondary marker can ever apply
    return [c for c in candidates if len(c) >= _MIN_NAME_LEN]


def _fold_diacritics(s: str) -> str:
    """Strip Vietnamese diacritics/tone marks to a bare ASCII-ish form,
    e.g. "Hòa Phát" -> "Hoa Phat", "Đá" -> "Da". đ/Đ are handled via an
    explicit translation table first since Unicode NFD decomposition
    doesn't split that letter (it's its own letter, not base+combining
    mark, unlike e.g. "á" = "a" + combining acute)."""
    s = s.translate(_DIACRITIC_MAP)
    nfd = unicodedata.normalize("NFD", s)
    return "".join(c for c in nfd if unicodedata.category(c) != "Mn")


def _fuzzy_window_ratio(candidate: str, question_words: list) -> float:
    """Best difflib ratio of `candidate` against any contiguous
    word-window of `question_words` with the same word count as
    `candidate` - a fuzzy SUBSTRING search, not a whole-string comparison
    (which would unfairly penalize a short candidate against a long
    question even for an otherwise-perfect match). A window only counts if
    it ALSO passes the per-word guard (_FUZZY_MIN_WORD_RATIO - see that
    constant's comment for the real false-positive case this blocks)."""
    cand_words = candidate.split()
    n = len(cand_words)
    if len(question_words) < n:
        return 0.0
    cand_low = candidate.lower()
    best = 0.0
    for i in range(len(question_words) - n + 1):
        window_words = question_words[i:i + n]
        window = " ".join(window_words).lower()
        ratio = SequenceMatcher(None, cand_low, window).ratio()
        if ratio <= best:
            continue
        word_ratios = [
            SequenceMatcher(None, cw.lower(), ww.lower()).ratio()
            for cw, ww in zip(cand_words, window_words)
        ]
        if min(word_ratios) < _FUZZY_MIN_WORD_RATIO:
            continue
        best = ratio
    return best


def _load_index() -> list:
    global _cache
    if _cache is not None:
        return _cache
    with get_session() as session:
        rows = session.execute(_COMPANIES_SQL).fetchall()
    _cache = [(ticker, candidate) for ticker, name in rows if name for candidate in _name_candidates(name)]
    # Inject hand-picked brand-name aliases (BRAND_NAME_ALIASES above) -
    # these tickers may not even have a `companies` row (VNZ doesn't, as of
    # 2026-08-05), so they're added on top of the SQL-derived rows rather
    # than requiring one.
    _cache.extend(
        (ticker, alias) for ticker, aliases in BRAND_NAME_ALIASES.items() for alias in aliases
    )
    return _cache


def is_known_ticker(ticker: Optional[str]) -> bool:
    """True if `ticker` is a real ticker this system knows about - either a
    `companies` table row, or a BRAND_NAME_ALIASES key. Used by
    query_builder._extract_ticker() to reject a bare-regex match that LOOKS
    like a ticker (3-4 uppercase letters) but isn't one, e.g. "VINA" being
    caught out of "VINA Game" - see that function's 2026-08-05 fix for the
    real failure this addresses.

    Deliberately does NOT swallow DB errors here (unlike
    resolve_company_name()) - the caller needs to tell "verified not a
    ticker" apart from "couldn't verify, DB unreachable" so it can fail
    OPEN (trust the regex match) in the latter case rather than silently
    breaking ticker extraction on a transient DB hiccup."""
    if not ticker:
        return False
    known = {t for t, _ in _load_index()} | set(BRAND_NAME_ALIASES.keys())
    return ticker in known


def _best_ticker(matches: list) -> Optional[str]:
    """Shared tie-break for resolve_company_name(): per ticker, keep its
    longest matching candidate; across tickers, the longest-candidate
    ticker wins; a genuine tie in length between 2 DIFFERENT tickers
    returns None (ambiguous - don't guess)."""
    best_per_ticker = {}
    for ticker, name in matches:
        if ticker not in best_per_ticker or len(name) > len(best_per_ticker[ticker]):
            best_per_ticker[ticker] = name
    ranked = sorted(best_per_ticker.items(), key=lambda tn: len(tn[1]), reverse=True)
    if len(ranked) > 1 and len(ranked[0][1]) == len(ranked[1][1]):
        return None
    return ranked[0][0]


def resolve_company_name(question: str) -> Optional[str]:
    """Best-effort single-ticker resolution for DAG/RAG/WHATIF (all of
    which need exactly one company). Tries EXACT -> DIACRITIC-FOLDED ->
    FUZZY, stopping at the first tier that finds anything (see module
    docstring) - a later, weaker tier never overrides an earlier, more
    reliable one. Returns None on no match at any tier, or a length-tied
    ambiguous match within the tier that fired."""
    index = _load_index()
    low = question.lower()

    exact = [(t, n) for t, n in index if n.lower() in low]
    if exact:
        return _best_ticker(exact)

    folded_q = _fold_diacritics(low)
    folded = [(t, n) for t, n in index if _fold_diacritics(n.lower()) in folded_q]
    if folded:
        return _best_ticker(folded)

    q_words = _WORD_RE.findall(question)
    fuzzy = [(t, n) for t, n in index if _fuzzy_window_ratio(n, q_words) >= _FUZZY_THRESHOLD]
    if not fuzzy:
        return None
    return _best_ticker(fuzzy)


def resolve_all_company_names(question: str) -> list:
    """Plural version for the COMPARE branch (rules._extract_all_tickers()),
    which needs 2+ distinct tickers. Unlike resolve_company_name(), doesn't
    stop at the first tier: exact + diacritic-folded matches are ALWAYS
    both collected (both deterministic, low false-positive risk), and the
    fuzzy tier is only added on top if still short of 2 tickers found -
    COMPARE's risk profile tolerates an extra match much better than a
    single-answer lookup elsewhere (it just adds one more line to a
    comparison, not a wrong number)."""
    index = _load_index()
    low = question.lower()
    seen = []

    for ticker, name in index:
        if ticker not in seen and name.lower() in low:
            seen.append(ticker)

    folded_q = _fold_diacritics(low)
    for ticker, name in index:
        if ticker not in seen and _fold_diacritics(name.lower()) in folded_q:
            seen.append(ticker)

    if len(seen) < 2:
        q_words = _WORD_RE.findall(question)
        for ticker, name in index:
            if ticker not in seen and _fuzzy_window_ratio(name, q_words) >= _FUZZY_THRESHOLD:
                seen.append(ticker)

    return seen


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "Tổng tài sản của Tập đoàn Hòa Phát năm 2025 là bao nhiêu?"
    print(f"Q: {q}")
    print("resolve_company_name ->", resolve_company_name(q))
    print("resolve_all_company_names ->", resolve_all_company_names(q))
