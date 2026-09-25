"""Bounded schema description for the DAG (structured lookup) path.

Grounded in the Text-to-SQL literature reviewed for the thesis, the DAG is
deliberately NOT a general-purpose NL2SQL system - it only resolves a
closed set of (company, period, indicator) lookups against the CORE_MS_CODES
defined in src/core/constants.py. This module is the single source of
truth for what's "in scope": which indicator names/aliases resolve to
which MS code.
"""
import re
import unicodedata

from src.core.constants import CORE_MS_CODES

# alias (normalized) -> (statement_type, ms_code)
INDICATOR_ALIASES: dict[str, tuple[str, str]] = {}


def _normalize(s: str) -> str:
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = re.sub(r"[^a-z0-9 ]", " ", s.lower())
    return re.sub(r"\s+", " ", s).strip()


def _register(alias: str, statement_type: str, ms_code: str):
    INDICATOR_ALIASES.setdefault(_normalize(alias), (statement_type, ms_code))


# 1) auto-register the official line item names from CORE_MS_CODES
for _stype, _codes in CORE_MS_CODES.items():
    for _ms_code, _name in _codes.items():
        _register(_name, _stype, _ms_code)

# 2) common short aliases people would actually type in a question - these
# take priority via setdefault ordering below (registered after official
# names so official names still win on exact collision, but most of these
# don't collide since phrasing differs from the formal line item text).
_COMMON_ALIASES = [
    ("tong tai san", "balance_sheet", "280"),
    ("tong cong tai san", "balance_sheet", "280"),
    ("tong nguon von", "balance_sheet", "440"),
    ("tai san ngan han", "balance_sheet", "100"),
    ("tai san dai han", "balance_sheet", "200"),
    ("tien mat", "balance_sheet", "110"),
    ("tien va tuong duong tien", "balance_sheet", "110"),
    ("hang ton kho", "balance_sheet", "140"),
    ("no phai tra", "balance_sheet", "300"),
    ("tong no", "balance_sheet", "300"),
    ("no ngan han", "balance_sheet", "310"),
    ("no dai han", "balance_sheet", "330"),
    ("von chu so huu", "balance_sheet", "400"),
    ("doanh thu", "income_statement", "10"),
    ("doanh thu thuan", "income_statement", "10"),
    ("gia von hang ban", "income_statement", "11"),
    ("loi nhuan gop", "income_statement", "20"),
    ("loi nhuan truoc thue", "income_statement", "50"),
    ("loi nhuan sau thue", "income_statement", "60"),
    ("lai co ban tren co phieu", "income_statement", "70"),
    ("eps", "income_statement", "70"),
    ("dong tien kinh doanh", "cash_flow_indirect", "20"),
    ("luu chuyen tien thuan tu hoat dong kinh doanh", "cash_flow_indirect", "20"),
    ("luu chuyen tien thuan tu hoat dong dau tu", "cash_flow_indirect", "30"),
    ("luu chuyen tien thuan tu hoat dong tai chinh", "cash_flow_indirect", "40"),
    ("tien cuoi ky", "cash_flow_indirect", "70"),
]
for _alias, _stype, _code in _COMMON_ALIASES:
    _register(_alias, _stype, _code)

# 3) English aliases - same bounded set, just phrased for English questions
# ("What is the total assets of CTD in 2024?"). Kept deliberately as full
# phrases (not single short words like "cash" or "assets" alone) to avoid
# ambiguous substring matches in resolve_indicator's fallback.
_ENGLISH_ALIASES = [
    ("total assets", "balance_sheet", "280"),
    ("total capital", "balance_sheet", "440"),
    ("total liabilities and equity", "balance_sheet", "440"),
    ("current assets", "balance_sheet", "100"),
    ("non current assets", "balance_sheet", "200"),
    ("long term assets", "balance_sheet", "200"),
    ("cash and cash equivalents", "balance_sheet", "110"),
    ("inventory", "balance_sheet", "140"),
    ("inventories", "balance_sheet", "140"),
    ("total liabilities", "balance_sheet", "300"),
    ("current liabilities", "balance_sheet", "310"),
    ("long term liabilities", "balance_sheet", "330"),
    ("non current liabilities", "balance_sheet", "330"),
    ("owners equity", "balance_sheet", "400"),
    ("shareholders equity", "balance_sheet", "400"),
    ("total equity", "balance_sheet", "400"),
    ("net revenue", "income_statement", "10"),
    ("net sales", "income_statement", "10"),
    ("revenue", "income_statement", "10"),
    ("cost of goods sold", "income_statement", "11"),
    ("cogs", "income_statement", "11"),
    ("gross profit", "income_statement", "20"),
    ("profit before tax", "income_statement", "50"),
    ("pretax profit", "income_statement", "50"),
    ("net profit", "income_statement", "60"),
    ("net income", "income_statement", "60"),
    ("profit after tax", "income_statement", "60"),
    ("earnings per share", "income_statement", "70"),
    ("operating cash flow", "cash_flow_indirect", "20"),
    ("cash flow from operating activities", "cash_flow_indirect", "20"),
    ("investing cash flow", "cash_flow_indirect", "30"),
    ("cash flow from investing activities", "cash_flow_indirect", "30"),
    ("financing cash flow", "cash_flow_indirect", "40"),
    ("cash flow from financing activities", "cash_flow_indirect", "40"),
    ("cash at end of period", "cash_flow_indirect", "70"),
    ("ending cash balance", "cash_flow_indirect", "70"),
]
for _alias, _stype, _code in _ENGLISH_ALIASES:
    _register(_alias, _stype, _code)

# 4) 2026-09-15 - Quang pasted the FULL standard Vietnamese BCTC chart of
# accounts and pointed out (correctly, after auditing) that only ~20 of
# CORE_MS_CODES's 45 line items had a natural VI/EN alias beyond step 1's
# auto-registered OFFICIAL name (which still carries its Roman-numeral/
# letter prefix, e.g. "ii au tu tai chinh ngan han" for code 120 - a string
# nobody would type in a real question). The other 25 lines were only
# reachable by typing that exact unwieldy official string verbatim -
# effectively unmatchable. This section closes that gap: one natural VI
# alias (official name MINUS the Roman-numeral/letter prefix, since that
# prefix is the only thing making the auto-registered name unnatural) +
# common VI/EN variants, per unaliased code. Full audit + reasoning logged
# in docs/PROGRESS.md's 2026-09-15 entry.
#
# IMPORTANT - written in NATIVE VIETNAMESE WITH DIACRITICS ON PURPOSE
# (unlike section 2 above, which was hand-typed as pre-stripped ASCII by
# convention): `_register()` calls `_normalize()` on whatever string it
# receives, and `resolve_indicator()` calls that SAME `_normalize()` on the
# incoming question - so registering the raw diacritic phrase here and
# letting `_register` do the normalizing guarantees this alias's key is
# whatever a real diacritic-typed question containing the same phrase would
# ALSO normalize to, even where `_normalize()`'s own known bug (NFD does
# NOT decompose "đ" the way it does "ư"/"ơ"/"ô", so "đ" gets silently
# stripped instead of becoming "d" - see id=3 in
# scripts/final_benchmark_100_en.py and its docstring for the original
# discovery) changes what the "expected" ASCII spelling would be. Example:
# "Tiền và tương đương tiền đầu kỳ" normalizes to "tien va tuong uong tien
# au ky" (đ silently dropped, NOT "duong"/"dau") - hand-typing a "corrected"
# ASCII guess here (like section 2's pre-existing "tien va tuong duong
# tien", which is why THAT alias never actually matches a real diacritic
# question) would silently break the very alias being added. Passing the
# native phrase sidesteps that trap entirely rather than repeating it.
_UNALIASED_MS_CODE_ALIASES_VI = [
    # -- balance_sheet (11 of the 25) --
    ("Đầu tư tài chính ngắn hạn", "balance_sheet", "120"),
    ("Các khoản phải thu ngắn hạn", "balance_sheet", "130"),
    ("Phải thu ngắn hạn", "balance_sheet", "130"),
    ("Tài sản sinh học ngắn hạn", "balance_sheet", "150"),
    ("Tài sản ngắn hạn khác", "balance_sheet", "160"),
    ("Các khoản phải thu dài hạn", "balance_sheet", "210"),
    ("Phải thu dài hạn", "balance_sheet", "210"),
    ("Tài sản cố định", "balance_sheet", "220"),
    ("Tài sản sinh học dài hạn", "balance_sheet", "230"),
    ("Bất động sản đầu tư", "balance_sheet", "240"),
    ("Tài sản dở dang dài hạn", "balance_sheet", "250"),
    ("Đầu tư tài chính dài hạn", "balance_sheet", "260"),
    ("Tài sản dài hạn khác", "balance_sheet", "270"),
    # -- income_statement (11 of the 25; "01" deliberately skipped - see
    # _ENGLISH_ALIASES-only note below) --
    ("Doanh thu hoạt động tài chính", "income_statement", "22"),
    ("Doanh thu tài chính", "income_statement", "22"),
    ("Chi phí tài chính", "income_statement", "23"),
    ("Chi phí bán hàng", "income_statement", "25"),
    ("Chi phí quản lý doanh nghiệp", "income_statement", "26"),
    ("Chi phí quản lý", "income_statement", "26"),
    ("Lợi nhuận thuần từ hoạt động kinh doanh", "income_statement", "30"),
    ("Thu nhập khác", "income_statement", "31"),
    ("Chi phí khác", "income_statement", "32"),
    ("Lợi nhuận khác", "income_statement", "40"),
    ("Chi phí thuế TNDN hiện hành", "income_statement", "51"),
    ("Chi phí thuế TNDN hoãn lại", "income_statement", "52"),
    ("Lãi suy giảm trên cổ phiếu", "income_statement", "71"),
    # -- cash_flow_indirect (2 of the 25) --
    ("Lưu chuyển tiền thuần trong kỳ", "cash_flow_indirect", "50"),
    ("Tiền đầu kỳ", "cash_flow_indirect", "60"),  # short form only - see
    # the docstring above for why "tương đương" is deliberately avoided
    # here (its "đ" is silently dropped by _normalize(), same bug as the
    # official name already auto-registered in step 1 for this exact code -
    # this short alias sidesteps that word entirely instead of relying on
    # it).
]
for _alias, _stype, _code in _UNALIASED_MS_CODE_ALIASES_VI:
    _register(_alias, _stype, _code)

# English counterparts for the same 25-code gap, same style as section 3.
# "01" (gross revenue, code income_statement/01) deliberately gets ONLY an
# English alias, not a Vietnamese one: its official name "Doanh thu bán
# hàng và cung cấp dịch vụ" differs from code "10"'s "Doanh thu THUẦN về
# bán hàng và cung cấp dịch vụ" by exactly the word "thuần" - a natural
# short VI alias like "doanh thu bán hàng" would be genuinely ambiguous in
# everyday usage (many speakers say this loosely to mean NET revenue, i.e.
# code 10), so adding it risks silently mis-resolving a real question to
# the wrong line. "gross revenue"/"gross sales" in English carry no such
# ambiguity with "net revenue"/"net sales" (already registered for code 10
# in section 3), so only the English side is added here.
_UNALIASED_MS_CODE_ALIASES_EN = [
    ("gross revenue", "income_statement", "01"),
    ("gross sales", "income_statement", "01"),
    ("short term financial investments", "balance_sheet", "120"),
    ("short term investments", "balance_sheet", "120"),
    ("short term receivables", "balance_sheet", "130"),
    ("short term biological assets", "balance_sheet", "150"),
    ("other short term assets", "balance_sheet", "160"),
    ("other current assets", "balance_sheet", "160"),
    ("long term receivables", "balance_sheet", "210"),
    ("fixed assets", "balance_sheet", "220"),
    ("long term biological assets", "balance_sheet", "230"),
    ("investment property", "balance_sheet", "240"),
    ("long term assets in progress", "balance_sheet", "250"),
    ("long term financial investments", "balance_sheet", "260"),
    ("other long term assets", "balance_sheet", "270"),
    ("other non current assets", "balance_sheet", "270"),
    ("financial income", "income_statement", "22"),
    ("finance income", "income_statement", "22"),
    ("financial expense", "income_statement", "23"),
    ("finance costs", "income_statement", "23"),
    ("finance expenses", "income_statement", "23"),
    ("selling expenses", "income_statement", "25"),
    ("general and administrative expenses", "income_statement", "26"),
    ("administrative expenses", "income_statement", "26"),
    ("net operating profit", "income_statement", "30"),
    ("other income", "income_statement", "31"),
    ("other expenses", "income_statement", "32"),
    ("other profit", "income_statement", "40"),
    ("current corporate income tax expense", "income_statement", "51"),
    ("current income tax expense", "income_statement", "51"),
    ("deferred corporate income tax expense", "income_statement", "52"),
    ("deferred income tax expense", "income_statement", "52"),
    ("diluted earnings per share", "income_statement", "71"),
    ("diluted eps", "income_statement", "71"),
    ("net cash flow during the period", "cash_flow_indirect", "50"),
    ("net increase in cash", "cash_flow_indirect", "50"),
    ("cash at beginning of period", "cash_flow_indirect", "60"),
    ("beginning cash balance", "cash_flow_indirect", "60"),
]
for _alias, _stype, _code in _UNALIASED_MS_CODE_ALIASES_EN:
    _register(_alias, _stype, _code)


def resolve_indicator(text: str):
    """Return (statement_type, ms_code, matched_alias) for the best match
    against the bounded alias table, or None if nothing matches. Exact
    normalized match first, then longest-substring containment as a
    fallback - still bounded to INDICATOR_ALIASES, never falls through to
    free-form SQL generation."""
    norm = _normalize(text)
    # 2026-08-06: khai triển viết tắt tài chính VN (LNST, VCSH, TTS...)
    # TRƯỚC khi khớp alias - xem src/structured/abbreviations.py cho nguyên
    # tắc chọn từ (>=3 ký tự, word-boundary, map về alias đã đăng ký).
    from src.structured.abbreviations import expand
    norm = expand(norm)
    if norm in INDICATOR_ALIASES:
        stype, code = INDICATOR_ALIASES[norm]
        return stype, code, norm
    best = None
    for alias, (stype, code) in INDICATOR_ALIASES.items():
        if alias and alias in norm and (best is None or len(alias) > len(best[2])):
            best = (stype, code, alias)
    return best
