"""Parameterized lookup against financial_ratios - the RATIO_LOOKUP path's
execution half (src/structured/ratio_schema.py is the classification half).
Mirrors src/structured/query_builder.py's ParsedQuery/execute() for
financial_line_items exactly, with two differences: (1) resolves against
ratio_schema.resolve_ratio() instead of schema.resolve_indicator(), (2)
queries financial_ratios (ticker+fiscal_year+period+ratio_name -> value),
which has no `ms_code`/`statement_type`/`consolidation` columns - it's a
flatter table than financial_line_items (see infra/postgres/migrations/
001_init_schema.sql's CREATE TABLE financial_ratios).
"""
import re
from dataclasses import dataclass
from typing import Optional

from sqlalchemy import text

from src.core.db import get_session
from src.structured.query_builder import (
    HALF_RE,
    NINE_MONTH_RE,
    QUARTER_RE_EN,
    QUARTER_RE_VI,
    YEAR_PERIOD_RE,
    YEAR_RE,
    _extract_ticker,
)
from src.structured.ratio_schema import resolve_ratio


@dataclass
class ParsedRatioQuery:
    ticker: Optional[str]
    fiscal_year: Optional[int]
    period: Optional[str]
    ratio_group: Optional[str]
    ratio_name: Optional[str]
    matched_alias: Optional[str]

    def missing_fields(self) -> list[str]:
        return [f for f in ("ticker", "fiscal_year", "period", "ratio_name") if getattr(self, f) is None]


def parse_ratio_question(question: str) -> ParsedRatioQuery:
    """Same ticker/period extraction as query_builder.parse_question() -
    deliberately reused, not reimplemented, so a ratio question and a
    line-item question resolve periods identically (same regexes, same
    YEAR/quarter/half/9-month/full-year precedence)."""
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
        period = "YEAR"
    else:
        period = None

    resolved = resolve_ratio(question)
    group, name, alias = resolved if resolved else (None, None, None)

    return ParsedRatioQuery(
        ticker=ticker,
        fiscal_year=int(year_match.group(1)) if year_match else None,
        period=period,
        ratio_group=group,
        ratio_name=name,
        matched_alias=alias,
    )


LOOKUP_RATIO = text("""
    SELECT value, unit, ratio_group
    FROM financial_ratios
    WHERE ticker = :ticker AND fiscal_year = :fiscal_year AND period = :period
      AND ratio_name = :ratio_name
    LIMIT 1
""")


def execute(parsed: ParsedRatioQuery) -> dict:
    missing = parsed.missing_fields()
    if missing:
        return {"ok": False, "error": f"Missing required fields: {missing}"}

    with get_session() as session:
        row = session.execute(LOOKUP_RATIO, {
            "ticker": parsed.ticker,
            "fiscal_year": parsed.fiscal_year,
            "period": parsed.period,
            "ratio_name": parsed.ratio_name,
        }).first()

    if row is None:
        return {"ok": False, "error": "No matching row in financial_ratios (chưa nạp dữ liệu cho kỳ/mã này)"}

    value, unit, ratio_group = row
    return {
        "ok": True,
        "value": value,
        "unit": unit,
        "ratio_group": ratio_group,
        "ratio_name": parsed.ratio_name,
        "ticker": parsed.ticker,
        "fiscal_year": parsed.fiscal_year,
        "period": parsed.period,
    }
