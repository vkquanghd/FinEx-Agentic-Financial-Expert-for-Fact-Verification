"""Ticker-scoped Vietstock structured-table ingestion.

This module collects rendered HTML tables from the company's Trading and
Financial pages. It deliberately stores the observed table shape and values
before attempting to map them into the thesis's narrower accounting schema.
That is important because Vietstock uses different statement layouts for
banks, insurers, securities firms, and ordinary operating companies.

The crawler uses the account owner's already-authenticated browser session;
it does not accept credentials or bypass Premium access controls. Charts are
not collected in this first pass. PDF/OCR chunking is intentionally separate
and can run later from the existing document corpus.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
from sqlalchemy import text

from src.core.db import get_session
from src.ingestion.vietstock_premium import (
    CompanyLink,
    browser_session,
    known_company_links,
    known_companies,
    safe_error,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FINANCIAL_SUBSECTION_WORDS = (
    "bctc", "báo cáo", "bao cao", "tình hình tài chính", "tinh hinh tai chinh",
    "kết quả kinh doanh", "ket qua kinh doanh", "lưu chuyển tiền tệ",
    "luu chuyen tien te", "thuyết minh", "thuyet minh", "chỉ số tài chính",
    "chi so tai chinh", "chỉ tiêu kế hoạch", "chi tieu ke hoach",
)
CHART_WORDS = ("biểu đồ", "bieu do", "chart", "graph")
SECTION_ROUTES = {
    "trading": "thong-ke-giao-dich.htm",
    "financial": "tai-chinh.htm",
}


@dataclass(frozen=True)
class StructuredTable:
    ticker: str
    section: str
    subsection: str
    source_url: str
    table_index: int
    title: str
    headers: list[str]
    rows: list[list[str]]
    table_hash: str

    def as_json(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "section": self.section,
            "subsection": self.subsection,
            "source_url": self.source_url,
            "table_index": self.table_index,
            "title": self.title,
            "headers": self.headers,
            "rows": self.rows,
            "table_hash": self.table_hash,
        }


def _normalise(value: str) -> str:
    value = unicodedata.normalize("NFKC", value or "")
    return re.sub(r"\s+", " ", value).strip()


def _ascii(value: str) -> str:
    value = unicodedata.normalize("NFKD", value or "")
    return "".join(ch for ch in value if not unicodedata.combining(ch)).lower()


def structured_tab_url(ticker: str, section: str) -> str:
    ticker = ticker.strip().upper()
    try:
        route = SECTION_ROUTES[section]
    except KeyError as exc:
        raise ValueError(f"Unsupported structured section: {section}") from exc
    return f"https://finance.vietstock.vn/{ticker}/{route}"


def _is_chart_label(value: str) -> bool:
    folded = _ascii(value)
    return any(word in folded for word in CHART_WORDS)


def _is_ticker_scoped_url(url: str, ticker: str) -> bool:
    parsed = urlparse(url)
    return parsed.netloc.lower() == "finance.vietstock.vn" and parsed.path.lower().startswith(
        f"/{ticker.lower()}/"
    )


def extract_financial_subsection_links(html: str, base_url: str, ticker: str) -> list[dict[str, str]]:
    """Return ticker-scoped financial sub-tabs and exclude chart links."""
    soup = BeautifulSoup(html, "html.parser")
    links: dict[str, dict[str, str]] = {}
    for anchor in soup.select("a[href]"):
        label = _normalise(anchor.get_text(" ", strip=True))
        href = str(anchor.get("href") or "").strip()
        if not href or href.startswith(("javascript:", "#")):
            continue
        absolute = urljoin(base_url, href)
        if not _is_ticker_scoped_url(absolute, ticker) or _is_chart_label(label):
            continue
        blob = _ascii(label)
        if any(_ascii(word) in blob for word in FINANCIAL_SUBSECTION_WORDS):
            links.setdefault(absolute, {"url": absolute, "title": label or "financial"})
    return list(links.values())


def _visible_table(table) -> bool:
    classes = " ".join(table.get("class") or []).lower()
    style = str(table.get("style") or "").replace(" ", "").lower()
    return "hidden" not in classes and "display:none" not in style and "visibility:hidden" not in style


def _table_title(table) -> str:
    caption = table.find("caption")
    if caption:
        return _normalise(caption.get_text(" ", strip=True))
    previous = table.find_previous(["h1", "h2", "h3", "h4", "h5", "h6", "legend"])
    return _normalise(previous.get_text(" ", strip=True)) if previous else ""


def parse_structured_tables(html: str, *, ticker: str, section: str,
                            subsection: str, source_url: str) -> list[StructuredTable]:
    """Parse non-empty rendered HTML tables without inventing column meaning."""
    soup = BeautifulSoup(html, "html.parser")
    tables: list[StructuredTable] = []
    for index, table in enumerate(soup.select("table")):
        if not _visible_table(table):
            continue
        raw_rows: list[list[str]] = []
        for tr in table.select("tr"):
            cells = tr.find_all(["th", "td"])
            values = [_normalise(cell.get_text(" ", strip=True)) for cell in cells]
            if values and any(values):
                raw_rows.append(values)
        if len(raw_rows) < 2:
            continue
        header_row = next((row for row in raw_rows if any("năm" in _ascii(cell) or "year" in _ascii(cell) for cell in row)), raw_rows[0])
        body_start = raw_rows.index(header_row) + 1
        body = raw_rows[body_start:]
        if not body:
            continue
        canonical = json.dumps({"title": _table_title(table), "headers": header_row, "rows": body}, ensure_ascii=False, sort_keys=True)
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        tables.append(StructuredTable(
            ticker=ticker.upper(), section=section, subsection=subsection,
            source_url=source_url, table_index=index, title=_table_title(table),
            headers=header_row, rows=body, table_hash=digest,
        ))
    return tables


CREATE_RUN = text("""
    INSERT INTO vietstock_structured_runs (scope, status, config)
    VALUES (:scope, 'running', CAST(:config AS jsonb))
    RETURNING id
""")
FINISH_RUN = text("""
    UPDATE vietstock_structured_runs
    SET status = :status, finished_at = now(), counters = CAST(:counters AS jsonb), error_summary = :error_summary
    WHERE id = :run_id
""")
UPSERT_TABLE = text("""
    INSERT INTO vietstock_structured_tables
        (ticker, section, subsection, source_url, table_index, table_title,
         headers, rows, table_hash, http_status, fetched_at)
    VALUES
        (:ticker, :section, :subsection, :source_url, :table_index, :table_title,
         CAST(:headers AS jsonb), CAST(:rows AS jsonb), :table_hash, :http_status, now())
    ON CONFLICT (ticker, source_url, table_index, table_hash) DO UPDATE SET
        subsection = EXCLUDED.subsection, table_title = EXCLUDED.table_title,
        headers = EXCLUDED.headers, rows = EXCLUDED.rows,
        http_status = EXCLUDED.http_status, fetched_at = now()
""")


def _new_run(config: dict[str, Any], scope: str = "structured_sync") -> int:
    with get_session() as session:
        return int(session.execute(CREATE_RUN, {"scope": scope, "config": json.dumps(config)}).scalar_one())


def _finish_run(run_id: int, status: str, counters: dict[str, int], error: str | None = None) -> None:
    with get_session() as session:
        session.execute(FINISH_RUN, {
            "run_id": run_id, "status": status, "counters": json.dumps(counters), "error_summary": error,
        })


def _persist_tables(tables: Iterable[StructuredTable], http_status: int | None) -> int:
    rows = list(tables)
    if not rows:
        return 0
    with get_session() as session:
        ticker_keys = {}
        for ticker in {table.ticker for table in rows}:
            matches = session.execute(text(
                "SELECT ticker FROM companies WHERE upper(trim(ticker)) = :ticker"
            ), {"ticker": ticker.strip().upper()}).scalars().all()
            if len(matches) != 1:
                raise ValueError(f"Expected one company key for {ticker}; found {len(matches)}")
            ticker_keys[ticker] = matches[0]
        for table in rows:
            session.execute(UPSERT_TABLE, {
                "ticker": ticker_keys[table.ticker], "section": table.section, "subsection": table.subsection,
                "source_url": table.source_url, "table_index": table.table_index,
                "table_title": table.title, "headers": json.dumps(table.headers, ensure_ascii=False),
                "rows": json.dumps(table.rows, ensure_ascii=False), "table_hash": table.table_hash,
                "http_status": http_status,
            })
    return len(rows)


def sync_structured(companies: Iterable[CompanyLink], *, sections: tuple[str, ...] = ("trading", "financial"),
                    headless: bool = True, cdp_url: str | None = None,
                    delay_seconds: float = 1.5, max_subsections: int = 20,
                    scope: str = "structured_sync") -> dict[str, int]:
    """Collect structured tables for selected companies, one browser page at a time."""
    companies = list(companies)
    if not sections:
        raise ValueError("At least one structured section is required")
    if scope not in {"structured_sync", "trading_daily"}:
        raise ValueError(f"Unsupported structured run scope: {scope}")
    run_id = _new_run(
        {"tickers": len(companies), "sections": sections, "delay_seconds": delay_seconds, "scope": scope},
        scope=scope,
    )
    counters = {"requested": len(companies), "completed": 0, "partial": 0, "failed": 0,
                "pages": 0, "subsections": 0, "tables": 0}
    try:
        with browser_session(
            headless=headless, cdp_url=cdp_url, cdp_no_defaults=True,
        ) as context:
            page = context.pages[0] if context.pages else context.new_page()
            for position, company in enumerate(companies, start=1):
                try:
                    company_tables = 0
                    company_subsections = 0
                    for section in sections:
                        base_url = structured_tab_url(company.ticker, section)
                        response = page.goto(base_url, wait_until="domcontentloaded", timeout=60_000)
                        page.wait_for_timeout(900)
                        if not response or response.status >= 400:
                            continue
                        counters["pages"] += 1
                        subsection_links = extract_financial_subsection_links(page.content(), base_url, company.ticker) if section == "financial" else []
                        targets = [{"url": base_url, "title": section}] + subsection_links[:max_subsections]
                        seen: set[str] = set()
                        for target in targets:
                            if target["url"] in seen:
                                continue
                            seen.add(target["url"])
                            if target["url"] != page.url:
                                page.goto(target["url"], wait_until="domcontentloaded", timeout=60_000)
                                page.wait_for_timeout(700)
                            tables = parse_structured_tables(
                                page.content(), ticker=company.ticker, section=section,
                                subsection=target["title"], source_url=target["url"],
                            )
                            company_tables += _persist_tables(tables, response.status)
                            company_subsections += 1
                    counters["tables"] += company_tables
                    counters["subsections"] += company_subsections
                    counters["completed"] += 1 if company_subsections else 0
                    counters["partial"] += 0 if company_subsections else 1
                    print(f"[{position}/{len(companies)}] {company.ticker}: {company_tables} tables from {company_subsections} pages", flush=True)
                except Exception as exc:
                    counters["failed"] += 1
                    print(f"[{position}/{len(companies)}] {company.ticker}: failed ({safe_error(exc)})", flush=True)
                if position < len(companies):
                    time.sleep(max(delay_seconds, 0))
        status = "completed" if counters["failed"] == 0 and counters["partial"] == 0 else "partial"
        _finish_run(run_id, status, counters)
        return counters
    except BaseException as exc:
        _finish_run(run_id, "failed", counters, safe_error(exc))
        raise


def selected_companies(tickers: str | None, all_known: bool) -> list[CompanyLink]:
    if all_known:
        return known_company_links()
    known = known_companies()
    wanted = {item.strip().upper() for item in (tickers or "").split(",") if item.strip()}
    unknown = wanted - set(known)
    if unknown:
        raise ValueError(f"Unknown ticker(s) in companies table: {', '.join(sorted(unknown))}")
    return [CompanyLink(ticker=ticker, name=known[ticker], url="") for ticker in sorted(wanted)]
