"""Authenticated, host-side Vietstock Finance synchronisation.

This module deliberately automates only pages and files that the account
owner can already access in a normal logged-in browser. It does not accept a
password, solve a CAPTCHA, replay a token outside the browser profile, or
try to bypass a Premium entitlement check.

The first ``login`` run opens Chromium with a persistent profile under
``data/private/vietstock_profile``. The owner signs in directly. Later runs
reuse that local browser profile and write only data observations and audit
metadata to Postgres. Browser state is never written to the database, logs,
or source control.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
import unicodedata
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urljoin, urlparse

from sqlalchemy import text

from src.core.db import get_session

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROFILE_DIR = PROJECT_ROOT / "data" / "private" / "vietstock_profile"
SNAPSHOT_DIR = PROJECT_ROOT / "data" / "raw" / "vietstock_premium" / "snapshots"
PDF_DIR = PROJECT_ROOT / "data" / "raw" / "pdf_disclosures" / "vietstock"
DIRECTORY_URL = "https://finance.vietstock.vn/doanh-nghiep-a-z?page={page}"

DOCUMENT_WORDS = ("tài liệu", "tai lieu", "document", "báo cáo", "bao cao", "report", "cbtt")
DOCUMENT_ROW_WORDS = DOCUMENT_WORDS + (
    "điều lệ", "dieu le", "annual", "financial", "governance", "đại hội", "dai hoi", "dhdcd",
)
NEWS_WORDS = ("tin tức", "tin tuc", "news", "sự kiện", "su kien", "event")
PDF_MAGIC = b"%PDF-"


class PlaywrightUnavailable(RuntimeError):
    """Raised with a concrete install command instead of a vague ImportError."""


@dataclass(frozen=True)
class CompanyLink:
    ticker: str
    name: str
    url: str


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _normalise_space(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def _safe_slug(value: str, fallback: str = "document") -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return value[:140] or fallback


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def safe_error(exc: BaseException) -> str:
    """Return a bounded operational error safe to persist or display.

    Playwright's detailed call log can include request headers. Persisting the
    raw exception would therefore risk retaining an authenticated session
    cookie in Postgres. Keep only the exception type and the message before
    that diagnostic block.
    """
    message = str(exc).split("Call log:", 1)[0].strip()
    return f"{type(exc).__name__}: {message[:400]}"


def _require_playwright():
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise PlaywrightUnavailable(
            "Playwright is not installed. Run `python -m pip install -r requirements.txt` "
            "then `python -m playwright install chromium` from the project virtual environment."
        ) from exc
    return sync_playwright


def extract_company_links(html: str, base_url: str,
                          valid_tickers: set[str] | None = None) -> list[CompanyLink]:
    """Extract unique ticker/company URLs from rendered directory HTML.

    The directory layout has changed over time, so this intentionally reads
    ordinary company detail links rather than depending on a brittle CSS
    class. It accepts URLs ending ``.htm`` with a leading ticker and ignores
    navigation/marketing links. Unit tests use this parser without network or
    browser dependencies.
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError as exc:  # pragma: no cover - BeautifulSoup is present today
        raise RuntimeError("BeautifulSoup4 is required to parse rendered Vietstock pages.") from exc

    soup = BeautifulSoup(html, "html.parser")
    found: dict[str, CompanyLink] = {}
    for anchor in soup.select("a[href]"):
        href = str(anchor.get("href") or "").strip()
        absolute = urljoin(base_url, href)
        path = urlparse(absolute).path
        match = re.match(r"/([A-Za-z0-9]{2,10})-[^/]+\.htm$", path, flags=re.I)
        if not match:
            continue
        ticker = match.group(1).upper()
        if valid_tickers is not None and ticker not in valid_tickers:
            continue
        if ticker in {"TIN", "HOME", "NEWS"}:
            continue
        label = _normalise_space(anchor.get_text(" ", strip=True))
        found.setdefault(ticker, CompanyLink(ticker=ticker, name=label or ticker, url=absolute))
    return list(found.values())


def company_detail_url(ticker: str, company_name: str) -> str:
    """Build the ordinary Vietstock detail URL from the project's company
    metadata, only for a user-requested smoke-test fallback.

    The resulting URL is not used to bulk-discover companies. It lets a known
    ticker such as VNM be tested even when directory pagination has not yet
    reached its alphabetical page.
    """
    normalized = unicodedata.normalize("NFKD", company_name)
    ascii_name = "".join(char for char in normalized if not unicodedata.combining(char))
    ascii_name = ascii_name.replace("đ", "d").replace("Đ", "D").lower()
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_name).strip("-")
    return f"https://finance.vietstock.vn/{ticker.upper()}-{slug}.htm"


def known_companies() -> dict[str, str]:
    """Return only project-approved tickers for directory validation."""
    with get_session() as session:
        rows = session.execute(text("SELECT ticker, company_name FROM companies")).all()
    return {str(row[0]).upper(): str(row[1]) for row in rows}


def known_company_links() -> list[CompanyLink]:
    """Project-approved bulk universe, independent of a fragile directory
    pager. URLs remain individually auditable because failed/redirected pages
    are recorded per ticker by ``sync_companies``."""
    return [
        CompanyLink(ticker=ticker, name=name, url=company_detail_url(ticker, name))
        for ticker, name in sorted(known_companies().items())
    ]


def company_tab_url(ticker: str, tab: str) -> str:
    """Return a deterministic, ticker-scoped Vietstock tab URL.

    Do not discover these URLs by scanning navigation. Vietstock embeds
    global category links in company pages, which are not company evidence.
    """
    ticker = ticker.strip().upper()
    routes = {
        "documents": "tai-tai-lieu.htm",
        "news": "tin-tuc-su-kien.htm",
    }
    try:
        return f"https://finance.vietstock.vn/{ticker}/{routes[tab]}"
    except KeyError as exc:
        raise ValueError(f"Unsupported company tab: {tab}") from exc


def is_company_tab_url(url: str, ticker: str, tab: str) -> bool:
    parsed = urlparse(url)
    expected_path = urlparse(company_tab_url(ticker, tab)).path.lower()
    return (
        parsed.netloc.lower() == "finance.vietstock.vn"
        and parsed.path.lower() == expected_path
    )


def _soup(html: str):
    try:
        from bs4 import BeautifulSoup
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("BeautifulSoup4 is required to parse rendered Vietstock pages.") from exc
    return BeautifulSoup(html, "html.parser")


def has_login_wall(html: str) -> bool:
    """Detect the protected-tab sign-in wall without treating normal nav as one."""
    soup = _soup(html)
    if soup.select_one("form[action*='Login' i], input[type='password']"):
        return True
    login_links = soup.select("a[href*='Account/Login' i], a[href*='/login' i]")
    text = _normalise_space(soup.get_text(" ", strip=True)).lower()
    return bool(login_links) and any(
        phrase in text for phrase in ("đăng nhập để", "vui lòng đăng nhập", "sign in to continue")
    )


def page_has_login_wall(page) -> bool:
    """Detect a login wall from rendered, visible controls only.

    Vietstock keeps hidden login forms in the protected page template even
    after a successful sign-in. HTML-only detection therefore gives false
    positives for a real authenticated browser session.
    """
    password_inputs = page.locator('input[type="password"]')
    if any(password_inputs.nth(i).is_visible() for i in range(password_inputs.count())):
        return True
    login_links = page.locator('a[href*="Account/Login"], a[href*="/login"]')
    visible_login = [login_links.nth(i) for i in range(login_links.count()) if login_links.nth(i).is_visible()]
    if not visible_login:
        return False
    visible_text = _normalise_space(page.locator("body").inner_text(timeout=15_000)).lower()
    return any(phrase in visible_text for phrase in ("đăng nhập để", "vui lòng đăng nhập", "sign in to continue"))


def _candidate_from_anchor(anchor, base_url: str) -> dict[str, str] | None:
    href = str(anchor.get("href") or "").strip()
    if not href or href.startswith(("javascript:", "#", "mailto:")):
        return None
    absolute = urljoin(base_url, href)
    return {
        "url": absolute,
        "title": _normalise_space(anchor.get_text(" ", strip=True)) or Path(urlparse(absolute).path).name,
    }


def parse_company_document_links(html: str, base_url: str) -> list[dict[str, str]]:
    """Parse disclosure links only from rows in a company document listing.

    Company document URLs can point to generic download paths, so filename
    matching is not a valid ownership test. The ownership boundary is the
    authenticated, ticker-scoped tab and its disclosure table; generic page
    navigation is intentionally ignored.
    """
    soup = _soup(html)
    documents: dict[str, dict[str, str]] = {}
    for anchor in soup.select("a.doc__ttl--text-link[href]"):
        candidate = _candidate_from_anchor(anchor, base_url)
        if candidate is not None and _is_pdf_url(candidate["url"]):
            documents.setdefault(candidate["url"], candidate)
    for row in soup.select("table tr, .stock-document__item, .document-list__item"):
        row_text = _normalise_space(row.get_text(" ", strip=True)).lower()
        for anchor in row.select("a[href]"):
            candidate = _candidate_from_anchor(anchor, base_url)
            if candidate is None:
                continue
            href = candidate["url"].lower()
            if ".pdf" in href or "download" in href or any(word in row_text for word in DOCUMENT_ROW_WORDS):
                documents.setdefault(candidate["url"], candidate)
    return list(documents.values())


def _is_pdf_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.netloc.lower() in {"static2.vietstock.vn", "finance.vietstock.vn"} and ".pdf" in parsed.path.lower()


def collect_company_document_links(page, base_url: str) -> list[dict[str, str]]:
    """Collect document URLs revealed by clicking controls inside listing rows.

    Vietstock may render a document row without its static2 PDF URL and add
    the URL only after the row's view/download control is clicked. Network
    requests and newly-rendered anchors are observed only during that scoped
    row interaction. Site-wide anchors and navigation are never clicked.
    """
    documents: dict[str, dict[str, str]] = {
        row["url"]: row for row in parse_company_document_links(page.content(), base_url)
    }
    requests_seen: list[str] = []

    def remember_request(request) -> None:
        url = request.url
        if _is_pdf_url(url):
            requests_seen.append(url)

    page.on("request", remember_request)
    try:
        rows = page.locator("table tr, .stock-document__item, .document-list__item, p:has(a.doc__ttl--text-link)")
        for index in range(rows.count()):
            row = rows.nth(index)
            row_text = _normalise_space(row.inner_text(timeout=3_000))
            if not row_text or not any(word in row_text.lower() for word in DOCUMENT_ROW_WORDS):
                continue
            controls = row.locator("a, button, [role='button']")
            selected = None
            for control_index in range(controls.count()):
                control = controls.nth(control_index)
                label = _normalise_space(control.inner_text(timeout=1_000))
                href = str(control.get_attribute("href") or "").strip()
                marker = f"{label} {href}".lower()
                if any(word in marker for word in ("tải", "download", "xem", "view", "chi tiết", "detail")) or not href or href in {"#", "javascript:void(0)"}:
                    selected = control
                    break
            if selected is None:
                continue
            before_pages = set(page.context.pages)
            before_url = page.url
            requests_seen.clear()
            try:
                with page.expect_download(timeout=2_500) as download_info:
                    selected.click(timeout=5_000)
                download = download_info.value
                if _is_pdf_url(download.url):
                    documents.setdefault(download.url, {"url": download.url, "title": row_text})
            except Exception:
                # A modal, a popup, or an XHR is more common than a direct
                # browser download. The click has still happened; inspect the
                # scoped DOM and request list below. Do not click again after
                # the download expectation times out.
                pass
            page.wait_for_timeout(500)
            for url in requests_seen:
                documents.setdefault(url, {"url": url, "title": row_text})
            for anchor in page.locator("a[href]").all():
                href = str(anchor.get_attribute("href") or "").strip()
                absolute = urljoin(page.url, href)
                if _is_pdf_url(absolute):
                    documents.setdefault(absolute, {
                        "url": absolute,
                        "title": _normalise_space(anchor.inner_text()) or row_text,
                    })
            for popup in [candidate for candidate in page.context.pages if candidate not in before_pages]:
                try:
                    popup_url = popup.url
                    if _is_pdf_url(popup_url):
                        documents.setdefault(popup_url, {"url": popup_url, "title": row_text})
                    popup.close()
                except Exception:
                    pass
            if page.url != before_url:
                try:
                    page.goto(before_url, wait_until="domcontentloaded", timeout=15_000)
                    page.wait_for_timeout(300)
                except Exception:
                    pass
    finally:
        page.remove_listener("request", remember_request)
    return list(documents.values())


def parse_company_news_links(html: str, base_url: str, *, overview: bool = False) -> list[dict[str, str]]:
    """Parse only Vietstock's company-news title elements, never nav links."""
    soup = _soup(html)
    selector = "a.overview-news__title" if overview else "a.stock-news__title"
    news: dict[str, dict[str, str]] = {}
    for anchor in soup.select(selector):
        candidate = _candidate_from_anchor(anchor, base_url)
        if candidate is not None:
            news.setdefault(candidate["url"], candidate)
    return list(news.values())


def extract_page_links(html: str, base_url: str) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Return document and news/event candidates visible in rendered HTML.

    A candidate is stored as a manifest item first. Download eligibility is
    determined later from the authenticated HTTP response, never inferred
    from the anchor label.
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("BeautifulSoup4 is required to parse rendered Vietstock pages.") from exc

    soup = BeautifulSoup(html, "html.parser")
    documents: dict[str, dict[str, str]] = {}
    news: dict[str, dict[str, str]] = {}
    for anchor in soup.select("a[href]"):
        href = str(anchor.get("href") or "").strip()
        if not href or href.startswith(("javascript:", "#", "mailto:")):
            continue
        absolute = urljoin(base_url, href)
        label = _normalise_space(anchor.get_text(" ", strip=True))
        blob = f"{label} {href}".lower()
        row = {"url": absolute, "title": label or Path(urlparse(absolute).path).name}
        if ".pdf" in blob or any(word in blob for word in DOCUMENT_WORDS):
            documents.setdefault(absolute, row)
        if any(word in blob for word in NEWS_WORDS):
            news.setdefault(absolute, row)
    return list(documents.values()), list(news.values())


def merge_link_candidates(*groups: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    """Deduplicate candidates by URL while retaining first visible title."""
    merged: dict[str, dict[str, str]] = {}
    for group in groups:
        for candidate in group:
            merged.setdefault(candidate["url"], candidate)
    return list(merged.values())


def is_direct_pdf(candidate: dict[str, str]) -> bool:
    return ".pdf" in urlparse(candidate["url"]).path.lower()


def parse_published_date(value: str) -> date | None:
    """Best-effort date parser. Absence remains NULL rather than guessed."""
    match = re.search(r"(?<!\d)(\d{1,2})[/-](\d{1,2})[/-](20\d{2})(?!\d)", value or "")
    if not match:
        return None
    day, month, year = map(int, match.groups())
    try:
        return date(year, month, day)
    except ValueError:
        return None


CREATE_RUN = text("""
    INSERT INTO vietstock_ingestion_runs (scope, status, requested_by, config)
    VALUES (:scope, 'running', :requested_by, CAST(:config AS jsonb))
    RETURNING id
""")
FINISH_RUN = text("""
    UPDATE vietstock_ingestion_runs
    SET status = :status, finished_at = now(), counters = CAST(:counters AS jsonb), error_summary = :error_summary
    WHERE id = :run_id
""")
UPSERT_TICKER = text("""
    INSERT INTO vietstock_ticker_sync
        (run_id, ticker, company_url, status, http_status, page_title, profile, metrics,
         raw_snapshot, document_count, news_count, error_message)
    VALUES
        (:run_id, :ticker, :company_url, :status, :http_status, :page_title,
         CAST(:profile AS jsonb), CAST(:metrics AS jsonb), :raw_snapshot,
         :document_count, :news_count, :error_message)
    ON CONFLICT (run_id, ticker) DO UPDATE SET
        status = EXCLUDED.status, http_status = EXCLUDED.http_status, page_title = EXCLUDED.page_title,
        profile = EXCLUDED.profile, metrics = EXCLUDED.metrics, raw_snapshot = EXCLUDED.raw_snapshot,
        document_count = EXCLUDED.document_count, news_count = EXCLUDED.news_count,
        error_message = EXCLUDED.error_message, fetched_at = now()
""")
UPSERT_DOCUMENT = text("""
    INSERT INTO vietstock_document_manifest
        (ticker, source_url, title, document_type, published_at, last_seen_at, remote_hash,
         local_path, local_hash, download_status, http_status, last_error)
    VALUES
        (:ticker, :source_url, :title, :document_type, :published_at, now(), :remote_hash,
         :local_path, :local_hash, :download_status, :http_status, :last_error)
    ON CONFLICT (ticker, source_url) DO UPDATE SET
        title = EXCLUDED.title, document_type = EXCLUDED.document_type,
        published_at = COALESCE(EXCLUDED.published_at, vietstock_document_manifest.published_at),
        last_seen_at = now(), remote_hash = COALESCE(EXCLUDED.remote_hash, vietstock_document_manifest.remote_hash),
        local_path = COALESCE(EXCLUDED.local_path, vietstock_document_manifest.local_path),
        local_hash = COALESCE(EXCLUDED.local_hash, vietstock_document_manifest.local_hash),
        download_status = EXCLUDED.download_status, http_status = EXCLUDED.http_status,
        last_error = EXCLUDED.last_error
""")
UPSERT_NEWS = text("""
    INSERT INTO vietstock_news_events (ticker, source_url, title, published_at, category, last_seen_at)
    VALUES (:ticker, :source_url, :title, :published_at, :category, now())
    ON CONFLICT (ticker, source_url) DO UPDATE SET
        title = EXCLUDED.title,
        published_at = COALESCE(EXCLUDED.published_at, vietstock_news_events.published_at),
        category = EXCLUDED.category, last_seen_at = now()
""")
SELECT_DOCUMENT_STATE = text("""
    SELECT download_status, http_status, remote_hash, local_path, local_hash, last_error
    FROM vietstock_document_manifest
    WHERE ticker = :ticker AND source_url = :source_url
""")


def _new_run(scope: str, config: dict[str, Any], requested_by: str | None = None) -> int:
    with get_session() as session:
        return int(session.execute(CREATE_RUN, {
            "scope": scope, "requested_by": requested_by,
            "config": json.dumps(config, ensure_ascii=True),
        }).scalar_one())


def _finish_run(run_id: int, status: str, counters: dict[str, int], error_summary: str | None = None) -> None:
    with get_session() as session:
        session.execute(FINISH_RUN, {
            "run_id": run_id, "status": status,
            "counters": json.dumps(counters, ensure_ascii=True),
            "error_summary": error_summary,
        })


def _persist_ticker(run_id: int, ticker: str, url: str, *, status: str,
                    http_status: int | None, title: str | None, profile: dict[str, Any],
                    metrics: dict[str, Any], raw_snapshot: str | None,
                    document_count: int, news_count: int, error_message: str | None) -> None:
    with get_session() as session:
        session.execute(UPSERT_TICKER, {
            "run_id": run_id, "ticker": ticker, "company_url": url, "status": status,
            "http_status": http_status, "page_title": title,
            "profile": json.dumps(profile, ensure_ascii=True),
            "metrics": json.dumps(metrics, ensure_ascii=True), "raw_snapshot": raw_snapshot,
            "document_count": document_count, "news_count": news_count,
            "error_message": error_message,
        })


def _save_snapshot(ticker: str, html: str) -> str:
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = _utcnow().strftime("%Y%m%dT%H%M%SZ")
    path = SNAPSHOT_DIR / f"{_safe_slug(ticker)}_{stamp}.html"
    path.write_text(html, encoding="utf-8")
    return str(path.relative_to(PROJECT_ROOT))


def _page_summary(page) -> tuple[dict[str, Any], dict[str, Any]]:
    """Extract bounded, presentation-safe observations without guessing fields.

    Values are text visible to an authenticated browser; unknown values are
    omitted. Full raw HTML is retained in a local snapshot for parser
    refinement, while DB ``profile``/``metrics`` remain small and inspectable.
    """
    title = _normalise_space(page.title())
    visible = _normalise_space(page.locator("body").inner_text(timeout=15_000))
    profile = {"page_title": title} if title else {}
    metrics: dict[str, Any] = {}
    for label in ("Giá", "Vốn hóa", "P/E", "EPS", "ROE", "ROA", "Khối lượng"):
        match = re.search(rf"{re.escape(label)}\s*[:\-]?\s*([^\n]{{1,80}})", visible, flags=re.I)
        if match:
            metrics[label] = _normalise_space(match.group(1))
    return profile, metrics


def _download_pdf(context, ticker: str, candidate: dict[str, str]) -> dict[str, Any]:
    """Download a discovered candidate through the authenticated context.

    It writes a file only when the response really begins with the PDF magic
    bytes. Login pages, HTML error pages and non-PDF links remain manifest
    rows, which prevents corrupt files from entering the PDF pipeline.
    """
    try:
        response = context.request.get(candidate["url"], timeout=60_000)
        status = response.status
        body = response.body()
        content_type = (response.headers.get("content-type") or "").lower()
        if status in {401, 403}:
            return {"download_status": "access_denied", "http_status": status, "last_error": "Premium access was denied."}
        if status >= 400:
            return {"download_status": "failed", "http_status": status, "last_error": f"HTTP {status}"}
        if not body.startswith(PDF_MAGIC):
            note = "Response is not a PDF"
            if "text/html" in content_type:
                note = "Response is HTML, likely a page rather than a downloadable PDF"
            return {"download_status": "not_pdf", "http_status": status, "last_error": note}
        digest = _sha256(body)
        ext = ".pdf"
        name = _safe_slug(Path(urlparse(candidate["url"]).path).name, fallback=f"{ticker}_{digest[:12]}{ext}")
        if not name.lower().endswith(ext):
            name = f"{name}{ext}"
        destination = PDF_DIR / ticker / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists() or _sha256(destination.read_bytes()) != digest:
            destination.write_bytes(body)
        return {
            "download_status": "downloaded", "http_status": status,
            "remote_hash": digest, "local_hash": digest,
            "local_path": str(destination.relative_to(PROJECT_ROOT)), "last_error": None,
        }
    except Exception as exc:  # network/browser lifecycle failures are per-document
        return {"download_status": "failed", "http_status": None, "last_error": safe_error(exc)}


def _persist_discovered_links(ticker: str, context, documents: Iterable[dict[str, str]],
                              news: Iterable[dict[str, str]], *, download_documents: bool,
                              refresh_downloads: bool = False) -> dict[str, int]:
    counters = {"documents_discovered": 0, "documents_downloaded": 0, "news_events": 0}
    with get_session() as session:
        for candidate in documents:
            result: dict[str, Any] = {"download_status": "discovered", "http_status": None, "last_error": None}
            existing = session.execute(SELECT_DOCUMENT_STATE, {
                "ticker": ticker, "source_url": candidate["url"],
            }).mappings().first()
            should_download = download_documents and (
                refresh_downloads or not existing or existing["download_status"] != "downloaded"
            )
            if should_download:
                result = _download_pdf(context, ticker, candidate)
            elif existing:
                # Preserve the known-good manifest state; discovery still
                # updates last_seen_at below without re-downloading bytes.
                result = dict(existing)
            session.execute(UPSERT_DOCUMENT, {
                "ticker": ticker, "source_url": candidate["url"], "title": candidate["title"],
                "document_type": "pdf_candidate" if ".pdf" in candidate["url"].lower() else "linked_document",
                "published_at": parse_published_date(candidate["title"]), "remote_hash": result.get("remote_hash"),
                "local_path": result.get("local_path"), "local_hash": result.get("local_hash"),
                "download_status": result["download_status"], "http_status": result.get("http_status"),
                "last_error": result.get("last_error"),
            })
            counters["documents_discovered"] += 1
            counters["documents_downloaded"] += int(should_download and result["download_status"] == "downloaded")
        for candidate in news:
            session.execute(UPSERT_NEWS, {
                "ticker": ticker, "source_url": candidate["url"], "title": candidate["title"],
                "published_at": parse_published_date(candidate["title"]), "category": "news_or_event",
            })
            counters["news_events"] += 1
    return counters


def browser_session(headless: bool = False, cdp_url: str | None = None,
                   cdp_no_defaults: bool = False):
    """Context manager yielding a persistent or normal-Chrome context.

    Browser binaries need a one-time ``python -m playwright install chromium``
    after requirements installation. This is intentionally host-side and is
    not suitable for the project's Docker/Airflow containers. When ``cdp_url``
    is supplied, Playwright connects to an already-running normal Chrome
    instance instead of launching its bundled Chromium. This lets Google
    OAuth run in the user's ordinary browser profile. ``cdp_no_defaults`` is
    opt-in for structured read-only attachment; the Premium document path
    keeps its existing browser-session behavior.
    """
    sync_playwright = _require_playwright()
    class _Session:
        def __enter__(self):
            self.playwright = sync_playwright().start()
            self.connected_over_cdp = bool(cdp_url)
            if cdp_url:
                try:
                    self.browser = self.playwright.chromium.connect_over_cdp(
                        cdp_url, no_defaults=cdp_no_defaults,
                    )
                except Exception as exc:
                    self.playwright.stop()
                    raise RuntimeError(
                        f"Cannot connect to Chrome CDP at {cdp_url}. Start a dedicated Chrome with "
                        "--remote-debugging-port=9222, verify `curl http://127.0.0.1:9222/json/version`, "
                        f"then retry ({type(exc).__name__}: {str(exc).split('Call log:', 1)[0].strip()[:240]})."
                    ) from exc
                if not self.browser.contexts:
                    raise RuntimeError("The CDP browser has no browser context.")
                self.context = self.browser.contexts[0]
                return self.context
            PROFILE_DIR.mkdir(parents=True, exist_ok=True)
            self.context = self.playwright.chromium.launch_persistent_context(
                str(PROFILE_DIR), headless=headless, accept_downloads=True,
                viewport={"width": 1440, "height": 1000},
            )
            return self.context

        def __exit__(self, exc_type, exc, tb):
            try:
                if not self.connected_over_cdp:
                    self.context.close()
            except Exception:
                # Preserve the original cancellation (for example Ctrl+C
                # while waiting for interactive login) instead of replacing
                # it with a noisy "connection closed" shutdown traceback.
                if exc_type is None and not self.connected_over_cdp:
                    raise
            finally:
                self.playwright.stop()
            return False
    return _Session()


def login(headless: bool = False, cdp_url: str | None = None) -> None:
    """Verify normal login and access to a protected company tab."""
    with browser_session(headless=headless, cdp_url=cdp_url) as context:
        page = context.pages[0] if context.pages else context.new_page()
        page.goto("https://finance.vietstock.vn/", wait_until="domcontentloaded", timeout=60_000)
        if cdp_url:
            print("Connected to normal Chrome over CDP. Sign in in that Chrome window, then return here and press Enter.")
        elif not headless:
            print("A Chromium window is open. Sign in normally, then return here and press Enter.")
        if cdp_url or not headless:
            try:
                input()
            except KeyboardInterrupt:
                print("Login session cancelled; no crawl was run.")
                return
        if not has_authenticated_document_access(headless=headless, context=context):
            print("The protected VNM documents tab still requires sign-in.")
            return
        print("Authenticated VNM Documents access verified.")


def has_authenticated_document_access(*, headless: bool = True, context=None,
                                      cdp_url: str | None = None) -> bool:
    """Return whether the persistent profile can access one protected tab.

    A bulk document run must prove this once before starting. The optional
    ``context`` allows ``login()`` to verify without opening a second browser.
    """
    if context is None:
        with browser_session(headless=headless, cdp_url=cdp_url) as owned_context:
            return has_authenticated_document_access(headless=headless, context=owned_context, cdp_url=cdp_url)
    page = context.pages[0] if context.pages else context.new_page()
    probe = company_tab_url("VNM", "documents")
    response = page.goto(probe, wait_until="domcontentloaded", timeout=60_000)
    page.wait_for_timeout(800)
    return bool(
        response
        and response.status < 400
        and is_company_tab_url(page.url, "VNM", "documents")
        and not page_has_login_wall(page)
    )


def discover_directory(max_pages: int = 25, headless: bool = True, cdp_url: str | None = None) -> list[CompanyLink]:
    if max_pages < 1:
        raise ValueError("max_pages must be at least 1")
    run_id = _new_run("directory", {"max_pages": max_pages})
    companies: dict[str, CompanyLink] = {}
    try:
        known = known_companies()
        with browser_session(headless=headless, cdp_url=cdp_url) as context:
            page = context.pages[0] if context.pages else context.new_page()
            seen_resolved_urls: set[str] = set()
            for page_number in range(1, max_pages + 1):
                url = DIRECTORY_URL.format(page=page_number)
                response = page.goto(url, wait_until="domcontentloaded", timeout=60_000)
                page.wait_for_timeout(1_000)
                if page.url in seen_resolved_urls:
                    raise RuntimeError(
                        f"Directory pagination did not advance: requested page {page_number} resolved again to {page.url}. "
                        "Use `sync --all-known` for the project universe; inspect the live pager before changing this crawler."
                    )
                seen_resolved_urls.add(page.url)
                html = page.content()
                page_companies = extract_company_links(html, url, set(known))
                if not page_companies:
                    if page_number == 1:
                        raise RuntimeError(
                            "No company detail links were found on the directory page. "
                            "Verify the logged-in session and update the parser from its local snapshot if Vietstock changed its markup."
                        )
                    break
                for company in page_companies:
                    companies.setdefault(company.ticker, company)
                if response and response.status >= 400:
                    raise RuntimeError(f"Directory page {page_number} returned HTTP {response.status}")
        values = list(companies.values())
        _finish_run(run_id, "completed", {"pages_requested": max_pages, "companies": len(values)})
        return values
    except Exception as exc:
        _finish_run(run_id, "failed", {"pages_requested": max_pages, "companies": len(companies)}, safe_error(exc))
        raise


def sync_companies(companies: Iterable[CompanyLink], *, download_documents: bool = False,
                   refresh_downloads: bool = False, headless: bool = True,
                   delay_seconds: float = 1.5, cdp_url: str | None = None) -> dict[str, int]:
    """Synchronise known company pages one at a time with polite pacing."""
    company_list = list(companies)
    run_id = _new_run("document_sync" if download_documents else "company_sync", {
        "tickers": len(company_list), "download_documents": download_documents,
        "refresh_downloads": refresh_downloads,
        "delay_seconds": delay_seconds,
    })
    counters = {"requested": len(company_list), "completed": 0, "partial": 0, "failed": 0,
                "documents_discovered": 0, "documents_downloaded": 0, "news_events": 0}
    try:
        with browser_session(headless=headless, cdp_url=cdp_url) as context:
            page = context.pages[0] if context.pages else context.new_page()
            for index, company in enumerate(company_list, start=1):
                try:
                    print(f"[{index}/{len(company_list)}] {company.ticker}: opening company page", flush=True)
                    response = page.goto(company.url, wait_until="domcontentloaded", timeout=60_000)
                    page.wait_for_timeout(1_000)
                    print(f"[{index}/{len(company_list)}] {company.ticker}: page HTTP {response.status if response else 'unknown'}; reading scoped tabs", flush=True)
                    html = page.content()
                    snapshot = _save_snapshot(company.ticker, html)
                    profile, metrics = _page_summary(page)
                    documents: list[dict[str, str]] = []
                    news = parse_company_news_links(html, company.url, overview=True)
                    partial_reason: str | None = None

                    documents_url = company_tab_url(company.ticker, "documents")
                    print(f"    documents: {documents_url}", flush=True)
                    documents_response = page.goto(documents_url, wait_until="domcontentloaded", timeout=60_000)
                    page.wait_for_timeout(700)
                    documents_html = page.content()
                    if not is_company_tab_url(page.url, company.ticker, "documents"):
                        partial_reason = "Company documents tab redirected outside the ticker-scoped URL."
                    elif documents_response and documents_response.status >= 400:
                        partial_reason = f"Company documents tab returned HTTP {documents_response.status}."
                    elif page_has_login_wall(page):
                        partial_reason = "Authenticated Vietstock session is required for company documents."
                    else:
                        documents = collect_company_document_links(page, documents_url)

                    news_url = company_tab_url(company.ticker, "news")
                    print(f"    news: {news_url}", flush=True)
                    news_response = page.goto(news_url, wait_until="domcontentloaded", timeout=60_000)
                    page.wait_for_timeout(700)
                    if is_company_tab_url(page.url, company.ticker, "news") and (not news_response or news_response.status < 400):
                        news = merge_link_candidates(news, parse_company_news_links(page.content(), news_url))
                    elif partial_reason is None:
                        partial_reason = "Company news tab was unavailable."
                    print(
                        f"[{index}/{len(company_list)}] {company.ticker}: "
                        f"{len(documents)} document candidates, {len(news)} news/event candidates",
                        flush=True,
                    )
                    link_counts = _persist_discovered_links(
                        company.ticker, context, documents, news, download_documents=download_documents,
                        refresh_downloads=refresh_downloads,
                    )
                    for key in ("documents_discovered", "documents_downloaded", "news_events"):
                        counters[key] += link_counts[key]
                    status = "completed" if response and response.status < 400 and partial_reason is None else "partial"
                    counters[status] += 1
                    _persist_ticker(
                        run_id, company.ticker, company.url, status=status,
                        http_status=response.status if response else None, title=profile.get("page_title"),
                        profile=profile, metrics=metrics, raw_snapshot=snapshot,
                        document_count=link_counts["documents_discovered"], news_count=link_counts["news_events"],
                        error_message=None if status == "completed" else (
                            partial_reason or "Company page did not return a successful HTTP status."
                        ),
                    )
                    print(f"[{index}/{len(company_list)}] {company.ticker}: {status}", flush=True)
                except Exception as exc:  # one ticker must not stop 200 others
                    counters["failed"] += 1
                    _persist_ticker(
                        run_id, company.ticker, company.url, status="failed", http_status=None,
                        title=None, profile={}, metrics={}, raw_snapshot=None, document_count=0, news_count=0,
                        error_message=safe_error(exc),
                    )
                    print(f"[{index}/{len(company_list)}] {company.ticker}: failed ({safe_error(exc)})", flush=True)
                if index < len(company_list):
                    time.sleep(max(delay_seconds, 0))
        final_status = "completed" if not counters["failed"] and not counters["partial"] else "partial"
        _finish_run(run_id, final_status, counters)
        return counters
    except BaseException as exc:
        # KeyboardInterrupt is a meaningful operational outcome. Persist it
        # so the Admin tab never leaves a run falsely marked "running".
        _finish_run(run_id, "failed", counters, safe_error(exc))
        raise
