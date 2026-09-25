#!/usr/bin/env python3
"""Host-side command line for authenticated Vietstock Premium synchronisation.

Examples:
  python -m scripts.vietstock_premium_sync login
  python -m scripts.vietstock_premium_sync login --cdp-url http://127.0.0.1:9222
  python -m scripts.vietstock_premium_sync discover --max-pages 2 --output data/private/vietstock_companies.json
  python -m scripts.vietstock_premium_sync sync --tickers VNM,HPG --documents --headed

The owner logs in in the opened browser. Never pass account credentials to
this command, environment variables, or source files.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ingestion.vietstock_premium import (
    CompanyLink,
    company_detail_url,
    discover_directory,
    has_authenticated_document_access,
    known_companies,
    known_company_links,
    login,
    safe_error,
    sync_companies,
)


def _load_companies(path: Path) -> list[CompanyLink]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return [CompanyLink(ticker=row["ticker"], name=row.get("name", row["ticker"]), url=row["url"]) for row in rows]


def _save_companies(path: Path, companies: list[CompanyLink]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([c.__dict__ for c in companies], ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    login_parser = sub.add_parser("login", help="Open a persistent browser for normal interactive login")
    login_parser.add_argument("--headless", action="store_true", help="Only for a pre-existing authenticated profile")
    login_parser.add_argument("--cdp-url", help="Connect to an already-running normal Chrome debugging endpoint")
    discover_parser = sub.add_parser("discover", help="Scan the public company directory using the logged-in profile")
    discover_parser.add_argument("--max-pages", type=int, default=25,
                                 help="Safety cap; scanning stops early when a page has no company detail links")
    discover_parser.add_argument("--output", type=Path, default=Path("data/private/vietstock_companies.json"))
    discover_parser.add_argument("--headed", action="store_true", help="Show Chromium while scanning")
    sync_parser = sub.add_parser("sync", help="Snapshot company pages and read ticker-scoped document/news tabs")
    sync_parser.add_argument("--companies-file", type=Path, default=Path("data/private/vietstock_companies.json"))
    selection = sync_parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--tickers", help="Comma-separated subset, e.g. VNM,HPG")
    selection.add_argument("--all-known", action="store_true",
                           help="Sync all companies in the project's approved companies table")
    sync_parser.add_argument("--documents", action="store_true", help="Download PDFs found in the authenticated ticker-scoped document listing")
    sync_parser.add_argument("--refresh-downloads", action="store_true",
                             help="Re-request already-downloaded document URLs; default fetches only new/unresolved links")
    sync_parser.add_argument("--delay-seconds", type=float, default=1.5)
    sync_parser.add_argument("--headed", action="store_true", help="Show Chromium while scanning")
    sync_parser.add_argument("--cdp-url", help="Connect to an already-running normal Chrome debugging endpoint")
    args = parser.parse_args()

    if args.command == "login":
        login(headless=args.headless, cdp_url=args.cdp_url)
        return
    if args.command == "discover":
        companies = discover_directory(args.max_pages, headless=not args.headed, cdp_url=None)
        _save_companies(args.output, companies)
        print(json.dumps({"companies": len(companies), "output": str(args.output)}, ensure_ascii=False))
        return
    if args.all_known:
        companies = known_company_links()
    else:
        companies = _load_companies(args.companies_file)
        selected = {ticker.strip().upper() for ticker in args.tickers.split(",") if ticker.strip()}
        discovered = {company.ticker: company for company in companies}
        known = known_companies()
        missing = selected - set(discovered)
        unknown = missing - set(known)
        if unknown:
            parser.error(f"Unknown ticker(s) not present in the project companies table: {', '.join(sorted(unknown))}")
        for ticker in sorted(missing):
            name = known[ticker]
            discovered[ticker] = CompanyLink(
                ticker=ticker, name=name, url=company_detail_url(ticker, name),
            )
            print(f"Using known-company URL fallback for {ticker}; it was absent from the current discovery file.", file=sys.stderr)
        companies = [discovered[ticker] for ticker in sorted(selected)]
    if not companies:
        parser.error("No companies selected. Run discover first or provide a non-empty companies file.")
    if args.all_known and args.documents:
        try:
            authenticated = has_authenticated_document_access(headless=not args.headed, cdp_url=args.cdp_url)
        except Exception as exc:
            parser.error(f"Could not verify the persistent Vietstock session: {safe_error(exc)}")
        if not authenticated:
            parser.error(
                "Bulk document sync was not started because the persistent browser profile cannot access "
                "the protected VNM Documents tab. Run `python -m scripts.vietstock_premium_sync login`, "
                "complete normal sign-in, and verify one ticker before retrying."
            )
    print(json.dumps(sync_companies(
        companies, download_documents=args.documents, refresh_downloads=args.refresh_downloads,
        headless=not args.headed,
        delay_seconds=args.delay_seconds,
        cdp_url=args.cdp_url,
    ), ensure_ascii=False))


if __name__ == "__main__":
    main()
