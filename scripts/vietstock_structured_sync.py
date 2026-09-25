#!/usr/bin/env python3
"""Collect Vietstock trading and financial HTML tables.

Run a single ticker first. Only after the resulting table counts and schema
are reviewed should ``--all-known`` be used for the 222-company universe.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ingestion.vietstock_premium import safe_error
from src.ingestion.vietstock_structured import selected_companies, sync_structured


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--tickers", help="Comma-separated tickers, e.g. VNM,AAA")
    selection.add_argument("--all-known", action="store_true")
    parser.add_argument("--sections", default="trading,financial", help="Comma-separated sections")
    parser.add_argument("--max-subsections", type=int, default=20)
    parser.add_argument("--delay-seconds", type=float, default=1.5)
    parser.add_argument(
        "--daily-trading", action="store_true",
        help="Crawl only the trading page and record this run as trading_daily",
    )
    parser.add_argument("--headed", action="store_true")
    parser.add_argument("--cdp-url")
    args = parser.parse_args()
    if args.daily_trading and args.sections != "trading,financial":
        parser.error("--daily-trading cannot be combined with a custom --sections value")
    sections = ("trading",) if args.daily_trading else tuple(
        item.strip() for item in args.sections.split(",") if item.strip()
    )
    companies = selected_companies(args.tickers, args.all_known)
    try:
        print(sync_structured(
            companies, sections=sections, max_subsections=args.max_subsections,
            headless=not args.headed, cdp_url=args.cdp_url, delay_seconds=args.delay_seconds,
            scope="trading_daily" if args.daily_trading else "structured_sync",
        ))
    except Exception as exc:
        parser.error(safe_error(exc))


if __name__ == "__main__":
    main()
