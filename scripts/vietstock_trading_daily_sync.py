#!/usr/bin/env python3
"""Daily Vietstock trading-table snapshot for the known ticker universe.

This is intentionally separate from the Premium PDF/document sync. It stores
the rendered trading tables in ``vietstock_structured_tables`` and preserves
history through the table hash. It is a daily browser snapshot, not a live
tick-by-tick market feed; normalized OHLCV remains in ``company_price_snapshot``.
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
    parser.add_argument("--delay-seconds", type=float, default=1.0)
    parser.add_argument("--headed", action="store_true")
    parser.add_argument("--cdp-url", default=None)
    args = parser.parse_args()
    try:
        print(sync_structured(
            selected_companies(args.tickers, args.all_known),
            sections=("trading",),
            headless=not args.headed,
            cdp_url=args.cdp_url,
            delay_seconds=args.delay_seconds,
            scope="trading_daily",
        ))
    except Exception as exc:
        parser.error(safe_error(exc))


if __name__ == "__main__":
    main()
