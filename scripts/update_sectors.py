"""Fill in the real Vietstock industry classification for the 200 candidate
companies that were seeded with sector = "Chưa phân loại" (scripts/seed_companies.py).

Source: data/reference/ticker_sector_map.csv, crawled from
finance.vietstock.vn/doanh-nghiep-a-z (24-category VS industry taxonomy,
2026-07-15) via Claude in Chrome - covers all 200 candidate tickers.

Deliberately does NOT touch the CORE_16 companies - they keep their
curated 4-sector thesis-specific labels ("Tiêu dùng thiết yếu", "Bất động
sản", "Công nghiệp chế biến", "Tiêu dùng không thiết yếu (Bán lẻ)"), which
are a different, coarser taxonomy chosen for the original 16-company
sample design - overwriting them with Vietstock's finer-grained categories
would break that intentional grouping.

Safe to re-run (plain UPDATE, idempotent).

Usage: python scripts/update_sectors.py
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import text
from src.core.db import get_session

MAP_CSV = Path(__file__).resolve().parents[1] / "data/reference/ticker_sector_map.csv"

UPDATE_SECTOR = text("UPDATE companies SET sector = :sector WHERE ticker = :ticker")

if __name__ == "__main__":
    with open(MAP_CSV, encoding="utf-8") as f:
        rows = [(row["ticker"].strip(), row["vietstock_sector"].strip()) for row in csv.DictReader(f)]

    with get_session() as session:
        updated = 0
        for ticker, sector in rows:
            result = session.execute(UPDATE_SECTOR, {"ticker": ticker, "sector": sector})
            updated += result.rowcount

    print(f"Updated sector for {updated}/{len(rows)} companies "
          f"(rows in CSV but not matched to a seeded ticker: {len(rows) - updated}).")
    print("Check remaining gaps with:")
    print("  SELECT COUNT(*) FROM companies WHERE sector = 'Chưa phân loại';")
