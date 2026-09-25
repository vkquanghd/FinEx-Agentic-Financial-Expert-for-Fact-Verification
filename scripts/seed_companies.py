"""Seed the `companies` table.

Two tiers:
  1. CORE_16 - the original curated sample (4 sectors x 4 companies, all
     HOSE, all using the standard non-financial-institution statement
     template - see the sector-swap discussion that dropped the 4 bank
     tickers VCB/BID/ACB/VPB in favor of MWG/PNJ/DGW/FRT). These get their
     real sector label.
  2. Everything else in data/reference/top200_candidate_companies.csv (the
     data-completeness ranking across all 2115 listed tickers) - these get
     seeded too, with sector = "Chưa phân loại", so that exporting BCTC/CSTC
     for ANY of the top-200 tickers "just works" against the ETL's FK check
     instead of being skipped. Re-classify sector later once you decide
     which of these actually go into the thesis's final evaluation set.

Safe to re-run (upsert on ticker, CORE_16 always wins on conflict).

Usage: python scripts/seed_companies.py
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import text
from src.core.db import get_session

CORE_16 = [
    ("VNM", "CTCP Sữa Việt Nam", "HOSE", "Tiêu dùng thiết yếu"),
    ("SAB", "Tổng Công ty cổ phần Bia - Rượu - Nước giải khát Sài Gòn", "HOSE", "Tiêu dùng thiết yếu"),
    ("MSN", "CTCP Tập đoàn Masan", "HOSE", "Tiêu dùng thiết yếu"),
    ("KDC", "CTCP Tập đoàn KIDO", "HOSE", "Tiêu dùng thiết yếu"),
    ("VIC", "Tập đoàn VINGROUP - CTCP", "HOSE", "Bất động sản"),
    ("VHM", "CTCP Vinhomes", "HOSE", "Bất động sản"),
    ("NVL", "CTCP Tập đoàn Đầu tư Địa ốc No Va", "HOSE", "Bất động sản"),
    ("KDH", "CTCP Đầu tư và Kinh doanh Nhà Khang Điền", "HOSE", "Bất động sản"),
    ("HPG", "CTCP Tập đoàn Hòa Phát", "HOSE", "Công nghiệp chế biến"),
    ("GAS", "Tổng Công ty Khí Việt Nam - CTCP", "HOSE", "Công nghiệp chế biến"),
    ("DPM", "Tổng công ty Phân bón và Hóa chất Dầu khí - CTCP", "HOSE", "Công nghiệp chế biến"),
    ("DCM", "CTCP - Tổng công ty Phân bón Dầu khí Cà Mau", "HOSE", "Công nghiệp chế biến"),
    ("MWG", "CTCP Đầu tư Thế giới Di động", "HOSE", "Tiêu dùng không thiết yếu (Bán lẻ)"),
    ("PNJ", "CTCP Vàng bạc Đá quý Phú Nhuận", "HOSE", "Tiêu dùng không thiết yếu (Bán lẻ)"),
    ("DGW", "CTCP Thế Giới Số", "HOSE", "Tiêu dùng không thiết yếu (Bán lẻ)"),
    ("FRT", "CTCP Bán lẻ Kỹ thuật số FPT", "HOSE", "Tiêu dùng không thiết yếu (Bán lẻ)"),
]

# Ad-hoc additions outside both CORE_16 and top200 CSV - grows only when a
# specific ticker gets asked about in the Chat demo and genuinely fails to
# resolve (same "add on real failure" philosophy as
# src/structured/company_lookup.py's BRAND_NAME_ALIASES and
# src/structured/query_builder.py's NON_TICKER_ABBREVIATIONS - not a
# preemptive attempt to cover more of the market).
#
# 2026-08-05: VNZ (VNG Corporation) - hit via "Vina Game" in the Chat demo;
# RAG already had VNZ's disclosure_chunks embedded (PDF ingestion doesn't
# require a companies row), but VNZ had NO companies row at all, so it was
# invisible to company_lookup's index, the "Chọn trực tiếp"/"Dữ liệu đã nạp"
# tab dropdowns, and any future structured/DAG query about it. company_name/
# exchange/sector below are best-effort from public listing info - VERIFY
# against Vietstock before citing in the thesis; this row's job is making
# VNZ resolvable, not being an authoritative data claim.
EXTRA_TICKERS = [
    ("VNZ", "CTCP VNG", "UPCOM", "Công nghệ thông tin"),
]

CANDIDATES_CSV = Path(__file__).resolve().parents[1] / "data/reference/top200_candidate_companies.csv"

UPSERT = text("""
    INSERT INTO companies (ticker, company_name, exchange, sector)
    VALUES (:ticker, :company_name, :exchange, :sector)
    ON CONFLICT (ticker) DO UPDATE SET
        company_name = EXCLUDED.company_name, exchange = EXCLUDED.exchange, sector = EXCLUDED.sector
""")


def load_candidates(csv_path: Path, exclude_tickers: set[str]):
    if not csv_path.exists():
        return []
    rows = []
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ticker = row["ticker"].strip()
            if ticker in exclude_tickers:
                continue
            rows.append((ticker, row["company_name"].strip(), row["exchange"].strip(), "Chưa phân loại"))
    return rows


if __name__ == "__main__":
    core_tickers = {t for t, *_ in CORE_16} | {t for t, *_ in EXTRA_TICKERS}
    extras = load_candidates(CANDIDATES_CSV, exclude_tickers=core_tickers)
    all_rows = CORE_16 + EXTRA_TICKERS + extras

    with get_session() as session:
        for ticker, name, exchange, sector in all_rows:
            session.execute(UPSERT, {
                "ticker": ticker, "company_name": name, "exchange": exchange, "sector": sector,
            })
    print(f"Seeded {len(CORE_16)} core companies + {len(EXTRA_TICKERS)} ad-hoc extras + "
          f"{len(extras)} candidate companies (from {CANDIDATES_CSV.name}) = {len(all_rows)} total.")
