"""One-off (2026-07-21): vietstock_etl.py's auto-register (see
_ensure_ticker_registered) creates a PLACEHOLDER row for any ticker it meets
that isn't already in `companies` (company_name=ticker, exchange='UNKNOWN',
sector='Chua phan loai'), so batch ingestion never silently drops data again.
This script overwrites that placeholder with real metadata for the 4 tickers
found this way in the 2026-07-21 scan of data/raw/vietstock_exports/
(AAN, VCB, VIB, VNZ - none of these were in top200_candidate_companies.csv
either, so seed_companies.py wouldn't have caught them).

NOTE on VCB/VIB: both are banks. Their Vietstock "Bao-cao-tai-chinh" export
uses the bank-specific statement template (e.g. "Tien gui tai NHNN", "Cho vay
khach hang" instead of "Hang ton kho") - this is exactly why the original
CORE_16 sample swapped VCB/BID/ACB/VPB out for MWG/PNJ/DGW/FRT. The ETL will
parse and load their rows without erroring (STATEMENT_HEADERS matches on the
same section-header text for both templates), but any downstream code that
looks up item_name values assuming the non-bank template (financial_ratios
comparisons, calculator.py, causal panel) will not find matching line items
for VCB/VIB and will silently get None/missing for those metrics. Decided
2026-07-21: keep VCB/VIB registered with real metadata (so RAG/DAG data is
available for reference and in case more banks get added later), but they
are OUT OF SCOPE for any RQ that assumes the non-bank ratio template until a
bank-specific parsing path is built (documented as Future Engineering Work).

Safe to re-run (upsert on ticker).

Usage: python3 -m scripts.fix_new_ticker_metadata
"""
from sqlalchemy import text
from src.core.db import get_session

REAL_METADATA = [
    ("VCB", "Ngân hàng TMCP Ngoại thương Việt Nam (Vietcombank)", "HOSE", "Ngân hàng"),
    ("VIB", "Ngân hàng TMCP Quốc tế Việt Nam", "HOSE", "Ngân hàng"),
    ("VNZ", "CTCP VNG (VNG Corporation)", "UPCOM", "Công nghệ thông tin"),
    ("AAN", "CTCP Lương thực A An", "HOSE", "Nông nghiệp - Lương thực"),
]

UPSERT = text("""
    INSERT INTO companies (ticker, company_name, exchange, sector)
    VALUES (:ticker, :company_name, :exchange, :sector)
    ON CONFLICT (ticker) DO UPDATE SET
        company_name = EXCLUDED.company_name, exchange = EXCLUDED.exchange, sector = EXCLUDED.sector
""")

if __name__ == "__main__":
    with get_session() as session:
        for ticker, name, exchange, sector in REAL_METADATA:
            session.execute(UPSERT, {
                "ticker": ticker, "company_name": name, "exchange": exchange, "sector": sector,
            })
    print(f"Updated real metadata for {len(REAL_METADATA)} ticker(s): "
          f"{', '.join(t for t, *_ in REAL_METADATA)}")
    print("VCB/VIB: registered but OUT OF SCOPE for ratio-based RQs until bank-template "
          "parsing is built (see docs/PROGRESS.md).")
