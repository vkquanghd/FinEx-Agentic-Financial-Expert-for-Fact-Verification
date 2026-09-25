"""One-off (2026-07-23): confirmed via scripts/investigate_aaaa_data.py that
'AAAA' is real Vietstock-exported data (verified by opening
data/raw/vietstock_exports/VietstockFinance_Bao-cao-tai-chinh_20260714-190643.xlsx
directly - cell C6/D6/E6 literally reads "AAAA" as Mã CK, not a parsing bug
on this project's side) for an unidentifiable insurance company (item
names like "Phải thu về hợp đồng bảo hiểm", "Thu phí nhận tái bảo hiểm"
confirm insurance, not a misfiled copy of AAA - 0% value match against
AAA's real data for the same fiscal years, ruling that hypothesis out).
No company name, exchange, or sector was ever recoverable
(`companies.company_name`='AAAA', exchange='UNKNOWN', sector='Chưa phân
loại'), and insurance companies are already out of this thesis's scope
regardless (see docs/PROGRESS.md's "Known gaps" - insurance uses a
separate chart of accounts, not Circular 200/CORE_MS_CODES). Decision
(confirmed with the project owner 2026-07-23): delete it - real but
unidentifiable, out-of-scope data left in only risks contaminating
cross-sectional analysis (e.g. the causal panel) with an unlabeled row.

Deletes in FK-safe order (financial_line_items/financial_ratios/
disclosures reference companies.ticker) and re-verifies the row counts
match what scripts/cleanup_aaaa_ticker.py and investigate_aaaa_data.py
already found (297/33/3) before deleting anything - if the counts have
changed since that investigation, stops instead of deleting blind.

Usage: python3 -m scripts.delete_aaaa_ticker
"""
from sqlalchemy import text

from src.core.db import engine

TICKER = "AAAA"
EXPECTED = {
    "financial_line_items": 297,
    "financial_ratios": 33,
    "disclosures": 3,
}

print(f"=== Re-verifying row counts for ticker='{TICKER}' before deleting ===\n")
with engine.connect() as conn:
    actual = {}
    for table in EXPECTED:
        n = conn.execute(text(f"SELECT COUNT(*) FROM {table} WHERE ticker = :t"), {"t": TICKER}).scalar()
        actual[table] = n
        match = "OK" if n == EXPECTED[table] else "CHANGED"
        print(f"  {table}: {n} row(s) (expected {EXPECTED[table]}) [{match}]")

if actual != EXPECTED:
    print("\nSTOPPED - row counts changed since the last investigation. "
          "Re-run scripts/investigate_aaaa_data.py before deleting.")
else:
    print(f"\nCounts match. Deleting all '{TICKER}' rows (FK-safe order)...")
    with engine.begin() as conn:
        for table in EXPECTED:
            result = conn.execute(text(f"DELETE FROM {table} WHERE ticker = :t"), {"t": TICKER})
            print(f"  deleted {result.rowcount} row(s) from {table}")
        result = conn.execute(text("DELETE FROM companies WHERE ticker = :t"), {"t": TICKER})
        print(f"  deleted {result.rowcount} row(s) from companies")
    print(f"\nDone. '{TICKER}' fully removed. Companies table should now have 221 rows "
          f"(was 222). Re-run python3 -m scripts.export_rag_missing_tickers to confirm "
          f"the 'Total tracked companies' count updates.")
