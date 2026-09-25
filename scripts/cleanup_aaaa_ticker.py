"""One-off (2026-07-23): "AAAA" has been flagged as a bogus ticker across
multiple sessions now (scripts/download_tinix_batch.py's docstring,
docs/PROGRESS.md) - confirmed via web search to not be a real Vietnamese
listing anywhere, unlike the genuine "AAA" (An Phat Bioplastics). It
slipped into `companies` as an auto-registered placeholder from a
malformed cell in one Vietstock export file
(data/raw/vietstock_exports/VietstockFinance_Bao-cao-tai-chinh_20260714-190643.xlsx).
Flagged as "worth deleting later" three separate times, never actually
done - this does it, but checks every table that could reference it FIRST
rather than deleting blind, since a wrong delete on a live project DB isn't
reversible without a backup.

Checks (real queries, not assumptions):
  - MySQL: financial_line_items, financial_ratios, disclosures,
    company_price_snapshot - any row with ticker='AAAA'?
  - Postgres/pgvector: disclosure_chunks - any row with ticker='AAAA'?
If ALL of the above are 0, deletes the `companies` row. If ANY is nonzero,
prints what was found and does NOT delete - a nonzero count would mean
AAAA has real ingested data somehow, which would change the conclusion
(worth a second look, not a silent delete).

Usage: python3 -m scripts.cleanup_aaaa_ticker
"""
from sqlalchemy import text

from src.core.db import engine as mysql_engine
from src.core import config

TICKER = "AAAA"

MYSQL_TABLES = ["financial_line_items", "financial_ratios", "disclosures", "company_price_snapshot"]

print(f"=== Checking references to ticker='{TICKER}' before any delete ===\n")

mysql_counts = {}
with mysql_engine.connect() as conn:
    for t in MYSQL_TABLES:
        n = conn.execute(text(f"SELECT COUNT(*) FROM {t} WHERE ticker = :t"), {"t": TICKER}).scalar()
        mysql_counts[t] = n
        print(f"  MySQL {t}: {n} row(s)")

from sqlalchemy import create_engine
PG_URL = (
    f"postgresql+psycopg2://{config.PGVECTOR_USER}:{config.PGVECTOR_PASSWORD}"
    f"@{config.PGVECTOR_HOST}:{config.PGVECTOR_PORT}/{config.PGVECTOR_DB}"
)
pg_engine = create_engine(PG_URL)
with pg_engine.connect() as conn:
    n_chunks = conn.execute(text("SELECT COUNT(*) FROM disclosure_chunks WHERE ticker = :t"), {"t": TICKER}).scalar()
print(f"  Postgres disclosure_chunks: {n_chunks} row(s)")

total = sum(mysql_counts.values()) + n_chunks
print()

if total == 0:
    print(f"No referencing data anywhere. Deleting '{TICKER}' from companies...")
    with mysql_engine.begin() as conn:
        result = conn.execute(text("DELETE FROM companies WHERE ticker = :t"), {"t": TICKER})
        print(f"Deleted {result.rowcount} row(s) from companies.")
else:
    print(f"STOPPED - found {total} referencing row(s) total, NOT deleting. "
          f"'{TICKER}' has real data somewhere - re-check before deciding what to do.")
