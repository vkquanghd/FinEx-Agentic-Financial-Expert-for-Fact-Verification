"""One-off (2026-07-20): after batch_ingest_vietstock.py's second run (214
Chi-so-tai-chinh files loaded), DAG coverage jumped dramatically. This prints
the EXACT current coverage summary + a ready-to-paste Python list literal of
every ticker still missing RAG, for scripts/download_tinix_batch.py's
TICKERS list - avoids manually transcribing 190+ tickers from chat output
(error-prone, exactly what we've been avoiding all session).

Usage: python3 -m scripts.export_rag_missing_tickers
"""
import pandas as pd
from src.core.db import engine
from src.core import config as cfg
from sqlalchemy import create_engine

pg_engine = create_engine(
    f"postgresql+psycopg2://{cfg.PGVECTOR_USER}:{cfg.PGVECTOR_PASSWORD}"
    f"@{cfg.PGVECTOR_HOST}:{cfg.PGVECTOR_PORT}/{cfg.PGVECTOR_DB}"
)

companies = pd.read_sql("SELECT ticker FROM companies", engine)
fli = pd.read_sql("SELECT DISTINCT ticker FROM financial_line_items", engine)
fratios = pd.read_sql("SELECT DISTINCT ticker FROM financial_ratios", engine)
prices = pd.read_sql("SELECT DISTINCT ticker FROM company_price_snapshot", engine)
chunks = pd.read_sql("SELECT DISTINCT ticker FROM disclosure_chunks", pg_engine)

n_total = len(companies)
n_dag = companies["ticker"].isin(fli["ticker"]).sum()
n_ratios = companies["ticker"].isin(fratios["ticker"]).sum()
n_price = companies["ticker"].isin(prices["ticker"]).sum()
n_rag = companies["ticker"].isin(chunks["ticker"]).sum()

print(f"Total tracked companies: {n_total}")
print(f"  has_dag:    {n_dag}/{n_total}")
print(f"  has_ratios: {n_ratios}/{n_total}")
print(f"  has_price:  {n_price}/{n_total}")
print(f"  has_rag:    {n_rag}/{n_total}")

missing_rag = sorted(set(companies["ticker"]) - set(chunks["ticker"]))
print(f"\n{len(missing_rag)} ticker(s) still missing RAG (disclosure_chunks).")
print("Paste this into scripts/download_tinix_batch.py's TICKERS list:\n")
print("TICKERS = [")
for i in range(0, len(missing_rag), 10):
    row = missing_rag[i:i + 10]
    print("    " + ", ".join(f'"{t}"' for t in row) + ",")
print("]")
