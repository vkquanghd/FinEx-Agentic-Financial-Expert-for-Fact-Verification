"""One-off (2026-07-20): lists the tickers sitting at 2/4 or 3/4 modality
coverage (DAG / Ratios / Price / RAG) from the data-inventory EDA notebook's
finding that only 1/216 tickers (CTD) has full coverage. These are the
cheapest targets to bring to full coverage - each is missing just 1-2
modalities, not starting from zero. Used to decide which tickers to
prioritize when expanding TiniX RAG downloads and/or manual Vietstock DAG
exports (see chat discussion 2026-07-20).

Usage: python3 -m scripts.list_partial_coverage_tickers
Delete this file once the question is answered.
"""
import pandas as pd
from src.core.db import engine
from src.core import config as cfg
from sqlalchemy import create_engine

pg_engine = create_engine(
    f"postgresql+psycopg2://{cfg.PGVECTOR_USER}:{cfg.PGVECTOR_PASSWORD}"
    f"@{cfg.PGVECTOR_HOST}:{cfg.PGVECTOR_PORT}/{cfg.PGVECTOR_DB}"
)

companies = pd.read_sql("SELECT ticker, company_name, sector, exchange FROM companies", engine)
fli = pd.read_sql("SELECT DISTINCT ticker FROM financial_line_items", engine)
fratios = pd.read_sql("SELECT DISTINCT ticker FROM financial_ratios", engine)
prices = pd.read_sql("SELECT DISTINCT ticker FROM company_price_snapshot", engine)
chunks = pd.read_sql("SELECT DISTINCT ticker FROM disclosure_chunks", pg_engine)

cov = companies.copy()
cov["has_dag"] = cov["ticker"].isin(fli["ticker"])
cov["has_ratios"] = cov["ticker"].isin(fratios["ticker"])
cov["has_price"] = cov["ticker"].isin(prices["ticker"])
cov["has_rag"] = cov["ticker"].isin(chunks["ticker"])
cov["modality_count"] = cov[["has_dag", "has_ratios", "has_price", "has_rag"]].sum(axis=1)

def missing(row):
    m = []
    if not row["has_dag"]: m.append("DAG")
    if not row["has_ratios"]: m.append("Ratios")
    if not row["has_price"]: m.append("Price")
    if not row["has_rag"]: m.append("RAG")
    return ",".join(m)

cov["missing"] = cov.apply(missing, axis=1)

print("=== 3/4 modality (missing exactly 1) ===")
sub = cov[cov["modality_count"] == 3].sort_values("ticker")
print(sub[["ticker", "company_name", "sector", "missing"]].to_string(index=False))

print("\n=== 2/4 modality (missing exactly 2) ===")
sub2 = cov[cov["modality_count"] == 2].sort_values("ticker")
print(sub2[["ticker", "company_name", "sector", "missing"]].to_string(index=False))

print(f"\nAll 14 DAG tickers: {sorted(fli['ticker'].unique().tolist())}")
print(f"All 3 RAG tickers: {sorted(chunks['ticker'].unique().tolist())}")
