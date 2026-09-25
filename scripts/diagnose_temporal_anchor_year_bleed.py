"""One-off (2026-07-22): verify_ambiguous_auditor_cluster.py confirmed 3 real
cases (AVG, CMD, AAV) where the RAG pipeline answered a "Công ty kiểm toán
của X năm 2025 là ai?" question with the CORRECT auditor for 2024 - but
mislabeled it in the answer text as "Theo BCTC năm 2025...". This is a
genuine candidate bug in Temporal Anchor Filtering (RQ4), not a benchmark/
metric issue like the other false-negatives found in the same batch.

Two competing explanations, and this script is the only way to tell them
apart without guessing:
  (a) DATA GAP: the 2025 audited BCTC simply isn't in disclosure_chunks yet
      for these 3 tickers (TiniX/PDF corpus coverage), so
      apply_temporal_anchor()'s documented fallback ("fall back to the
      unfiltered pool rather than returning nothing if that year isn't
      present" - see rag_pipeline.py) correctly used whatever it had
      (2024), and the bug is narrower: the model's own year-labeling in the
      answer doesn't reflect the actual chunk year it was given, even
      though generation.py's SYSTEM_PROMPT rule 5 explicitly requires it to.
  (b) RETRIEVAL BUG: 2025 chunks DO exist, but temporal_anchor scoring
      still let 2024 chunks win, which would be a real regression in the
      filtering logic itself.

This queries disclosure_chunks directly (source_file naming encodes the
year, same extract_year() regex vector_store.py already uses) for the 3
tickers, to see which years actually exist before deciding which fix (if
either) is warranted.

Usage: python3 -m scripts.diagnose_temporal_anchor_year_bleed
"""
import re

import pandas as pd
from sqlalchemy import create_engine

from src.core import config

PG_URL = (
    f"postgresql+psycopg2://{config.PGVECTOR_USER}:{config.PGVECTOR_PASSWORD}"
    f"@{config.PGVECTOR_HOST}:{config.PGVECTOR_PORT}/{config.PGVECTOR_DB}"
)
pg_engine = create_engine(PG_URL)

TICKERS = ["AVG", "CMD", "AAV"]

YEAR_RE = re.compile(r"(?<!\d)(20\d{2})(?!\d)")


def extract_year(source_file: str):
    m = YEAR_RE.search(source_file or "")
    return int(m.group(1)) if m else None


ticker_list = ",".join(f"'{t}'" for t in TICKERS)
df = pd.read_sql(
    f"""
    SELECT ticker, source_file, COUNT(*) AS n_chunks
    FROM disclosure_chunks
    WHERE ticker IN ({ticker_list})
    GROUP BY ticker, source_file
    ORDER BY ticker, source_file
    """,
    pg_engine,
)

df["year"] = df["source_file"].apply(extract_year)
print(f"{len(df)} distinct (ticker, source_file) row(s) for {df['ticker'].nunique()}/{len(TICKERS)} ticker(s)\n")

for ticker in TICKERS:
    sub = df[df["ticker"] == ticker]
    if sub.empty:
        print(f"{ticker}: NO CHUNKS AT ALL in disclosure_chunks")
        continue
    years = sorted(sub["year"].dropna().unique().astype(int).tolist())
    print(f"{ticker}: years present = {years}  (from {sub['source_file'].nunique()} source file(s), {sub['n_chunks'].sum()} chunk(s) total)")
    for _, row in sub.iterrows():
        print(f"    {row['source_file']}  year={row['year']}  n_chunks={row['n_chunks']}")
    print()
