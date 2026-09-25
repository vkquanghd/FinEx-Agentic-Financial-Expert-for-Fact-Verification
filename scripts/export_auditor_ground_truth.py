"""One-off (2026-07-21): exports real (ticker, fiscal_year, auditor) ground
truth from the `disclosures` table - populated directly from Vietstock's
Bao-cao-tai-chinh row 10 ("auditor") by
src.ingestion.vietstock_etl.parse_bao_cao_tai_chinh() - to build a larger,
systematic set of "who is the auditor of X in year Y" RAG benchmark items.

Why from `disclosures` and not hand-picked: the existing benchmark
(data/eval/benchmark.jsonl) has exactly ONE auditor item
(rag_pnj_auditor_2024), manually verified against one real run. Guessing
auditor names for more tickers would risk building a benchmark with WRONG
expected answers baked in - `disclosures.auditor` is the actual ground
truth Vietstock itself reports, no guessing needed.

Filters to YEAR-period, non-null auditor, most recent fiscal year per
ticker (freshest data, most likely to also be in the RAG/TiniX corpus
since TiniX skews toward recent filings) - then samples across DIFFERENT
auditor names so the benchmark isn't 200 near-duplicate "PwC" questions,
before printing ready-to-paste JSONL lines.

Usage: python3 -m scripts.export_auditor_ground_truth
"""
import json
import random

import pandas as pd

from src.core.db import engine

df = pd.read_sql(
    """
    SELECT ticker, fiscal_year, auditor
    FROM disclosures
    WHERE period = 'YEAR' AND auditor IS NOT NULL AND auditor != ''
    ORDER BY ticker, fiscal_year DESC
    """,
    engine,
)
print(f"Total (ticker, year, auditor) rows with period=YEAR: {len(df)}")

latest = df.loc[df.groupby("ticker")["fiscal_year"].idxmax()]
print(f"Tickers with at least one YEAR-period auditor value: {len(latest)}")

print("\nDistinct auditor names found (sanity check before sampling):")
print(latest["auditor"].value_counts().to_string())

random.seed(42)
by_auditor = {}
for _, row in latest.iterrows():
    by_auditor.setdefault(row["auditor"], []).append(row)

SAMPLE_PER_AUDITOR = 3
sample_rows = []
for auditor, rows in by_auditor.items():
    random.shuffle(rows)
    sample_rows.extend(rows[:SAMPLE_PER_AUDITOR])

print(f"\n{len(sample_rows)} sampled item(s) (up to {SAMPLE_PER_AUDITOR} per distinct auditor name) "
      f"across {len(by_auditor)} distinct auditor(s).")
print("Paste these lines into data/eval/benchmark.jsonl:\n")

for row in sample_rows:
    ticker, year, auditor = row["ticker"], int(row["fiscal_year"]), row["auditor"]
    item = {
        "id": f"rag_{ticker.lower()}_auditor_{year}",
        "category": "rag_faithfulness",
        "question": f"Công ty kiểm toán của {ticker} năm {year} là ai?",
        "ticker": ticker,
        "expected_branch": "rag",
        "expected_keywords": [auditor],
        "expected_year": year,
        "notes": f"Ground truth from disclosures.auditor (Vietstock Bao-cao-tai-chinh row 10, "
                 f"loaded by vietstock_etl.py) - NOT yet verified against a real RAG answer, "
                 f"only that this is what the source document itself reports. Auto-generated "
                 f"2026-07-21 by scripts/export_auditor_ground_truth.py.",
    }
    print(json.dumps(item, ensure_ascii=False))
