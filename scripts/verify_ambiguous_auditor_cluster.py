"""One-off (2026-07-22): a real run_eval.py pass over the 68-item RAG
faithfulness set scored 23 items as 0.0. Spot-checking the actual generated
answers (not just the score) showed most look like genuine correct answers
using the auditor's full legal name instead of the short code stored in
`disclosures.auditor` (e.g. ground truth "EY" vs generated "Ernst & Young
Việt Nam") - a false negative of the substring-match metric, not a system
failure.

But one cluster doesn't cleanly resolve that way: AVG (ground truth "IAV")
and CVN/CAP/CMD (ground truth "NVT") were ALL answered by the RAG pipeline
as "CÔNG TY TNHH KIỂM TOÁN NHÂN TÂM VIỆT" - which doesn't obviously match
either "IAV" or "NVT" as an abbreviation. Also checking D11 (ground truth
"A&C"), which got a garbled non-answer ("M TOÁN ĐỘC LẬP"), and BRR/APSC
(ground truth "IAV", answered with two DIFFERENT company names from each
other and from AVG's "Nhân Tâm Việt" answer) - these three needed a
different auditor than AVG even though the benchmark's `expected_keywords`
say the same "IAV" for all three.

This script pulls the REAL disclosures.auditor value from MySQL for this
whole cluster (not re-deriving it from what the LLM said, and not from the
original sampling script's cached output) - the only way to tell whether
this is a metric-formatting issue, a genuine RAG retrieval error (wrong
company's document surfaced), or a benchmark ground-truth sampling mistake,
without guessing.

Usage: python3 -m scripts.verify_ambiguous_auditor_cluster
"""
import pandas as pd

from src.core.db import engine

TICKERS = ["AVG", "CVN", "CAP", "CMD", "BRR", "APSC", "D11", "AAV"]

ticker_list = ",".join(f"'{t}'" for t in TICKERS)
df = pd.read_sql(
    f"""
    SELECT ticker, period, fiscal_year, auditor
    FROM disclosures
    WHERE ticker IN ({ticker_list}) AND period = 'YEAR' AND auditor IS NOT NULL
    ORDER BY ticker, fiscal_year DESC
    """,
    engine,
)

print(f"{len(df)} row(s) found for {df['ticker'].nunique()}/{len(TICKERS)} ticker(s)\n")
for ticker in TICKERS:
    sub = df[df["ticker"] == ticker]
    if sub.empty:
        print(f"{ticker}: NO ROWS in disclosures (benchmark item may have sampled from a stale export)")
        continue
    for _, row in sub.iterrows():
        print(f"{ticker}  {row['fiscal_year']}  auditor='{row['auditor']}'")
