"""One-off (2026-07-21): same idea as export_auditor_ground_truth.py, applied
to the "whatif_exact_match" benchmark category - `data/eval/benchmark.jsonl`
had exactly ONE such item (wf_ctd_cogs5_2025, CTD only). This picks a
diverse sample of OTHER tickers with valid revenue+COGS data and computes
their real "COGS +5%" scenario by calling the ACTUAL
`src.structured.calculator.sensitivity_what_if()` function directly - not
reimplementing the arithmetic separately, so there is zero risk of the
benchmark's "expected" values drifting from what the real code actually
computes. This tests the SAME thing the existing item already does
(ceteris-paribus gross-profit sensitivity), just across many more tickers,
which matters because different companies can hit edge cases a single
ticker won't (e.g. COGS >= revenue already, very small revenue bases,
negative gross profit even before the shock).

Note on what this is (and isn't) testing: since the "ground truth" here IS
the function's own output, this is a REGRESSION test (did the calculator's
arithmetic change unexpectedly across a future code edit / did the Router
correctly dispatch these questions to the whatif branch), not an
independent verification that the accounting math itself is correct - that
independent check was already done once by hand for the CTD example (see
its "notes" field in benchmark.jsonl). Combined with a wide ticker sample,
this is still valuable: it catches Router misclassification and data
plumbing bugs (wrong ms_code, wrong period) that a single fixed example
can't surface, even though it can't catch "the formula itself is wrong in
a way that's consistent across all tickers."

Usage: python3 -m scripts.export_whatif_ground_truth
"""
import json
import random

import pandas as pd

from src.core.db import engine
from src.structured.calculator import Period, sensitivity_what_if

df = pd.read_sql(
    """
    SELECT ticker, fiscal_year, ms_code, value
    FROM financial_line_items
    WHERE statement_type = 'income_statement' AND period = 'YEAR' AND ms_code IN ('10', '11')
    """,
    engine,
)
print(f"Total income_statement/YEAR revenue+COGS rows: {len(df)}")

pivot = df.pivot_table(index=["ticker", "fiscal_year"], columns="ms_code", values="value", aggfunc="first")
pivot = pivot.rename(columns={"10": "revenue", "11": "cogs"}).dropna(subset=["revenue", "cogs"])
pivot = pivot[(pivot["revenue"] != 0)]
pivot = pivot.reset_index()
print(f"(ticker, year) pairs with both non-null revenue+COGS, revenue!=0: {len(pivot)}")

latest = pivot.loc[pivot.groupby("ticker")["fiscal_year"].idxmax()]
print(f"Distinct tickers with at least one valid YEAR: {len(latest)}")

EXCLUDE_TICKERS = {"CTD"}  # already in benchmark.jsonl (wf_ctd_cogs5_2025)
latest = latest[~latest["ticker"].isin(EXCLUDE_TICKERS)]

random.seed(42)
SAMPLE_N = 25
sample = latest.sample(n=min(SAMPLE_N, len(latest)), random_state=42)

print(f"\n{len(sample)} sampled ticker(s). Calling the REAL sensitivity_what_if() "
      f"for each (not reimplementing the math) - paste the printed lines into "
      f"data/eval/benchmark.jsonl:\n")

n_ok = 0
for _, row in sample.iterrows():
    ticker, year = row["ticker"], int(row["fiscal_year"])
    result = sensitivity_what_if(ticker, Period(year, "YEAR"), cogs_pct_change=5.0)
    if not result["ok"]:
        print(f"# SKIPPED {ticker}/{year}: {result.get('error')}")
        continue
    item = {
        "id": f"wf_{ticker.lower()}_cogs5_{year}",
        "category": "whatif_exact_match",
        "question": f"Nếu COGS của {ticker} tăng 5% trong {year}, lợi nhuận gộp sẽ ra sao?",
        "ticker": ticker,
        "expected_branch": "whatif",
        "expected_fields": {
            "revenue_before": result["revenue_before"],
            "cogs_before": result["cogs_before"],
            "cogs_pct_change": 5.0,
            "gross_profit_before": result["gross_profit_before"],
            "gross_profit_after": result["gross_profit_after"],
        },
        "tolerance_pct": 0.5,
        "notes": f"Expected values computed by calling the real "
                 f"src.structured.calculator.sensitivity_what_if() directly (not reimplemented) "
                 f"2026-07-21 - regression test, not an independent math verification (see "
                 f"scripts/export_whatif_ground_truth.py's module docstring). "
                 f"gross_profit_before={result['gross_profit_before']:.0f}.",
    }
    print(json.dumps(item, ensure_ascii=False))
    n_ok += 1

print(f"\n{n_ok}/{len(sample)} produced a valid item (rest skipped - see reasons above).")
