"""One-off (2026-07-23): scripts/cleanup_aaaa_ticker.py refused to delete
'AAAA' from `companies` because it found real referencing data (297
financial_line_items, 33 financial_ratios, 3 disclosures) - contradicting
the earlier assumption (download_tinix_batch.py's docstring,
docs/PROGRESS.md) that AAAA was just an empty placeholder row with no real
data attached. That assumption was wrong, or incomplete - this investigates
which, before deciding anything:

  1. What do AAAA's actual financial_line_items/financial_ratios/
     disclosures rows contain (fiscal years, item names/values, any text
     hints of a real company)?
  2. Given the known root cause (a malformed cell in
     data/raw/vietstock_exports/VietstockFinance_Bao-cao-tai-chinh_20260714-190643.xlsx),
     the leading hypothesis is that this is actually AAA's (An Phat
     Bioplastics) real data, misfiled under "AAAA" by an off-by-one/
     extra-character parsing bug - checks this directly by comparing
     AAAA's rows against AAA's rows for the same fiscal years.

Usage: python3 -m scripts.investigate_aaaa_data
"""
import pandas as pd
from sqlalchemy import text

from src.core.db import engine

print("=== AAAA's financial_line_items (sample) ===")
df_items = pd.read_sql(
    "SELECT fiscal_year, period, item_name, value FROM financial_line_items "
    "WHERE ticker = 'AAAA' ORDER BY fiscal_year, item_name LIMIT 30",
    engine,
)
print(df_items.to_string(index=False))

print(f"\n=== AAAA's financial_line_items: fiscal years present ===")
df_years = pd.read_sql(
    "SELECT fiscal_year, period, COUNT(*) AS n FROM financial_line_items "
    "WHERE ticker = 'AAAA' GROUP BY fiscal_year, period ORDER BY fiscal_year",
    engine,
)
print(df_years.to_string(index=False))

print("\n=== AAAA's disclosures rows ===")
df_disc = pd.read_sql(
    "SELECT * FROM disclosures WHERE ticker = 'AAAA'",
    engine,
)
print(df_disc.to_string(index=False))

print("\n=== AAAA's financial_ratios (sample) ===")
df_ratios = pd.read_sql(
    "SELECT fiscal_year, period, ratio_name, value FROM financial_ratios "
    "WHERE ticker = 'AAAA' ORDER BY fiscal_year, ratio_name LIMIT 20",
    engine,
)
print(df_ratios.to_string(index=False))

print("\n=== Does AAA (the real ticker) have data for the SAME fiscal years? ===")
years = df_years["fiscal_year"].unique().tolist()
if years:
    years_list = ",".join(str(y) for y in years)
    df_aaa = pd.read_sql(
        f"SELECT fiscal_year, period, COUNT(*) AS n FROM financial_line_items "
        f"WHERE ticker = 'AAA' AND fiscal_year IN ({years_list}) GROUP BY fiscal_year, period ORDER BY fiscal_year",
        engine,
    )
    print(df_aaa.to_string(index=False))

    print("\n=== Value-level comparison: AAAA vs AAA for one shared (fiscal_year, item_name) sample ===")
    df_compare = pd.read_sql(
        f"""
        SELECT a.fiscal_year, a.item_name, a.value AS aaaa_value, b.value AS aaa_value
        FROM financial_line_items a
        JOIN financial_line_items b
          ON a.item_name = b.item_name AND a.fiscal_year = b.fiscal_year AND a.period = b.period
        WHERE a.ticker = 'AAAA' AND b.ticker = 'AAA'
        LIMIT 20
        """,
        engine,
    )
    print(df_compare.to_string(index=False))
    if not df_compare.empty:
        match_rate = (df_compare["aaaa_value"] == df_compare["aaa_value"]).mean()
        print(f"\nExact value match rate (AAAA vs AAA, same item/year): {match_rate:.0%}")
else:
    print("  No fiscal years found for AAAA - nothing to compare.")

print("\n=== companies row for AAAA and AAA (side by side) ===")
df_companies = pd.read_sql(
    "SELECT * FROM companies WHERE ticker IN ('AAAA', 'AAA')",
    engine,
)
print(df_companies.to_string(index=False))
