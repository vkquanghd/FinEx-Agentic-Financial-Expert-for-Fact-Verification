"""Batch-run vietstock_etl.py over EVERY file already sitting in
data/raw/vietstock_exports/ - discovered 2026-07-20 that ~84
"Bao-cao-tai-chinh" + ~84 "Chi-so-tai-chinh" files were already downloaded
(2026-07-14 to 2026-07-18) but only 14 distinct tickers ever actually made
it into financial_line_items/financial_ratios (per notebooks/
01_data_inventory_eda.ipynb's finding) - meaning most of these files were
never run through the ETL at all. vietstock_etl.py's loaders are safe to
call repeatedly (UPSERT + auto-detects ticker from the file itself, skips
anything not in `companies`), so this just needs to iterate the directory.

Usage: python3 -m scripts.batch_ingest_vietstock
"""
from pathlib import Path

from src.ingestion.vietstock_etl import load_bao_cao_tai_chinh, load_chi_so_tai_chinh

EXPORT_DIR = Path("data/raw/vietstock_exports")

bao_cao_files = sorted(EXPORT_DIR.glob("*Bao-cao-tai-chinh*.xlsx"))
chi_so_files = sorted(EXPORT_DIR.glob("*Chi-so-tai-chinh*.xlsx"))

print(f"Found {len(bao_cao_files)} 'Bao-cao-tai-chinh' file(s), {len(chi_so_files)} 'Chi-so-tai-chinh' file(s)\n")

print("=" * 60)
print("Loading Bao-cao-tai-chinh (financial_line_items + disclosures)")
print("=" * 60)
n_ok, n_err, n_empty = 0, 0, 0
for f in bao_cao_files:
    if f.stat().st_size == 0:
        print(f"{f.name}: SKIPPED (0 bytes - failed/incomplete download)")
        n_empty += 1
        continue
    try:
        load_bao_cao_tai_chinh(f)
        n_ok += 1
    except Exception as e:
        print(f"{f.name}: ERROR - {type(e).__name__}: {e}")
        n_err += 1

print(f"\nBao-cao-tai-chinh: {n_ok} loaded, {n_err} errors, {n_empty} empty/skipped\n")

print("=" * 60)
print("Loading Chi-so-tai-chinh (financial_ratios)")
print("=" * 60)
n_ok2, n_err2, n_empty2 = 0, 0, 0
for f in chi_so_files:
    if f.stat().st_size == 0:
        print(f"{f.name}: SKIPPED (0 bytes - failed/incomplete download)")
        n_empty2 += 1
        continue
    try:
        load_chi_so_tai_chinh(f)
        n_ok2 += 1
    except Exception as e:
        print(f"{f.name}: ERROR - {type(e).__name__}: {e}")
        n_err2 += 1

print(f"\nChi-so-tai-chinh: {n_ok2} loaded, {n_err2} errors, {n_empty2} empty/skipped")
print("\nDone. Re-run scripts/list_partial_coverage_tickers.py to see the updated coverage.")
