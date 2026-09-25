"""Batch-ingest every BCTC export sitting in data/raw/vietstock_exports/.

src/ingestion/vietstock_etl.py only loads ONE file at a time and needs to be
told which of the 2 formats it is (bao_cao_tai_chinh / chi_so_tai_chinh).
This wrapper infers the format from Vietstock's own filename convention
("VietstockFinance_Bao-cao-tai-chinh_*.xlsx" / "..._Chi-so-tai-chinh_*.xlsx")
and loads everything in the folder in one call - built so the Airflow
refresh DAG (infra/airflow/dags/thesis_refresh_dag.py) has a single command
to run instead of looping over files itself.

Both underlying loaders are UPSERTS, so re-running on files already loaded
is safe (values just get overwritten with the same numbers) - this is what
makes it OK to run on a schedule against a folder that accumulates files
over time rather than only ever seeing brand-new ones.

Usage:
    python scripts/ingest_bctc_folder.py [folder]
    (folder defaults to data/raw/vietstock_exports/)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingestion.vietstock_etl import load_bao_cao_tai_chinh, load_chi_so_tai_chinh

DEFAULT_DIR = Path(__file__).resolve().parents[1] / "data/raw/vietstock_exports"


def ingest_folder(folder: Path):
    files = sorted(p for p in folder.glob("*.xlsx") if not p.name.startswith("~"))
    n_bctc = n_ratio = n_skipped = 0
    for path in files:
        name_lower = path.name.lower()
        try:
            if "bao-cao-tai-chinh" in name_lower:
                load_bao_cao_tai_chinh(path)
                n_bctc += 1
            elif "chi-so-tai-chinh" in name_lower:
                load_chi_so_tai_chinh(path)
                n_ratio += 1
            else:
                print(f"{path.name}: SKIPPED - filename doesn't match "
                      f"'Bao-cao-tai-chinh' or 'Chi-so-tai-chinh' pattern, "
                      f"don't know which loader to use")
                n_skipped += 1
        except Exception as e:  # keep going on a bad file, same policy as macro_etl.py
            print(f"{path.name}: SKIPPED - unexpected error: {e}")
            n_skipped += 1
    print(f"\nDone: {n_bctc} bao_cao_tai_chinh + {n_ratio} chi_so_tai_chinh loaded, "
          f"{n_skipped} skipped, out of {len(files)} files in {folder}")
    return n_skipped == 0


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DIR
    ok = ingest_folder(target)
    sys.exit(0 if ok else 1)
