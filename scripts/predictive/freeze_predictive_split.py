"""Freezes the train/validation/test split for `predictive_features` rows
into `predictive_split_manifest` (Task #18, roadmap item 4, 2026-07-30 -
see infra/postgres/migrations/011_predictive_split_manifest.sql and
docs/PROGRESS.md for the full rationale).

Split rule (time-based, exactly as specified): a row's information_cutoff
year determines its split -
    < TRAIN_END_YEAR + 1 (i.e. <= 2022)         -> train
    == VALIDATION_YEAR (2023)                    -> validation
    >= TEST_START_YEAR (2024)                     -> test
Rows with information_cutoff IS NULL (unknown disclosure date) or
eligible_for_training = FALSE (no known target_value yet) are excluded
entirely - same "unknown means excluded, not safe" convention used
throughout this codebase (see feature_builder.py's leakage-check comments).

FREEZE, not recompute: once a (ticker, fiscal_year, period, consolidation,
target_col, horizon) key is written under a given split_manifest_version,
re-running this script NEVER changes its split - only genuinely NEW keys
(a ticker/year predictive_features has never seen before under this
version) get inserted. If a previously-frozen row's information_cutoff has
since changed (e.g. a corrected disclosure_date) such that it would now
compute to a DIFFERENT split than what's frozen, this script does NOT
silently update it - it prints a loud warning and skips, unless
--allow-reassign is passed explicitly. This is the whole point of this
table existing separately from predictive_features (see the migration's
comment) - do not defeat it by making --allow-reassign the default.

Usage:
    python3 -m scripts.predictive.freeze_predictive_split
    python3 -m scripts.predictive.freeze_predictive_split --target net_margin --horizon 1
    python3 -m scripts.predictive.freeze_predictive_split --csv-out data/processed/predictive_split_manifest.csv
"""
import argparse
import csv
from pathlib import Path

from sqlalchemy import text

from src.core.db import get_session

TRAIN_END_YEAR = 2022     # information_cutoff year <= this -> train
VALIDATION_YEAR = 2023    # information_cutoff year == this -> validation
TEST_START_YEAR = 2024    # information_cutoff year >= this -> test

# Auto-derived, same pattern as CHUNKING_VERSION/INGEST_VERSION elsewhere in
# this project - change any of the three boundary years above and this
# changes automatically, so a genuinely different split rule can never be
# silently confused with the one currently frozen in the DB.
SPLIT_MANIFEST_VERSION = f"time_train_lte{TRAIN_END_YEAR}_val{VALIDATION_YEAR}_test_gte{TEST_START_YEAR}_v1"

SELECT_ELIGIBLE_FEATURES = text("""
    SELECT feature_version, ticker, fiscal_year, period, consolidation, target_col, horizon, information_cutoff
    FROM predictive_features
    WHERE feature_version = :feature_version
      AND target_col = :target_col
      AND horizon = :horizon
      AND eligible_for_training = TRUE
      AND information_cutoff IS NOT NULL
""")

SELECT_FROZEN_KEYS = text("""
    SELECT ticker, fiscal_year, period, consolidation, split, information_cutoff
    FROM predictive_split_manifest
    WHERE split_manifest_version = :split_manifest_version
      AND target_col = :target_col
      AND horizon = :horizon
""")

INSERT_SPLIT_ROW = text("""
    INSERT INTO predictive_split_manifest
        (split_manifest_version, feature_version, ticker, fiscal_year, period, consolidation,
         target_col, horizon, information_cutoff, split)
    VALUES
        (:split_manifest_version, :feature_version, :ticker, :fiscal_year, :period, :consolidation,
         :target_col, :horizon, :information_cutoff, :split)
""")

# --allow-reassign path only - explicit, separate statement so a normal run
# (no flag) can NEVER execute an UPDATE against this table, by construction.
UPDATE_SPLIT_ROW_REASSIGN = text("""
    UPDATE predictive_split_manifest
    SET split = :split, information_cutoff = :information_cutoff
    WHERE split_manifest_version = :split_manifest_version
      AND ticker = :ticker AND fiscal_year = :fiscal_year AND period = :period
      AND consolidation IS NOT DISTINCT FROM :consolidation
      AND target_col = :target_col AND horizon = :horizon
""")


def _assign_split(information_cutoff) -> str:
    year = information_cutoff.year
    if year <= TRAIN_END_YEAR:
        return "train"
    if year == VALIDATION_YEAR:
        return "validation"
    return "test"  # year >= TEST_START_YEAR by construction of the three boundaries above


def freeze_split(feature_version: str = "predfeat_v1", target_col: str = "net_margin",
                  horizon: int = 1, allow_reassign: bool = False) -> dict:
    """Returns {"n_new", "n_unchanged", "n_reassigned", "n_conflicts_skipped",
    "counts_by_split"}."""
    n_new = n_unchanged = n_reassigned = n_conflicts_skipped = 0
    counts_by_split = {"train": 0, "validation": 0, "test": 0}

    with get_session() as session:
        eligible_rows = session.execute(SELECT_ELIGIBLE_FEATURES, {
            "feature_version": feature_version, "target_col": target_col, "horizon": horizon,
        }).fetchall()
        frozen = {
            (r[0], r[1], r[2], r[3]): {"split": r[4], "information_cutoff": r[5]}
            for r in session.execute(SELECT_FROZEN_KEYS, {
                "split_manifest_version": SPLIT_MANIFEST_VERSION, "target_col": target_col, "horizon": horizon,
            }).fetchall()
        }

        for row in eligible_rows:
            _fv, ticker, fiscal_year, period, consolidation, _tc, _h, information_cutoff = row
            key = (ticker, fiscal_year, period, consolidation)
            computed_split = _assign_split(information_cutoff)

            existing = frozen.get(key)
            if existing is None:
                session.execute(INSERT_SPLIT_ROW, {
                    "split_manifest_version": SPLIT_MANIFEST_VERSION, "feature_version": feature_version,
                    "ticker": ticker, "fiscal_year": fiscal_year, "period": period, "consolidation": consolidation,
                    "target_col": target_col, "horizon": horizon,
                    "information_cutoff": information_cutoff, "split": computed_split,
                })
                n_new += 1
                counts_by_split[computed_split] += 1
                continue

            if existing["split"] == computed_split:
                n_unchanged += 1
                counts_by_split[existing["split"]] += 1
                continue

            # Frozen split disagrees with what the CURRENT information_cutoff
            # would compute - either information_cutoff changed upstream
            # (a corrected disclosure_date) or SPLIT_MANIFEST_VERSION's
            # boundary years changed without actually bumping the version
            # string (shouldn't happen - the version is auto-derived - but
            # checked here defensively anyway).
            if allow_reassign:
                session.execute(UPDATE_SPLIT_ROW_REASSIGN, {
                    "split_manifest_version": SPLIT_MANIFEST_VERSION,
                    "ticker": ticker, "fiscal_year": fiscal_year, "period": period, "consolidation": consolidation,
                    "target_col": target_col, "horizon": horizon,
                    "information_cutoff": information_cutoff, "split": computed_split,
                })
                print(f"  [REASSIGNED] {ticker} FY{fiscal_year} {period}: "
                      f"{existing['split']} -> {computed_split} (--allow-reassign)")
                n_reassigned += 1
                counts_by_split[computed_split] += 1
            else:
                print(f"  [SKIPPED - frozen] {ticker} FY{fiscal_year} {period}: frozen as "
                      f"'{existing['split']}' but current information_cutoff computes to "
                      f"'{computed_split}' - pass --allow-reassign to override this explicitly.")
                n_conflicts_skipped += 1
                counts_by_split[existing["split"]] += 1

    return {
        "split_manifest_version": SPLIT_MANIFEST_VERSION,
        "n_new": n_new, "n_unchanged": n_unchanged,
        "n_reassigned": n_reassigned, "n_conflicts_skipped": n_conflicts_skipped,
        "counts_by_split": counts_by_split,
    }


def export_csv(csv_path: Path, feature_version: str, target_col: str, horizon: int):
    """Snapshot the current frozen manifest to CSV, sorted deterministically -
    a portable, git-diffable artifact alongside the DB table (the user's
    roadmap said "freeze into a file OR table" - this script does both)."""
    with get_session() as session:
        rows = session.execute(text("""
            SELECT ticker, fiscal_year, period, consolidation, split, information_cutoff, frozen_at
            FROM predictive_split_manifest
            WHERE split_manifest_version = :v AND feature_version = :fv
              AND target_col = :tc AND horizon = :h
            ORDER BY ticker, fiscal_year, period, consolidation
        """), {"v": SPLIT_MANIFEST_VERSION, "fv": feature_version, "tc": target_col, "h": horizon}).fetchall()

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ticker", "fiscal_year", "period", "consolidation", "split", "information_cutoff", "frozen_at"])
        w.writerows(rows)
    print(f"CSV snapshot written: {csv_path} ({len(rows)} row(s))")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-version", default="predfeat_v1")
    ap.add_argument("--target", default="net_margin")
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--allow-reassign", action="store_true",
                     help="Allow this run to change a PREVIOUSLY FROZEN row's split assignment "
                          "if its information_cutoff has changed since it was frozen. Off by "
                          "default - see this script's module docstring for why.")
    ap.add_argument("--csv-out", type=Path, default=Path("data/processed/predictive_split_manifest.csv"))
    args = ap.parse_args()

    summary = freeze_split(args.feature_version, args.target, args.horizon, args.allow_reassign)
    print(f"\nsplit_manifest_version = {summary['split_manifest_version']}")
    print(f"  {summary['n_new']} new row(s) frozen, {summary['n_unchanged']} already frozen (no-op), "
          f"{summary['n_reassigned']} reassigned, {summary['n_conflicts_skipped']} conflicts skipped "
          f"(frozen split disagrees with current data - pass --allow-reassign to override).")
    for split, n in summary["counts_by_split"].items():
        print(f"  {split:12s}: {n}")
    if summary["counts_by_split"]["validation"] == 0 or summary["counts_by_split"]["test"] == 0:
        print("  WARNING: validation or test split is EMPTY - every model trained against this "
              "split_manifest_version will have nothing to validate/test against. Check that "
              "predictive_features actually has rows with information_cutoff in "
              f"{VALIDATION_YEAR}/{TEST_START_YEAR}+ before trusting any reported metric.")

    export_csv(args.csv_out, args.feature_version, args.target, args.horizon)


if __name__ == "__main__":
    main()
