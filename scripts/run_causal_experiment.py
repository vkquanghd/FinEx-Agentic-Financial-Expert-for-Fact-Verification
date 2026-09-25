"""Runs estimate_ate()/estimate_ate_dml() (src/causal/treatment_effect.py)
and logs the FULL experiment - inputs (treatment/outcome/covariates,
estimator, nuisance model, random seed, bootstrap iterations, code commit,
which dataset it ran against) and outputs (n, point estimate, CI,
confidence label) - into `causal_experiment_runs` (see
infra/postgres/migrations/009_causal_experiment_runs.sql).

WHY this script exists given treatment_effect.py's functions already
return a full result dict: that dict lives only in whatever process
called it (a notebook cell, a REPL, this script's own stdout) - nothing
records WHICH exact estimator/seed/nuisance-model/dataset combination
produced a given number once that process exits. Two runs of
estimate_ate() months apart, even against the identical
--snapshot-version, are otherwise indistinguishable in the thesis
write-up. This script is the thin DB-writing layer around the pure
estimator functions - same split as scripts/build_predictive_features.py
around src/predictive/feature_builder.py, and
scripts/build_causal_panel_snapshot.py around
src/causal/panel_builder.py: the src/ module stays free of DB imports
(besides its own live-view queries), the scripts/ file owns persistence.

Unlike causal_panel_snapshot (immutable, ON CONFLICT DO NOTHING) and
predictive_features (target columns only updatable), causal_experiment_
runs is a pure APPEND-ONLY LOG - every call to this script inserts a NEW
row, including failed/refused runs (ok=FALSE + error_message), so "how
many times, and how, has this treatment/outcome pair been estimated" is
answerable from the table itself.

Usage:
    python3 -m scripts.run_causal_experiment
    python3 -m scripts.run_causal_experiment --treatment leverage --outcome net_margin --estimator both
    python3 -m scripts.run_causal_experiment --snapshot-version panel_v1_20260729 --estimator t_learner
"""
import argparse
import subprocess
from datetime import date

from sqlalchemy import text

from src.causal.panel_builder import load_panel_from_snapshot
from src.causal.treatment_effect import MIN_N_EXPLORATORY, estimate_ate, estimate_ate_dml
from src.core.db import get_session

INSERT_RUN = text("""
    INSERT INTO causal_experiment_runs
        (snapshot_version, source_dataset_version, treatment_name, outcome_name, covariates,
         estimator, nuisance_model, random_seed, bootstrap_iterations, min_sample_threshold,
         code_commit, n, n_treated, n_control, point_estimate, ci_low, ci_high,
         confidence_label, ok, error_message)
    VALUES
        (:snapshot_version, :source_dataset_version, :treatment_name, :outcome_name, :covariates,
         :estimator, :nuisance_model, :random_seed, :bootstrap_iterations, :min_sample_threshold,
         :code_commit, :n, :n_treated, :n_control, :point_estimate, :ci_low, :ci_high,
         :confidence_label, :ok, :error_message)
""")


def _git_commit() -> str:
    """Best-effort short git SHA of the current working tree - None if git
    isn't installed, this isn't a git checkout, or any other failure. Same
    don't-fabricate-just-say-what's-missing pattern as the rest of this
    project's optional metadata (e.g. estimate_ate_dml()'s econml-not-
    installed message)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, check=True,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def _row_for_t_learner(result: dict, seed: int, n_bootstrap: int) -> dict:
    return {
        "estimator": "t_learner",
        "nuisance_model": "Ridge(alpha=1.0)",
        "random_seed": seed,
        "bootstrap_iterations": n_bootstrap,
        "n": result.get("n"),
        "n_treated": result.get("n_treated"),
        "n_control": result.get("n_control"),
        "point_estimate": result.get("ate_t_learner"),
        "ci_low": result["ate_95ci"][0] if result.get("ate_95ci") else None,
        "ci_high": result["ate_95ci"][1] if result.get("ate_95ci") else None,
        "confidence_label": result.get("confidence"),
        "ok": bool(result.get("ok")),
        "error_message": result.get("error"),
    }


def _row_for_dml(result: dict) -> dict:
    return {
        "estimator": "dml",
        "nuisance_model": "RidgeCV/RidgeCV",
        "random_seed": 42,  # hardcoded inside estimate_ate_dml() itself, not caller-configurable today
        "bootstrap_iterations": None,  # analytic CI (est.ate_interval), no bootstrap
        "n": result.get("n"),
        "n_treated": None,  # DML doesn't split treated/control - continuous treatment
        "n_control": None,
        "point_estimate": result.get("ate_dml_marginal"),
        "ci_low": result["ate_dml_marginal_95ci"][0] if result.get("ate_dml_marginal_95ci") else None,
        "ci_high": result["ate_dml_marginal_95ci"][1] if result.get("ate_dml_marginal_95ci") else None,
        "confidence_label": result.get("confidence"),
        "ok": bool(result.get("ok")),
        "error_message": result.get("error"),
    }


def run_experiment(
    treatment: str = "leverage",
    outcome: str = "net_margin",
    covariates: tuple = ("sector", "fiscal_year"),
    estimator: str = "t_learner",
    snapshot_version: str = None,
    n_bootstrap: int = 200,
    seed: int = 42,
) -> list:
    """Runs the requested estimator(s) and writes ONE row per estimator
    into causal_experiment_runs (logs failed/refused attempts too).
    Returns the list of row-dicts written, for callers (tests, a future
    notebook) to assert on without re-querying the DB."""
    if estimator not in ("t_learner", "dml", "both"):
        raise ValueError(f"estimator must be 't_learner', 'dml', or 'both' - got {estimator!r}")

    if snapshot_version:
        panel = load_panel_from_snapshot(snapshot_version)
        source_dataset_version = snapshot_version
    else:
        panel = None  # estimate_ate()/estimate_ate_dml() call build_panel() live themselves
        source_dataset_version = f"live_{date.today():%Y%m%d}"

    common = {
        "snapshot_version": snapshot_version,
        "source_dataset_version": source_dataset_version,
        "treatment_name": treatment,
        "outcome_name": outcome,
        "covariates": ",".join(covariates),
        "min_sample_threshold": MIN_N_EXPLORATORY,
        "code_commit": _git_commit(),
    }

    rows = []
    if estimator in ("t_learner", "both"):
        result = estimate_ate(treatment_col=treatment, outcome_col=outcome, covariates=covariates,
                               panel=panel, n_bootstrap=n_bootstrap, seed=seed)
        rows.append({**common, **_row_for_t_learner(result, seed, n_bootstrap)})
    if estimator in ("dml", "both"):
        result = estimate_ate_dml(treatment_col=treatment, outcome_col=outcome, covariates=covariates, panel=panel)
        rows.append({**common, **_row_for_dml(result)})

    with get_session() as session:
        for row in rows:
            session.execute(INSERT_RUN, row)

    for row in rows:
        status = "OK" if row["ok"] else f"REFUSED ({row['error_message']})"
        print(f"[{row['estimator']}] {treatment} -> {outcome} | n={row['n']} | "
              f"point_estimate={row['point_estimate']} | {status}")

    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--treatment", default="leverage")
    ap.add_argument("--outcome", default="net_margin")
    ap.add_argument("--covariates", default="sector,fiscal_year",
                     help="Comma-separated covariate column names (default: 'sector,fiscal_year').")
    ap.add_argument("--estimator", choices=["t_learner", "dml", "both"], default="t_learner")
    ap.add_argument("--snapshot-version", default=None,
                     help="If given, loads panel from causal_panel_snapshot instead of calling "
                          "build_panel() live (see src/causal/panel_builder.py's load_panel_from_snapshot()).")
    ap.add_argument("--n-bootstrap", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    covariates = tuple(c.strip() for c in args.covariates.split(",") if c.strip())
    run_experiment(
        treatment=args.treatment, outcome=args.outcome, covariates=covariates,
        estimator=args.estimator, snapshot_version=args.snapshot_version,
        n_bootstrap=args.n_bootstrap, seed=args.seed,
    )


if __name__ == "__main__":
    main()
