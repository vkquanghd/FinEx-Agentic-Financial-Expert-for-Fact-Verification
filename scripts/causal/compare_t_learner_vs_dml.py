"""Task #21 (roadmap item 7, 2026-07-30): runs T-learner (estimate_ate) and
LinearDML (estimate_ate_dml) on the EXACT SAME panel/snapshot and the SAME
base covariates, then reports what the roadmap asked for by name: sample
size, effect estimate, confidence interval, overlap, sensitivity to
covariates, and whether the two methods agree in SIGN.

Explicit instruction from the roadmap, quoted directly: "Nếu hai phương
pháp cho kết quả khác nhau, đừng cố che giấu. Đây chính là phần error/
methodological analysis có giá trị." (if the two methods disagree, don't
try to hide it - that IS the valuable error/methodological analysis). This
script's report NEVER suppresses a sign disagreement - `sign_agreement`
below is always computed and always printed, whichever way it comes out,
and a disagreement adds a loud `disagreement_warning` field rather than
being quietly dropped.

Overlap: a T-learner/DML comparison is only meaningful if treated and
control units are actually comparable on the covariates used to adjust for
confounding - if they don't overlap (e.g. every high-leverage company also
happens to be in one sector, every low-leverage company in another), BOTH
estimators are extrapolating outside data they actually saw, and any
disagreement between them says more about non-overlap than about which
method is "more correct". Measured via a logistic regression propensity
score P(treated | covariates) - reusing src.causal.treatment_effect's own
_design_matrix() so the propensity model sees EXACTLY the same covariate
encoding both estimators use, not a re-derived approximation of it - and
reporting the fraction of units with propensity in the standard [0.1, 0.9]
common-support band (Rosenbaum & Rubin-style, not a novel threshold).

Sensitivity to covariates: re-runs BOTH estimators, via
scripts.run_causal_experiment.run_experiment() (which also logs every
variant into causal_experiment_runs - see that table), across a small fixed
set of covariate-set VARIANTS and reports how much the point estimate
moves. The default variant set's first entry intentionally matches this
script's own default `--covariates`, so the "main" result above also ends
up logged into causal_experiment_runs via the sensitivity sweep, without
needing two separate DB-writing code paths for what is really the same run.

No new DB table (per the roadmap's "stop expanding the architecture") -
reuses causal_experiment_runs (already exists, Task #11) for the individual
estimator runs, and writes ONE extra JSON file for the comparison itself
(sign agreement, overlap, sensitivity table) that has nowhere else to live.

Usage:
    python3 -m scripts.causal.compare_t_learner_vs_dml
    python3 -m scripts.causal.compare_t_learner_vs_dml --treatment leverage --outcome net_margin
    python3 -m scripts.causal.compare_t_learner_vs_dml --snapshot-version panel_v1_20260729
"""
import argparse
import json
from pathlib import Path

import numpy as np

from src.causal.panel_builder import build_panel, load_panel_from_snapshot
from src.causal.treatment_effect import _design_matrix, estimate_ate, estimate_ate_dml
from scripts.run_causal_experiment import run_experiment

BASE_COVARIATES = ("sector", "fiscal_year")
SENSITIVITY_COVARIATE_SETS = {
    "base (sector, fiscal_year)": ("sector", "fiscal_year"),
    "base + cpi": ("sector", "fiscal_year", "cpi_annual_avg"),
    "sector only": ("sector",),
    "fiscal_year only": ("fiscal_year",),
}


def _overlap_diagnostic(panel, treatment_col: str, covariates: tuple) -> dict:
    """Propensity-score common-support check - see module docstring."""
    from sklearn.linear_model import LogisticRegression

    needed = [treatment_col] + list(covariates)
    df = panel.dropna(subset=needed).reset_index(drop=True)
    if df.empty:
        return {"ok": False, "error": "Không còn dòng nào sau khi loại NaN trên treatment/covariates."}

    median = df[treatment_col].median()
    treated = (df[treatment_col] > median).to_numpy().astype(int)
    if treated.sum() < 5 or (len(treated) - treated.sum()) < 5:
        return {
            "ok": False,
            "error": f"Không đủ dữ liệu ở 1 trong 2 nhóm treated/control (n_treated={int(treated.sum())}, "
                     f"n_control={int(len(treated) - treated.sum())}) để ước lượng propensity score.",
        }

    X = _design_matrix(df, list(covariates))
    # Standardize before fitting - _design_matrix() mixes raw-scale numeric
    # covariates (e.g. fiscal_year, ~2019-2025) with 0/1 one-hot columns in
    # the SAME matrix; an unstandardized large-offset numeric column can
    # push a logistic regression toward numerical instability/poor
    # convergence regardless of solver, purely as an optimization artifact
    # unrelated to actual covariate overlap - found while testing this
    # function against a synthetic fixture including fiscal_year as a raw
    # covariate. Standardizing per-column (skipping any column with zero
    # variance, e.g. an all-zero one-hot level) keeps every covariate on a
    # comparable scale before it reaches LogisticRegression.
    col_std = X.std(axis=0)
    col_std_safe = np.where(col_std > 1e-12, col_std, 1.0)
    X_scaled = (X - X.mean(axis=0)) / col_std_safe

    clf = LogisticRegression(max_iter=1000).fit(X_scaled, treated)
    propensity = clf.predict_proba(X_scaled)[:, 1]
    in_band = (propensity >= 0.1) & (propensity <= 0.9)
    return {
        "ok": True,
        "n": len(df),
        "propensity_min": float(propensity.min()),
        "propensity_max": float(propensity.max()),
        "propensity_mean": float(propensity.mean()),
        "fraction_in_well_overlapped_band_0.1_0.9": float(in_band.mean()),
        "note": (
            "Tỷ lệ đơn vị có propensity score (P(treated|covariates)) nằm trong [0.1, 0.9] - "
            "tỷ lệ THẤP nghĩa là nhóm treated/control tách biệt rõ theo covariates, cả T-learner "
            "lẫn DML đều đang ngoại suy (extrapolate) ra ngoài vùng dữ liệu thực sự quan sát được, "
            "bất kể 2 phương pháp có đồng thuận hay không."
        ),
    }


def compare(treatment: str = "leverage", outcome: str = "net_margin", covariates: tuple = BASE_COVARIATES,
            snapshot_version: str = None, n_bootstrap: int = 200, seed: int = 42) -> dict:
    panel = load_panel_from_snapshot(snapshot_version) if snapshot_version else build_panel()

    t_result = estimate_ate(treatment_col=treatment, outcome_col=outcome, covariates=covariates,
                             panel=panel, n_bootstrap=n_bootstrap, seed=seed)
    dml_result = estimate_ate_dml(treatment_col=treatment, outcome_col=outcome, covariates=covariates, panel=panel)
    overlap = _overlap_diagnostic(panel, treatment, covariates)

    sign_agreement = None
    if t_result.get("ok") and dml_result.get("ok"):
        t_sign = np.sign(t_result["ate_t_learner"])
        dml_sign = np.sign(dml_result["translated_group_contrast"])
        sign_agreement = bool(t_sign == dml_sign) if t_sign != 0 and dml_sign != 0 else None

    sensitivity = []
    for label, cov_set in SENSITIVITY_COVARIATE_SETS.items():
        rows = run_experiment(treatment=treatment, outcome=outcome, covariates=cov_set,
                               estimator="both", snapshot_version=snapshot_version,
                               n_bootstrap=n_bootstrap, seed=seed)
        entry = {"covariates": label}
        for row in rows:
            entry[row["estimator"]] = {"point_estimate": row["point_estimate"], "ok": row["ok"],
                                        "error": row.get("error_message")}
        sensitivity.append(entry)

    report = {
        "treatment": treatment, "outcome": outcome, "covariates": list(covariates),
        "snapshot_version": snapshot_version,
        "t_learner": t_result, "dml": dml_result,
        "overlap": overlap,
        "sign_agreement": sign_agreement,
        "sensitivity_to_covariates": sensitivity,
    }

    if sign_agreement is False:
        report["disagreement_warning"] = (
            "T-learner và DML KHÔNG đồng thuận về DẤU của hiệu ứng - theo yêu cầu tường minh của "
            "roadmap, điều này KHÔNG bị che giấu ở đây. Đây chính là phần phân tích lỗi/phương pháp "
            "luận có giá trị cho phần error analysis của luận văn, không phải một lỗi cần sửa trước "
            "khi báo cáo."
        )

    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--treatment", default="leverage")
    ap.add_argument("--outcome", default="net_margin")
    ap.add_argument("--covariates", default="sector,fiscal_year")
    ap.add_argument("--snapshot-version", default=None)
    ap.add_argument("--n-bootstrap", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--report", type=Path, default=Path("reports/causal_t_learner_vs_dml.json"))
    args = ap.parse_args()

    covariates = tuple(c.strip() for c in args.covariates.split(",") if c.strip())
    report = compare(treatment=args.treatment, outcome=args.outcome, covariates=covariates,
                      snapshot_version=args.snapshot_version, n_bootstrap=args.n_bootstrap, seed=args.seed)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n=== T-learner vs DML: {args.treatment} -> {args.outcome} ===")
    if report["t_learner"].get("ok"):
        tl = report["t_learner"]
        print(f"  T-learner: n={tl['n']} ate={tl['ate_t_learner']:.6f} 95%CI={tl['ate_95ci']}")
    else:
        print(f"  T-learner: REFUSED ({report['t_learner'].get('error')})")
    if report["dml"].get("ok"):
        dm = report["dml"]
        print(f"  DML: n={dm['n']} marginal_effect={dm['ate_dml_marginal']:.6f} "
              f"translated_group_contrast={dm['translated_group_contrast']:.6f} 95%CI={dm['ate_dml_marginal_95ci']}")
    else:
        print(f"  DML: REFUSED ({report['dml'].get('error')})")
    print(f"  Sign agreement: {report['sign_agreement']}")
    if report.get("disagreement_warning"):
        print(f"  *** {report['disagreement_warning']} ***")
    if report["overlap"].get("ok"):
        print(f"  Overlap: {report['overlap']['fraction_in_well_overlapped_band_0.1_0.9']:.1%} of units "
              f"have propensity in [0.1, 0.9]")
    else:
        print(f"  Overlap: could not compute ({report['overlap'].get('error')})")
    print("\n  Sensitivity to covariate choice:")
    for entry in report["sensitivity_to_covariates"]:
        t = entry.get("t_learner", {})
        d = entry.get("dml", {})
        print(f"    {entry['covariates']}: T-learner={t.get('point_estimate')} DML={d.get('point_estimate')}")
    print(f"\nReport written: {args.report}")


if __name__ == "__main__":
    main()
