"""M0 (mean/median) + M1 (Ridge) baselines for the PREDICTIVE branch (Task
#19, roadmap item 5, 2026-07-30) - the FIRST real model results for this
branch, deliberately narrow in scope per the roadmap: exactly ONE target
(net_margin, horizon=1 fiscal year) and NO deep learning yet. M2 (Random
Forest) / M3 (XGBoost) + SHAP live in scripts/predictive/train_advanced_
models.py (Task #20) - this one stays at "does a linear model beat a
constant guess" before anything fancier is justified.

Uses scripts.predictive.freeze_predictive_split's FROZEN
predictive_split_manifest (via src.predictive.split_loader) for train/
validation/test membership - see that module's docstring for why every
model MUST go through this shared loader instead of re-deriving its own
split. Ridge's regularization strength (alpha) is selected by lowest
validation-set MAE among a small fixed grid, then refit on train only and
evaluated ONCE on test - the test set is never used for model selection.

Metric/breakdown/design-matrix logic lives in src/predictive/model_utils.py,
shared with train_advanced_models.py so both scripts use the EXACT same
MAE/RMSE/R2/MAPE-suppression and train-only-quantile-bucketing logic - see
that module's docstring for why this is factored out rather than copied.

No new DB table for these results (deliberately, per the 2026-07-30
roadmap's explicit "stop expanding the architecture" instruction) - just a
JSON report file, meant to be read directly or turned into a table for the
thesis write-up.

Usage:
    python3 -m scripts.predictive.train_baselines
    python3 -m scripts.predictive.train_baselines --report reports/predictive_baselines_net_margin_h1.json
"""
import argparse
import json
import math
from pathlib import Path

import pandas as pd

from src.predictive.model_utils import (
    breakdown_metrics, bucket_by_train_quantiles, build_matrices, compute_metrics, load_and_split,
)

RIDGE_ALPHAS = (0.1, 1.0, 10.0, 100.0)


def train_and_evaluate(feature_version: str = "predfeat_v1", target_col: str = "net_margin",
                        horizon: int = 1, ridge_alphas=RIDGE_ALPHAS) -> dict:
    train, val, test = load_and_split(feature_version, target_col, horizon)
    X_train, X_val, X_test, y_train, y_val, y_test = build_matrices(train, val, test)

    results = {
        "feature_version": feature_version, "target_col": target_col, "horizon": horizon,
        "n_train": len(train), "n_validation": len(val), "n_test": len(test),
        "predictor_columns": list(X_train.columns), "models": {},
    }

    # --- M0: constant-prediction baselines ---
    from sklearn.dummy import DummyRegressor
    for strategy in ("mean", "median"):
        model = DummyRegressor(strategy=strategy).fit(X_train, y_train)
        pred_test = model.predict(X_test)
        results["models"][f"M0_{strategy}"] = {
            "description": f"DummyRegressor(strategy='{strategy}') - predicts the TRAIN-set "
                            f"{strategy} of target_value for every test row, ignoring all features. "
                            f"Any real model (M1+) that fails to beat this is not adding value.",
            "test_metrics": compute_metrics(y_test, pred_test),
        }

    # --- M1: Ridge, alpha selected on validation, refit on train, evaluated ONCE on test ---
    from sklearn.linear_model import Ridge
    best_alpha, best_val_mae, best_model = None, math.inf, None
    alpha_search = []
    for alpha in ridge_alphas:
        model = Ridge(alpha=alpha).fit(X_train, y_train)
        val_mae = compute_metrics(y_val, model.predict(X_val))["mae"]
        alpha_search.append({"alpha": alpha, "validation_mae": val_mae})
        if val_mae < best_val_mae:
            best_alpha, best_val_mae, best_model = alpha, val_mae, model

    pred_test_m1 = best_model.predict(X_test)
    results["models"]["M1_ridge"] = {
        "description": f"Ridge(alpha={best_alpha}) - alpha selected by lowest validation MAE among "
                        f"{list(ridge_alphas)}, refit on train only, evaluated once on test.",
        "alpha_search": alpha_search,
        "selected_alpha": best_alpha,
        "test_metrics": compute_metrics(y_test, pred_test_m1),
    }

    # --- breakdowns (on M1, the strongest model here) by sector / year /
    #     company size / target magnitude, per the roadmap's explicit
    #     reporting requirement ---
    test_bd = test.copy()
    test_bd["y_pred"] = pred_test_m1
    train_assets = pd.to_numeric(train["total_assets"], errors="coerce")
    test_assets = pd.to_numeric(test_bd["total_assets"], errors="coerce")
    test_bd["size_bucket"] = bucket_by_train_quantiles(train_assets, test_assets, ["small", "mid", "large"])
    train_abs_target = train["target_value"].abs()
    test_bd["magnitude_bucket"] = bucket_by_train_quantiles(
        train_abs_target, test_bd["target_value"].abs(), ["low", "mid", "high", "very_high"]
    )
    results["models"]["M1_ridge"]["breakdown_by_sector"] = breakdown_metrics(test_bd, "target_value", "y_pred", "sector")
    results["models"]["M1_ridge"]["breakdown_by_fiscal_year"] = breakdown_metrics(test_bd, "target_value", "y_pred", "fiscal_year")
    results["models"]["M1_ridge"]["breakdown_by_company_size"] = breakdown_metrics(test_bd, "target_value", "y_pred", "size_bucket")
    results["models"]["M1_ridge"]["breakdown_by_target_magnitude"] = breakdown_metrics(test_bd, "target_value", "y_pred", "magnitude_bucket")

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-version", default="predfeat_v1")
    ap.add_argument("--target", default="net_margin")
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--report", type=Path, default=Path("reports/predictive_baselines_net_margin_h1.json"))
    args = ap.parse_args()

    results = train_and_evaluate(args.feature_version, args.target, args.horizon)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n=== M0/M1 baseline results (target={args.target}, horizon={args.horizon}) ===")
    print(f"n_train={results['n_train']} n_validation={results['n_validation']} n_test={results['n_test']}")
    for name, info in results["models"].items():
        tm = info["test_metrics"]
        mape_str = f" MAPE={tm['mape']:.1f}%" if tm["mape"] is not None else " MAPE=n/a (target near zero)"
        print(f"  {name}: MAE={tm['mae']:.4f} RMSE={tm['rmse']:.4f} R2={tm['r2']:.4f}{mape_str}")
    print(f"\nReport written: {args.report}")


if __name__ == "__main__":
    main()
