"""M2 (Random Forest) + M3 (XGBoost) for the PREDICTIVE branch (Task #20,
roadmap item 6, 2026-07-30), plus a post-hoc SHAP explanation of M3.

Builds on scripts/predictive/train_baselines.py (Task #19, M0/M1) - same
target (net_margin, horizon=1), same FROZEN predictive_split_manifest split
(via src.predictive.model_utils.load_and_split -> split_loader), same
metric/breakdown logic (src/predictive/model_utils.py) - so M0 through M3
are directly comparable on identical data, not just "similar" data. This is
also the "Predictive model ablation (Ridge vs XGBoost; with vs without
macro features)" ablation point from the roadmap's item 9 - --no-macro
below runs the SAME M3 XGBoost with the 5 macro columns dropped, for a
direct with/without comparison in one report.

RandomForestRegressor (M2) needs no hyperparameter search here - a single
reasonable default configuration (matching this project's stated
philosophy of only reaching for a tuned/complex model once a simple one is
shown to be insufficient). XGBoost (M3) uses early stopping against the
FROZEN validation split (not test) to pick n_estimators - same
never-touch-test-for-model-selection discipline as M1's alpha search.

SHAP (roadmap item 4): explains what M3 learned - which features push a
PARTICULAR prediction up or down - it is EXPLICITLY NOT a causal
explanation. A feature with high SHAP importance for forecasting net_margin
is not thereby shown to CAUSE net_margin to change; that question belongs
to src/causal/treatment_effect.py's T-learner/DML estimators (Task #21),
which use a completely different identification strategy (potential
outcomes, not "what did this trained model attend to"). Anyone reading the
SHAP output in the report JSON below should read the "reminder" field
before interpreting a feature as more than "the model relied on this
column."

No new DB table for these results either (same "stop expanding the
architecture" reasoning as train_baselines.py) - JSON report only.

Usage:
    python3 -m scripts.predictive.train_advanced_models
    python3 -m scripts.predictive.train_advanced_models --no-macro
    python3 -m scripts.predictive.train_advanced_models --report reports/predictive_advanced_net_margin_h1.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.predictive.model_utils import (
    NUMERIC_PREDICTORS, breakdown_metrics, bucket_by_train_quantiles, compute_metrics, load_and_split,
    prepare_design_matrix,
)

MACRO_COLUMNS = ["cpi_annual_avg", "gdp_growth_annual_avg", "gold_price_annual_avg",
                 "usdvnd_annual_avg", "oil_price_annual_avg"]

SHAP_REMINDER = (
    "SHAP explains what the FITTED MODEL relied on to produce a prediction - "
    "it is NOT a causal explanation. A high-SHAP feature is not shown to CAUSE "
    "net_margin to change; see src/causal/treatment_effect.py's T-learner/DML "
    "estimators (a different identification strategy entirely) for causal claims."
)


def _build_matrices_with_macro_toggle(train, val, test, include_macro: bool):
    """Same as model_utils.build_matrices(), but optionally drops the 5
    macro columns from NUMERIC_PREDICTORS first - used for the
    with/without-macro-features ablation (roadmap item 9)."""
    predictors = NUMERIC_PREDICTORS if include_macro else [c for c in NUMERIC_PREDICTORS if c not in MACRO_COLUMNS]

    def _design(df, train_medians=None):
        numeric = df[predictors].apply(pd.to_numeric, errors="coerce")
        if train_medians is None:
            train_medians = numeric.median()
        numeric = numeric.fillna(train_medians)
        sector_dummies = pd.get_dummies(df["sector"].fillna("UNKNOWN"), prefix="sector")
        X = pd.concat([numeric.reset_index(drop=True), sector_dummies.reset_index(drop=True)], axis=1)
        return X, train_medians

    X_train, train_medians = _design(train)
    X_val, _ = _design(val, train_medians)
    X_test, _ = _design(test, train_medians)
    X_val = X_val.reindex(columns=X_train.columns, fill_value=0)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
    y_train = train["target_value"].to_numpy(dtype=float)
    y_val = val["target_value"].to_numpy(dtype=float)
    y_test = test["target_value"].to_numpy(dtype=float)
    return X_train, X_val, X_test, y_train, y_val, y_test


def _breakdowns(train, test, y_pred_test) -> dict:
    test_bd = test.copy()
    test_bd["y_pred"] = y_pred_test
    train_assets = pd.to_numeric(train["total_assets"], errors="coerce")
    test_assets = pd.to_numeric(test_bd["total_assets"], errors="coerce")
    test_bd["size_bucket"] = bucket_by_train_quantiles(train_assets, test_assets, ["small", "mid", "large"])
    train_abs_target = train["target_value"].abs()
    test_bd["magnitude_bucket"] = bucket_by_train_quantiles(
        train_abs_target, test_bd["target_value"].abs(), ["low", "mid", "high", "very_high"]
    )
    return {
        "breakdown_by_sector": breakdown_metrics(test_bd, "target_value", "y_pred", "sector"),
        "breakdown_by_fiscal_year": breakdown_metrics(test_bd, "target_value", "y_pred", "fiscal_year"),
        "breakdown_by_company_size": breakdown_metrics(test_bd, "target_value", "y_pred", "size_bucket"),
        "breakdown_by_target_magnitude": breakdown_metrics(test_bd, "target_value", "y_pred", "magnitude_bucket"),
    }


def train_and_evaluate(feature_version: str = "predfeat_v1", target_col: str = "net_margin",
                        horizon: int = 1, include_macro: bool = True, compute_shap: bool = True,
                        random_state: int = 42) -> dict:
    train, val, test = load_and_split(feature_version, target_col, horizon)
    X_train, X_val, X_test, y_train, y_val, y_test = _build_matrices_with_macro_toggle(
        train, val, test, include_macro=include_macro)

    results = {
        "feature_version": feature_version, "target_col": target_col, "horizon": horizon,
        "include_macro_features": include_macro,
        "n_train": len(train), "n_validation": len(val), "n_test": len(test),
        "predictor_columns": list(X_train.columns), "models": {},
    }

    # --- M2: Random Forest, single reasonable default config (no search -
    #     a search is only justified once a simple config is shown wanting) ---
    from sklearn.ensemble import RandomForestRegressor
    rf = RandomForestRegressor(
        n_estimators=300, max_depth=6, min_samples_leaf=5, random_state=random_state, n_jobs=-1,
    ).fit(X_train, y_train)
    pred_test_rf = rf.predict(X_test)
    results["models"]["M2_random_forest"] = {
        "description": "RandomForestRegressor(n_estimators=300, max_depth=6, min_samples_leaf=5) - "
                        "single default config, no hyperparameter search.",
        "test_metrics": compute_metrics(y_test, pred_test_rf),
        **_breakdowns(train, test, pred_test_rf),
    }

    # --- M3: XGBoost, n_estimators picked via early stopping against the
    #     FROZEN validation split (never test) ---
    import xgboost as xgb
    xgb_model = xgb.XGBRegressor(
        n_estimators=1000, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.0, reg_lambda=1.0, random_state=random_state, early_stopping_rounds=30, n_jobs=-1,
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    pred_test_xgb = xgb_model.predict(X_test)
    results["models"]["M3_xgboost"] = {
        "description": f"XGBRegressor(max_depth=4, learning_rate=0.05) - n_estimators selected by early "
                        f"stopping (best_iteration={xgb_model.best_iteration}) against the validation split.",
        "best_iteration": int(xgb_model.best_iteration),
        "test_metrics": compute_metrics(y_test, pred_test_xgb),
        **_breakdowns(train, test, pred_test_xgb),
    }

    # --- SHAP on M3 (the strongest model here) - see module docstring for
    #     the explicit not-causal caveat ---
    if compute_shap:
        import shap
        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(X_test)
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        importance = sorted(
            zip(X_test.columns, mean_abs_shap.tolist()), key=lambda kv: kv[1], reverse=True
        )
        results["models"]["M3_xgboost"]["shap"] = {
            "reminder": SHAP_REMINDER,
            "mean_abs_shap_by_feature_top20": importance[:20],
        }

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-version", default="predfeat_v1")
    ap.add_argument("--target", default="net_margin")
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--no-macro", action="store_true",
                     help="Drop the 5 macro covariates from the predictor set - with/without-macro "
                          "ablation point (roadmap item 9). Run once with and once without to compare.")
    ap.add_argument("--no-shap", action="store_true", help="Skip the SHAP explanation step (faster dry-run).")
    ap.add_argument("--report", type=Path, default=Path("reports/predictive_advanced_net_margin_h1.json"))
    args = ap.parse_args()

    results = train_and_evaluate(
        args.feature_version, args.target, args.horizon,
        include_macro=not args.no_macro, compute_shap=not args.no_shap,
    )

    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n=== M2/M3 results (target={args.target}, horizon={args.horizon}, "
          f"include_macro_features={results['include_macro_features']}) ===")
    print(f"n_train={results['n_train']} n_validation={results['n_validation']} n_test={results['n_test']}")
    for name, info in results["models"].items():
        tm = info["test_metrics"]
        mape_str = f" MAPE={tm['mape']:.1f}%" if tm["mape"] is not None else " MAPE=n/a (target near zero)"
        print(f"  {name}: MAE={tm['mae']:.4f} RMSE={tm['rmse']:.4f} R2={tm['r2']:.4f}{mape_str}")
    if "shap" in results["models"].get("M3_xgboost", {}):
        print("\nTop-5 |SHAP| features for M3_xgboost (NOT a causal ranking - see SHAP_REMINDER in the report):")
        for feat, val in results["models"]["M3_xgboost"]["shap"]["mean_abs_shap_by_feature_top20"][:5]:
            print(f"  {feat:30s} {val:.5f}")
    print(f"\nReport written: {args.report}")


if __name__ == "__main__":
    main()
