"""Shared model-evaluation utilities for the PREDICTIVE branch (Tasks #19
M0/M1 and #20 M2/M3+SHAP, 2026-07-30 roadmap items 5-6) - extracted out of
scripts/predictive/train_baselines.py so train_advanced_models.py (M2/M3)
reuses the EXACT same metric/breakdown/design-matrix logic rather than a
copy that could silently drift (e.g. a MAPE-suppression fix landing in one
script's copy but not the other) - same
duplication-is-a-real-risk-here reasoning already applied elsewhere in this
project (see src/retrieval/text_splitter.py's comment on why it does vs
doesn't share code with scripts/pdf_ingest_router.py).
"""
import numpy as np
import pandas as pd

from src.predictive.feature_builder import FEATURE_COLUMNS

NUMERIC_PREDICTORS = (
    FEATURE_COLUMNS
    + [f"{c}_lag1" for c in FEATURE_COLUMNS]
    + ["revenue_growth_yoy", "net_profit_growth_yoy", "total_assets_growth_yoy"]
    + ["cpi_annual_avg", "gdp_growth_annual_avg", "gold_price_annual_avg", "usdvnd_annual_avg", "oil_price_annual_avg"]
)


def _mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def _rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return float("nan")
    return float(1 - ss_res / ss_tot)


def compute_metrics(y_true, y_pred) -> dict:
    """MAE/RMSE/R2 always; MAPE only if EVERY true value in this slice is
    comfortably away from zero (per the roadmap: "MAPE only if target isn't
    near zero") - net_margin is frequently at/near zero (loss-making
    companies, thin-margin sectors), where MAPE explodes or divides by ~0
    and would mislead more than inform."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    metrics = {
        "n": int(len(y_true)), "mae": _mae(y_true, y_pred),
        "rmse": _rmse(y_true, y_pred), "r2": _r2(y_true, y_pred),
    }
    if len(y_true) and np.all(np.abs(y_true) > 0.01):
        metrics["mape"] = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
    else:
        metrics["mape"] = None
    return metrics


def bucket_by_train_quantiles(train_values: pd.Series, values: pd.Series, labels: list) -> pd.Series:
    """Bucket `values` using quantile EDGES computed only from
    `train_values` - a company-size or target-magnitude bucket boundary is
    never chosen using information from the validation/test set."""
    quantiles = np.linspace(0, 1, len(labels) + 1)
    edges = np.unique(train_values.quantile(quantiles).to_numpy())
    if len(edges) < 2:
        return pd.Series(["all"] * len(values), index=values.index)
    edges = edges.astype(float)
    edges[0], edges[-1] = -np.inf, np.inf
    return pd.cut(values, bins=edges, labels=labels[: len(edges) - 1], include_lowest=True)


def breakdown_metrics(df: pd.DataFrame, y_true_col: str, y_pred_col: str, group_col: str) -> dict:
    out = {}
    for group_value, sub in df.groupby(group_col, observed=True):
        if len(sub) == 0:
            continue
        out[str(group_value)] = compute_metrics(sub[y_true_col], sub[y_pred_col])
    return out


def prepare_design_matrix(df: pd.DataFrame, train_medians: pd.Series = None):
    """One-hot `sector` + median-impute numeric predictors. `train_medians`
    is None only when `df` IS the train split (medians are computed fresh
    from it); pass the TRAIN split's medians through unchanged for
    validation/test so their imputation never uses their own distribution -
    same leakage discipline as bucket_by_train_quantiles above."""
    numeric = df[NUMERIC_PREDICTORS].apply(pd.to_numeric, errors="coerce")
    if train_medians is None:
        train_medians = numeric.median()
    numeric = numeric.fillna(train_medians)
    sector_dummies = pd.get_dummies(df["sector"].fillna("UNKNOWN"), prefix="sector")
    X = pd.concat([numeric.reset_index(drop=True), sector_dummies.reset_index(drop=True)], axis=1)
    return X, train_medians


def load_and_split(feature_version: str, target_col: str, horizon: int):
    """Loads via src.predictive.split_loader.load_features_with_split() and
    returns (train, validation, test) DataFrames - raises SystemExit with a
    clear message if the load is empty or any split is empty, same guard
    used identically by every model script (train_baselines.py,
    train_advanced_models.py) so the failure mode is consistent."""
    from src.predictive.split_loader import load_features_with_split
    df = load_features_with_split(feature_version=feature_version, target_col=target_col, horizon=horizon)
    if df.empty:
        raise SystemExit(
            "load_features_with_split() returned 0 rows - has "
            "scripts/predictive/freeze_predictive_split.py been run for this "
            "feature_version/target_col/horizon?"
        )
    train = df[df["split"] == "train"].reset_index(drop=True)
    val = df[df["split"] == "validation"].reset_index(drop=True)
    test = df[df["split"] == "test"].reset_index(drop=True)
    if train.empty or val.empty or test.empty:
        raise SystemExit(
            f"Empty split(s) - train={len(train)}, validation={len(val)}, test={len(test)}. "
            f"Cannot select (needs validation) or evaluate (needs test) a model with an empty split."
        )
    return train, val, test


def build_matrices(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame):
    """Returns (X_train, X_val, X_test, y_train, y_val, y_test) with
    consistent one-hot columns across all three splits (see
    prepare_design_matrix's docstring) and TRAIN-only median imputation."""
    X_train, train_medians = prepare_design_matrix(train)
    X_val, _ = prepare_design_matrix(val, train_medians)
    X_test, _ = prepare_design_matrix(test, train_medians)
    X_val = X_val.reindex(columns=X_train.columns, fill_value=0)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
    y_train = train["target_value"].to_numpy(dtype=float)
    y_val = val["target_value"].to_numpy(dtype=float)
    y_test = test["target_value"].to_numpy(dtype=float)
    return X_train, X_val, X_test, y_train, y_val, y_test
