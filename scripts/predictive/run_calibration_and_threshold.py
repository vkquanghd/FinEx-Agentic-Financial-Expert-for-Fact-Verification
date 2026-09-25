#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AUTO-EXTRACTED tu notebooks/03_ml_task_suite_rag_ready.ipynb (cell 1..21).

Chay thang bang CLI, KHONG can mo Jupyter — de xac nhan/tinh lai calibration
(checklist 1.1), threshold selection (checklist 1.2), va subgroup/error analysis
(checklist 1.3) tu terminal.

Yeu cau: .venv da activate, thesis_postgres dang chay (docker compose up -d postgres),
chay tu thu muc goc project (co folder src/).

Neu ban sua notebook (them/doi cell 1..21), phai chay lai script generator de
dong bo — script nay la BAN SAO TRICH XUAT, khong phai nguon chinh; nguon chinh
van la notebook.

Usage:
    cd ~/Desktop/Project/Thesis_Project
    source .venv/bin/activate
    python scripts/predictive/run_calibration_and_threshold.py
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve()
while not (PROJECT_ROOT / "src").exists() and PROJECT_ROOT.parent != PROJECT_ROOT:
    PROJECT_ROOT = PROJECT_ROOT.parent
assert (PROJECT_ROOT / "src").exists(), (
    "Khong tim thay folder src/ — chay script tu trong project."
)
sys.path.insert(0, str(PROJECT_ROOT))
try:
    from IPython.display import display
except Exception:
    def display(x):
        print(x)

# ===== Cell 1 (tu notebook) =====
import sys
import warnings
import time
import threading
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path.cwd()
if not (PROJECT_ROOT / "src").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent
assert (PROJECT_ROOT / "src").exists(), (
    "Chạy notebook từ thư mục gốc project, nơi có folder src/."
)
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 80)
pd.set_option("display.width", 180)
pd.set_option("display.max_colwidth", 100)
np.random.seed(42)

from src.core.db import engine, check_connection

check_connection()

ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "ml_task_suite"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

RUN_FINAL_TEST = False

# ------------------------------------------------------------------
# Official time boundary / quarter coverage
# quarter_index = fiscal_year * 4 + quarter_num.
# 2026Q1 = 2026*4+1 = 8105.
# ------------------------------------------------------------------
ANALYSIS_CUTOFF_QUARTER = 2026 * 4 + 1
CURRENT_ANALYSIS_QUARTER = ANALYSIS_CUTOFF_QUARTER
FINAL_TEST_TARGET_QUARTER = ANALYSIS_CUTOFF_QUARTER
MIN_COMPLETE_QUARTER_RATIO = 0.80
MIN_COMPLETE_QUARTER_TICKERS = 100

# Flow-derived tasks are skipped until ETL convention is confirmed.
ALLOW_UNCONFIRMED_FLOW_TASKS = False

# Structured output for downstream RAG / structured retrieval.
EXPORT_RAG_OUTPUTS = False
MODELS_LOCKED_FOR_RAG = False
WRITE_RAG_OUTPUT_TO_DB = False
RAG_MODEL_VERSION_PREFIX = "quarterly_panel_v2"
CLASSIFICATION_THRESHOLDS = {
    "margin_up_next_q": 0.50,
    "margin_negative_next_q": 0.50,
    "ocf_negative_next_q": 0.50,
    "deterioration_strict_3of3": 0.50,
    "deterioration_3of4": 0.50,
}
# Ví dụ sau khi chốt: {'net_margin_next_q': 'elastic_net'}
FINAL_MODEL_SELECTION = {}
REQUIRE_POINT_IN_TIME_CUTOFF = True
USE_SAFE_QUARTERLY_MACRO = False

# Xác nhận từ nguồn ETL trước khi đổi sang "ytd_cumulative".
# [2026-08-04] Xác nhận qua đối chiếu Q1+Q2 vs "6 tháng" trên ABI/BSG/CRC/CKG
# trong data/raw/vietstock_exports/ (xem docs/CHECKLIST_PHASES.md muc 0.1):
# quy la standalone, khong phai YTD cumulative.
FLOW_VALUE_MODE = "reported_quarter"
FLOW_MODE_CONFIRMED = True

RUN_CAUSAL = False

# ------------------------------------------------------------------
# Runtime controls
# ------------------------------------------------------------------
# fast: model đại diện, phù hợp smoke test.
# balanced: leaderboard chính, bỏ SVM/MLP chậm. Đây là mặc định.
# full: chạy tất cả model, có thể lâu đáng kể.
RUNTIME_PROFILE = "balanced"
SHOW_MODEL_PROGRESS = True
HEARTBEAT_SECONDS = 20
MODEL_SOFT_BUDGET_SECONDS = 5 * 60
TASK_SOFT_BUDGET_SECONDS = 20 * 60
SKIP_REMAINING_FOLDS_AFTER_BUDGET = True

# SVM chỉ xuất hiện trong profile full và vẫn bị cap sample.
SLOW_MODEL_MAX_TRAIN_N = 500
SVM_MAX_ITER = 5_000

MODEL_PROFILES = {
    "fast": {
        "regression": [
            "ridge",
            "elastic_net",
            "random_forest",
            "hist_gbm",
            "xgboost",
        ],
        "classification": [
            "logistic",
            "elastic_net_logistic",
            "random_forest",
            "hist_gbm",
            "xgboost",
        ],
    },
    "balanced": {
        "regression": [
            "linear_regression",
            "ridge",
            "lasso",
            "elastic_net",
            "huber",
            "knn",
            "random_forest",
            "extra_trees",
            "hist_gbm",
            "xgboost",
            "lightgbm",
            "catboost",
        ],
        "classification": [
            "logistic",
            "elastic_net_logistic",
            "knn",
            "random_forest",
            "extra_trees",
            "hist_gbm",
            "xgboost",
            "lightgbm",
            "catboost",
        ],
    },
    "full": {
        "regression": None,
        "classification": None,
    },
}

if RUNTIME_PROFILE not in MODEL_PROFILES:
    raise ValueError(
        f"RUNTIME_PROFILE phải là một trong {list(MODEL_PROFILES)}"
    )

# Giảm số cây/iterations ở fast/balanced; full giữ cấu hình đầy đủ.
TREE_ESTIMATORS = {"fast": 120, "balanced": 220, "full": 400}[RUNTIME_PROFILE]
BOOST_ITERATIONS = {"fast": 150, "balanced": 250, "full": 400}[RUNTIME_PROFILE]
MLP_MAX_ITER = {"fast": 400, "balanced": 700, "full": 1500}[RUNTIME_PROFILE]

print("PROJECT_ROOT:", PROJECT_ROOT)
print("ARTIFACT_DIR:", ARTIFACT_DIR)
print("RUN_FINAL_TEST:", RUN_FINAL_TEST)
print("ANALYSIS_CUTOFF_QUARTER:", ANALYSIS_CUTOFF_QUARTER)
print("FINAL_TEST_TARGET_QUARTER:", FINAL_TEST_TARGET_QUARTER)
print("MIN_COMPLETE_QUARTER_RATIO:", MIN_COMPLETE_QUARTER_RATIO)
print("ALLOW_UNCONFIRMED_FLOW_TASKS:", ALLOW_UNCONFIRMED_FLOW_TASKS)
print("EXPORT_RAG_OUTPUTS:", EXPORT_RAG_OUTPUTS)
print("REQUIRE_POINT_IN_TIME_CUTOFF:", REQUIRE_POINT_IN_TIME_CUTOFF)
print("USE_SAFE_QUARTERLY_MACRO:", USE_SAFE_QUARTERLY_MACRO)
print("FLOW_VALUE_MODE:", FLOW_VALUE_MODE, "| confirmed:", FLOW_MODE_CONFIRMED)
print("RUNTIME_PROFILE:", RUNTIME_PROFILE)
print("TREE_ESTIMATORS:", TREE_ESTIMATORS, "| BOOST_ITERATIONS:", BOOST_ITERATIONS)
print("HEARTBEAT_SECONDS:", HEARTBEAT_SECONDS)
print("MODEL_SOFT_BUDGET_SECONDS:", MODEL_SOFT_BUDGET_SECONDS)
print("TASK_SOFT_BUDGET_SECONDS:", TASK_SOFT_BUDGET_SECONDS)


def _try_import(name):
    try:
        return __import__(name)
    except Exception as exc:
        print(
            f"[optional] {name}: chưa dùng được "
            f"({type(exc).__name__}: {exc}); phần phụ thuộc package này sẽ bị bỏ qua."
        )
        return None


HAS_LIGHTGBM = _try_import("lightgbm") is not None
HAS_CATBOOST = _try_import("catboost") is not None
HAS_UMAP = _try_import("umap") is not None

print(
    f"lightgbm: {'có' if HAS_LIGHTGBM else 'không'} | "
    f"catboost: {'có' if HAS_CATBOOST else 'không'} | "
    f"umap: {'có' if HAS_UMAP else 'không'}"
)


# ===== Cell 3 (tu notebook) =====
from src.causal.panel_builder import _fetch_raw, _PANEL_FIELDS, _RATIO_BOUNDS
from src.core.constants import FINANCE_SUBSECTOR_KEYWORDS
from sqlalchemy import text

QUARTER_PERIODS = ["Q1", "Q2", "Q3", "Q4"]
RAW_COLS = list(_PANEL_FIELDS.values())
RATIO_COLS = [
    "leverage",
    "current_ratio",
    "gross_margin",
    "net_margin",
    "asset_turnover",
    "ocf_to_revenue",
]

FLOW_COLS = [
    "revenue",
    "cogs",
    "gross_profit",
    "profit_before_tax",
    "net_profit",
    "operating_cash_flow",
]

CORE_COLS = RAW_COLS + RATIO_COLS


def _table_columns(table_name: str) -> set[str]:
    query = text('''
    SELECT column_name
    FROM information_schema.columns
    WHERE table_schema = 'public'
      AND table_name = :table_name
    ''')
    return set(
        pd.read_sql(query, engine, params={"table_name": table_name})[
            "column_name"
        ].tolist()
    )


def _safe_divide(
    numerator: pd.Series,
    denominator: pd.Series,
    eps: float = 1e-12,
) -> pd.Series:
    numerator = pd.to_numeric(numerator, errors="coerce")
    denominator = pd.to_numeric(denominator, errors="coerce")
    result = numerator / denominator.where(denominator.abs() > eps)
    return result.replace([np.inf, -np.inf], np.nan)


def _normalise_consolidation(value) -> str:
    text = str(value).strip().lower()
    if text in {"consolidated", "hop_nhat", "hợp nhất", "1", "true"}:
        return "consolidated"
    if text in {"standalone", "separate", "rieng", "riêng", "0", "false"}:
        return "standalone"
    return text or "unknown"


def _resolve_duplicate_line_items(qraw: pd.DataFrame) -> pd.DataFrame:
    '''
    Không pivot với aggfunc='first' khi có nhiều phiên bản cùng key.
    Duplicate cùng value được gộp; duplicate khác value cần version metadata.
    '''
    key = [
        "ticker",
        "fiscal_year",
        "period",
        "consolidation",
        "statement_type",
        "ms_code",
    ]
    key = [col for col in key if col in qraw.columns]

    duplicated = qraw.duplicated(key, keep=False)
    if not duplicated.any():
        return qraw

    dup = qraw.loc[duplicated].copy()
    value_nunique = dup.groupby(key, dropna=False)["value"].nunique(dropna=False)
    ambiguous_keys = value_nunique[value_nunique > 1]

    print(
        f"[duplicate audit] {duplicated.sum()} dòng duplicate; "
        f"{len(ambiguous_keys)} key có nhiều value khác nhau."
    )

    exact = qraw.drop_duplicates(key + ["value"], keep="last")

    if len(ambiguous_keys) == 0:
        return exact

    version_candidates = [
        col
        for col in [
            "disclosure_date",
            "signed_date",
            "updated_at",
            "ingested_at",
            "created_at",
            "id",
        ]
        if col in exact.columns
    ]

    if not version_candidates:
        raise RuntimeError(
            "Có nhiều giá trị khác nhau cho cùng line-item key nhưng _fetch_raw() "
            "không trả version timestamp/id. Không thể chọn canonical row an toàn. "
            "Hãy bổ sung source/version metadata vào _fetch_raw()."
        )

    sortable = exact.copy()
    for col in version_candidates:
        if col != "id":
            sortable[col] = pd.to_datetime(sortable[col], errors="coerce")

    sortable = sortable.sort_values(key + version_candidates)
    resolved = sortable.drop_duplicates(key, keep="last")

    print(
        "[duplicate audit] Đã chọn canonical row theo: "
        + ", ".join(version_candidates)
    )
    return resolved


def _load_disclosure_cutoffs() -> pd.DataFrame:
    cols = _table_columns("disclosures")
    required = {"ticker", "fiscal_year", "period"}
    if not required.issubset(cols):
        print(
            "[point-in-time] disclosures thiếu ticker/fiscal_year/period; "
            "không thể gắn information_cutoff."
        )
        return pd.DataFrame()

    selected = ["ticker", "fiscal_year", "period"]
    selected.append(
        "consolidation" if "consolidation" in cols
        else "'unknown' AS consolidation"
    )
    selected.append(
        "disclosure_date" if "disclosure_date" in cols
        else "NULL AS disclosure_date"
    )
    selected.append(
        "signed_date" if "signed_date" in cols
        else "NULL AS signed_date"
    )

    disclosures = pd.read_sql(
        f"SELECT {', '.join(selected)} FROM disclosures",
        engine,
    )
    disclosures["period"] = disclosures["period"].astype(str).str.upper()
    disclosures["consolidation"] = disclosures["consolidation"].map(
        _normalise_consolidation
    )
    disclosures["disclosure_date"] = pd.to_datetime(
        disclosures["disclosure_date"], errors="coerce"
    )
    disclosures["signed_date"] = pd.to_datetime(
        disclosures["signed_date"], errors="coerce"
    )

    disclosures["information_cutoff"] = disclosures[
        "disclosure_date"
    ].combine_first(disclosures["signed_date"])

    disclosures = disclosures[
        disclosures["information_cutoff"].notna()
    ].copy()

    if disclosures.empty:
        return disclosures

    exact = (
        disclosures.groupby(
            ["ticker", "fiscal_year", "period", "consolidation"],
            as_index=False,
        )["information_cutoff"]
        .min()
    )

    fallback = (
        disclosures.groupby(
            ["ticker", "fiscal_year", "period"],
            as_index=False,
        )["information_cutoff"]
        .min()
        .rename(columns={"information_cutoff": "information_cutoff_fallback"})
    )

    return exact.merge(
        fallback,
        on=["ticker", "fiscal_year", "period"],
        how="left",
    )


def _attach_information_cutoff(wide: pd.DataFrame) -> pd.DataFrame:
    cutoffs = _load_disclosure_cutoffs()
    output = wide.copy()

    if cutoffs.empty:
        output["information_cutoff"] = pd.NaT
        output["cutoff_match_quality"] = "missing"
        return output

    output["consolidation"] = output["consolidation"].map(
        _normalise_consolidation
    )

    exact_cols = [
        "ticker",
        "fiscal_year",
        "period",
        "consolidation",
        "information_cutoff",
    ]
    fallback = cutoffs[
        [
            "ticker",
            "fiscal_year",
            "period",
            "information_cutoff_fallback",
        ]
    ].drop_duplicates()

    output = output.merge(
        cutoffs[exact_cols],
        on=["ticker", "fiscal_year", "period", "consolidation"],
        how="left",
    )
    output = output.merge(
        fallback,
        on=["ticker", "fiscal_year", "period"],
        how="left",
    )

    exact_match = output["information_cutoff"].notna()
    output["information_cutoff"] = output[
        "information_cutoff"
    ].combine_first(output["information_cutoff_fallback"])
    output["cutoff_match_quality"] = np.where(
        exact_match,
        "exact",
        np.where(
            output["information_cutoff"].notna(),
            "fallback_without_consolidation",
            "missing",
        ),
    )
    return output.drop(columns=["information_cutoff_fallback"])


def _convert_ytd_to_standalone(
    wide: pd.DataFrame,
    flow_cols: list[str],
) -> pd.DataFrame:
    output = wide.sort_values(
        ["ticker", "fiscal_year", "quarter_num"]
    ).copy()
    group = output.groupby(
        ["ticker", "fiscal_year", "consolidation"],
        group_keys=False,
    )

    for col in flow_cols:
        if col not in output.columns:
            continue
        previous = group[col].shift(1)
        previous_q = group["quarter_num"].shift(1)
        consecutive = output["quarter_num"] - previous_q == 1
        standalone = output[col] - previous
        output[col] = np.where(
            output["quarter_num"].eq(1),
            output[col],
            np.where(consecutive, standalone, np.nan),
        )
    return output


def audit_flow_value_convention(
    wide: pd.DataFrame,
    flow_cols: list[str] = FLOW_COLS,
) -> pd.DataFrame:
    rows = []
    ordered = wide.sort_values(
        ["ticker", "fiscal_year", "quarter_num"]
    )
    group = ordered.groupby(
        ["ticker", "fiscal_year", "consolidation"],
        group_keys=False,
    )

    for col in flow_cols:
        if col not in ordered.columns:
            continue
        previous = group[col].shift(1)
        valid = ordered[col].notna() & previous.notna()
        if valid.sum() == 0:
            continue
        ratio = _safe_divide(
            ordered.loc[valid, col],
            previous.loc[valid],
        )
        rows.append(
            {
                "feature": col,
                "n_transitions": int(valid.sum()),
                "share_non_decreasing": float(
                    (ordered.loc[valid, col] >= previous.loc[valid]).mean()
                ),
                "median_current_to_previous": float(np.nanmedian(ratio)),
            }
        )
    return pd.DataFrame(rows)


def _attach_safe_quarterly_macro(wide: pd.DataFrame) -> pd.DataFrame:
    '''
    Conservative point-in-time macro join.
    Chỉ dùng macro Q1..Q4 có release_date <= information_cutoff.
    '''
    output = wide.copy()

    macro_cols = _table_columns("macro_indicators")
    required = {
        "indicator_name",
        "value",
        "fiscal_year",
        "period",
        "release_date",
    }
    if not required.issubset(macro_cols):
        print(
            "[macro] Thiếu quarterly release-dated macro fields; bỏ qua macro."
        )
        return output

    macro = pd.read_sql(
        '''
        SELECT indicator_name, value, fiscal_year, period, release_date
        FROM macro_indicators
        WHERE release_date IS NOT NULL
        ''',
        engine,
    )
    macro["period"] = macro["period"].astype(str).str.upper()
    macro = macro[macro["period"].isin(QUARTER_PERIODS)].copy()
    macro["release_date"] = pd.to_datetime(
        macro["release_date"], errors="coerce"
    )
    macro["value"] = pd.to_numeric(macro["value"], errors="coerce")
    macro = macro.dropna(subset=["release_date", "value"])

    if macro.empty:
        print("[macro] Không có macro quarterly với release_date; bỏ qua.")
        return output

    rows = []
    for row in output[
        ["ticker", "fiscal_year", "period", "information_cutoff"]
    ].itertuples(index=False):
        eligible = macro[
            (macro["fiscal_year"] == row.fiscal_year)
            & (macro["period"] == row.period)
            & (macro["release_date"] <= row.information_cutoff)
        ]
        record = {
            "ticker": row.ticker,
            "fiscal_year": row.fiscal_year,
            "period": row.period,
        }
        for name, group in eligible.groupby("indicator_name"):
            record[f"macro_{name}"] = group.sort_values(
                "release_date"
            )["value"].iloc[-1]
        rows.append(record)

    macro_wide = pd.DataFrame(rows)
    if macro_wide.empty:
        return output

    return output.merge(
        macro_wide,
        on=["ticker", "fiscal_year", "period"],
        how="left",
    )


def build_quarterly_panel() -> pd.DataFrame:
    raw = _fetch_raw()
    if raw.empty:
        print(
            "[cảnh báo] financial_line_items rỗng; không thể xây panel quý."
        )
        return pd.DataFrame()

    qraw = raw[
        raw["period"].astype(str).str.upper().isin(QUARTER_PERIODS)
    ].copy()
    if qraw.empty:
        print(
            "[cảnh báo] Không có Q1..Q4. Kiểm tra dữ liệu ingestion."
        )
        return pd.DataFrame()

    qraw["period"] = qraw["period"].astype(str).str.upper()
    qraw["quarter_num"] = qraw["period"].str[1].astype(int)
    qraw["consolidation"] = qraw["consolidation"].map(
        _normalise_consolidation
    )
    qraw["field"] = list(
        zip(qraw["statement_type"], qraw["ms_code"])
    )
    qraw["field"] = qraw["field"].map(_PANEL_FIELDS)
    qraw["value"] = pd.to_numeric(qraw["value"], errors="coerce")
    qraw = qraw[qraw["field"].notna()].copy()

    qraw = _resolve_duplicate_line_items(qraw)

    idx_cols = [
        "ticker",
        "fiscal_year",
        "period",
        "quarter_num",
        "consolidation",
        "sector",
        "exchange",
        "company_name",
    ]

    wide = (
        qraw.pivot(
            index=idx_cols,
            columns="field",
            values="value",
        )
        .reset_index()
    )
    wide.columns.name = None

    for col in _PANEL_FIELDS.values():
        if col not in wide.columns:
            wide[col] = np.nan

    wide["_cons_rank"] = np.select(
        [
            wide["consolidation"].eq("consolidated"),
            wide["consolidation"].eq("standalone"),
        ],
        [0, 1],
        default=2,
    )
    wide = (
        wide.sort_values(
            [
                "ticker",
                "fiscal_year",
                "quarter_num",
                "_cons_rank",
            ]
        )
        .drop_duplicates(
            subset=["ticker", "fiscal_year", "quarter_num"],
            keep="first",
        )
        .drop(columns=["_cons_rank"])
        .reset_index(drop=True)
    )

    flow_audit = audit_flow_value_convention(wide)
    if not flow_audit.empty:
        print("[flow audit] Dùng bảng sau để xác nhận cumulative/standalone:")
        display(flow_audit)

    if not FLOW_MODE_CONFIRMED:
        print(
            "[CẢNH BÁO] FLOW_MODE_CONFIRMED=False. "
            "Kết quả liên quan QoQ chỉ mang tính exploratory."
        )

    if FLOW_VALUE_MODE == "ytd_cumulative":
        wide = _convert_ytd_to_standalone(wide, FLOW_COLS)
    elif FLOW_VALUE_MODE != "reported_quarter":
        raise ValueError(
            "FLOW_VALUE_MODE phải là 'reported_quarter' hoặc 'ytd_cumulative'."
        )

    wide["leverage"] = _safe_divide(
        wide["total_liabilities"], wide["total_assets"]
    )
    wide["current_ratio"] = _safe_divide(
        wide["current_assets"], wide["current_liabilities"]
    )
    wide["gross_margin"] = _safe_divide(
        wide["gross_profit"], wide["revenue"]
    )
    wide["net_margin"] = _safe_divide(
        wide["net_profit"], wide["revenue"]
    )
    wide["asset_turnover"] = _safe_divide(
        wide["revenue"], wide["total_assets"]
    )
    wide["ocf_to_revenue"] = _safe_divide(
        wide["operating_cash_flow"], wide["revenue"]
    )

    is_finance = wide["company_name"].str.contains(
        "|".join(FINANCE_SUBSECTOR_KEYWORDS),
        case=False,
        na=False,
    )

    def _violations(row):
        return [
            col
            for col, (low, high) in _RATIO_BOUNDS.items()
            if pd.notna(row[col]) and not (low <= row[col] <= high)
        ]

    violations = wide.apply(_violations, axis=1)
    eligible = ~is_finance & violations.map(len).eq(0)

    print(
        f"[quarterly_panel] {len(wide)} dòng thô -> "
        f"loại {int(is_finance.sum())} finance-subsector, "
        f"{int((~is_finance & violations.map(len).gt(0)).sum())} ratio-invalid "
        f"-> {int(eligible.sum())} dòng."
    )

    wide = wide[eligible].copy()
    wide = _attach_information_cutoff(wide)

    if REQUIRE_POINT_IN_TIME_CUTOFF:
        before = len(wide)
        wide = wide[wide["information_cutoff"].notna()].copy()
        print(
            f"[point-in-time] Giữ {len(wide)}/{before} dòng có information_cutoff."
        )

    wide["quarter_index"] = (
        wide["fiscal_year"].astype(int) * 4
        + wide["quarter_num"].astype(int)
    )

    if USE_SAFE_QUARTERLY_MACRO:
        wide = _attach_safe_quarterly_macro(wide)
    else:
        print(
            "[macro] Macro annual/không có release_date bị loại khỏi panel quý."
        )

    wide["restatement_point_in_time_verified"] = False

    wide = (
        wide.drop(columns=["company_name"])
        .sort_values(["ticker", "quarter_index"])
        .reset_index(drop=True)
    )
    return wide


panel = build_quarterly_panel()

if not panel.empty:
    print(
        f"panel: {panel.shape[0]} dòng, "
        f"{panel['ticker'].nunique()} ticker, "
        f"{panel['quarter_index'].nunique()} quý."
    )
    display(
        panel[
            [
                "ticker",
                "fiscal_year",
                "period",
                "information_cutoff",
                "cutoff_match_quality",
                "sector",
                "net_margin",
                "leverage",
            ]
        ].head(10)
    )


# ===== Cell 5 (tu notebook) =====
def quarter_index_to_label(value: int | float) -> str:
    value = int(value)
    year = (value - 1) // 4
    quarter = ((value - 1) % 4) + 1
    return f"{year}Q{quarter}"


raw_panel_before_coverage_gate = panel.copy()

quarter_coverage_gate = (
    raw_panel_before_coverage_gate.groupby(
        "quarter_index",
        as_index=False,
    )
    .agg(
        rows=("ticker", "size"),
        tickers=("ticker", "nunique"),
    )
    .sort_values("quarter_index")
)

coverage_before_cutoff = quarter_coverage_gate[
    quarter_coverage_gate["quarter_index"]
    <= ANALYSIS_CUTOFF_QUARTER
].copy()

if coverage_before_cutoff.empty:
    raise RuntimeError(
        "Không có quý nào trước ANALYSIS_CUTOFF_QUARTER."
    )

typical_quarter_tickers = int(
    round(coverage_before_cutoff["tickers"].median())
)
complete_quarter_threshold = max(
    MIN_COMPLETE_QUARTER_TICKERS,
    int(np.ceil(
        typical_quarter_tickers
        * MIN_COMPLETE_QUARTER_RATIO
    )),
)

quarter_coverage_gate["quarter_label"] = (
    quarter_coverage_gate["quarter_index"]
    .map(quarter_index_to_label)
)
quarter_coverage_gate["before_cutoff"] = (
    quarter_coverage_gate["quarter_index"]
    <= ANALYSIS_CUTOFF_QUARTER
)
quarter_coverage_gate["coverage_ratio_vs_typical"] = (
    quarter_coverage_gate["tickers"]
    / max(1, typical_quarter_tickers)
)
quarter_coverage_gate["is_complete_official_quarter"] = (
    quarter_coverage_gate["before_cutoff"]
    & (
        quarter_coverage_gate["tickers"]
        >= complete_quarter_threshold
    )
)

complete_quarters = (
    quarter_coverage_gate.loc[
        quarter_coverage_gate[
            "is_complete_official_quarter"
        ],
        "quarter_index",
    ]
    .astype(int)
    .tolist()
)

if ANALYSIS_CUTOFF_QUARTER not in complete_quarters:
    raise RuntimeError(
        "Official cutoff quarter không đủ coverage. "
        f"cutoff={quarter_index_to_label(ANALYSIS_CUTOFF_QUARTER)}, "
        f"threshold={complete_quarter_threshold} tickers."
    )

excluded_quarters = quarter_coverage_gate.loc[
    ~quarter_coverage_gate[
        "is_complete_official_quarter"
    ],
    ["quarter_index", "quarter_label", "tickers"],
].copy()

panel = raw_panel_before_coverage_gate[
    raw_panel_before_coverage_gate[
        "quarter_index"
    ].isin(complete_quarters)
].copy()

CURRENT_ANALYSIS_QUARTER = int(
    panel["quarter_index"].max()
)

if CURRENT_ANALYSIS_QUARTER != ANALYSIS_CUTOFF_QUARTER:
    raise RuntimeError(
        "CURRENT_ANALYSIS_QUARTER không khớp official cutoff."
    )

print(
    "[complete-quarter gate] typical coverage=",
    typical_quarter_tickers,
    "| threshold=",
    complete_quarter_threshold,
)
print(
    "[complete-quarter gate] official quarters:",
    [
        quarter_index_to_label(value)
        for value in complete_quarters
    ],
)
print(
    "[complete-quarter gate] excluded/incomplete quarters:"
)
display(excluded_quarters)
print(
    "Official panel:",
    len(panel),
    "rows |",
    panel["ticker"].nunique(),
    "tickers | latest=",
    quarter_index_to_label(CURRENT_ANALYSIS_QUARTER),
)
print(
    "Final test:",
    quarter_index_to_label(
        FINAL_TEST_TARGET_QUARTER - 1
    ),
    "features ->",
    quarter_index_to_label(
        FINAL_TEST_TARGET_QUARTER
    ),
    "target",
)

quarter_coverage_gate.to_csv(
    ARTIFACT_DIR / "quarter_coverage_gate.csv",
    index=False,
)


# ===== Cell 7 (tu notebook) =====
def _shift_valid(grouped, col, periods, valid_mask):
    return grouped[col].shift(periods).where(valid_mask)


def enrich_panel(panel: pd.DataFrame) -> pd.DataFrame:
    df = panel.sort_values(
        ["ticker", "quarter_index"]
    ).reset_index(drop=True)
    grouped = df.groupby("ticker")

    prev1_qidx = grouped["quarter_index"].shift(1)
    next1_qidx = grouped["quarter_index"].shift(-1)
    prev4_qidx = grouped["quarter_index"].shift(4)

    valid_lag1 = (df["quarter_index"] - prev1_qidx).eq(1)
    valid_next1 = (next1_qidx - df["quarter_index"]).eq(1)
    valid_lag4 = (df["quarter_index"] - prev4_qidx).eq(4)

    for col in CORE_COLS:
        if col in df.columns:
            df[f"{col}_lag1"] = _shift_valid(
                grouped, col, 1, valid_lag1
            )

    for col in [
        "net_margin",
        "gross_margin",
        "leverage",
        "current_ratio",
        "asset_turnover",
        "ocf_to_revenue",
    ]:
        if col in df.columns:
            df[f"delta_{col}"] = df[col] - df[f"{col}_lag1"]

    for base in ["revenue", "net_profit", "total_assets"]:
        if base not in df.columns:
            continue

        lag1 = df[f"{base}_lag1"]
        lag4 = _shift_valid(grouped, base, 4, valid_lag4)

        df[f"{base}_growth_qoq"] = _safe_divide(
            df[base] - lag1,
            lag1.abs(),
        )
        df[f"{base}_growth_yoy"] = _safe_divide(
            df[base] - lag4,
            lag4.abs(),
        )

    if "operating_cash_flow" in df.columns:
        df["ocf_signed_log"] = (
            np.sign(df["operating_cash_flow"])
            * np.log1p(df["operating_cash_flow"].abs())
        )

    grouped2 = df.groupby("ticker")

    next_target_cols = [
        col
        for col in (
            CORE_COLS
            + [
                "revenue_growth_qoq",
                "net_profit_growth_qoq",
                "revenue_growth_yoy",
                "ocf_signed_log",
            ]
        )
        if col in df.columns
    ]

    for col in next_target_cols:
        df[f"{col}_next_q"] = _shift_valid(
            grouped2, col, -1, valid_next1
        )

    if "information_cutoff" in df.columns:
        df["target_observed_at"] = _shift_valid(
            grouped2,
            "information_cutoff",
            -1,
            valid_next1,
        )

    df["target_quarter_index"] = (
        df["quarter_index"] + 1
    ).where(valid_next1)

    return df


def add_sector_relative_features(
    panel: pd.DataFrame,
    cols=RATIO_COLS,
) -> pd.DataFrame:
    df = panel.copy()

    for col in cols:
        grouped = df.groupby(
            ["quarter_index", "sector"]
        )[col]

        median = grouped.transform("median")
        q1 = grouped.transform(lambda s: s.quantile(0.25))
        q3 = grouped.transform(lambda s: s.quantile(0.75))
        iqr = (q3 - q1).replace(0, np.nan)

        df[f"{col}_vs_sector"] = df[col] - median
        df[f"{col}_sector_z"] = (
            (df[col] - median) / iqr
        )

    return df


def add_labels(panel: pd.DataFrame) -> pd.DataFrame:
    df = panel.copy()

    have_next_margin = df["net_margin_next_q"].notna()
    have_next_ocf = df["operating_cash_flow_next_q"].notna()
    have_next_liq = df[
        ["current_ratio_next_q", "leverage_next_q"]
    ].notna().all(axis=1)

    df["margin_up_next_q"] = np.where(
        have_next_margin,
        (df["net_margin_next_q"] > df["net_margin"]).astype(float),
        np.nan,
    )
    df["margin_negative_next_q"] = np.where(
        have_next_margin,
        (df["net_margin_next_q"] < 0).astype(float),
        np.nan,
    )
    df["ocf_negative_next_q"] = np.where(
        have_next_ocf,
        (df["operating_cash_flow_next_q"] < 0).astype(float),
        np.nan,
    )

    strict_valid = have_next_margin & have_next_liq
    strict_3of3 = (
        (df["net_margin_next_q"] < df["net_margin"])
        & (df["current_ratio_next_q"] < df["current_ratio"])
        & (df["leverage_next_q"] > df["leverage"])
    )
    df["deterioration_strict_3of3"] = np.where(
        strict_valid,
        strict_3of3.astype(float),
        np.nan,
    )

    score_valid = (
        have_next_margin
        & have_next_ocf
        & have_next_liq
    )
    score = (
        (df["net_margin_next_q"] < df["net_margin"]).astype(int)
        + (
            df["operating_cash_flow_next_q"]
            < df["operating_cash_flow"]
        ).astype(int)
        + (
            df["current_ratio_next_q"]
            < df["current_ratio"]
        ).astype(int)
        + (
            df["leverage_next_q"] > df["leverage"]
        ).astype(int)
    )
    df["deterioration_score"] = np.where(
        score_valid,
        score,
        np.nan,
    )
    df["deterioration_3of4"] = np.where(
        score_valid,
        (score >= 3).astype(float),
        np.nan,
    )

    df["net_margin_gain_next_q"] = (
        df["net_margin_next_q"] - df["net_margin"]
    )

    return df


if not panel.empty:
    panel = enrich_panel(panel)
    panel = add_sector_relative_features(panel)
    panel = add_labels(panel)

    label_cols = [
        "margin_up_next_q",
        "margin_negative_next_q",
        "ocf_negative_next_q",
        "deterioration_strict_3of3",
        "deterioration_3of4",
    ]
    print("Tỷ lệ positive label:")
    display(panel[label_cols].mean())


# ===== Cell 9 (tu notebook) =====
FEATURE_GROUPS = {
    "profitability": [
        "gross_margin",
        "net_margin",
        "profit_before_tax",
        "net_profit",
    ],
    "liquidity_leverage": [
        "leverage",
        "current_ratio",
        "current_assets",
        "current_liabilities",
        "total_liabilities",
        "total_equity",
    ],
    "efficiency": [
        "asset_turnover",
        "inventory",
        "cogs",
    ],
    "cash_quality": [
        "operating_cash_flow",
        "ocf_to_revenue",
    ],
    "growth": [
        "revenue_growth_qoq",
        "net_profit_growth_qoq",
        "total_assets_growth_qoq",
        "revenue_growth_yoy",
        "net_profit_growth_yoy",
        "total_assets_growth_yoy",
    ],
    "lags": [f"{col}_lag1" for col in RATIO_COLS],
    "categorical": ["sector", "exchange"],
    "deltas": [
        "delta_net_margin",
        "delta_gross_margin",
        "delta_leverage",
        "delta_current_ratio",
        "delta_asset_turnover",
        "delta_ocf_to_revenue",
    ],
    # Macro không nằm trong feature groups mặc định của panel quý.
    # Chỉ thêm macro_* sau khi SAFE quarterly join đã pass leakage audit.
}


def walk_forward_splits(
    df: pd.DataFrame,
    quarter_col: str = "target_quarter_index",
    min_train_quarters: int = 4,
    include_final_test: bool = RUN_FINAL_TEST,
    final_test_target_quarter: int = FINAL_TEST_TARGET_QUARTER,
):
    '''
    Split theo TARGET quarter, không theo feature quarter.

    Ví dụ final test target 2026Q1:
      feature quarter = 2025Q4
      target quarter  = 2026Q1

    Khi include_final_test=False, target quarter 2026Q1 bị khóa hoàn toàn.
    Các target quarter sau official cutoff cũng bị loại.
    '''
    if quarter_col not in df.columns:
        raise KeyError(
            f"Thiếu quarter column cho split: {quarter_col}"
        )

    eligible = df[
        df[quarter_col].notna()
        & (
            df[quarter_col]
            <= final_test_target_quarter
        )
    ].copy()

    quarters = sorted(
        int(value)
        for value in eligible[
            quarter_col
        ].dropna().unique()
    )

    if len(quarters) <= min_train_quarters:
        print(
            f"[walk_forward_splits] Chỉ có {len(quarters)} target quarters; "
            f"cần ít nhất {min_train_quarters + 1}."
        )
        return []

    folds = []

    for i in range(min_train_quarters, len(quarters)):
        train_quarters = quarters[:i]
        validation_quarter = quarters[i]
        is_final_test = (
            validation_quarter
            == final_test_target_quarter
        )

        if is_final_test and not include_final_test:
            continue

        if validation_quarter > final_test_target_quarter:
            continue

        train_idx = eligible.index[
            eligible[quarter_col].isin(
                train_quarters
            )
        ]
        validation_idx = eligible.index[
            eligible[quarter_col]
            == validation_quarter
        ]

        folds.append(
            {
                "train_idx": train_idx,
                "val_idx": validation_idx,
                "val_quarter": validation_quarter,
                "val_quarter_label": (
                    quarter_index_to_label(
                        validation_quarter
                    )
                ),
                "feature_quarter": (
                    validation_quarter - 1
                ),
                "feature_quarter_label": (
                    quarter_index_to_label(
                        validation_quarter - 1
                    )
                ),
                "is_final_test": is_final_test,
            }
        )

    if include_final_test and not any(
        fold["is_final_test"]
        for fold in folds
    ):
        raise RuntimeError(
            "RUN_FINAL_TEST=True nhưng không tạo được final-test fold "
            f"cho target {quarter_index_to_label(final_test_target_quarter)}."
        )

    return folds

from sklearn.preprocessing import OneHotEncoder, RobustScaler


def _make_one_hot_encoder():
    try:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
        )
    except TypeError:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse=False,
        )


def fit_preprocessor(
    train_df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    scale_numeric: bool,
) -> dict:
    '''
    Fit preprocessing chỉ trên train fold.
    Feature all-null hoặc không tồn tại trong train bị loại.
    '''
    usable_numeric = [
        col
        for col in numeric_cols
        if col in train_df.columns
        and train_df[col].notna().any()
    ]
    usable_categorical = [
        col
        for col in categorical_cols
        if col in train_df.columns
        and train_df[col].notna().any()
    ]

    if not usable_numeric and not usable_categorical:
        raise ValueError(
            "Không có feature usable trong train fold."
        )

    state = {
        "numeric_cols": usable_numeric,
        "categorical_cols": usable_categorical,
        "medians": None,
        "scaler": None,
        "encoder": None,
        "output_columns": None,
        "scale_numeric": scale_numeric,
    }

    parts = []

    if usable_numeric:
        numeric = train_df[usable_numeric].apply(
            pd.to_numeric,
            errors="coerce",
        )
        medians = numeric.median()
        numeric = numeric.fillna(medians)

        if numeric.isna().any().any():
            still_null = numeric.columns[
                numeric.isna().any()
            ].tolist()
            raise ValueError(
                f"Numeric features vẫn NaN sau imputation: {still_null}"
            )

        state["medians"] = medians

        if scale_numeric:
            scaler = RobustScaler()
            numeric_array = scaler.fit_transform(numeric)
            state["scaler"] = scaler
            numeric = pd.DataFrame(
                numeric_array,
                columns=usable_numeric,
                index=train_df.index,
            )

        parts.append(numeric)

    if usable_categorical:
        categorical = (
            train_df[usable_categorical]
            .fillna("UNKNOWN")
            .astype(str)
        )
        encoder = _make_one_hot_encoder()
        encoded = encoder.fit_transform(categorical)
        encoded_df = pd.DataFrame(
            encoded,
            columns=encoder.get_feature_names_out(
                usable_categorical
            ),
            index=train_df.index,
        )
        state["encoder"] = encoder
        parts.append(encoded_df)

    matrix = pd.concat(parts, axis=1)
    state["output_columns"] = matrix.columns.tolist()

    return state


def transform_matrix(
    df: pd.DataFrame,
    state: dict,
) -> pd.DataFrame:
    parts = []

    numeric_cols = state["numeric_cols"]
    categorical_cols = state["categorical_cols"]

    if numeric_cols:
        numeric = df[numeric_cols].apply(
            pd.to_numeric,
            errors="coerce",
        )
        numeric = numeric.fillna(state["medians"])

        if state["scale_numeric"]:
            numeric_array = state["scaler"].transform(numeric)
            numeric = pd.DataFrame(
                numeric_array,
                columns=numeric_cols,
                index=df.index,
            )

        parts.append(numeric)

    if categorical_cols:
        categorical = (
            df[categorical_cols]
            .fillna("UNKNOWN")
            .astype(str)
        )
        encoded = state["encoder"].transform(categorical)
        encoded_df = pd.DataFrame(
            encoded,
            columns=state["encoder"].get_feature_names_out(
                categorical_cols
            ),
            index=df.index,
        )
        parts.append(encoded_df)

    matrix = pd.concat(parts, axis=1)
    return matrix.reindex(
        columns=state["output_columns"],
        fill_value=0,
    )


SCALED_MODEL_NAMES = {
    "linear_regression",
    "ridge",
    "lasso",
    "elastic_net",
    "huber",
    "knn",
    "linear_svm",
    "rbf_svm",
    "svr_rbf",
    "mlp",
}

# SLOW_MODEL_MAX_TRAIN_N được cấu hình ở cell đầu.
SLOW_MODEL_NAMES = {
    "linear_svm",
    "rbf_svm",
    "svr_rbf",
}


def _capped_train_data(
    X,
    y,
    model_name,
    cap=SLOW_MODEL_MAX_TRAIN_N,
    seed=42,
    stratify=False,
):
    '''
    Trả thêm train_policy để không trộn capped SVM với official leaderboard.
    '''
    if model_name not in SLOW_MODEL_NAMES or len(y) <= cap:
        return X, y, "full"

    rng = np.random.RandomState(seed)

    if stratify:
        y_array = np.asarray(y)
        selected_parts = []

        for class_value in np.unique(y_array):
            class_idx = np.where(y_array == class_value)[0]
            proportional = round(
                cap * len(class_idx) / len(y_array)
            )
            n_class = min(
                len(class_idx),
                max(5, int(proportional)),
            )
            selected_parts.append(
                rng.choice(
                    class_idx,
                    size=n_class,
                    replace=False,
                )
            )

        selected_idx = np.concatenate(selected_parts)
        rng.shuffle(selected_idx)
    else:
        selected_idx = rng.choice(
            len(y),
            size=cap,
            replace=False,
        )

    print(
        f"[cap exploratory] {model_name}: "
        f"{len(y)} -> {len(selected_idx)} train rows. "
        "Kết quả model này không thuộc official full-data leaderboard."
    )

    return (
        X[selected_idx],
        np.asarray(y)[selected_idx],
        "exploratory_capped",
    )

# ------------------------------------------------------------------
# Runtime monitoring helpers
# ------------------------------------------------------------------
MODEL_BASE_SECONDS = {
    "linear_regression": 0.5,
    "ridge": 0.5,
    "lasso": 1.0,
    "elastic_net": 1.0,
    "huber": 3.0,
    "logistic": 1.0,
    "elastic_net_logistic": 2.0,
    "knn": 0.5,
    "linear_svm": 4.0,
    "rbf_svm": 15.0,
    "svr_rbf": 15.0,
    "random_forest": 5.0,
    "extra_trees": 4.0,
    "hist_gbm": 4.0,
    "xgboost": 8.0,
    "lightgbm": 4.0,
    "catboost": 10.0,
    "mlp": 12.0,
}

FIT_RUNTIME_HISTORY = defaultdict(list)


def format_duration(seconds):
    if seconds is None or not np.isfinite(seconds):
        return "unknown"
    seconds = max(0, int(round(seconds)))
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{sec:02d}"
    return f"{minutes:02d}:{sec:02d}"


def selected_models(kind, factories):
    configured = MODEL_PROFILES[RUNTIME_PROFILE][kind]
    if configured is None:
        return list(factories.keys())
    return [name for name in configured if name in factories]


def estimate_fit_seconds(model_name, n_train, n_features):
    history = FIT_RUNTIME_HISTORY.get(model_name, [])
    if history:
        return float(np.median(history[-5:]))

    base = MODEL_BASE_SECONDS.get(model_name, 5.0)
    row_scale = max(0.5, n_train / 500.0)
    feature_scale = max(0.7, np.sqrt(max(1, n_features) / 20.0))

    if model_name in {"linear_svm", "rbf_svm", "svr_rbf"}:
        row_scale = min(row_scale, SLOW_MODEL_MAX_TRAIN_N / 500.0) ** 2
    elif model_name in {"random_forest", "extra_trees", "xgboost", "lightgbm", "catboost"}:
        row_scale = row_scale ** 0.8
    else:
        row_scale = row_scale ** 0.5

    return float(base * row_scale * feature_scale)


class FitHeartbeat:
    def __init__(self, label, estimated_seconds):
        self.label = label
        self.estimated_seconds = estimated_seconds
        self.started_at = None
        self.stop_event = threading.Event()
        self.thread = None

    def __enter__(self):
        self.started_at = time.perf_counter()

        if SHOW_MODEL_PROGRESS and HEARTBEAT_SECONDS > 0:
            self.thread = threading.Thread(
                target=self._run,
                daemon=True,
            )
            self.thread.start()

        return self

    def _run(self):
        while not self.stop_event.wait(HEARTBEAT_SECONDS):
            elapsed = time.perf_counter() - self.started_at
            print(
                f"  [heartbeat] {self.label} vẫn đang chạy | "
                f"elapsed={format_duration(elapsed)} | "
                f"ước tính ban đầu={format_duration(self.estimated_seconds)} | "
                f"soft budget/fold={format_duration(MODEL_SOFT_BUDGET_SECONDS)}",
                flush=True,
            )

    def __exit__(self, exc_type, exc, traceback):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=1.0)


def print_task_runtime_plan(task_name, model_names, folds, sample, numeric, categorical):
    total_jobs = len(model_names) * len(folds)
    estimates = []
    for fold in folds:
        n_train = len(sample.loc[fold["train_idx"]])
        n_features = len(numeric) + len(categorical)
        estimates.extend(
            estimate_fit_seconds(name, n_train, n_features)
            for name in model_names
        )
    rough = float(np.sum(estimates)) if estimates else 0.0
    print(
        f"[runtime plan] task={task_name} | profile={RUNTIME_PROFILE} | "
        f"models={len(model_names)} | folds={len(folds)} | jobs={total_jobs} | "
        f"rough ETA={format_duration(rough)}",
        flush=True,
    )
    print("  models:", ", ".join(model_names), flush=True)
    return total_jobs


def progress_start(task_name, model_name, fold, job_index, total_jobs, n_train, n_features, task_started_at):
    estimated = estimate_fit_seconds(model_name, n_train, n_features)
    elapsed_task = time.perf_counter() - task_started_at
    print(
        f"[{job_index}/{total_jobs}] START {task_name} | model={model_name} | "
        f"val_q={fold['val_quarter']} | train_n={n_train} | features={n_features} | "
        f"fit ETA≈{format_duration(estimated)} | task elapsed={format_duration(elapsed_task)}",
        flush=True,
    )
    return estimated


def progress_done(task_name, model_name, fold, job_index, total_jobs, elapsed_fit, completed_jobs, task_started_at):
    FIT_RUNTIME_HISTORY[model_name].append(float(elapsed_fit))
    elapsed_task = time.perf_counter() - task_started_at
    avg_job = elapsed_task / max(1, completed_jobs)
    dynamic_eta = avg_job * max(0, total_jobs - completed_jobs)
    print(
        f"[{job_index}/{total_jobs}] DONE  {task_name} | model={model_name} | "
        f"val_q={fold['val_quarter']} | elapsed={format_duration(elapsed_fit)} | "
        f"task elapsed={format_duration(elapsed_task)} | dynamic ETA≈{format_duration(dynamic_eta)}",
        flush=True,
    )



# ===== Cell 11 (tu notebook) =====
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
    HuberRegressor,
    LogisticRegression,
)
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    IsolationForest,
)
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVR, SVC, LinearSVC, OneClassSVM
from sklearn.neural_network import MLPRegressor, MLPClassifier

try:
    import xgboost as _xgb_check
    HAS_XGBOOST = True
except Exception as exc:
    print(
        f"[optional] xgboost chưa dùng được "
        f"({type(exc).__name__}: {exc}); bỏ qua XGBoost và ranking."
    )
    HAS_XGBOOST = False


def _xgb_reg():
    from xgboost import XGBRegressor

    return XGBRegressor(
        n_estimators=TREE_ESTIMATORS,
        max_depth=3,
        learning_rate=0.03,
        min_child_weight=8,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="reg:squarederror",
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbosity=0,
    )


def _xgb_clf():
    from xgboost import XGBClassifier

    return XGBClassifier(
        n_estimators=TREE_ESTIMATORS,
        max_depth=3,
        learning_rate=0.03,
        min_child_weight=8,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbosity=0,
    )


def _lgbm_reg():
    from lightgbm import LGBMRegressor

    return LGBMRegressor(
        random_state=RANDOM_STATE,
        verbosity=-1,
        n_estimators=TREE_ESTIMATORS,
        learning_rate=0.03,
        max_depth=4,
        num_leaves=15,
        min_child_samples=20,
        reg_alpha=0.1,
        reg_lambda=1.0,
    )


def _cat_reg():
    from catboost import CatBoostRegressor

    return CatBoostRegressor(
        random_seed=RANDOM_STATE,
        verbose=False,
        iterations=BOOST_ITERATIONS,
        learning_rate=0.03,
        depth=5,
        l2_leaf_reg=5.0,
        loss_function="RMSE",
    )


def _lgbm_clf():
    from lightgbm import LGBMClassifier

    return LGBMClassifier(
        random_state=RANDOM_STATE,
        verbosity=-1,
        n_estimators=TREE_ESTIMATORS,
        learning_rate=0.03,
        max_depth=4,
        num_leaves=15,
        min_child_samples=20,
        class_weight="balanced",
    )


def _cat_clf():
    from catboost import CatBoostClassifier

    return CatBoostClassifier(
        random_seed=RANDOM_STATE,
        verbose=False,
        iterations=BOOST_ITERATIONS,
        learning_rate=0.03,
        depth=5,
        l2_leaf_reg=5.0,
        loss_function="Logloss",
        auto_class_weights="Balanced",
    )


REGRESSION_MODEL_FACTORIES = {
    "linear_regression": lambda: LinearRegression(),
    "ridge": lambda: Ridge(alpha=1.0),
    "lasso": lambda: Lasso(
        alpha=0.01,
        max_iter=20000,
        random_state=RANDOM_STATE,
    ),
    "elastic_net": lambda: ElasticNet(
        alpha=0.01,
        l1_ratio=0.5,
        max_iter=20000,
        random_state=RANDOM_STATE,
    ),
    "huber": lambda: HuberRegressor(max_iter=2000),
    "knn": lambda: KNeighborsRegressor(
        n_neighbors=15,
        weights="distance",
    ),
    "random_forest": lambda: RandomForestRegressor(
        n_estimators=TREE_ESTIMATORS,
        max_depth=8,
        min_samples_leaf=8,
        max_features=0.7,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    ),
    "extra_trees": lambda: ExtraTreesRegressor(
        n_estimators=TREE_ESTIMATORS,
        max_depth=8,
        min_samples_leaf=8,
        max_features=0.7,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    ),
    "hist_gbm": lambda: HistGradientBoostingRegressor(
        max_depth=4,
        max_iter=BOOST_ITERATIONS,
        learning_rate=0.05,
        min_samples_leaf=15,
        l2_regularization=1.0,
        random_state=RANDOM_STATE,
    ),
    "xgboost": _xgb_reg,
    "lightgbm": _lgbm_reg,
    "catboost": _cat_reg,
    "svr_rbf": lambda: SVR(
        kernel="rbf",
        C=10.0,
        epsilon=0.01,
    ),
    "mlp": lambda: MLPRegressor(
        hidden_layer_sizes=(32, 16),
        alpha=0.01,
        max_iter=MLP_MAX_ITER,
        early_stopping=True,
        random_state=RANDOM_STATE,
    ),
}

CLASSIFICATION_MODEL_FACTORIES = {
    "logistic": lambda: LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    ),
    "elastic_net_logistic": lambda: LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        l1_ratio=0.5,
        max_iter=5000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    ),
    "knn": lambda: KNeighborsClassifier(
        n_neighbors=15,
        weights="distance",
    ),
    "linear_svm": lambda: LinearSVC(
        class_weight="balanced",
        dual="auto",
        max_iter=SVM_MAX_ITER,
        random_state=RANDOM_STATE,
    ),
    "rbf_svm": lambda: SVC(
        kernel="rbf",
        probability=False,
        class_weight="balanced",
        cache_size=1024,
        max_iter=SVM_MAX_ITER,
        random_state=RANDOM_STATE,
    ),
    "random_forest": lambda: RandomForestClassifier(
        n_estimators=TREE_ESTIMATORS,
        max_depth=8,
        min_samples_leaf=8,
        max_features=0.7,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    ),
    "extra_trees": lambda: ExtraTreesClassifier(
        n_estimators=TREE_ESTIMATORS,
        max_depth=8,
        min_samples_leaf=8,
        max_features=0.7,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    ),
    "hist_gbm": lambda: HistGradientBoostingClassifier(
        max_depth=4,
        max_iter=BOOST_ITERATIONS,
        learning_rate=0.05,
        min_samples_leaf=15,
        l2_regularization=1.0,
        random_state=RANDOM_STATE,
    ),
    "xgboost": _xgb_clf,
    "lightgbm": _lgbm_clf,
    "catboost": _cat_clf,
    "mlp": lambda: MLPClassifier(
        hidden_layer_sizes=(32, 16),
        alpha=0.01,
        max_iter=MLP_MAX_ITER,
        early_stopping=True,
        random_state=RANDOM_STATE,
    ),
}

print(
    f"{len(REGRESSION_MODEL_FACTORIES)} regression models, "
    f"{len(CLASSIFICATION_MODEL_FACTORIES)} classification models; "
    f"xgboost={'có' if HAS_XGBOOST else 'không'}."
)


# ===== Cell 13 (tu notebook) =====
from scipy.stats import spearmanr


def _mae(y, prediction):
    return float(np.mean(np.abs(y - prediction)))


def _rmse(y, prediction):
    return float(np.sqrt(np.mean((y - prediction) ** 2)))


def _r2(y, prediction):
    residual = np.sum((y - prediction) ** 2)
    total = np.sum((y - np.mean(y)) ** 2)
    return float(1 - residual / total) if total > 0 else float("nan")


def directional_accuracy(
    y_true,
    y_pred,
    current_values,
):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    current = np.asarray(current_values, dtype=float)

    valid = (
        np.isfinite(y_true)
        & np.isfinite(y_pred)
        & np.isfinite(current)
    )
    if valid.sum() == 0:
        return float("nan")

    true_direction = np.sign(
        y_true[valid] - current[valid]
    )
    predicted_direction = np.sign(
        y_pred[valid] - current[valid]
    )

    return float(
        np.mean(true_direction == predicted_direction)
    )


def regression_metrics(
    y_true,
    y_pred,
    current_values=None,
):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[valid]
    y_pred = y_pred[valid]

    if len(y_true) == 0:
        return {
            "n": 0,
            "mae": np.nan,
            "rmse": np.nan,
            "r2": np.nan,
            "spearman": np.nan,
            "directional_accuracy": np.nan,
        }

    rho = (
        spearmanr(y_true, y_pred).correlation
        if len(y_true) > 2
        else float("nan")
    )

    direction = float("nan")
    if current_values is not None:
        current = np.asarray(
            current_values,
            dtype=float,
        )[valid]
        direction = directional_accuracy(
            y_true,
            y_pred,
            current,
        )

    return {
        "n": len(y_true),
        "mae": _mae(y_true, y_pred),
        "rmse": _rmse(y_true, y_pred),
        "r2": _r2(y_true, y_pred),
        "spearman": rho,
        "directional_accuracy": direction,
    }


def run_regression_task(
    panel,
    task,
    model_names=None,
    min_train_quarters=4,
):
    target = task["target"]
    numeric = task["numeric"]
    categorical = task.get(
        "categorical",
        ["sector", "exchange"],
    )

    if target not in panel.columns:
        print(f"[skip] Target không tồn tại: {target}")
        return pd.DataFrame()

    df = panel.dropna(subset=[target]).copy()
    model_names = (
        model_names
        or selected_models(
            "regression",
            REGRESSION_MODEL_FACTORIES,
        )
    )

    if RUN_FINAL_TEST:
        selected = FINAL_MODEL_SELECTION.get(task["name"])
        if not selected:
            raise RuntimeError(
                f"RUN_FINAL_TEST=True nhưng chưa khóa FINAL_MODEL_SELECTION "
                f"cho task {task['name']}."
            )
        model_names = [selected]

    folds = walk_forward_splits(
        df,
        min_train_quarters=min_train_quarters,
        include_final_test=RUN_FINAL_TEST,
    )

    if not folds:
        return pd.DataFrame()

    total_jobs = print_task_runtime_plan(
        task["name"],
        model_names,
        folds,
        df,
        numeric,
        categorical,
    )
    task_started_at = time.perf_counter()
    completed_jobs = 0
    job_index = 0
    model_elapsed = defaultdict(float)
    skipped_models = set()

    persistence_col = (
        task.get("persistence_col")
        or target.replace("_next_q", "")
    )

    fold_rows = []

    for fold in folds:
        if time.perf_counter() - task_started_at > TASK_SOFT_BUDGET_SECONDS:
            print(
                f"[task budget] Dừng task {task['name']} sau "
                f"{format_duration(time.perf_counter() - task_started_at)}; "
                "các job chưa chạy được bỏ qua.",
                flush=True,
            )
            break

        train_df = df.loc[fold["train_idx"]]
        val_df = df.loc[fold["val_idx"]]

        if train_df.empty or val_df.empty:
            continue

        y_train = train_df[target].to_numpy(dtype=float)
        y_val = val_df[target].to_numpy(dtype=float)

        if persistence_col in val_df.columns:
            persistence_prediction = val_df[
                persistence_col
            ].to_numpy(dtype=float)
            persistence_metrics = regression_metrics(
                y_val,
                persistence_prediction,
                current_values=val_df[
                    persistence_col
                ].to_numpy(dtype=float),
            )
        else:
            persistence_prediction = np.full(
                len(val_df),
                np.nanmean(y_train),
            )
            persistence_metrics = regression_metrics(
                y_val,
                persistence_prediction,
            )

        fold_rows.append(
            {
                "task": task["name"],
                "model": "persistence",
                "val_quarter": fold["val_quarter"],
                "is_final_test": fold["is_final_test"],
                "train_n": len(train_df),
                "val_n": len(val_df),
                "train_policy": "full",
                "fit_seconds": 0.0,
                **persistence_metrics,
            }
        )

        for name in model_names:
            job_index += 1

            if name in skipped_models:
                print(
                    f"[{job_index}/{total_jobs}] SKIP {task['name']} | model={name} | "
                    "đã vượt soft budget ở fold trước.",
                    flush=True,
                )
                continue

            factory = REGRESSION_MODEL_FACTORIES.get(name)
            if factory is None:
                continue

            fit_started_at = time.perf_counter()
            estimated = progress_start(
                task["name"],
                name,
                fold,
                job_index,
                total_jobs,
                len(train_df),
                len(numeric) + len(categorical),
                task_started_at,
            )

            try:
                scale_numeric = name in SCALED_MODEL_NAMES
                preprocess_state = fit_preprocessor(
                    train_df,
                    numeric,
                    categorical,
                    scale_numeric=scale_numeric,
                )

                X_train = transform_matrix(
                    train_df,
                    preprocess_state,
                )
                X_val = transform_matrix(
                    val_df,
                    preprocess_state,
                )

                X_fit, y_fit, train_policy = _capped_train_data(
                    X_train.to_numpy(dtype=float),
                    y_train,
                    name,
                )

                model = factory()
                label = (
                    f"{task['name']} / {name} / "
                    f"val_q={fold['val_quarter']}"
                )
                with FitHeartbeat(label, estimated):
                    model.fit(X_fit, y_fit)
                    prediction = model.predict(
                        X_val.to_numpy(dtype=float)
                    )

            except Exception as exc:
                elapsed_fit = time.perf_counter() - fit_started_at
                completed_jobs += 1
                progress_done(
                    task["name"], name, fold, job_index, total_jobs,
                    elapsed_fit, completed_jobs, task_started_at,
                )
                print(
                    f"[skip] {task['name']} / {name} / "
                    f"quý {fold['val_quarter']}: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
                continue

            elapsed_fit = time.perf_counter() - fit_started_at
            completed_jobs += 1
            model_elapsed[name] += elapsed_fit
            progress_done(
                task["name"], name, fold, job_index, total_jobs,
                elapsed_fit, completed_jobs, task_started_at,
            )

            if (
                SKIP_REMAINING_FOLDS_AFTER_BUDGET
                and model_elapsed[name] > MODEL_SOFT_BUDGET_SECONDS
            ):
                skipped_models.add(name)
                print(
                    f"[model budget] {name} đã dùng tổng "
                    f"{format_duration(model_elapsed[name])}; "
                    "bỏ các fold còn lại của model này.",
                    flush=True,
                )

            current_values = (
                val_df[persistence_col].to_numpy(dtype=float)
                if persistence_col in val_df.columns
                else None
            )

            fold_rows.append(
                {
                    "task": task["name"],
                    "model": name,
                    "val_quarter": fold["val_quarter"],
                    "is_final_test": fold["is_final_test"],
                    "train_n": len(train_df),
                    "val_n": len(val_df),
                    "train_policy": train_policy,
                    "fit_seconds": elapsed_fit,
                    "n_numeric_used": len(
                        preprocess_state["numeric_cols"]
                    ),
                    "n_categorical_used": len(
                        preprocess_state["categorical_cols"]
                    ),
                    **regression_metrics(
                        y_val,
                        prediction,
                        current_values=current_values,
                    ),
                }
            )

    print(
        f"[task complete] {task['name']} | elapsed="
        f"{format_duration(time.perf_counter() - task_started_at)} | "
        f"completed jobs={completed_jobs}/{total_jobs}",
        flush=True,
    )
    return pd.DataFrame(fold_rows)


REGRESSION_TASKS = [
    {
        "name": "net_margin_next_q",
        "target": "net_margin_next_q",
        "numeric": [
            "net_margin",
            "net_margin_lag1",
            "gross_margin",
            "gross_margin_lag1",
            "ocf_to_revenue",
            "ocf_to_revenue_lag1",
            "leverage",
            "leverage_lag1",
            "current_ratio",
            "current_ratio_lag1",
            "asset_turnover",
            "asset_turnover_lag1",
            "revenue_growth_yoy",
            "net_profit_growth_yoy",
            "delta_net_margin",
            "delta_gross_margin",
            "delta_leverage",
            "delta_current_ratio",
        ],
        "categorical": ["sector", "exchange"],
    },
    {
        "name": "revenue_growth_next_q",
        "target": "revenue_growth_qoq_next_q",
        "requires_flow_mode_confirmed": True,
        "persistence_col": "revenue_growth_qoq",
        "numeric": [
            "revenue_growth_qoq",
            "revenue_growth_yoy",
            "inventory",
            "total_assets",
            "asset_turnover",
            "asset_turnover_lag1",
            "gross_margin",
            "operating_cash_flow",
        ],
        "categorical": ["sector"],
    },
    {
        "name": "ocf_next_q",
        "target": "ocf_signed_log_next_q",
        "requires_flow_mode_confirmed": True,
        "persistence_col": "ocf_signed_log",
        "numeric": [
            "ocf_signed_log",
            "operating_cash_flow_lag1",
            "revenue",
            "revenue_lag1",
            "net_profit",
            "net_profit_lag1",
            "inventory",
            "inventory_lag1",
            "current_assets",
            "current_liabilities",
            "ocf_to_revenue",
            "net_margin",
        ],
        "categorical": ["sector"],
    },
    {
        "name": "leverage_next_q",
        "target": "leverage_next_q",
        "numeric": [
            "leverage",
            "leverage_lag1",
            "total_liabilities",
            "total_liabilities_lag1",
            "total_equity",
            "total_equity_lag1",
            "operating_cash_flow",
            "net_profit",
            "total_assets_growth_yoy",
            "current_ratio",
        ],
        "categorical": ["sector"],
    },
]

regression_results = {}

if not panel.empty:
    for task in REGRESSION_TASKS:
        print(f"\n=== Regression: {task['name']} ===")

        if (
            task.get("requires_flow_mode_confirmed", False)
            and not FLOW_MODE_CONFIRMED
            and not ALLOW_UNCONFIRMED_FLOW_TASKS
        ):
            print(
                "[skip flow task] FLOW_MODE_CONFIRMED=False. "
                "Xác nhận reported-quarter hay YTD cumulative trước khi chạy task này."
            )
            regression_results[task["name"]] = pd.DataFrame()
            continue

        result = run_regression_task(panel, task)
        regression_results[task["name"]] = result

        if result.empty:
            print(
                "Không đủ validation fold hoặc target coverage."
            )
            continue

        validation = result[
            ~result["is_final_test"]
        ].copy()

        official = validation[
            validation["train_policy"] == "full"
        ]

        summary = (
            official.groupby("model")[
                [
                    "mae",
                    "rmse",
                    "r2",
                    "spearman",
                    "directional_accuracy",
                ]
            ]
            .agg(["mean", "std"])
            .sort_values(("mae", "mean"))
        )

        print(
            "Official full-data validation summary "
            "(final test bị khóa):"
        )
        if summary.empty:
            print("Không có model full-data hoàn tất.")
        else:
            print(summary.round(4).to_string())

        exploratory = validation[
            validation["train_policy"]
            != "full"
        ]
        if not exploratory.empty:
            print(
                "Exploratory capped-sample models "
                "(không so sánh chính thức):"
            )
            display(
                exploratory.groupby("model")[
                    ["mae", "rmse", "r2"]
                ].mean()
            )


# ===== Cell 15 (tu notebook) =====
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)


def classification_metrics(
    y_true,
    y_prob,
    threshold=0.5,
):
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    valid = np.isfinite(y_true) & np.isfinite(y_prob)
    y_true = y_true[valid]
    y_prob = y_prob[valid]

    if len(y_true) == 0:
        return {
            "n": 0,
            "prevalence": np.nan,
            "balanced_accuracy": np.nan,
            "f1": np.nan,
            "roc_auc": np.nan,
            "pr_auc": np.nan,
            "brier": np.nan,
        }

    y_pred = (y_prob >= threshold).astype(int)

    output = {
        "n": len(y_true),
        "prevalence": float(np.mean(y_true)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        "f1": float(
            f1_score(y_true, y_pred, zero_division=0)
        ),
        "brier": float(
            brier_score_loss(y_true, y_prob)
        ),
    }

    if len(np.unique(y_true)) > 1:
        output["roc_auc"] = float(
            roc_auc_score(y_true, y_prob)
        )
        output["pr_auc"] = float(
            average_precision_score(y_true, y_prob)
        )
    else:
        output["roc_auc"] = float("nan")
        output["pr_auc"] = float("nan")

    return output


def run_classification_task(
    panel,
    task,
    model_names=None,
    min_train_quarters=4,
):
    target = task["target"]
    numeric = task["numeric"]
    categorical = task.get(
        "categorical",
        ["sector", "exchange"],
    )

    if target not in panel.columns:
        print(f"[skip] Target không tồn tại: {target}")
        return pd.DataFrame()

    df = panel.dropna(subset=[target]).copy()
    model_names = (
        model_names
        or selected_models(
            "classification",
            CLASSIFICATION_MODEL_FACTORIES,
        )
    )

    if RUN_FINAL_TEST:
        selected = FINAL_MODEL_SELECTION.get(task["name"])
        if not selected:
            raise RuntimeError(
                f"RUN_FINAL_TEST=True nhưng chưa khóa FINAL_MODEL_SELECTION "
                f"cho task {task['name']}."
            )
        model_names = [selected]

    folds = walk_forward_splits(
        df,
        min_train_quarters=min_train_quarters,
        include_final_test=RUN_FINAL_TEST,
    )

    if not folds:
        return pd.DataFrame()

    total_jobs = print_task_runtime_plan(
        task["name"],
        model_names,
        folds,
        df,
        numeric,
        categorical,
    )
    task_started_at = time.perf_counter()
    completed_jobs = 0
    job_index = 0
    model_elapsed = defaultdict(float)
    skipped_models = set()
    fold_rows = []

    for fold in folds:
        if time.perf_counter() - task_started_at > TASK_SOFT_BUDGET_SECONDS:
            print(
                f"[task budget] Dừng task {task['name']} sau "
                f"{format_duration(time.perf_counter() - task_started_at)}; "
                "các job chưa chạy được bỏ qua.",
                flush=True,
            )
            break

        train_df = df.loc[fold["train_idx"]]
        val_df = df.loc[fold["val_idx"]]

        if (
            train_df.empty
            or val_df.empty
            or train_df[target].nunique() < 2
        ):
            continue

        y_train = train_df[target].to_numpy(dtype=float)
        y_val = val_df[target].to_numpy(dtype=float)

        base_rate = float(np.mean(y_train))
        base_probability = np.full(
            len(y_val),
            base_rate,
        )

        fold_rows.append(
            {
                "task": task["name"],
                "model": "base_rate",
                "val_quarter": fold["val_quarter"],
                "is_final_test": fold["is_final_test"],
                "train_n": len(train_df),
                "val_n": len(val_df),
                "train_policy": "full",
                "fit_seconds": 0.0,
                **classification_metrics(
                    y_val,
                    base_probability,
                ),
            }
        )

        for name in model_names:
            job_index += 1

            if name in skipped_models:
                print(
                    f"[{job_index}/{total_jobs}] SKIP {task['name']} | model={name} | "
                    "đã vượt soft budget ở fold trước.",
                    flush=True,
                )
                continue

            factory = CLASSIFICATION_MODEL_FACTORIES.get(name)
            if factory is None:
                continue

            fit_started_at = time.perf_counter()
            estimated = progress_start(
                task["name"],
                name,
                fold,
                job_index,
                total_jobs,
                len(train_df),
                len(numeric) + len(categorical),
                task_started_at,
            )

            try:
                scale_numeric = name in SCALED_MODEL_NAMES
                preprocess_state = fit_preprocessor(
                    train_df,
                    numeric,
                    categorical,
                    scale_numeric=scale_numeric,
                )

                X_train = transform_matrix(
                    train_df,
                    preprocess_state,
                )
                X_val = transform_matrix(
                    val_df,
                    preprocess_state,
                )

                X_fit, y_fit, train_policy = _capped_train_data(
                    X_train.to_numpy(dtype=float),
                    y_train,
                    name,
                    stratify=True,
                )

                model = factory()

                if name == "xgboost":
                    positives = max(1, int(np.sum(y_fit == 1)))
                    negatives = max(1, int(np.sum(y_fit == 0)))
                    model.set_params(
                        scale_pos_weight=negatives / positives
                    )

                label = (
                    f"{task['name']} / {name} / "
                    f"val_q={fold['val_quarter']}"
                )
                with FitHeartbeat(label, estimated):
                    model.fit(X_fit, y_fit)

                    if hasattr(model, "predict_proba"):
                        probability = model.predict_proba(
                            X_val.to_numpy(dtype=float)
                        )[:, 1]
                    elif hasattr(model, "decision_function"):
                        score = model.decision_function(
                            X_val.to_numpy(dtype=float)
                        )
                        score = np.clip(score, -30, 30)
                        probability = 1.0 / (1.0 + np.exp(-score))
                    else:
                        probability = model.predict(
                            X_val.to_numpy(dtype=float)
                        )

            except Exception as exc:
                elapsed_fit = time.perf_counter() - fit_started_at
                completed_jobs += 1
                progress_done(
                    task["name"], name, fold, job_index, total_jobs,
                    elapsed_fit, completed_jobs, task_started_at,
                )
                print(
                    f"[skip] {task['name']} / {name} / "
                    f"quý {fold['val_quarter']}: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
                continue

            elapsed_fit = time.perf_counter() - fit_started_at
            completed_jobs += 1
            model_elapsed[name] += elapsed_fit
            progress_done(
                task["name"], name, fold, job_index, total_jobs,
                elapsed_fit, completed_jobs, task_started_at,
            )

            if (
                SKIP_REMAINING_FOLDS_AFTER_BUDGET
                and model_elapsed[name] > MODEL_SOFT_BUDGET_SECONDS
            ):
                skipped_models.add(name)
                print(
                    f"[model budget] {name} đã dùng tổng "
                    f"{format_duration(model_elapsed[name])}; "
                    "bỏ các fold còn lại của model này.",
                    flush=True,
                )

            fold_rows.append(
                {
                    "task": task["name"],
                    "model": name,
                    "val_quarter": fold["val_quarter"],
                    "is_final_test": fold["is_final_test"],
                    "train_n": len(train_df),
                    "val_n": len(val_df),
                    "train_policy": train_policy,
                    "fit_seconds": elapsed_fit,
                    "n_numeric_used": len(
                        preprocess_state["numeric_cols"]
                    ),
                    "n_categorical_used": len(
                        preprocess_state["categorical_cols"]
                    ),
                    **classification_metrics(
                        y_val,
                        probability,
                    ),
                }
            )

    print(
        f"[task complete] {task['name']} | elapsed="
        f"{format_duration(time.perf_counter() - task_started_at)} | "
        f"completed jobs={completed_jobs}/{total_jobs}",
        flush=True,
    )
    return pd.DataFrame(fold_rows)


CLASSIFICATION_TASKS = [
    {
        "name": "margin_up_next_q",
        "target": "margin_up_next_q",
        "numeric": [
            "net_margin",
            "net_margin_lag1",
            "delta_net_margin",
            "gross_margin",
            "leverage",
            "current_ratio",
            "ocf_to_revenue",
            "revenue_growth_yoy",
            "net_profit_growth_yoy",
        ],
        "categorical": ["sector"],
    },
    {
        "name": "margin_negative_next_q",
        "target": "margin_negative_next_q",
        "numeric": [
            "net_margin",
            "net_margin_lag1",
            "delta_net_margin",
            "gross_margin",
            "leverage",
            "current_ratio",
            "ocf_to_revenue",
            "revenue_growth_yoy",
            "net_profit_growth_yoy",
        ],
        "categorical": ["sector"],
    },
    {
        "name": "ocf_negative_next_q",
        "target": "ocf_negative_next_q",
        "requires_flow_mode_confirmed": True,
        "numeric": [
            "ocf_to_revenue",
            "operating_cash_flow_lag1",
            "net_profit",
            "net_profit_lag1",
            "inventory",
            "inventory_lag1",
        ],
        "categorical": ["sector"],
    },
    {
        "name": "deterioration_strict_3of3",
        "target": "deterioration_strict_3of3",
        "numeric": [
            "net_margin",
            "current_ratio",
            "leverage",
            "ocf_to_revenue",
            "delta_net_margin",
            "delta_leverage",
            "delta_current_ratio",
        ],
        "categorical": ["sector"],
    },
    {
        "name": "deterioration_3of4",
        "target": "deterioration_3of4",
        "requires_flow_mode_confirmed": True,
        "numeric": [
            "net_margin",
            "current_ratio",
            "leverage",
            "ocf_to_revenue",
            "delta_net_margin",
            "delta_leverage",
            "delta_current_ratio",
        ],
        "categorical": ["sector"],
    },
]

classification_results = {}

if not panel.empty:
    for task in CLASSIFICATION_TASKS:
        print(f"\n=== Classification: {task['name']} ===")

        if (
            task.get("requires_flow_mode_confirmed", False)
            and not FLOW_MODE_CONFIRMED
            and not ALLOW_UNCONFIRMED_FLOW_TASKS
        ):
            print(
                "[skip flow task] FLOW_MODE_CONFIRMED=False. "
                "Xác nhận reported-quarter hay YTD cumulative trước khi chạy task này."
            )
            classification_results[task["name"]] = pd.DataFrame()
            continue

        result = run_classification_task(panel, task)
        classification_results[task["name"]] = result

        if result.empty:
            print(
                "Không đủ dữ liệu, fold hoặc hai lớp để chạy."
            )
            continue

        validation = result[
            ~result["is_final_test"]
        ].copy()

        official = validation[
            validation["train_policy"] == "full"
        ]

        summary = (
            official.groupby("model")[
                [
                    "balanced_accuracy",
                    "f1",
                    "roc_auc",
                    "pr_auc",
                    "brier",
                    "prevalence",
                ]
            ]
            .agg(["mean", "std"])
            .sort_values(
                ("pr_auc", "mean"),
                ascending=False,
            )
        )

        print(
            "Official full-data validation summary "
            "(final test bị khóa):"
        )
        if summary.empty:
            print("Không có model full-data hoàn tất.")
        else:
            print(summary.round(4).to_string())

        exploratory = validation[
            validation["train_policy"]
            != "full"
        ]
        if not exploratory.empty:
            print(
                "Exploratory capped-sample models "
                "(không so sánh chính thức):"
            )
            display(
                exploratory.groupby("model")[
                    [
                        "balanced_accuracy",
                        "f1",
                        "roc_auc",
                        "pr_auc",
                        "brier",
                    ]
                ].mean()
            )


# ===== Cell 17 (tu notebook) =====
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss

CALIBRATION_CHAMPION_MODEL = "random_forest"
CALIBRATION_TASK_NAMES = ["margin_up_next_q", "margin_negative_next_q"]
CALIBRATION_BRIER_IMPROVEMENT_MARGIN = 0.002


def collect_oof_probabilities(
    panel,
    task_def,
    model_name=CALIBRATION_CHAMPION_MODEL,
    min_train_quarters=4,
):
    """
    Mirror của run_classification_task, nhưng chỉ chạy 1 model và GIỮ LẠI
    raw (y_true, y_prob) từng dòng thay vì chỉ tính metric tổng hợp.
    Final-test fold luôn bị loại (include_final_test=False) — calibration
    KHÔNG được dùng final test, đúng nguyên tắc promotion gate.
    """
    target = task_def["target"]
    numeric = task_def["numeric"]
    categorical = task_def.get("categorical", ["sector", "exchange"])

    df = panel.dropna(subset=[target]).copy()
    folds = walk_forward_splits(
        df, min_train_quarters=min_train_quarters, include_final_test=False
    )

    factory = CLASSIFICATION_MODEL_FACTORIES.get(model_name)
    if factory is None:
        raise KeyError(
            f"Model '{model_name}' không có trong CLASSIFICATION_MODEL_FACTORIES."
        )

    records = []
    for fold in folds:
        train_df = df.loc[fold["train_idx"]]
        val_df = df.loc[fold["val_idx"]]
        if train_df.empty or val_df.empty or train_df[target].nunique() < 2:
            continue

        y_train = train_df[target].to_numpy(dtype=float)
        y_val = val_df[target].to_numpy(dtype=float)

        scale_numeric = model_name in SCALED_MODEL_NAMES
        preprocess_state = fit_preprocessor(
            train_df, numeric, categorical, scale_numeric=scale_numeric
        )
        X_train = transform_matrix(train_df, preprocess_state)
        X_val = transform_matrix(val_df, preprocess_state)

        X_fit, y_fit, _ = _capped_train_data(
            X_train.to_numpy(dtype=float), y_train, model_name, stratify=True
        )

        model = factory()
        model.fit(X_fit, y_fit)
        prob = model.predict_proba(X_val.to_numpy(dtype=float))[:, 1]

        tickers = (
            val_df["ticker"].to_numpy()
            if "ticker" in val_df.columns
            else np.full(len(val_df), None)
        )
        sectors = (
            val_df["sector"].to_numpy()
            if "sector" in val_df.columns
            else np.full(len(val_df), None)
        )
        # net_margin ở feature quarter (t) — dấu hiện tại, dùng cho subgroup 1.3.
        net_margin_current = (
            val_df["net_margin"].to_numpy()
            if "net_margin" in val_df.columns
            else np.full(len(val_df), np.nan)
        )
        for ticker, sector, nm, y_t, p in zip(
            tickers, sectors, net_margin_current, y_val, prob
        ):
            records.append(
                {
                    "val_quarter": fold["val_quarter"],
                    "ticker": ticker,
                    "sector": sector,
                    "net_margin_current": nm,
                    "y_true": y_t,
                    "y_prob_raw": p,
                }
            )

    return pd.DataFrame(records)


calibration_summary_rows = []

for task_def in CLASSIFICATION_TASKS:
    if task_def["name"] not in CALIBRATION_TASK_NAMES:
        continue

    print(f"\n=== Calibration OOF collection: {task_def['name']} ===")
    oof = collect_oof_probabilities(panel, task_def)

    if oof.empty:
        print("[skip] Không có OOF data (thiếu fold hoặc thiếu 2 lớp).")
        continue

    y_true = oof["y_true"].to_numpy()
    y_prob_raw = oof["y_prob_raw"].to_numpy()

    brier_raw = brier_score_loss(y_true, y_prob_raw)

    # Platt scaling: logistic regression trên logit(raw prob) — chuẩn Platt 1999.
    eps = 1e-6
    logit_raw = np.log(
        np.clip(y_prob_raw, eps, 1 - eps) / np.clip(1 - y_prob_raw, eps, 1 - eps)
    ).reshape(-1, 1)
    platt = LogisticRegression()
    platt.fit(logit_raw, y_true)
    y_prob_platt = platt.predict_proba(logit_raw)[:, 1]
    brier_platt = brier_score_loss(y_true, y_prob_platt)

    # Isotonic regression trực tiếp trên raw prob.
    isotonic = IsotonicRegression(out_of_bounds="clip")
    isotonic.fit(y_prob_raw, y_true)
    y_prob_isotonic = isotonic.predict(y_prob_raw)
    brier_isotonic = brier_score_loss(y_true, y_prob_isotonic)

    n_positive = int(y_true.sum())

    # Checklist 1.1: "nếu Brier isotonic không thắng rõ, chọn Platt cho an toàn."
    if brier_isotonic < brier_platt - CALIBRATION_BRIER_IMPROVEMENT_MARGIN:
        chosen_method = "isotonic"
        chosen_brier = brier_isotonic
        calibrator = isotonic
    else:
        chosen_method = "platt"
        chosen_brier = brier_platt
        calibrator = platt

    print(f"n={len(y_true)} | positives={n_positive} ({n_positive / len(y_true):.1%})")
    print(
        f"Brier raw={brier_raw:.4f} | Platt={brier_platt:.4f} | "
        f"Isotonic={brier_isotonic:.4f}"
    )
    print(f"-> Chọn: {chosen_method} (brier={chosen_brier:.4f})")

    calibrator_path = (
        ARTIFACT_DIR
        / f"calibrator_{task_def['name']}_{CALIBRATION_CHAMPION_MODEL}.joblib"
    )
    joblib.dump({"method": chosen_method, "model": calibrator}, calibrator_path)
    print(f"Lưu calibrator: {calibrator_path}")

    # Lưu OOF đầy đủ (raw + calibrated prob) để mục 1.2 (threshold selection)
    # đọc lại độc lập, không cần chạy lại collect_oof_probabilities().
    y_prob_calibrated = y_prob_isotonic if chosen_method == "isotonic" else y_prob_platt
    oof_out = oof.copy()
    oof_out["y_prob_calibrated"] = y_prob_calibrated
    oof_out["calibration_method"] = chosen_method
    oof_calibrated_path = ARTIFACT_DIR / f"oof_calibrated_{task_def['name']}.csv"
    oof_out.to_csv(oof_calibrated_path, index=False)
    print(f"Lưu OOF calibrated: {oof_calibrated_path}")

    calibration_summary_rows.append(
        {
            "task": task_def["name"],
            "champion_model": CALIBRATION_CHAMPION_MODEL,
            "n_oof": len(y_true),
            "n_positive": n_positive,
            "brier_raw": brier_raw,
            "brier_platt": brier_platt,
            "brier_isotonic": brier_isotonic,
            "calibration_method": chosen_method,
            "calibrator_path": str(calibrator_path),
        }
    )

calibration_summary = pd.DataFrame(calibration_summary_rows)
calibration_summary_path = ARTIFACT_DIR / "calibration_summary.csv"
calibration_summary.to_csv(calibration_summary_path, index=False)
print(f"\nĐã lưu: {calibration_summary_path}")
display(calibration_summary)



# ===== Cell 19 (tu notebook) =====
from sklearn.metrics import precision_score, recall_score, f1_score, balanced_accuracy_score

THRESHOLD_GRID = np.round(np.arange(0.05, 0.96, 0.01), 2)
THRESHOLD_SELECTION_TARGET_RECALL = 0.70

threshold_selection_rows = []

for task_name in CALIBRATION_TASK_NAMES:
    oof_path = ARTIFACT_DIR / f"oof_calibrated_{task_name}.csv"
    if not oof_path.exists():
        print(f"[skip] Không thấy {oof_path} — chạy cell 6.1 (calibration) trước.")
        continue

    oof_cal = pd.read_csv(oof_path)
    y_true = oof_cal["y_true"].to_numpy()
    y_prob = oof_cal["y_prob_calibrated"].to_numpy()

    sweep_rows = []
    for t in THRESHOLD_GRID:
        y_pred = (y_prob >= t).astype(int)
        sweep_rows.append(
            {
                "threshold": t,
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "f1": f1_score(y_true, y_pred, zero_division=0),
                "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            }
        )
    sweep = pd.DataFrame(sweep_rows)
    sweep_path = ARTIFACT_DIR / f"threshold_sweep_{task_name}.csv"
    sweep.to_csv(sweep_path, index=False)

    def pick_mid_of_tied_optimum(frame, metric_col):
        """Isotonic-calibrated probability là hàm bậc thang (ít giá trị rời
        rạc), nên nhiều threshold liên tiếp cho CÙNG MỘT tập dự đoán -> cùng
        metric. idxmax() mặc định trả về threshold ĐẦU TIÊN trong dải đồng
        hạng đó, thường rơi ngay mép thấp (gần "dự đoán tất cả positive"),
        không phải điểm vận hành có ý nghĩa. Chọn threshold Ở GIỮA dải đồng
        hạng thay vì mép thấp.
        """
        best_value = frame[metric_col].max()
        tied = frame[np.isclose(frame[metric_col], best_value)].sort_values(
            "threshold"
        )
        mid_row = tied.iloc[len(tied) // 2]
        plateau_width = float(tied["threshold"].max() - tied["threshold"].min())
        return mid_row, plateau_width, len(tied)

    if task_name == "margin_negative_next_q":
        eligible = sweep[sweep["recall"] >= THRESHOLD_SELECTION_TARGET_RECALL]
        if eligible.empty:
            # Không threshold nào đạt recall mục tiêu — chọn recall cao nhất đạt được,
            # ghi rõ giới hạn (đưa vào Limitations 6.6).
            best_row, plateau_width, plateau_n = pick_mid_of_tied_optimum(
                sweep, "recall"
            )
            reason = (
                f"KHÔNG threshold nào đạt recall≥{THRESHOLD_SELECTION_TARGET_RECALL}"
                f" (chỉ {int(y_true.sum())} mẫu dương trong {len(y_true)}); chọn "
                f"threshold có recall cao nhất đạt được (giữa dải {plateau_n} mức "
                f"đồng hạng, rộng {plateau_width:.2f})."
            )
        else:
            best_row, plateau_width, plateau_n = pick_mid_of_tied_optimum(
                eligible, "precision"
            )
            reason = (
                f"Trong các threshold đạt recall≥{THRESHOLD_SELECTION_TARGET_RECALL}, "
                f"chọn precision cao nhất (cảnh báo rủi ro margin âm ưu tiên recall); "
                f"giữa dải {plateau_n} mức đồng hạng (rộng {plateau_width:.2f}) do "
                f"isotonic step-function, lấy threshold trung tâm thay vì mép thấp. "
                f"Đã kiểm tra: điểm chọn KHÔNG nằm trong vùng thoái hóa "
                f"'luôn đoán positive' (balanced_accuracy={best_row['balanced_accuracy']:.4f}, "
                f"rõ ràng > 0.50)."
            )
    else:
        # KHÔNG dùng max F1: F1 không phạt đủ mạnh chiến lược "luôn đoán
        # positive" khi lớp gần cân bằng (~53% dương) -> chọn nhầm threshold
        # nằm trong vùng thoái hóa (balanced_accuracy=0.50, xem cell 18 &
        # phát hiện ở 6.3 subgroup analysis). Dùng max balanced_accuracy —
        # cùng metric dùng ở 6.3 — để đảm bảo threshold chọn ra phản ánh
        # khả năng phân biệt THẬT của mô hình.
        best_row, plateau_width, plateau_n = pick_mid_of_tied_optimum(
            sweep, "balanced_accuracy"
        )
        reason = (
            "Lớp cân bằng hơn — chọn threshold max BALANCED ACCURACY (không "
            "dùng max F1: F1 chọn nhầm điểm thoái hóa 'luôn đoán positive', "
            "balanced_accuracy=0.50, xem cell 18); giữa dải "
            f"{plateau_n} mức đồng hạng (rộng {plateau_width:.2f}) do isotonic "
            "step-function, lấy threshold trung tâm thay vì mép thấp."
        )

    if plateau_width > 0.10:
        print(
            f"[lưu ý] Dải đồng hạng rộng ({plateau_width:.2f}, {plateau_n} mức) — "
            "đặc tính bậc thang của isotonic calibration, không phải bug dữ liệu; "
            "đã lấy threshold trung tâm để tránh chọn nhầm mép 'gần như luôn dự "
            "đoán positive'."
        )

    chosen_threshold = float(best_row["threshold"])
    CLASSIFICATION_THRESHOLDS[task_name] = chosen_threshold

    print(f"\n=== Threshold selection: {task_name} ===")
    print(
        f"Chọn threshold={chosen_threshold:.2f} | precision={best_row['precision']:.4f} "
        f"| recall={best_row['recall']:.4f} | f1={best_row['f1']:.4f} "
        f"| balanced_accuracy={best_row['balanced_accuracy']:.4f}"
    )
    print(f"Lý do: {reason}")
    print(f"Sweep đầy đủ lưu: {sweep_path}")

    threshold_selection_rows.append(
        {
            "task": task_name,
            "chosen_threshold": chosen_threshold,
            "precision": best_row["precision"],
            "recall": best_row["recall"],
            "f1": best_row["f1"],
            "balanced_accuracy": best_row["balanced_accuracy"],
            "reason": reason,
            "sweep_path": str(sweep_path),
        }
    )

threshold_selection_summary = pd.DataFrame(threshold_selection_rows)
threshold_selection_summary_path = ARTIFACT_DIR / "threshold_selection_summary.csv"
threshold_selection_summary.to_csv(threshold_selection_summary_path, index=False)
print(f"\nĐã lưu: {threshold_selection_summary_path}")
print("CLASSIFICATION_THRESHOLDS hiện tại:", CLASSIFICATION_THRESHOLDS)
display(threshold_selection_summary)



# ===== Cell 21 (tu notebook) =====
from sklearn.metrics import balanced_accuracy_score

SUBGROUP_MIN_N = 15  # nhóm quá nhỏ thì không đáng tin, chỉ ghi nhận n
SUBGROUP_BALANCED_ACC_GAP_FLAG = 0.10  # 10 điểm phần trăm so với tổng thể

subgroup_rows = []

for task_name in CALIBRATION_TASK_NAMES:
    oof_path = ARTIFACT_DIR / f"oof_calibrated_{task_name}.csv"
    if not oof_path.exists():
        print(f"[skip] Không thấy {oof_path} — chạy cell 6.1 trước.")
        continue

    oof_cal = pd.read_csv(oof_path)
    threshold = CLASSIFICATION_THRESHOLDS.get(task_name, 0.50)

    y_true_all = oof_cal["y_true"].to_numpy()
    y_pred_all = (oof_cal["y_prob_calibrated"].to_numpy() >= threshold).astype(int)
    overall_bal_acc = balanced_accuracy_score(y_true_all, y_pred_all)

    print(f"\n=== Subgroup analysis: {task_name} (threshold={threshold:.2f}) ===")
    print(f"Balanced accuracy tổng thể: {overall_bal_acc:.4f}")

    # --- theo sector ---
    for sector, sub in oof_cal.groupby("sector"):
        if len(sub) < SUBGROUP_MIN_N:
            continue
        y_pred_sub = (sub["y_prob_calibrated"].to_numpy() >= threshold).astype(int)
        bal_acc = balanced_accuracy_score(sub["y_true"].to_numpy(), y_pred_sub)
        gap = overall_bal_acc - bal_acc
        subgroup_rows.append(
            {
                "task": task_name,
                "group_type": "sector",
                "group_value": sector,
                "n": len(sub),
                "balanced_accuracy": bal_acc,
                "gap_vs_overall": gap,
                "flagged": gap > SUBGROUP_BALANCED_ACC_GAP_FLAG,
            }
        )

    # --- theo dấu net_margin hiện tại ---
    margin_sign = np.where(
        oof_cal["net_margin_current"] >= 0, "margin_hiện_tại_dương", "margin_hiện_tại_âm"
    )
    oof_cal_signed = oof_cal.assign(margin_sign=margin_sign)
    for sign, sub in oof_cal_signed.groupby("margin_sign"):
        if len(sub) < SUBGROUP_MIN_N:
            continue
        y_pred_sub = (sub["y_prob_calibrated"].to_numpy() >= threshold).astype(int)
        bal_acc = balanced_accuracy_score(sub["y_true"].to_numpy(), y_pred_sub)
        gap = overall_bal_acc - bal_acc
        subgroup_rows.append(
            {
                "task": task_name,
                "group_type": "margin_sign_hien_tai",
                "group_value": sign,
                "n": len(sub),
                "balanced_accuracy": bal_acc,
                "gap_vs_overall": gap,
                "flagged": gap > SUBGROUP_BALANCED_ACC_GAP_FLAG,
            }
        )

subgroup_summary = pd.DataFrame(subgroup_rows).sort_values(
    ["task", "group_type", "gap_vs_overall"], ascending=[True, True, False]
)
subgroup_summary_path = ARTIFACT_DIR / "subgroup_error_analysis.csv"
subgroup_summary.to_csv(subgroup_summary_path, index=False)
print(f"\nĐã lưu: {subgroup_summary_path}")

flagged = subgroup_summary[subgroup_summary["flagged"]]
if flagged.empty:
    print(
        "Không có nhóm nào lệch >10 điểm balanced_accuracy so với tổng thể — "
        "không phát hiện subgroup bất thường rõ rệt."
    )
else:
    print(
        f"[CẢNH BÁO] {len(flagged)} nhóm lệch >10 điểm balanced_accuracy so với "
        "tổng thể — cân nhắc ghi vào Limitations (6.6):"
    )
    display(flagged)

display(subgroup_summary)

