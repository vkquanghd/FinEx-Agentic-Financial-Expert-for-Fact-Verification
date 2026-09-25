"""Cross-sectional average treatment effect (ATE) estimation across the
company panel - e.g. "does higher leverage cause lower net margin,
controlling for sector and year?".

Method: T-learner (Künzel et al. 2019, "meta-learners" - the same family of
estimator CausalML/EconML wrap). Two regularized regression models are
fit, one on treated units and one on control units, both predicting the
outcome from covariates. ATE = mean over ALL units of (predicted outcome
under treatment - predicted outcome under control). This needs MANY
cross-sectional units, not a long time series - which matches the actual
panel shape here (216 companies x periods), unlike single-company
time-series causal discovery (PCMCI etc.), which is NOT implemented here
because 15-20 periods per company is too short for it to be reliable.

Honesty-by-construction: every result carries an explicit confidence label
driven by actual sample size (see _confidence_label). Nothing here claims
more certainty than the current data supports - as more Vietstock exports
are loaded, panel size n grows and the label upgrades automatically.

If the `causalml` package is installed, its BaseTRegressor is used as a
cross-check against the hand-rolled T-learner below (optional, not
required - kept dependency-light since causalml is a heavy/fragile
install; the estimator implemented here is not a placeholder, it is the
same T-learner method causalml wraps).

**R-learner / Double Machine Learning cross-check (added 2026-07-26,
`estimate_ate_dml()` below), via EconML's `LinearDML`.** Added after a
review of this module raised a real methodological limitation of the
T-learner above: `estimate_ate()` binarizes a continuous treatment (e.g.
leverage) at its median before fitting two separate Ridge models - this
throws away real variance/signal in the treatment variable (a company at
the 51st percentile of leverage is treated identically to one at the 99th),
and can be more sensitive to regularization bias when the true effect is
small relative to outcome variance. `estimate_ate_dml()` is a SEPARATE,
ADDITIONAL estimator, not a replacement:
- Keeps the treatment CONTINUOUS (no median split) - this is the actual
  methodological upgrade, not just "a different library".
- `LinearDML` is EconML's implementation of the R-learner idea (Nie & Wager
  2017; the residual-on-residual regression traces back to Robinson 1988) -
  citing it as "R-learner + DML" in the thesis is accurate, they are the
  same underlying estimator here, not two separate things bolted together.
- Deliberately uses `RidgeCV`/`LassoCV` as BOTH DML nuisance models
  (model_y, model_t) - the SAME regularized-linear model family as the
  T-learner's two Ridge models above, not a fancier gradient-boosted/forest
  nuisance model. This is a controlled-comparison choice: if the nuisance
  model family were also changed, a difference between the two ATE results
  could come from either "T-learner vs DML" or "Ridge vs boosted trees" -
  keeping the nuisance family fixed isolates the actual variable being
  compared (the estimation STRATEGY, binarized-two-model vs continuous-
  residualized), consistent with this file's own "no unlabeled hyperparameter
  tuning" stance elsewhere (see MIN_N_* thresholds, `_confidence_label`).
- Optional dependency, same as causalml above - NOT pinned as a hard
  requirement (heavy install: numba/shap/etc as transitive deps), see
  requirements.txt's comment. `estimate_ate_dml()` returns `ok=False` with
  an explicit message if `econml` isn't installed, same "don't fabricate,
  say what's missing" pattern as everywhere else in this project.
- NOT wired into src/router/query_router.py's `causal` branch - the Router
  still calls `estimate_ate()` (T-learner) only, same reasoning as Nhánh B
  (Text-to-SQL Agent) being kept standalone: this is a robustness/
  comparison tool for the thesis write-up, not a second live code path the
  grading demo depends on having `econml` installed to run.

**Neo vào literature (thêm 2026-09-04, cập nhật 2026-09-04 sang trích dẫn
2023-nay theo yêu cầu, vá lỗ hổng nêu ở
docs/RQ_audit_truoc_khi_training_2026-08-14.md dòng 154-157: "Không có neo
literature... Đây là chỗ giám khảo quant sẽ xoáy mạnh nhất").** Trích dẫn
dưới đây đã XÁC MINH qua WebSearch 2026-09-04 (không phải nhớ lại từ
training data), năm xuất bản 2023-nay:

- Abdeljawad, I., & Jaradat, M. N. (2025). "Revisiting the pecking order
  theory: insights from an emerging market economy." International Review
  of Economics, 72, article 33. https://doi.org/10.1007/s12232-025-00509-7
  (xuất bản 27/10/2025). Kiểm định thực nghiệm lý thuyết pecking-order (POT)
  trên panel doanh nghiệp phi tài chính niêm yết tại một thị trường mới nổi
  (Palestine Exchange, 29 công ty, 2005-2019) - kết quả: **lợi nhuận
  (profitability) có quan hệ NGHỊCH với đòn bẩy** (profitability và cơ hội
  tăng trưởng tương quan âm với leverage), khớp dự đoán gốc của lý thuyết
  pecking-order (Myers 1984 - vẫn là nguồn gốc lý thuyết, chỉ không dùng làm
  trích dẫn bằng chứng chính vì ngoài khung 2023-nay). Bài báo 2025 này được
  chọn làm trích dẫn NEO chính vì: (a) thời gian xuất bản trong khung yêu
  cầu, (b) cùng bối cảnh thị trường mới nổi/thanh khoản mỏng như Việt Nam
  (không phải Mỹ/Tây Âu), (c) cùng chiều đo (profitability vs leverage) như
  cặp `leverage -> net_margin` đang xét ở đây.

**Đối chiếu với số liệu THẬT đã đo** (`docs/eval_run2_results.md` dòng
32-38, cặp leverage -> net_margin, panel thật):

    T-learner ATE  = -0.0971  [95% CI: -0.1871, -0.0264]
    DML (quy đổi)  = -0.1456

**Cả hai estimator đều cho dấu ÂM** - đòn bẩy càng cao, biên lợi nhuận ròng
càng thấp. Đây **khớp hướng** với kết quả thực nghiệm 2025 nêu trên (và với
dự đoán gốc của pecking-order theory, Myers 1984). Cách phát biểu AN TOÀN
cho thesis: nói "khớp hướng với bằng chứng thực nghiệm gần đây (Abdeljawad &
Jaradat, 2025) trên thị trường mới nổi", KHÔNG nói "chứng minh pecking-order
theory đúng" - một ước lượng ATE cắt ngang (cross-sectional) trên panel 9
quý, 2 cặp, không đủ để kiểm định giữa pecking-order và trade-off theory
(hai lý thuyết có thể cùng dự đoán chiều quan hệ này qua các cơ chế khác
nhau).

**Negative-control cho cặp trên** (Yang, Q., Yang, Z., Cai, X., Zhao, H.,
Jia, J., & Sun, F. (2023/2024). "Advances in methodologies of negative
controls: a scoping review." Journal of Clinical Epidemiology, 166, 111227.
PubMed: 38040387 - xem src/causal/negative_control.py): trước khi tin dấu ÂM
này, chạy CÙNG pipeline trên một cặp lý thuyết dự đoán KHÔNG có hiệu ứng
(`current_ratio -> gross_margin`) - nếu KTC 95% của cặp đó cũng không chứa
0, đó là cảnh báo cả pipeline (không riêng leverage->net_margin) có
confounding chưa kiểm soát.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder

from src.causal.panel_builder import build_panel

MIN_N_EXPLORATORY = 10   # below this, refuse to estimate at all
MIN_N_PRELIMINARY = 30
MIN_N_STANDARD = 100

# DML-specific: number of cross-fitting folds. Lower than EconML's often-
# cited default of 5 because this project's panel (216 companies x however
# many fiscal years are loaded) is small - 3 folds keeps each fold large
# enough to fit a RidgeCV nuisance model without starving it of rows, at
# the cost of slightly higher variance in the cross-fit estimate. A round,
# documented judgment call, not fit against labelled data (none exists),
# same honesty as every other untuned constant in this module.
DML_CV_FOLDS = 3


def _confidence_label(n: int) -> str:
    if n < MIN_N_PRELIMINARY:
        return f"exploratory (n={n} < {MIN_N_PRELIMINARY}) - minh hoạ, KHÔNG đủ mạnh để kết luận nhân quả"
    if n < MIN_N_STANDARD:
        return f"preliminary (n={n} < {MIN_N_STANDARD}) - xu hướng sơ bộ, nên diễn giải thận trọng"
    return f"standard power (n={n} >= {MIN_N_STANDARD}) - đủ cỡ mẫu cho ước lượng thông thường"


def _design_matrix(df: pd.DataFrame, covariates: list[str]) -> np.ndarray:
    # Bug found live 2026-09-07: the old check `df[c].dtype == object or
    # str(df[c].dtype).startswith("category")` misses pandas' newer nullable
    # "string" dtype (e.g. StringDtype from a modern SQLAlchemy/pyarrow read
    # path) - a text column read with that dtype fell through into num_cols
    # below, and `.to_numpy(dtype=float)` crashed with
    # "ValueError: could not convert string to float: 'Chưa phân loại'"
    # (the default placeholder for companies.sector, src/ingestion/
    # vietstock_etl.py). `pd.api.types.is_numeric_dtype()` is the robust,
    # idiomatic check - it correctly excludes object/string/category/bool
    # dtypes regardless of backend, instead of enumerating dtype spellings
    # by hand. See tests/test_causal/test_design_matrix_dtype.py.
    cat_cols = [c for c in covariates if not pd.api.types.is_numeric_dtype(df[c])]
    num_cols = [c for c in covariates if c not in cat_cols]

    parts = []
    if num_cols:
        parts.append(df[num_cols].to_numpy(dtype=float))
    if cat_cols:
        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        parts.append(enc.fit_transform(df[cat_cols].astype(str)))
    if not parts:
        return np.ones((len(df), 1))
    return np.hstack(parts)


def estimate_ate(
    treatment_col: str = "leverage",
    outcome_col: str = "net_margin",
    covariates: tuple[str, ...] = ("sector", "fiscal_year"),
    panel: pd.DataFrame = None,
    n_bootstrap: int = 200,
    seed: int = 42,
) -> dict:
    """Estimate ATE of `treatment_col` (split at its median into high/low)
    on `outcome_col`, adjusting for `covariates`, across the company panel.
    Returns a dict with the estimate, a bootstrap 95% CI, sample sizes, and
    an explicit confidence label - never a bare number with no caveat.

    Beyond "sector"/"fiscal_year", panel_builder.build_panel() also joins
    real macro covariates - "cpi_annual_avg" and "gdp_growth_annual_avg"
    (see src/causal/macro_lookup.py) - which can be added here, e.g.
    covariates=("sector", "fiscal_year", "cpi_annual_avg"), to control for
    a shared year-level macro confounder instead of just a bare year dummy.
    """
    if panel is None:
        panel = build_panel()

    needed = [treatment_col, outcome_col] + list(covariates)
    df = panel.dropna(subset=needed).reset_index(drop=True)
    n = len(df)

    if n < MIN_N_EXPLORATORY:
        return {
            "ok": False,
            "error": f"n={n} < {MIN_N_EXPLORATORY} - không đủ dữ liệu để ước lượng, kể cả ở mức thăm dò.",
        }

    median = df[treatment_col].median()
    treated = (df[treatment_col] > median).to_numpy()
    n_treated, n_control = int(treated.sum()), int((~treated).sum())
    if min(n_treated, n_control) < 5:
        return {
            "ok": False,
            "error": f"Nhóm treated/control quá lệch (n_treated={n_treated}, n_control={n_control}) - cần thêm dữ liệu.",
        }

    # Defense-in-depth, added 2026-09-07 after a live crash: everything from
    # here down does real numeric work (dtype conversion, Ridge fitting,
    # bootstrap) on data this function doesn't fully control the shape of
    # (panel_builder.py's real DB output). `_design_matrix`'s dtype bug above
    # is now fixed at the ROOT, but this function should never again let an
    # unexpected internal error escape as a raw Python traceback all the way
    # to the Streamlit/FastAPI UI - every OTHER branch in this router
    # already returns {"ok": False, "error": ...} on failure instead of
    # raising (see src/structured/calculator.py's pre-checks,
    # src/router/query_router.py's _format_causal() which already handles
    # "ok": False gracefully) - this brings the causal branch to the same
    # standard instead of being the one exception that can crash the page.
    try:
        X = _design_matrix(df, list(covariates))
        y = df[outcome_col].to_numpy(dtype=float)

        def _t_learner_ate(idx: np.ndarray) -> float:
            Xi, yi, ti = X[idx], y[idx], treated[idx]
            if ti.sum() < 3 or (~ti).sum() < 3:
                return np.nan
            m1 = Ridge(alpha=1.0).fit(Xi[ti], yi[ti])
            m0 = Ridge(alpha=1.0).fit(Xi[~ti], yi[~ti])
            return float(np.mean(m1.predict(Xi) - m0.predict(Xi)))

        all_idx = np.arange(n)
        point_estimate = _t_learner_ate(all_idx)

        rng = np.random.default_rng(seed)
        boot = []
        for _ in range(n_bootstrap):
            sample = rng.choice(all_idx, size=n, replace=True)
            v = _t_learner_ate(sample)
            if not np.isnan(v):
                boot.append(v)
        ci_low, ci_high = (np.percentile(boot, [2.5, 97.5]) if boot else (np.nan, np.nan))

        naive_diff = float(y[treated].mean() - y[~treated].mean())

        result = {
            "ok": True,
            "treatment": treatment_col,
            "outcome": outcome_col,
            "covariates": list(covariates),
            "median_split_at": float(median),
            "n": n,
            "n_treated": n_treated,
            "n_control": n_control,
            "ate_t_learner": point_estimate,
            "ate_95ci": [float(ci_low), float(ci_high)],
            "naive_mean_diff": naive_diff,
            "confidence": _confidence_label(n),
            "method": "T-learner (Ridge), bootstrap 95% CI, covariate-adjusted for sector + fiscal_year",
        }
    except Exception as e:
        return {
            "ok": False,
            "error": f"Lỗi nội bộ khi ước lượng T-learner ({type(e).__name__}): {e}",
        }

    # Optional cross-check via causalml - fail-soft on ANY error (not just
    # ImportError), same "an optional side-computation must never crash the
    # main result" pattern used elsewhere in this router (e.g. the FTS
    # lookup in query_router.py's auditor branch).
    try:
        from causalml.inference.meta import BaseTRegressor
        from sklearn.linear_model import Ridge as _Ridge
        tl = BaseTRegressor(learner=_Ridge(alpha=1.0))
        te, lb, ub = tl.estimate_ate(X=X, treatment=treated.astype(int), y=y)
        result["causalml_crosscheck"] = {"ate": float(te[0]), "ci": [float(lb[0]), float(ub[0])]}
    except ImportError:
        result["causalml_crosscheck"] = None  # optional package not installed, T-learner above already valid
    except Exception:
        result["causalml_crosscheck"] = None  # optional cross-check failed - main T-learner result still valid

    return result


def estimate_ate_dml(
    treatment_col: str = "leverage",
    outcome_col: str = "net_margin",
    covariates: tuple[str, ...] = ("sector", "fiscal_year"),
    panel: pd.DataFrame = None,
) -> dict:
    """R-learner / Double Machine Learning estimate of the effect of
    `treatment_col` on `outcome_col`, via EconML's `LinearDML` - see this
    module's docstring for the full methodological rationale (continuous
    treatment, shared Ridge/Lasso nuisance family with estimate_ate(), why
    this is a SEPARATE cross-check rather than a replacement for the
    T-learner above).

    Unlike estimate_ate(), the treatment is NOT split at its median - the
    whole point of this estimator is to use its continuous variation. This
    means the resulting effect is not directly the same quantity as the
    T-learner's "high vs low group" contrast: `LinearDML` (with no
    treatment featurizer, as used here) estimates a single CONSTANT
    MARGINAL EFFECT - the expected change in `outcome_col` per +1.0 unit
    change in `treatment_col`, holding covariates fixed - not a treated-
    vs-control group difference. Returned as `ate_dml_marginal`.

    To let a reader compare this against estimate_ate()'s group-contrast
    number without conflating the two, `translated_group_contrast` also
    reports `ate_dml_marginal * (mean(treatment | above median) -
    mean(treatment | below median))` - i.e. "what the marginal effect
    would imply for a gap the same size as the T-learner's median split",
    which is only a faithful translation UNDER the assumption LinearDML
    itself makes here (a constant/linear marginal effect across the range
    of treatment_col) - stated explicitly in the result, not hidden.

    Returns {"ok": False, "error": ...} if `econml` isn't installed, or if
    there isn't enough data (same MIN_N_* thresholds as estimate_ate(), for
    a like-for-like sample-size floor between the two estimators)."""
    try:
        from econml.dml import LinearDML
    except ImportError:
        return {
            "ok": False,
            "error": (
                "Chưa cài đặt `econml` (pip install econml) - "
                "estimate_ate_dml() không chạy được. estimate_ate() (T-learner) "
                "ở trên KHÔNG cần econml, vẫn dùng được bình thường."
            ),
        }

    if panel is None:
        panel = build_panel()

    needed = [treatment_col, outcome_col] + list(covariates)
    df = panel.dropna(subset=needed).reset_index(drop=True)
    n = len(df)

    if n < MIN_N_EXPLORATORY:
        return {
            "ok": False,
            "error": f"n={n} < {MIN_N_EXPLORATORY} - không đủ dữ liệu để ước lượng, kể cả ở mức thăm dò.",
        }

    from sklearn.linear_model import RidgeCV

    cv = min(DML_CV_FOLDS, n)  # can't cross-fit with more folds than rows
    if cv < 2:
        return {
            "ok": False,
            "error": f"n={n} quá nhỏ để chia {DML_CV_FOLDS}-fold cross-fitting cho DML.",
        }

    # Same defense-in-depth as estimate_ate() above (2026-09-07) - never let
    # an unexpected internal error (dtype issue, LinearDML fit failure, etc.)
    # escape as a raw traceback; return the standard {"ok": False, "error"}
    # shape every other branch already uses.
    try:
        W = _design_matrix(df, list(covariates))
        T = df[treatment_col].to_numpy(dtype=float)
        Y = df[outcome_col].to_numpy(dtype=float)

        # Both nuisance models are RidgeCV - deliberately the SAME regularized-
        # linear family as estimate_ate()'s two Ridge models, not a fancier
        # learner for either the outcome or the treatment nuisance (see module
        # docstring for why this is a controlled-comparison choice).
        est = LinearDML(
            model_y=RidgeCV(),
            model_t=RidgeCV(),
            discrete_treatment=False,
            cv=cv,
            random_state=42,
        )
        est.fit(Y, T, W=W)

        marginal_ate = float(est.ate())
        ci_low, ci_high = est.ate_interval(alpha=0.05)
        ci_low, ci_high = float(ci_low), float(ci_high)

        median = df[treatment_col].median()
        treated_mean = df.loc[df[treatment_col] > median, treatment_col].mean()
        control_mean = df.loc[df[treatment_col] <= median, treatment_col].mean()
        implied_gap = float(treated_mean - control_mean)
        translated = marginal_ate * implied_gap
    except Exception as e:
        return {
            "ok": False,
            "error": f"Lỗi nội bộ khi ước lượng DML ({type(e).__name__}): {e}",
        }

    return {
        "ok": True,
        "treatment": treatment_col,
        "outcome": outcome_col,
        "covariates": list(covariates),
        "n": n,
        "cv_folds": cv,
        "ate_dml_marginal": round(marginal_ate, 6),
        "ate_dml_marginal_95ci": [round(ci_low, 6), round(ci_high, 6)],
        "translated_group_contrast": round(translated, 4),
        "translated_group_contrast_note": (
            f"= marginal_effect * (trung bình {treatment_col} nhóm trên median "
            f"- trung bình nhóm dưới median) = {marginal_ate:.6f} * {implied_gap:.4f}. "
            "CHỈ là quy đổi để so sánh trực quan với estimate_ate() (T-learner) - "
            "giả định hiệu ứng biên KHÔNG đổi trên toàn khoảng giá trị của "
            f"{treatment_col} (đúng theo thiết kế của LinearDML không dùng "
            "treatment featurizer ở đây), một giả định thực sự, không phải "
            "một phép tính trung lập."
        ),
        "confidence": _confidence_label(n),
        "method": (
            f"R-learner / LinearDML (EconML), nuisance models RidgeCV/RidgeCV "
            f"(cùng họ Ridge với T-learner ở estimate_ate(), có chủ đích - xem "
            f"docstring module), {cv}-fold cross-fitting, treatment GIỮ LIÊN TỤC "
            f"(không median-split)."
        ),
    }


if __name__ == "__main__":
    import json
    import sys

    args = [a for a in sys.argv[1:] if a != "--dml"]
    want_dml = "--dml" in sys.argv[1:]
    treatment = args[0] if len(args) > 0 else "leverage"
    outcome = args[1] if len(args) > 1 else "net_margin"

    print("=== estimate_ate() - T-learner (Ridge, median-split treatment) ===")
    res = estimate_ate(treatment_col=treatment, outcome_col=outcome)
    print(json.dumps(res, indent=2, ensure_ascii=False))

    if want_dml:
        print("\n=== estimate_ate_dml() - R-learner/DML (EconML LinearDML, continuous treatment) ===")
        res_dml = estimate_ate_dml(treatment_col=treatment, outcome_col=outcome)
        print(json.dumps(res_dml, indent=2, ensure_ascii=False))
