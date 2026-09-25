"""Chạy negative-control cho RQ6 (2026-09-04) - vá lỗ hổng
"không có kiểm định giả định nhân quả" nêu ở
docs/RQ_audit_truoc_khi_training_2026-08-14.md dòng 154-166.

Chạy CÙNG pipeline (estimate_ate() + estimate_ate_dml(), cùng covariate)
trên cặp treatment/outcome mà lý thuyết dự đoán KHÔNG có hiệu ứng (xem
src/causal/negative_control.py's NEGATIVE_CONTROL_PAIRS + rationale), rồi in
diễn giải theo phương pháp Lipsitch et al. (2010). Dùng lại nguyên
scripts/run_causal_experiment.py's run_experiment() để MỌI lượt chạy (kể cả
negative-control) đều được ghi vào causal_experiment_runs - "cặp này đã được
kiểm tra bao nhiêu lần, kết quả gì" trả lời được từ DB, không phải từ trí
nhớ của người chạy.

CẦN Postgres thật - không chạy được trong sandbox này. Xem
tests/test_causal/test_negative_control.py cho phần logic thuần Python
(không cần DB) đã được test riêng.

Usage:
    python3 -m scripts.run_negative_control_check
    python3 -m scripts.run_negative_control_check --estimator both
"""
import argparse

from scripts.run_causal_experiment import run_experiment
from src.causal.negative_control import NEGATIVE_CONTROL_PAIRS, interpret_negative_control
from src.causal.treatment_effect import estimate_ate, estimate_ate_dml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--estimator", choices=["t_learner", "dml", "both"], default="both")
    ap.add_argument("--covariates", default="sector,fiscal_year")
    args = ap.parse_args()
    covariates = tuple(c.strip() for c in args.covariates.split(",") if c.strip())

    print(f"=== Negative-control check cho RQ6 ({len(NEGATIVE_CONTROL_PAIRS)} cặp) ===\n")

    for pair in NEGATIVE_CONTROL_PAIRS:
        label = f"{pair['treatment']} -> {pair['outcome']}"
        print(f"--- {label} ---")
        print(f"Lý do chọn cặp này: {pair['rationale']}\n")

        # Ghi lại vào causal_experiment_runs (cùng cơ chế log như mọi lượt
        # chạy causal khác) VÀ lấy dict kết quả để diễn giải ngay tại đây -
        # gọi trực tiếp estimate_ate()/estimate_ate_dml() thay vì chỉ dựa
        # vào run_experiment() vì hàm đó chỉ IN ra, không TRẢ VỀ dict đầy đủ
        # theo dạng interpret_negative_control() cần.
        run_experiment(treatment=pair["treatment"], outcome=pair["outcome"],
                       covariates=covariates, estimator=args.estimator)

        if args.estimator in ("t_learner", "both"):
            result = estimate_ate(treatment_col=pair["treatment"], outcome_col=pair["outcome"],
                                  covariates=covariates)
            verdict = interpret_negative_control(result, pair_label=label)
            print(f"  {verdict['message']}\n")

        if args.estimator in ("dml", "both"):
            result = estimate_ate_dml(treatment_col=pair["treatment"], outcome_col=pair["outcome"],
                                      covariates=covariates)
            verdict = interpret_negative_control(result, pair_label=label)
            print(f"  {verdict['message']}\n")


if __name__ == "__main__":
    main()
