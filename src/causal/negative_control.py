"""Negative-control kiểm tra cho RQ6 (2026-09-04) — vá lỗ hổng đã nêu ở
`docs/RQ_audit_truoc_khi_training_2026-08-14.md` dòng 154-166: RQ6 "không có
kiểm định giả định nhân quả... không có placebo/negative control".

Phương pháp: Yang, Q., Yang, Z., Cai, X., Zhao, H., Jia, J., & Sun, F.
(2023/2024). "Advances in methodologies of negative controls: a scoping
review." Journal of Clinical Epidemiology, 166, 111227.
(PubMed: https://pubmed.ncbi.nlm.nih.gov/38040387/ — chấp nhận đăng online
30/11/2023, xác minh qua WebSearch 2026-09-04, không phải nhớ từ training
data; trích dẫn cập nhật 2026-09-04 để nằm trong khung 2023-nay theo yêu
cầu, thay cho trích dẫn gốc Lipsitch et al. 2010 — bài báo 2023 này là một
scoping review tổng hợp 37 bài phương pháp luận negative-control 2010-2023,
bao trùm cả ý tưởng gốc của Lipsitch et al.). Ý tưởng cốt lõi (không đổi so
với bản gốc, chỉ đổi nguồn trích dẫn): chọn một cặp treatment/outcome mà LÝ
THUYẾT dự đoán KHÔNG có quan hệ nhân quả, rồi chạy CÙNG pipeline ước lượng
(cùng estimator, cùng covariate, cùng dữ liệu) như với cặp thật. Nếu khoảng
tin cậy 95% KHÔNG chứa 0, đó là tín hiệu cảnh báo confounding/bias chưa
kiểm soát trong toàn bộ pipeline — không riêng gì cặp negative-control đó.

**Vì sao `current_ratio -> gross_margin` được chọn** (không phải một cặp bất
kỳ): cả hai cột đã có sẵn trong `panel_builder.build_panel()`
(src/causal/panel_builder.py dòng 176-177), không cần dữ liệu mới, và không
thưa dữ liệu như các cột vĩ mô (gold/forex — xem cảnh báo "very sparse" ở
panel_builder.py dòng 109-114). Lý do lý thuyết: current_ratio phản ánh
quyết định QUẢN TRỊ VỐN LƯU ĐỘNG (tài trợ ngắn hạn), còn gross_margin phản
ánh cấu trúc GIÁ BÁN/GIÁ VỐN (quyết định vận hành/định giá sản phẩm) — hai
miền quyết định khác nhau của doanh nghiệp, không có cơ chế nhân quả trực
tiếp nối chúng sau khi đã kiểm soát sector + fiscal_year. Đây là suy luận
miền (domain reasoning), KHÔNG có một bài báo cụ thể khẳng định "current
ratio không ảnh hưởng gross margin" — nói rõ điều này thay vì gán một trích
dẫn không tồn tại (đúng ràng buộc research-mode: không trích dẫn bịa).

Chạy negative control (CẦN Postgres thật — không chạy được trong sandbox):
    python3 -m scripts.run_negative_control_check

Test không cần DB (dùng panel giả lập + hàm ước lượng THẬT của
treatment_effect.py, không mock):
    python3 -m unittest tests.test_causal.test_negative_control -v
"""

# Cặp negative-control đã chọn. Mỗi entry PHẢI dùng đúng tên cột mà
# src/causal/panel_builder.py's build_panel() thực sự trả về — có test tĩnh
# đối chiếu (tests/test_causal/test_negative_control.py) để bắt lỗi gõ sai
# tên cột TRƯỚC khi tốn một lượt chạy DB thật.
NEGATIVE_CONTROL_PAIRS = [
    {
        "treatment": "current_ratio",
        "outcome": "gross_margin",
        "rationale": (
            "current_ratio phản ánh quyết định quản trị vốn lưu động (tài trợ "
            "ngắn hạn); gross_margin phản ánh cấu trúc giá bán/giá vốn (quyết "
            "định vận hành). Hai miền quyết định khác nhau, không có cơ chế "
            "nhân quả trực tiếp sau khi kiểm soát sector + fiscal_year."
        ),
    },
]


def interpret_negative_control(result: dict, pair_label: str = "") -> dict:
    """Diễn giải MỘT kết quả từ estimate_ate() hoặc estimate_ate_dml()
    (src/causal/treatment_effect.py) theo đúng logic negative-control tổng
    hợp ở Yang et al. (2023/2024, J Clin Epidemiol — xem docstring module):
    khoảng tin cậy 95% CÓ chứa 0 -> không phát hiện confounding rõ ràng qua
    kiểm tra này (không có nghĩa là "chứng minh không có confounding" —
    negative control chỉ phát hiện được LOẠI confounding mà nó tình cờ nhạy
    với, đúng giới hạn phương pháp được nêu trong scoping review nói trên).
    CI KHÔNG chứa 0 -> cảnh báo, cần xem lại pipeline ước lượng (covariate,
    dạng hàm, v.v.) trước khi tin cặp treatment/outcome THẬT đang xét.

    Nhận cả hai dạng dict trả về:
      - estimate_ate() (T-learner): có khoá "ate_95ci" = [low, high]
      - estimate_ate_dml() (DML): có khoá "ate_dml_marginal_95ci" = [low, high]
    """
    if not result.get("ok"):
        return {
            "pair": pair_label,
            "verdict": "khong_chay_duoc",
            "message": f"Không ước lượng được: {result.get('error', '(không rõ lỗi)')}",
        }

    if "ate_95ci" in result:
        ci_low, ci_high = result["ate_95ci"]
        estimator_name = "T-learner"
    elif "ate_dml_marginal_95ci" in result:
        ci_low, ci_high = result["ate_dml_marginal_95ci"]
        estimator_name = "DML"
    else:
        return {
            "pair": pair_label,
            "verdict": "khong_ro_dang_ket_qua",
            "message": "Dict không có 'ate_95ci' lẫn 'ate_dml_marginal_95ci' - "
                       "không phải kết quả từ estimate_ate()/estimate_ate_dml().",
        }

    import math
    if any(v is None or (isinstance(v, float) and math.isnan(v)) for v in (ci_low, ci_high)):
        return {
            "pair": pair_label,
            "verdict": "ci_khong_xac_dinh",
            "message": f"[{estimator_name}] CI không tính được (NaN) - thường do "
                       "cỡ mẫu bootstrap quá nhỏ, không đủ để kết luận.",
        }

    ci_contains_zero = ci_low <= 0 <= ci_high

    if ci_contains_zero:
        verdict = "khong_phat_hien_confounding"
        message = (
            f"[{estimator_name}] KTC 95% [{ci_low:.4f}, {ci_high:.4f}] chứa 0 - "
            "đúng như negative-control kỳ vọng. Đây KHÔNG chứng minh pipeline "
            "hoàn toàn không có confounding (negative control chỉ phát hiện "
            "được loại nhiễu mà nó tình cờ nhạy với - Yang et al. 2023/2024), "
            "nhưng là một dấu hiệu ủng hộ."
        )
    else:
        verdict = "CANH_BAO_confounding"
        message = (
            f"[{estimator_name}] KTC 95% [{ci_low:.4f}, {ci_high:.4f}] KHÔNG chứa 0 "
            "dù lý thuyết dự đoán không có hiệu ứng - CẢNH BÁO: xem lại covariate/"
            "dạng hàm của pipeline ước lượng trước khi tin các cặp treatment/"
            "outcome THẬT (vd leverage->net_margin) chạy CÙNG pipeline này."
        )

    return {
        "pair": pair_label,
        "estimator": estimator_name,
        "ci": [ci_low, ci_high],
        "ci_contains_zero": ci_contains_zero,
        "verdict": verdict,
        "message": message,
    }
