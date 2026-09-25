"""Test cho src/causal/negative_control.py (2026-09-04).

Vá lỗ hổng nêu ở docs/RQ_audit_truoc_khi_training_2026-08-14.md dòng
154-166: RQ6 "không có kiểm định giả định nhân quả... không có placebo/
negative control". KHÔNG cần Postgres/mạng - ba nhóm test:

  1. `interpret_negative_control()` trên dict TỰ TẠO mô phỏng đúng hình dạng
     estimate_ate()/estimate_ate_dml() thật trả về (không mock các hàm đó,
     chỉ mô phỏng OUTPUT của chúng để test hàm diễn giải độc lập).
  2. Đối chiếu TĨNH: tên cột trong NEGATIVE_CONTROL_PAIRS phải khớp đúng
     cột mà panel_builder.py THẬT SỰ tính ra - bắt lỗi gõ sai tên cột TRƯỚC
     khi tốn một lượt chạy DB thật (đây chính là kiểu lỗi
     "TypeError khi chạy" mà một script cần DB rất khó phát hiện sớm).
  3. Test đầu-cuối THẬT (không mock): dựng một panel giả lập bằng
     pandas/numpy thuần (không cần DB) với outcome KHÔNG phụ thuộc treatment
     theo thiết kế (nhiễu ngẫu nhiên độc lập) - gọi TRỰC TIẾP
     estimate_ate() thật (không phải bản giả), rồi xác nhận
     interpret_negative_control() đọc đúng kết quả. Đây là bằng chứng cả
     pipeline ước lượng THẬT hoạt động đúng trên một ca đã biết trước đáp án
     (đáp án đúng = không có hiệu ứng), không chỉ kiểm tra hàm diễn giải một
     mình.

Chạy:
    python3 -m unittest tests.test_causal.test_negative_control -v
"""
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.causal.negative_control import (  # noqa: E402
    NEGATIVE_CONTROL_PAIRS,
    interpret_negative_control,
)

PANEL_BUILDER_SRC = (ROOT / "src" / "causal" / "panel_builder.py").read_text(encoding="utf-8")


class TestInterpretNegativeControl(unittest.TestCase):
    def test_ci_containing_zero_is_reassuring_not_alarming(self):
        result = {"ok": True, "ate_95ci": [-0.05, 0.03]}
        v = interpret_negative_control(result, pair_label="fake_a -> fake_b")
        self.assertEqual(v["verdict"], "khong_phat_hien_confounding")
        self.assertTrue(v["ci_contains_zero"])
        self.assertIn("KHÔNG chứng minh", v["message"])  # không quá tự tin

    def test_ci_not_containing_zero_raises_warning(self):
        result = {"ok": True, "ate_95ci": [0.02, 0.09]}
        v = interpret_negative_control(result, pair_label="fake_a -> fake_b")
        self.assertEqual(v["verdict"], "CANH_BAO_confounding")
        self.assertFalse(v["ci_contains_zero"])
        self.assertIn("CẢNH BÁO", v["message"])

    def test_negative_ci_not_containing_zero_also_flagged(self):
        """CI hoàn toàn âm (vd [-0.09, -0.02]) cũng phải bị gắn cờ - không
        chỉ trường hợp CI hoàn toàn dương."""
        result = {"ok": True, "ate_95ci": [-0.09, -0.02]}
        v = interpret_negative_control(result)
        self.assertEqual(v["verdict"], "CANH_BAO_confounding")

    def test_dml_result_shape_also_supported(self):
        """estimate_ate_dml() dùng khoá 'ate_dml_marginal_95ci', khác với
        estimate_ate()'s 'ate_95ci' - phải nhận diện đúng cả hai dạng."""
        result = {"ok": True, "ate_dml_marginal_95ci": [-0.01, 0.01]}
        v = interpret_negative_control(result)
        self.assertEqual(v["estimator"], "DML")
        self.assertEqual(v["verdict"], "khong_phat_hien_confounding")

    def test_failed_estimate_reported_not_crashed(self):
        result = {"ok": False, "error": "n=3 < 10 - không đủ dữ liệu"}
        v = interpret_negative_control(result)
        self.assertEqual(v["verdict"], "khong_chay_duoc")
        self.assertIn("không đủ dữ liệu", v["message"])

    def test_nan_ci_reported_not_crashed(self):
        """bootstrap có thể trả NaN nếu mọi lần resample đều lệch nhóm quá
        mức (xem estimate_ate()'s `boot` rỗng -> (nan, nan))."""
        result = {"ok": True, "ate_95ci": [float("nan"), float("nan")]}
        v = interpret_negative_control(result)
        self.assertEqual(v["verdict"], "ci_khong_xac_dinh")

    def test_unrecognized_dict_shape_reported_not_crashed(self):
        v = interpret_negative_control({"ok": True, "some_other_key": 1})
        self.assertEqual(v["verdict"], "khong_ro_dang_ket_qua")


class TestPairsMatchRealPanelColumns(unittest.TestCase):
    """Đối chiếu TĨNH - đọc trực tiếp mã nguồn panel_builder.py để lấy đúng
    tập cột nó tính ra, KHÔNG hard-code lại danh sách riêng (nếu ai thêm/bớt
    cột derived trong panel_builder.py mà quên cập nhật test này, test vẫn
    tự đúng vì đọc lại chính file đó mỗi lần chạy)."""

    @staticmethod
    def _known_ratio_columns() -> set:
        # _RATIO_BOUNDS = {...} trong panel_builder.py liệt kê ĐÚNG các cột
        # tỉ số derived mà build_panel() trả về - lấy trực tiếp từ đó thay
        # vì đoán tên.
        m = re.search(r"_RATIO_BOUNDS\s*=\s*\{(.*?)\}", PANEL_BUILDER_SRC, re.DOTALL)
        assert m, "không tìm thấy _RATIO_BOUNDS trong panel_builder.py - file đã đổi cấu trúc?"
        return set(re.findall(r'"(\w+)":\s*\(', m.group(1)))

    def test_ratio_bounds_extraction_sane(self):
        """Tự kiểm tra bước trích xuất ở trên không rỗng/sai - nếu regex bị
        hỏng do panel_builder.py đổi format, test này báo trước thay vì để
        các test khác âm thầm pass vì tập rỗng."""
        cols = self._known_ratio_columns()
        self.assertIn("leverage", cols)
        self.assertIn("net_margin", cols)
        self.assertGreaterEqual(len(cols), 5)

    def test_every_negative_control_pair_uses_real_columns(self):
        known = self._known_ratio_columns()
        for pair in NEGATIVE_CONTROL_PAIRS:
            with self.subTest(pair=f"{pair['treatment']}->{pair['outcome']}"):
                self.assertIn(pair["treatment"], known,
                             f"'{pair['treatment']}' không phải cột thật trong panel_builder.py")
                self.assertIn(pair["outcome"], known,
                             f"'{pair['outcome']}' không phải cột thật trong panel_builder.py")

    def test_every_pair_has_a_stated_rationale(self):
        for pair in NEGATIVE_CONTROL_PAIRS:
            self.assertTrue(pair.get("rationale", "").strip(),
                            f"cặp {pair['treatment']}->{pair['outcome']} thiếu rationale")


class TestEndToEndOnSyntheticKnownNullPanel(unittest.TestCase):
    """Gọi THẬT estimate_ate() (không mock) trên panel giả lập KHÔNG cần DB
    - outcome sinh độc lập với treatment, đáp án đúng ĐÃ BIẾT TRƯỚC là
    'không có hiệu ứng'. Đây là bằng chứng cả pipeline hoạt động đúng, không
    chỉ interpret_negative_control() một mình."""

    def test_known_null_relationship_ci_usually_contains_zero(self):
        import numpy as np
        import pandas as pd

        from src.causal.treatment_effect import estimate_ate

        rng = np.random.default_rng(7)
        n = 400
        # outcome CHỦ Ý không phụ thuộc treatment - chỉ phụ thuộc nhiễu +
        # một hằng số theo sector, đúng cấu trúc "đáp án đúng đã biết".
        sectors = rng.choice(["A", "B", "C"], size=n)
        years = rng.choice([2023, 2024, 2025], size=n)
        treatment = rng.normal(1.0, 0.3, size=n)
        sector_effect = {"A": 0.0, "B": 0.02, "C": -0.01}
        outcome = np.array([sector_effect[s] for s in sectors]) + rng.normal(0, 0.05, size=n)

        panel = pd.DataFrame({
            "fake_treatment": treatment, "fake_outcome": outcome,
            "sector": sectors, "fiscal_year": years,
        })

        result = estimate_ate(treatment_col="fake_treatment", outcome_col="fake_outcome",
                              covariates=("sector", "fiscal_year"), panel=panel, n_bootstrap=200, seed=7)

        self.assertTrue(result["ok"])
        verdict = interpret_negative_control(result, pair_label="fake_treatment -> fake_outcome")
        # Với seed cố định + n=400 + hiệu ứng thật = 0, kỳ vọng hợp lý là CI
        # chứa 0 - đây LÀ điều test này xác nhận cho seed cụ thể này (không
        # phải khẳng định phổ quát bootstrap luôn đúng mọi seed).
        self.assertTrue(verdict["ci_contains_zero"],
                        f"seed=7, n=400, hiệu ứng thật=0 nhưng CI không chứa 0: {verdict}")
        self.assertEqual(verdict["verdict"], "khong_phat_hien_confounding")


if __name__ == "__main__":
    unittest.main()
