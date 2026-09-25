"""Regression test cho lỗi crash THẬT gặp trên demo trực tiếp (2026-09-07):

    ValueError: could not convert string to float: 'Chưa phân loại'

Câu hỏi gây lỗi: "Đòn bẩy có ảnh hưởng đến biên lợi nhuận ròng của các công
ty niêm yết Việt Nam không?" -> router phân loại ĐÚNG vào nhánh causal (đã
xác minh trong src/router/rules.py, không phải lỗi định tuyến) -> crash bên
trong estimate_ate() -> route() không có try/except -> lộ traceback thô ra
UI (Streamlit/FastAPI).

Nguyên nhân gốc: src/causal/treatment_effect.py's _design_matrix() dùng
`df[c].dtype == object or str(df[c].dtype).startswith("category")` để tách
cột chữ (one-hot) khỏi cột số. Kiểm tra này BỎ SÓT pandas dtype "string"
(StringDtype - kiểu nullable mới hơn, có thể xuất hiện khi đọc DB qua
SQLAlchemy/pyarrow) - cột `sector` (giá trị mặc định "Chưa phân loại", xem
src/ingestion/vietstock_etl.py) lọt vào nhánh số, rồi
`.to_numpy(dtype=float)` crash đúng như log lỗi.

Hai lớp vá (cả hai đều test ở đây):
1. Vá GỐC: `_design_matrix()` đổi sang `pd.api.types.is_numeric_dtype()` -
   đúng cho MỌI dtype chữ (object/string/category), không liệt kê tay.
2. Vá PHÒNG THỦ: estimate_ate()/estimate_ate_dml() giờ bọc try/except quanh
   toàn bộ phần tính toán, trả {"ok": False, "error": ...} thay vì crash -
   giống đúng quy ước mọi nhánh khác trong router đã dùng (xem
   src/structured/calculator.py, _format_causal() đã xử lý "ok": False sẵn
   từ trước, chỉ chưa từng có gì gọi tới nó khi lỗi thật xảy ra).

Chạy phần KHÔNG cần sklearn (chạy được trong sandbox này):
    python3 -m unittest tests.test_causal.test_design_matrix_dtype_fix.TestIsNumericDtypeClassification -v

Chạy phần CẦN sklearn (chạy trên máy thật của bạn):
    python3 -m unittest tests.test_causal.test_design_matrix_dtype_fix -v
"""
import unittest

import pandas as pd


class TestIsNumericDtypeClassification(unittest.TestCase):
    """Kiểm tra đúng LOGIC phân loại cột (không cần sklearn/OneHotEncoder) -
    tái hiện chính xác điều kiện dùng trong _design_matrix() sau khi vá, để
    xác nhận nó xử lý đúng CẢ dtype cũ (object) LẪN dtype mới (StringDtype)
    đã gây crash thật."""

    def test_classic_object_dtype_string_column_is_not_numeric(self):
        # Đây là trường hợp code CŨ đã xử lý đúng - phải tiếp tục đúng sau vá.
        df = pd.DataFrame({"sector": pd.array(["Ngân hàng", "Chưa phân loại"], dtype=object)})
        self.assertEqual(str(df["sector"].dtype), "object")
        self.assertFalse(pd.api.types.is_numeric_dtype(df["sector"]))

    def test_modern_string_dtype_column_is_not_numeric(self):
        # Đây CHÍNH XÁC là trường hợp code CŨ xử lý SAI (bug thật đã crash
        # trên demo) - "Chưa phân loại" ở dtype "string" thay vì "object".
        df = pd.DataFrame({"sector": pd.array(["Ngân hàng", "Chưa phân loại"], dtype="string")})
        self.assertEqual(str(df["sector"].dtype), "string")
        # Điều kiện CŨ (chỉ so == object) sẽ trả False ở đây -> bug.
        self.assertFalse(df["sector"].dtype == object)  # xác nhận đúng là bug thật, không phải suy đoán
        # Điều kiện MỚI (is_numeric_dtype) phải trả False đúng - không phải numeric.
        self.assertFalse(pd.api.types.is_numeric_dtype(df["sector"]))

    def test_category_dtype_column_is_not_numeric(self):
        df = pd.DataFrame({"sector": pd.array(["A", "B", "A"], dtype="category")})
        self.assertFalse(pd.api.types.is_numeric_dtype(df["sector"]))

    def test_int_and_float_columns_are_numeric(self):
        df = pd.DataFrame({"fiscal_year": [2023, 2024, 2025], "leverage": [0.3, 0.5, 0.7]})
        self.assertTrue(pd.api.types.is_numeric_dtype(df["fiscal_year"]))
        self.assertTrue(pd.api.types.is_numeric_dtype(df["leverage"]))

    def test_nullable_integer_dtype_is_still_numeric(self):
        # pandas' nullable Int64 (khác int64 thường) - phải vẫn được coi là số.
        df = pd.DataFrame({"fiscal_year": pd.array([2023, 2024, None], dtype="Int64")})
        self.assertTrue(pd.api.types.is_numeric_dtype(df["fiscal_year"]))


class TestDesignMatrixEndToEnd(unittest.TestCase):
    """Gọi THẬT _design_matrix() (cần sklearn cho OneHotEncoder) trên đúng
    kịch bản đã crash: cột sector ở dtype "string" chứa "Chưa phân loại"."""

    def test_string_dtype_sector_column_no_longer_crashes(self):
        try:
            from src.causal.treatment_effect import _design_matrix
        except ImportError as e:
            self.skipTest(f"cần sklearn cài trên máy thật: {e}")

        df = pd.DataFrame({
            "sector": pd.array(["Ngân hàng", "Chưa phân loại", "Bán lẻ", "Chưa phân loại"], dtype="string"),
            "fiscal_year": [2023, 2024, 2023, 2024],
        })
        # Trước khi vá: dòng dưới ném ValueError "could not convert string to
        # float: 'Chưa phân loại'". Sau khi vá: phải chạy xong, không lỗi.
        X = _design_matrix(df, ["sector", "fiscal_year"])
        self.assertEqual(X.shape[0], 4)
        # 1 cột số (fiscal_year) + N cột one-hot cho sector (>=2 giá trị khác nhau)
        self.assertGreaterEqual(X.shape[1], 3)


class TestEstimateAteNeverRaises(unittest.TestCase):
    """Xác nhận lớp vá PHÒNG THỦ: estimate_ate() không còn để lộ traceback
    thô ra ngoài dù _design_matrix() có lỗi vì lý do KHÁC trong tương lai
    (không phải mock cụ thể bug đã vá, mà kiểm tra cơ chế try/except chung)."""

    def test_broken_design_matrix_returns_error_dict_not_raise(self):
        try:
            import src.causal.treatment_effect as te_module
        except ImportError as e:
            self.skipTest(f"cần sklearn cài trên máy thật: {e}")

        import pandas as pd

        panel = pd.DataFrame({
            "leverage": [0.1, 0.2, 0.3, 0.8, 0.9, 0.7, 0.4, 0.5, 0.6, 0.15],
            "net_margin": [0.05, 0.04, 0.03, -0.01, -0.02, 0.0, 0.02, 0.01, 0.0, 0.045],
            "sector": ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"],
            "fiscal_year": [2023] * 10,
        })

        original = te_module._design_matrix

        def _boom(df, covariates):
            raise RuntimeError("giả lập lỗi nội bộ bất kỳ, không riêng bug dtype")

        te_module._design_matrix = _boom
        try:
            result = te_module.estimate_ate(
                treatment_col="leverage", outcome_col="net_margin",
                covariates=("sector", "fiscal_year"), panel=panel, n_bootstrap=10,
            )
        finally:
            te_module._design_matrix = original

        self.assertFalse(result["ok"])
        self.assertIn("Lỗi nội bộ", result["error"])
        self.assertIn("RuntimeError", result["error"])


if __name__ == "__main__":
    unittest.main()
