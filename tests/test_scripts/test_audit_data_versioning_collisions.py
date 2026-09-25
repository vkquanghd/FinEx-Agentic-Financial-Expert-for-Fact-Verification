"""Test cho scripts/audit_data_versioning_collisions.py (2026-08-14).

KHÔNG cần DB/mạng - script thật cần Postgres (đã có sẵn dữ liệu real
data/raw/pdf_disclosures/ trong repo, không cần kết nối tới đâu cả). Hai
việc được test tách biệt:

  1. classify_audit_tier / classify_period_tier: đúng trên TÊN FILE THẬT lấy
     trực tiếp từ `ls data/raw/pdf_disclosures/` (không phải ví dụ bịa) -
     nếu ai đổi regex mà sai lệch với các ca thật này, test đỏ ngay.
  2. find_collisions: logic gộp nhóm + phát hiện va chạm, test bằng dữ liệu
     GIẢ LẬP nhỏ (không cần DB) - một ca có va chạm thật (rõ ràng bằng mắt),
     một ca không.

Chạy:
    python3 -m unittest tests.test_scripts.test_audit_data_versioning_collisions -v
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.audit_data_versioning_collisions import (  # noqa: E402
    classify_audit_tier,
    classify_period_tier,
    find_collisions,
)


class TestClassifyAuditTier(unittest.TestCase):
    """Tên file lấy thật từ `ls data/raw/pdf_disclosures/` ngày 2026-08-14 -
    không phải ví dụ tự nghĩ ra."""

    def test_audited_annual_files_detected(self):
        files = [
            "1_aah_2026_4_3_08378a4_vi_baocaotaichinh_kiemtoan_2025.pdf",
            "2_kts_2025_9_9_e5d2e55_kts_en_auditedfinancialstatements_20242025.pdf",
            "1.ACV_2021.4.12_1f69703_ACVBCTC_Hop_nhat_nam_2020_da_duoc_kiem_toancompressed",
            "1_agf_2024_3_29_4129fa5_bctc_nam_2023_da_kiem_toansign",
        ]
        for f in files:
            with self.subTest(f=f):
                self.assertEqual(classify_audit_tier(f), "audited")

    def test_reviewed_semiannual_files_detected(self):
        files = [
            "4_kts_2026_2_25_eb25435_vi_kts_baocaotaichinh_soat_xet_31_12_2025.pdf",
            "AAA_Baocaotaichinh_6T_2024_Soatxet_Hopnhat.pdf",
            "2_kts_2026_2_25_607d96d_en_kts_reviewedfinancialstatements_31_12_2025.pdf",
        ]
        for f in files:
            with self.subTest(f=f):
                self.assertEqual(classify_audit_tier(f), "reviewed")

    def test_plain_quarterly_files_default_to_unaudited(self):
        """BCTC quý không có tín hiệu kiểm toán/soát xét trong tên file -
        đây LÀ hành vi đúng theo quy ước thị trường VN (quý không bắt buộc
        kiểm toán), không phải lỗ hổng của regex."""
        files = [
            "AAM_Baocaotaichinh_Q4_2024_1.pdf",
            "1_aas_2024_7_24_627631e_bctc_quy_2_2024_cks.pdf",
            "1_kts_2025_10_23_23347e6_vi_baocaotaichinh_q1_2026.pdf",
            "AAT_Baocaotaichinh_Q3_2024.pdf",
        ]
        for f in files:
            with self.subTest(f=f):
                self.assertEqual(classify_audit_tier(f), "unaudited_quarterly")

    def test_non_financial_docs_are_unknown(self):
        files = [
            "SLS_Nghiquyet_DHDCD thuong nien_2024.pdf",
            "1_aav_2025_8_13_1dab6c7_vn_baocaoquantri_062025.pdf",
        ]
        for f in files:
            with self.subTest(f=f):
                self.assertEqual(classify_audit_tier(f), "unknown")


class TestClassifyPeriodTier(unittest.TestCase):
    def test_quarterly_files(self):
        for f in ["AAM_Baocaotaichinh_Q4_2024_1.pdf",
                  "1_aas_2024_7_24_627631e_bctc_quy_2_2024_cks.pdf",
                  "2_aav_2026_2_4_42974a7_vi_baocaotaichinh_q4_2025.pdf"]:
            with self.subTest(f=f):
                self.assertEqual(classify_period_tier(f), "quy")

    def test_semiannual_files(self):
        for f in ["AAA_Baocaotaichinh_6T_2024_Soatxet_Hopnhat.pdf",
                  "1_aav_2025_9_4_0cfdebf_vi_baocaotaichinhbannien_2025.pdf"]:
            with self.subTest(f=f):
                self.assertEqual(classify_period_tier(f), "ban_nien")

    def test_annual_defaults_to_nam(self):
        for f in ["1_aah_2026_4_3_08378a4_vi_baocaotaichinh_kiemtoan_2025.pdf",
                  "1_aav_2026_5_4_730ede5_vi_baocaotaichinh_2025.pdf"]:
            with self.subTest(f=f):
                self.assertEqual(classify_period_tier(f), "nam")


class TestFindCollisions(unittest.TestCase):
    """Logic gộp nhóm + phát hiện va chạm - dữ liệu giả lập, không cần DB."""

    def test_detects_real_collision_case(self):
        """Mô phỏng đúng kịch bản lo ngại trong §8: cùng ticker+năm+kỳ NĂM,
        một bản không có tín hiệu kiểm toán (unknown) và một bản có
        'kiemtoan' trong tên - nếu cả hai từng nạp vào disclosures, bản sau
        đã ghi đè bản trước."""
        rows = [
            {"ticker": "ABC", "fiscal_year": 2024, "source_file": "ABC_baocaotaichinh_2024_sobo.pdf"},
            {"ticker": "ABC", "fiscal_year": 2024, "source_file": "ABC_baocaotaichinh_2024_da_kiem_toan.pdf"},
        ]
        got = find_collisions(rows)
        self.assertEqual(len(got), 1)
        self.assertEqual(got[0]["ticker"], "ABC")
        self.assertEqual(got[0]["fiscal_year"], 2024)
        self.assertEqual(set(got[0]["audit_tiers_found"]), {"unknown", "audited"})

    def test_no_collision_when_different_period(self):
        """Q4 (chưa kiểm toán, mặc định) và BCTC năm (đã kiểm toán) KHÔNG va
        chạm trong disclosures thật vì period khác nhau ('Q4' vs 'YEAR') -
        không cùng UNIQUE key nên không hề ghi đè nhau."""
        rows = [
            {"ticker": "XYZ", "fiscal_year": 2024, "source_file": "XYZ_baocaotaichinh_Q4_2024.pdf"},
            {"ticker": "XYZ", "fiscal_year": 2024, "source_file": "XYZ_baocaotaichinh_2024_kiemtoan.pdf"},
        ]
        got = find_collisions(rows)
        self.assertEqual(got, [])

    def test_no_collision_when_same_tier_twice(self):
        """Hai bản CÙNG mức kiểm toán (vd 2 file audited trùng, do đăng lại)
        không phải va chạm audit_status - chỉ là trùng lặp file, vấn đề khác."""
        rows = [
            {"ticker": "DEF", "fiscal_year": 2024, "source_file": "DEF_baocaotaichinh_2024_kiemtoan_v1.pdf"},
            {"ticker": "DEF", "fiscal_year": 2024, "source_file": "DEF_baocaotaichinh_2024_kiemtoan_v2.pdf"},
        ]
        got = find_collisions(rows)
        self.assertEqual(got, [])

    def test_empty_input(self):
        self.assertEqual(find_collisions([]), [])


if __name__ == "__main__":
    unittest.main()
