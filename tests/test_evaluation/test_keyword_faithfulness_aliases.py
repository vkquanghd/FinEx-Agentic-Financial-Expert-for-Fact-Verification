"""Test hồi quy cho keyword_faithfulness() sau khi sửa 2026-08-07.

Bối cảnh: lần chạy eval đầy đủ đầu tiên (2026-08-07) báo faithfulness
0.6594. Đọc tay 23 câu bị chấm 0 thì thấy 11 câu TRẢ LỜI HOÀN TOÀN ĐÚNG
nhưng bị chấm 0, vì ground truth (`disclosures.auditor`, từ Vietstock) lưu
VIẾT TẮT còn RAG trích TÊN PHÁP LÝ ĐẦY ĐỦ từ tài liệu. Faithfulness thật
là 0.8188. Xem docs/eval_run1_faithfulness_analysis.md.

Mỗi ca dưới đây lấy TỪ DỮ LIỆU THẬT của lần chạy đó, không phải ví dụ bịa.

Hai nhóm test quan trọng NGANG NHAU:
  - nhóm "phải khớp": các ca từng bị chấm sai, giờ phải đúng.
  - nhóm "KHÔNG được khớp": các ca hệ thống trả lời SAI HÃNG - nếu bảng
    alias quá rộng và làm chúng khớp thì metric sẽ thổi phồng kết quả, tệ
    hơn nhiều so với việc bỏ sót. Đây là lưới an toàn cho mọi lần mở rộng
    AUDITOR_ALIASES về sau.

Chạy: python3 -m unittest discover -s tests/test_evaluation
Không cần DB/LLM/mạng.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.evaluation.metrics import keyword_faithfulness


class TestAuditorAliasMatching(unittest.TestCase):
    """11 ca THẬT từng bị chấm 0 oan (eval run 1, 2026-08-07)."""

    REAL_CASES = [
        # (id, viết tắt trong ground truth, trích đoạn câu trả lời thật)
        ("rag_ctd", "EY", "Công ty kiểm toán của CTD năm 2025 là Công ty TNHH Ernst & Young Việt Nam"),
        ("rag_vib", "EY", "là **Công ty Trách nhiệm Hữu hạn Ernst & Young Việt Nam**"),
        ("rag_csc", "VAE", "là **Công ty TNHH Kiểm toán và Định giá Việt Nam**, thành viên của LEA Global"),
        ("rag_cvn", "NVT", "là **CÔNG TY TNHH KIỂM TOÁN NHÂN TÂM VIỆT**"),
        ("rag_cap", "NVT", "là **CÔNG TY TNHH KIỂM TOÁN NHÂN TÂM VIỆT**"),
        ("rag_brr", "IAV", "là **International Auditing and Valuation Company Limited**"),
        ("rag_apsc", "IAV", "là **Công ty TNHH Kiểm toán và Định giá Quốc tế**"),
        # 3 ca chỉ khác DẤU CÁCH / lỗi OCR tách chữ:
        ("rag_bbs", "AnViet", "là **An Viet Auditing Company Limited** (Công ty TNHH Kiểm toán An Việt)"),
        ("rag_ckd", "iCPA", "là **i CPA**. [Nguồn 4: 'i CPA since 2001 INTERNATIONAL AUDITING CO., LTD']"),
        ("rag_amc", "iCPA", "là **i CPA**. Theo báo cáo kiểm toán độc lập số 031801/2025/BCKT-i CPA"),
        ("rag_hsg", "PWC", "là Công ty TNHH Pw C (Việt Nam)"),
    ]

    def test_real_false_negatives_now_match(self):
        for case_id, keyword, answer in self.REAL_CASES:
            with self.subTest(case=case_id, keyword=keyword):
                result = keyword_faithfulness(answer, [keyword])
                self.assertEqual(
                    result["score"], 1.0,
                    f"{case_id}: '{keyword}' phải khớp được với câu trả lời "
                    f"(đây là ca THẬT từng bị chấm 0 oan). missing={result['missing']}",
                )


class TestAliasMustNotOvermatch(unittest.TestCase):
    """Lưới an toàn: alias KHÔNG được rộng tới mức chấm đúng cho câu SAI.

    Cả 2 ca dưới đây là lỗi THẬT của hệ thống trong eval run 1 - phải TIẾP
    TỤC bị chấm 0. Nếu ai đó mở rộng AUDITOR_ALIASES quá tay, test này vỡ
    trước khi con số faithfulness bị thổi phồng trong luận văn.
    """

    def test_wrong_firm_still_fails(self):
        # rag_avg: ground truth IAV, hệ thống trả lời nhầm sang NVT.
        result = keyword_faithfulness(
            "Công ty kiểm toán của AVG năm 2025 là **CÔNG TY TNHH KIỂM TOÁN NHÂN TÂM VIỆT**",
            ["IAV"],
        )
        self.assertEqual(result["score"], 0.0,
                         "Trả lời sai hãng (NVT thay vì IAV) KHÔNG được chấm đúng")

    def test_different_firm_still_fails(self):
        # rag_ame: ground truth AVA, hệ thống trả lời AFC.
        result = keyword_faithfulness(
            "Công ty kiểm toán của AME năm 2025 là **AFC Vietnam Auditing Co., Ltd.**",
            ["AVA"],
        )
        self.assertEqual(result["score"], 0.0,
                         "Trả lời hãng khác (AFC thay vì AVA) KHÔNG được chấm đúng")

    def test_not_found_answer_still_fails(self):
        result = keyword_faithfulness("Không tìm thấy thông tin trong tài liệu được cung cấp.", ["BDO"])
        self.assertEqual(result["score"], 0.0)

    def test_icpa_does_not_cross_match_iav(self):
        """Cả 2 hãng đều có 'Quốc tế'/'International' trong tên - KHÔNG được
        khớp chéo. Đây là lý do AUDITOR_ALIASES cố ý KHÔNG khai
        'kiểm toán quốc tế' cho iCPA."""
        result = keyword_faithfulness("là **Công ty TNHH Kiểm toán và Định giá Quốc tế**", ["iCPA"])
        self.assertEqual(result["score"], 0.0,
                         "IAV không được tính là iCPA chỉ vì cùng có chữ 'Quốc tế'")


class TestOriginalBehaviourPreserved(unittest.TestCase):
    """Lớp khớp chuỗi con nguyên bản phải giữ nguyên - bản sửa chỉ được làm
    điểm TĂNG, không bao giờ đổi kết quả vốn đã đúng."""

    def test_plain_substring_still_works(self):
        self.assertEqual(keyword_faithfulness("Doanh thu đạt 1.234 tỷ đồng", ["1.234"])["score"], 1.0)

    def test_case_insensitive(self):
        self.assertEqual(keyword_faithfulness("Công ty kiểm toán là KPMG", ["kpmg"])["score"], 1.0)

    def test_partial_credit(self):
        r = keyword_faithfulness("Chỉ có KPMG ở đây", ["KPMG", "Deloitte"])
        self.assertEqual(r["score"], 0.5)
        self.assertEqual(r["missing"], ["Deloitte"])

    def test_empty_keywords_returns_none(self):
        self.assertIsNone(keyword_faithfulness("bất kỳ", [])["score"])

    def test_short_keyword_not_squash_matched(self):
        """Từ khoá <3 ký tự KHÔNG dùng khớp bỏ-dấu-cách (dễ khớp bừa vào
        giữa từ khác) - để bảng alias lo, nơi có ngữ cảnh rõ hơn."""
        # 'AB' không được khớp bừa vào 'A B C D' qua cơ chế squash
        self.assertEqual(keyword_faithfulness("Chỉ tiêu X Y Z", ["QQ"])["score"], 0.0)


if __name__ == "__main__":
    unittest.main()
