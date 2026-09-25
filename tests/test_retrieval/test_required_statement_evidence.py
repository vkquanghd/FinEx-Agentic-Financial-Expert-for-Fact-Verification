"""Exact requested financial labels supplement broad RAG retrieval cheaply."""
import unittest
from unittest.mock import patch

from src.retrieval import rag_pipeline


class TestRequiredStatementEvidence(unittest.TestCase):
    def test_compound_english_question_prepends_label_evidence(self):
        base = [{"ticker": "VNM", "source_file": "vnm_2024.pdf", "page": 55,
                 "chunk_index": 0, "chunk_text": "base"}]
        revenue = {"ticker": "VNM", "source_file": "vnm_2024.pdf", "page": 10,
                   "chunk_index": 0, "chunk_text": "Báo cáo kết quả hoạt động kinh doanh Mẫu B 02"}
        profit = {"ticker": "VNM", "source_file": "vnm_2024.pdf", "page": 11,
                  "chunk_index": 0, "chunk_text": "Báo cáo kết quả hoạt động kinh doanh Mẫu B 02"}
        with patch.object(rag_pipeline, "lexical_search", return_value=[revenue, profit]) as search:
            result = rag_pipeline._prepend_required_statement_evidence(
                base, "Compare revenue and net profit for VNM in 2024.", "VNM", 2024, 5,
            )
        self.assertEqual([item["page"] for item in result], [10, 11, 55])
        self.assertEqual(
            [call.args[0] for call in search.call_args_list],
            ['"Báo cáo kết quả hoạt động kinh doanh" "Mẫu B 02"'],
        )

    def test_unrelated_question_does_not_run_fts_supplement(self):
        base = [{"ticker": "VNM", "source_file": "vnm_2024.pdf", "page": 55,
                 "chunk_index": 0, "chunk_text": "base"}]
        with patch.object(rag_pipeline, "lexical_search") as search:
            result = rag_pipeline._prepend_required_statement_evidence(base, "What is VNM strategy?", "VNM", 2024, 5)
        self.assertEqual(result, base)
        search.assert_not_called()


if __name__ == "__main__":
    unittest.main()
