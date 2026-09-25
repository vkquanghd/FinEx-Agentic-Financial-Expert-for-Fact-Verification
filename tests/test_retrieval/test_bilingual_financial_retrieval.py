"""Unit coverage for English-query retrieval expansion over Vietnamese filings."""
import unittest
from unittest.mock import patch

from src.retrieval.rag_pipeline import classify_doctype, expand_financial_terms_for_retrieval


class TestBilingualFinancialRetrieval(unittest.TestCase):
    def test_english_financial_disclosure_filters_financial_statements(self):
        self.assertEqual(
            classify_doctype("Cite the supporting financial disclosure pages."),
            "financial_statement",
        )

    def test_revenue_and_net_profit_add_official_vietnamese_labels(self):
        result = expand_financial_terms_for_retrieval("Compare revenue and net profit for VNM.")
        self.assertIn("Doanh thu thuần", result)
        self.assertIn("Lợi nhuận sau thuế", result)

    def test_vietnamese_question_is_not_mutated(self):
        question = "So sánh doanh thu và lợi nhuận sau thuế của VNM."
        self.assertEqual(expand_financial_terms_for_retrieval(question), question)

    def test_answer_uses_expanded_query_without_name_error(self):
        from src.retrieval import rag_pipeline

        seen = []
        chunks = [{
            "ticker": "VNM", "doctype": "financial_statement", "source_file": "vnm_2024.pdf",
            "page": 10, "year": 2024, "similarity": 0.9, "chunk_text": "Doanh thu thuần.",
        }]
        with patch.object(rag_pipeline, "classify_market_regime", return_value={"regime": "stable"}), \
             patch.object(rag_pipeline, "similarity_search", side_effect=lambda query, **_: seen.append(query) or chunks), \
             patch.object(rag_pipeline, "answer_with_citation", return_value={"answer": "ok", "sources": []}), \
             patch.object(rag_pipeline, "verify_numeric_claims", return_value={"label": "no_numeric_claim", "claims": []}):
            result = rag_pipeline.answer("Compare revenue and net profit for VNM in 2024.", ticker="VNM")

        self.assertEqual(result["answer"], "ok")
        self.assertIn("Doanh thu thuần", seen[0])
        self.assertIn("Lợi nhuận sau thuế", seen[0])


if __name__ == "__main__":
    unittest.main()
