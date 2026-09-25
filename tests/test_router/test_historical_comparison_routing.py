"""Regression coverage for prior-year disclosure comparisons.

These questions contain ``compare`` but are historical evidence requests,
not what-if calculations.  They must not return ``whatif_unsupported``.
"""
import unittest

from src.router.rules import classify_intent


class TestHistoricalComparisonRouting(unittest.TestCase):
    def test_english_compound_prior_year_request_uses_rag(self):
        question = (
            "For VNM in 2024, compare revenue and net profit with the previous year, "
            "explain the change, and cite the supporting disclosure pages."
        )
        intent = classify_intent(question)
        self.assertEqual(intent.branch, "rag")
        self.assertEqual(intent.ticker, "VNM")

    def test_vietnamese_compound_prior_year_request_uses_rag(self):
        question = (
            "Với VNM năm 2024, hãy so sánh doanh thu và lợi nhuận ròng với năm trước, "
            "giải thích biến động và trích dẫn các trang tài liệu công bố hỗ trợ."
        )
        intent = classify_intent(question)
        self.assertEqual(intent.branch, "rag")
        self.assertEqual(intent.ticker, "VNM")

    def test_simple_prior_year_indicator_keeps_exact_growth_path(self):
        intent = classify_intent("So sánh doanh thu của VNM năm 2024 với năm trước")
        self.assertEqual(intent.branch, "whatif")
        self.assertEqual(intent.whatif_kind, "growth_rate")
        self.assertEqual(intent.periods, [(2023, "YEAR"), (2024, "YEAR")])


if __name__ == "__main__":
    unittest.main()
