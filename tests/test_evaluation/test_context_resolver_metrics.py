"""Unit tests for src/evaluation/context_resolver_metrics.py.

    python3 -m unittest tests.test_evaluation.test_context_resolver_metrics -v
"""
import unittest

from src.evaluation.context_resolver_metrics import (
    context_resolver_summary, entity_carryover_accuracy, exact_match_question,
    indicator_carryover_accuracy, latency_stats, recall_at_k,
    routing_accuracy_post_rewrite, temporal_carryover_accuracy, token_f1,
)


class TestEntityCarryoverAccuracy(unittest.TestCase):
    def test_all_correct(self):
        records = [{"predicted_ticker": "HPG", "gold_ticker": "HPG"},
                   {"predicted_ticker": "PNJ", "gold_ticker": "PNJ"}]
        self.assertEqual(entity_carryover_accuracy(records), 1.0)

    def test_none_equals_none_counts_as_match(self):
        records = [{"predicted_ticker": None, "gold_ticker": None}]
        self.assertEqual(entity_carryover_accuracy(records), 1.0)

    def test_mismatch_scored_zero(self):
        records = [{"predicted_ticker": "HPG", "gold_ticker": "PNJ"}]
        self.assertEqual(entity_carryover_accuracy(records), 0.0)

    def test_empty_records_returns_zero(self):
        self.assertEqual(entity_carryover_accuracy([]), 0.0)

    def test_temporal_and_indicator_field_variants(self):
        year_records = [{"predicted_year": 2025, "gold_year": 2025}]
        self.assertEqual(temporal_carryover_accuracy(year_records), 1.0)
        ind_records = [{"predicted_indicator": "revenue", "gold_indicator": "cogs"}]
        self.assertEqual(indicator_carryover_accuracy(ind_records), 0.0)


class TestRoutingAccuracyPostRewrite(unittest.TestCase):
    def test_mixed_correctness(self):
        records = [
            {"predicted_route": "rag", "gold_route": "rag"},
            {"predicted_route": "dag", "gold_route": "whatif"},
        ]
        self.assertEqual(routing_accuracy_post_rewrite(records), 0.5)


class TestTokenF1(unittest.TestCase):
    def test_identical_strings_f1_is_one(self):
        self.assertEqual(token_f1("Công ty kiểm toán của CTD", "Công ty kiểm toán của CTD"), 1.0)

    def test_no_overlap_f1_is_zero(self):
        self.assertEqual(token_f1("hoàn toàn khác nhau", "một hai ba bốn năm sáu"), 0.0)

    def test_partial_overlap_between_zero_and_one(self):
        f1 = token_f1("Công ty kiểm toán của CTD năm 2025", "Công ty kiểm toán của CTD")
        self.assertGreater(f1, 0.0)
        self.assertLess(f1, 1.0)

    def test_both_empty_is_one(self):
        self.assertEqual(token_f1("", ""), 1.0)

    def test_one_empty_is_zero(self):
        self.assertEqual(token_f1("", "something"), 0.0)


class TestExactMatchQuestion(unittest.TestCase):
    def test_exact_match_ignores_case_and_whitespace(self):
        self.assertTrue(exact_match_question("  CTD năm 2025  ", "ctd năm  2025"))

    def test_no_match(self):
        self.assertFalse(exact_match_question("CTD năm 2025", "PNJ năm 2025"))


class TestRecallAtK(unittest.TestCase):
    def test_uses_predicted_ticker_when_no_candidates(self):
        records = [{"predicted_ticker": "HPG", "gold_ticker": "HPG"},
                   {"predicted_ticker": "PNJ", "gold_ticker": "GAS"}]
        self.assertEqual(recall_at_k(records, k=1), 0.5)

    def test_uses_candidate_list_when_present(self):
        records = [{"predicted_ticker_candidates": ["PNJ", "HPG"], "gold_ticker": "HPG"}]
        self.assertEqual(recall_at_k(records, k=1), 0.0)
        self.assertEqual(recall_at_k(records, k=2), 1.0)


class TestLatencyStats(unittest.TestCase):
    def test_computes_mean_p50_p95(self):
        records = [{"latency_ms": v} for v in [10, 20, 30, 40, 100]]
        stats = latency_stats(records)
        self.assertEqual(stats["n"], 5)
        self.assertAlmostEqual(stats["mean"], 40.0)
        self.assertEqual(stats["p50"], 30)

    def test_empty_records(self):
        stats = latency_stats([])
        self.assertEqual(stats["n"], 0)
        self.assertIsNone(stats["mean"])


class TestContextResolverSummary(unittest.TestCase):
    def test_aggregates_across_categories(self):
        records = [
            {"error_category": "pronoun_reference", "predicted_ticker": "HPG", "gold_ticker": "HPG",
             "predicted_route": "dag", "gold_route": "dag",
             "predicted_standalone": "Tổng tài sản HPG 2025", "gold_standalone_question": "Tổng tài sản HPG 2025",
             "latency_ms": 50},
            {"error_category": "self_contained", "predicted_ticker": "PNJ", "gold_ticker": "PNJ",
             "predicted_route": "rag", "gold_route": "whatif",
             "predicted_standalone": "Câu khác", "gold_standalone_question": "Câu gốc",
             "latency_ms": 30},
        ]
        summary = context_resolver_summary(records)
        self.assertEqual(summary["n"], 2)
        self.assertEqual(summary["ticker_carryover_accuracy"], 1.0)
        self.assertEqual(summary["routing_accuracy_post_rewrite"], 0.5)
        self.assertIn("pronoun_reference", summary["per_error_category_n"])
        self.assertEqual(summary["per_error_category_n"]["pronoun_reference"], 1)

    def test_empty_records_does_not_crash(self):
        summary = context_resolver_summary([])
        self.assertEqual(summary["n"], 0)
        self.assertEqual(summary["mean_token_f1"], 0.0)


if __name__ == "__main__":
    unittest.main()
