"""Unit tests for src/evaluation/metrics.py.

Every function in metrics.py is pure (no DB/LLM/network calls, except
llm_judge_faithfulness which is intentionally NOT tested here - it needs a
real Ollama instance, see its own docstring), so this whole file runs with
zero external dependencies - stdlib `unittest` only, no pytest, no DB, no
network. Safe to run anywhere, including a sandbox with nothing else set up:

    python3 -m unittest tests.test_evaluation.test_metrics -v

Several test cases here reproduce EXACT real-world numbers observed during
real runs on the project owner's machine (2026-07-28, see docs/PROGRESS.md)
rather than made-up numbers, so a future regression in these functions
would be caught against a real, previously-verified data point.
"""
import unittest

from src.evaluation.metrics import (
    absa_structural_check,
    baseline_dag_only_summary,
    baseline_rag_only_summary,
    causal_estimator_comparison,
    exact_match,
    keyword_faithfulness,
    nhanh_a_vs_b_summary,
    regime_ablation_summary,
    routing_accuracy,
    sparse_coverage_summary,
    synthetic_causal_calibration,
    temporal_anchor_error_rate,
)


class TestRoutingAccuracy(unittest.TestCase):
    def test_perfect_score(self):
        predicted = ["dag", "rag", "whatif", "causal"]
        expected = ["dag", "rag", "whatif", "causal"]
        r = routing_accuracy(predicted, expected)
        self.assertEqual(r["accuracy"], 1.0)
        self.assertEqual(r["macro_f1"], 1.0)
        self.assertEqual(r["n"], 4)

    def test_real_2026_07_19_routing_result(self):
        """Reproduces the real 8/8, macro_f1=1.0 result from the first real
        --routing-only run (2026-07-19) and again 2026-07-28."""
        predicted = ["dag", "rag", "whatif", "whatif", "whatif", "whatif_unsupported", "causal", "compare"]
        expected = predicted[:]
        r = routing_accuracy(predicted, expected)
        self.assertEqual(r["accuracy"], 1.0)
        self.assertEqual(r["macro_f1"], 1.0)
        self.assertEqual(r["per_class"]["whatif"]["support"], 3)

    def test_imperfect_score_per_class(self):
        predicted = ["dag", "dag", "rag"]
        expected = ["dag", "rag", "rag"]
        r = routing_accuracy(predicted, expected)
        self.assertAlmostEqual(r["accuracy"], 2 / 3, places=4)
        self.assertEqual(r["per_class"]["dag"]["precision"], 0.5)
        self.assertEqual(r["per_class"]["dag"]["recall"], 1.0)

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            routing_accuracy(["dag"], ["dag", "rag"])

    def test_empty_input(self):
        r = routing_accuracy([], [])
        self.assertEqual(r, {"accuracy": None, "macro_f1": None, "n": 0, "per_class": {}})


class TestExactMatch(unittest.TestCase):
    def test_exact_int_match(self):
        self.assertTrue(exact_match(24884531092428, 24884531092428))

    def test_within_tolerance(self):
        self.assertTrue(exact_match(100.4, 100.0, tolerance_pct=0.5))

    def test_outside_tolerance(self):
        self.assertFalse(exact_match(102.0, 100.0, tolerance_pct=0.5))

    def test_none_actual_or_expected(self):
        self.assertFalse(exact_match(None, 5))
        self.assertFalse(exact_match(5, None))

    def test_zero_expected(self):
        self.assertTrue(exact_match(0, 0))
        self.assertFalse(exact_match(1, 0))

    def test_decimal_vs_float_real_bug_2026_07_28(self):
        """Real bug found running run_nhanh_a_vs_b() for real (2026-07-28):
        financial_line_items.value is Postgres NUMERIC -> comes back as
        decimal.Decimal, which raised TypeError against benchmark.jsonl's
        Python float gold values before this fix. See PROGRESS.md."""
        import decimal
        self.assertTrue(exact_match(decimal.Decimal("73434196003"), 73434196003.0))
        self.assertFalse(exact_match(decimal.Decimal("73434196003"), 73000000000.0))

    def test_uncoercible_value_is_no_match_not_crash(self):
        self.assertFalse(exact_match("garbage", 5))
        self.assertFalse(exact_match(5, "garbage"))

    def test_whatif_exact_match_regression_calculator_precast_floats(self):
        """calculator.py's _get_value() already float()-casts at the DB
        boundary, so plain float-vs-int/float comparisons (the 26/26
        whatif_exact_match passing case) must keep working after the
        Decimal fix above."""
        self.assertTrue(exact_match(815142288987.0, 815142288987))


class TestKeywordFaithfulness(unittest.TestCase):
    def test_all_found(self):
        r = keyword_faithfulness("Công ty kiểm toán là PwC Việt Nam", ["PwC"])
        self.assertEqual(r["score"], 1.0)
        self.assertEqual(r["missing"], [])

    def test_none_found(self):
        r = keyword_faithfulness("Không tìm thấy thông tin", ["PwC"])
        self.assertEqual(r["score"], 0.0)
        self.assertEqual(r["missing"], ["PwC"])

    def test_no_expected_keywords(self):
        r = keyword_faithfulness("bất kỳ câu trả lời nào", [])
        self.assertIsNone(r["score"])

    def test_case_insensitive(self):
        r = keyword_faithfulness("audited by pwc", ["PwC"])
        self.assertEqual(r["score"], 1.0)


class TestTemporalAnchorErrorRate(unittest.TestCase):
    def test_all_correct(self):
        results = [{"expected_year": 2024, "actual_year": 2024}, {"expected_year": 2025, "actual_year": 2025}]
        r = temporal_anchor_error_rate(results)
        self.assertEqual(r["error_rate"], 0.0)

    def test_real_2026_07_28_result(self):
        """Reproduces the real 2/2 error_rate=1.0 result (blocked entirely
        on empty disclosure_chunks post-Postgres-migration, actual_year=None
        for every question - see docs/PROGRESS.md)."""
        results = [{"expected_year": 2024, "actual_year": None}, {"expected_year": 2017, "actual_year": None}]
        r = temporal_anchor_error_rate(results)
        self.assertEqual(r, {"error_rate": 1.0, "n": 2, "errors": 2})

    def test_no_year_labeled_items_excluded(self):
        results = [{"expected_year": None, "actual_year": 2020}]
        r = temporal_anchor_error_rate(results)
        self.assertEqual(r, {"error_rate": None, "n": 0})


class TestNhanhAVsBSummary(unittest.TestCase):
    def test_real_2026_07_28_distribution(self):
        """Reproduces the real 26-ticker run after the Decimal fix: 24
        zero-row Nhánh B failures, 2 real successes, 1 qualitative_only item
        excluded from accuracy - see docs/PROGRESS.md."""
        detail = []
        for i in range(24):
            detail.append({"id": f"zero{i}", "nhanh_a_correct": True, "nhanh_b_correct": False,
                            "nhanh_b_gate_failed": None, "nhanh_b_row_count": 0,
                            "nhanh_b_gate2b_would_flag": None})
        for i in range(2):
            detail.append({"id": f"ok{i}", "nhanh_a_correct": True, "nhanh_b_correct": True,
                            "nhanh_b_gate_failed": None, "nhanh_b_row_count": 1,
                            "nhanh_b_gate2b_would_flag": None})
        detail.append({"id": "qual", "qualitative_only": True, "nhanh_a_correct": None,
                        "nhanh_b_correct": None, "nhanh_b_gate_failed": None,
                        "nhanh_b_row_count": 0, "nhanh_b_gate2b_would_flag": None})
        r = nhanh_a_vs_b_summary(detail)
        self.assertEqual(r["n"], 27)
        self.assertEqual(r["n_scored_for_accuracy"], 26)
        self.assertEqual(r["nhanh_a_accuracy"], 1.0)
        self.assertAlmostEqual(r["nhanh_b_accuracy"], 0.0769, places=4)
        self.assertEqual(r["n_success_zero_rows"], 25)
        self.assertAlmostEqual(r["success_zero_rows_rate"], 0.9259, places=4)

    def test_gate2b_would_have_caught_rate(self):
        detail = [
            {"id": "1", "nhanh_a_correct": True, "nhanh_b_correct": True,
             "nhanh_b_gate_failed": None, "nhanh_b_gate2b_would_flag": None},
            {"id": "2", "nhanh_a_correct": True, "nhanh_b_correct": False,
             "nhanh_b_gate_failed": "execution", "nhanh_b_gate2b_would_flag": True},
            {"id": "3", "nhanh_a_correct": True, "nhanh_b_correct": False,
             "nhanh_b_gate_failed": "execution", "nhanh_b_gate2b_would_flag": False},
        ]
        r = nhanh_a_vs_b_summary(detail)
        self.assertEqual(r["gate2b_would_have_caught_rate"], 0.5)

    def test_all_errored_returns_n_zero(self):
        r = nhanh_a_vs_b_summary([{"id": "1", "error": "boom"}])
        self.assertEqual(r, {"n": 0})


class TestRegimeAblationSummary(unittest.TestCase):
    def test_order_changed(self):
        w_on = [{"market_regime": "bear", "sources": [{"source_file": "a"}, {"source_file": "b"}]}]
        w_off = [{"market_regime": "stable", "sources": [{"source_file": "b"}, {"source_file": "a"}]}]
        r = regime_ablation_summary(w_on, w_off)
        self.assertEqual(r["regime_active_rate"], 1.0)
        self.assertEqual(r["rerank_order_changed_rate"], 1.0)

    def test_real_2026_07_28_result(self):
        """Both regime_ablation items blocked on empty disclosure_chunks -
        market_regime=None (early-return path), sources=[] both sides."""
        w_on = [{"market_regime": None, "sources": []}, {"market_regime": None, "sources": []}]
        w_off = [{"market_regime": None, "sources": []}, {"market_regime": None, "sources": []}]
        r = regime_ablation_summary(w_on, w_off)
        self.assertEqual(r["n"], 2)
        self.assertEqual(r["regime_active_rate"], 0.0)
        self.assertEqual(r["rerank_order_changed_rate"], 0.0)

    def test_empty_input(self):
        self.assertEqual(regime_ablation_summary([], []), {"n": 0})


class TestCausalEstimatorComparison(unittest.TestCase):
    def test_both_ok_ci_contains(self):
        t_res = {"ok": True, "ate_t_learner": 0.05, "ate_95ci": [0.01, 0.09]}
        d_res = {"ok": True, "ate_dml_marginal": 0.002, "ate_dml_marginal_95ci": [0.0005, 0.0035],
                 "translated_group_contrast": 0.045}
        r = causal_estimator_comparison(t_res, d_res)
        self.assertTrue(r["ok"])
        self.assertTrue(r["t_learner_ci_contains_dml_translated_point"])

    def test_dml_not_ok_econml_missing(self):
        """Reproduces the real 2026-07-28 result: T-learner succeeds, DML
        fails only because econml isn't installed (optional dependency)."""
        t_res = {"ok": True, "ate_t_learner": -0.0975, "ate_95ci": [-0.188, -0.027]}
        d_res = {"ok": False, "error": "Chưa cài đặt `econml`..."}
        r = causal_estimator_comparison(t_res, d_res)
        self.assertFalse(r["ok"])
        self.assertEqual(r["t_learner"]["ate_t_learner"], -0.0975)


class TestAbsaStructuralCheck(unittest.TestCase):
    def test_valid_result(self):
        result = {"ok": True, "aspects": {
            "profitability": {"sentiment": "positive", "evidence": "loi nhuan tang"},
            "liquidity_leverage": {"sentiment": "not_mentioned", "evidence": None},
            "growth_outlook": {"sentiment": "neutral", "evidence": "on dinh"},
            "governance": {"sentiment": "positive", "evidence": "HDQT moi"},
            "risk_legal": {"sentiment": "negative", "evidence": "tranh chap"},
        }}
        r = absa_structural_check(result)
        self.assertTrue(r["structurally_valid"])

    def test_missing_aspect_and_evidence(self):
        result = {"ok": True, "aspects": {"profitability": {"sentiment": "positive", "evidence": None}}}
        r = absa_structural_check(result)
        self.assertFalse(r["structurally_valid"])
        self.assertIn("liquidity_leverage", r["missing_aspects"])
        self.assertIn("profitability", r["missing_evidence_aspects"])

    def test_real_2026_07_28_result(self):
        """Reproduces the real result for all 5 absa_eval tickers: blocked
        on empty disclosure_chunks, ok=False for every ticker."""
        result = {"ok": False, "problems": ["Không tìm thấy đoạn trích nào cho mã HPG trong pgvector."]}
        r = absa_structural_check(result)
        self.assertEqual(r, {"ok": False, "structurally_valid": False,
                              "problems": ["Không tìm thấy đoạn trích nào cho mã HPG trong pgvector."]})


class TestBaselineDagOnlySummary(unittest.TestCase):
    def test_splits_by_category(self):
        detail = [
            {"category": "whatif_exact_match", "answerable": True, "correct": True},
            {"category": "whatif_exact_match", "answerable": True, "correct": False},
            {"category": "rag_faithfulness", "answerable": False, "correct": None},
            {"category": "rag_faithfulness", "answerable": False, "correct": None},
            {"category": "whatif_exact_match", "error": "boom"},
        ]
        r = baseline_dag_only_summary(detail)
        self.assertEqual(r["whatif_exact_match"]["n"], 2)
        self.assertEqual(r["whatif_exact_match"]["answerable_rate"], 1.0)
        self.assertEqual(r["whatif_exact_match"]["correct_rate"], 0.5)
        self.assertEqual(r["rag_faithfulness"]["answerable_rate"], 0.0)
        self.assertIsNone(r["rag_faithfulness"]["correct_rate"])

    def test_real_2026_07_28_result_after_whatif_answer_field_fix(self):
        """Reproduces the real result AFTER the WHATIF_ANSWER_FIELD scoring
        fix (2026-07-28): DAG-only answers nearly all whatif questions
        (finds the right line item via resolve_indicator()) but is almost
        never CORRECT (can't apply the hypothetical COGS shock) -
        correct_rate = 1/26, not the misleading 25/26 the first (buggy)
        version of this scoring produced. See docs/PROGRESS.md."""
        detail = [{"category": "whatif_exact_match", "answerable": True, "correct": (i == 0)}
                  for i in range(26)]
        r = baseline_dag_only_summary(detail)
        self.assertEqual(r["whatif_exact_match"]["n"], 26)
        self.assertEqual(r["whatif_exact_match"]["answerable_rate"], 1.0)
        self.assertAlmostEqual(r["whatif_exact_match"]["correct_rate"], 1 / 26, places=4)


class TestBaselineRagOnlySummary(unittest.TestCase):
    def test_basic(self):
        detail = [{"any_expected_field_found": False}, {"any_expected_field_found": True}]
        r = baseline_rag_only_summary(detail)
        self.assertEqual(r["any_expected_field_found_rate"], 0.5)

    def test_real_2026_07_28_result(self):
        """Reproduces the real result: 0/26, RAG-only never stumbles onto
        the exact scenario-computed gross_profit_after figure."""
        detail = [{"any_expected_field_found": False} for _ in range(26)]
        r = baseline_rag_only_summary(detail)
        self.assertEqual(r["n"], 26)
        self.assertEqual(r["any_expected_field_found_rate"], 0.0)

    def test_empty(self):
        self.assertEqual(baseline_rag_only_summary([]), {"n": 0})


class TestSparseCoverageSummary(unittest.TestCase):
    """Added 2026-09-08 alongside the sparse_coverage_no_hallucination
    benchmark category - see sparse_coverage_summary()'s docstring."""

    def test_empty_input(self):
        self.assertEqual(sparse_coverage_summary([]), {"n": 0})

    def test_all_degraded_correctly(self):
        detail = [{"id": "a", "degraded_correctly": True},
                   {"id": "b", "degraded_correctly": True}]
        result = sparse_coverage_summary(detail)
        self.assertEqual(result["n"], 2)
        self.assertEqual(result["degraded_correctly_rate"], 1.0)
        self.assertEqual(result["n_possible_hallucinations"], 0)

    def test_mixed_results(self):
        """3/4 admitted missing data, 1/4 answered confidently despite
        near-zero real coverage - the possible-hallucination case."""
        detail = [
            {"id": "a", "degraded_correctly": True},
            {"id": "b", "degraded_correctly": True},
            {"id": "c", "degraded_correctly": True},
            {"id": "d", "degraded_correctly": False},
        ]
        result = sparse_coverage_summary(detail)
        self.assertEqual(result["n"], 4)
        self.assertEqual(result["degraded_correctly_rate"], 0.75)
        self.assertEqual(result["n_possible_hallucinations"], 1)

    def test_none_degraded_correctly(self):
        detail = [{"id": "a", "degraded_correctly": False}]
        result = sparse_coverage_summary(detail)
        self.assertEqual(result["degraded_correctly_rate"], 0.0)
        self.assertEqual(result["n_possible_hallucinations"], 1)


class TestSyntheticCausalCalibration(unittest.TestCase):
    def test_bias_computation(self):
        t_res = {"ok": True, "ate_t_learner": -0.04}
        d_res = {"ok": True, "ate_dml_marginal": -0.078, "translated_group_contrast": -0.039}
        r = synthetic_causal_calibration(t_res, d_res, true_marginal_effect=-0.08, true_group_contrast=-0.04)
        self.assertAlmostEqual(r["t_learner_bias"], 0.0, places=9)
        self.assertAlmostEqual(r["dml_marginal_bias"], 0.002, places=9)

    def test_real_2026_07_28_result_t_learner_only(self):
        """Reproduces the real result: T-learner ran for real on the
        synthetic DGP with a very small bias (~-0.001); DML unavailable
        (econml not installed) so the aggregate is ok=False but the
        T-learner half is fully readable from the raw dicts."""
        t_res = {"ok": True, "ate_t_learner": -0.033952}
        d_res = {"ok": False, "error": "Chưa cài đặt `econml`..."}
        r = synthetic_causal_calibration(t_res, d_res, true_marginal_effect=-0.08,
                                          true_group_contrast=-0.032944)
        self.assertFalse(r["ok"])
        self.assertEqual(r["t_learner"]["ate_t_learner"], -0.033952)


if __name__ == "__main__":
    unittest.main()
