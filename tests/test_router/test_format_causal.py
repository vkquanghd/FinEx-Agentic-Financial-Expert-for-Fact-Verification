"""Unit tests for query_router._format_causal() - added 2026-09-12 after
Quang ran a real causal query live and pushed back: "Đây đâu phải là 1 câu
trả lời" (this isn't an answer) - the old version printed only the raw
ATE/CI/n statistics with no sentence answering the actual question asked
("X ảnh hưởng đến Y như thế nào?"). Same "giải thích được" (explainability)
requirement discussed for P1 (Eldifrawi et al. 2024) applies here: a correct
number with no interpretation is not an answer.

Pure string-formatting logic, no DB/Ollama needed for the tests themselves -
but importing query_router.py pulls in the same heavy DB-dependent import
chain as rag_pipeline.py (src.causal.market_regime -> market_factors ->
sqlalchemy), so this still needs the project's normal Python env to run.

    python3 -m unittest tests.test_router.test_format_causal -v
"""
import unittest

from src.router.query_router import _format_causal


class TestFormatCausal(unittest.TestCase):
    def setUp(self):
        # Real numbers from Quang's live run (2026-09-12) - leverage -> net_margin,
        # exact same figures quoted throughout PROGRESS.md/the PDF report.
        self.real_result = {
            "ok": True, "treatment": "leverage", "outcome": "net_margin",
            "covariates": ["sector", "fiscal_year"], "ate_t_learner": -0.0971,
            "ate_95ci": [-0.1871, -0.0264], "n": 349, "n_treated": 174, "n_control": 175,
            "confidence": "standard power (n=349 >= 100)",
            "naive_mean_diff": -0.0972,
        }

    def test_starts_with_a_natural_language_sentence_not_raw_stats(self):
        """The regression this test guards against: output must NOT start
        with "Ước lượng ATE" (the old, stats-first behavior) - it must start
        with an actual answer sentence."""
        out = _format_causal(self.real_result)
        self.assertFalse(out.startswith("Ước lượng ATE"))
        self.assertIn("Có quan hệ", out)

    def test_negative_ate_reports_direction_and_comparative_consistently(self):
        """Regression guard for a real bug caught before shipping: with a
        negative ATE, the sentence must say "NGHỊCH CHIỀU" AND "thấp hơn" -
        not the contradictory "NGHỊCH CHIỀU ... cao hơn" that an earlier
        draft produced."""
        out = _format_causal(self.real_result)
        self.assertIn("NGHỊCH CHIỀU", out)
        self.assertIn("thấp hơn", out)
        self.assertNotIn("cao hơn", out.split("\n\n")[0])  # not in the summary sentence

    def test_positive_ate_reports_direction_and_comparative_consistently(self):
        positive = dict(self.real_result, ate_t_learner=0.05, ate_95ci=[0.01, 0.09])
        out = _format_causal(positive)
        self.assertIn("THUẬN CHIỀU", out)
        self.assertIn("cao hơn", out)
        self.assertNotIn("thấp hơn", out.split("\n\n")[0])

    def test_leverage_net_margin_pair_cites_literature(self):
        out = _format_causal(self.real_result)
        self.assertIn("Myers 1984", out)
        self.assertIn("Abdeljawad", out)

    def test_other_pair_does_not_get_leverage_net_margin_citation(self):
        """The Myers/Abdeljawad citation is specific evidence for leverage->
        net_margin - must NOT be misapplied to an unrelated pair (e.g. the
        negative-control pair current_ratio->gross_margin)."""
        other = dict(self.real_result, treatment="current_ratio", outcome="gross_margin")
        out = _format_causal(other)
        self.assertNotIn("Myers", out)
        self.assertNotIn("Abdeljawad", out)
        self.assertIn("liên hệ có kiểm soát nhiễu đã biết", out)

    def test_technical_numbers_unchanged_from_before(self):
        """The old bare-stats string must still be present verbatim
        (labeled "Chi tiết kỹ thuật:") - this patch only ADDS an
        explanation, it must not change or drop any existing number."""
        out = _format_causal(self.real_result)
        self.assertIn(
            "Chi tiết kỹ thuật: Ước lượng ATE (T-learner) của 'leverage' lên 'net_margin' "
            "(kiểm soát ['sector', 'fiscal_year']): -0.0971 (KTC 95%: [-0.1871, -0.0264]), "
            "n=349 (treated=174, control=175). Mức độ tin cậy: standard power (n=349 >= 100). "
            "So sánh naive (không kiểm soát biến gây nhiễu): -0.0972.",
            out,
        )

    def test_unknown_column_falls_back_to_raw_name(self):
        """A concept pair outside the 6 known _CAUSAL_DISPLAY_NAMES entries
        must not crash - falls back to the raw column name."""
        unknown = dict(self.real_result, treatment="some_new_ratio", outcome="net_margin")
        out = _format_causal(unknown)
        self.assertIn("some_new_ratio", out)

    def test_not_ok_result_returns_error_message_unchanged(self):
        out = _format_causal({"ok": False, "error": "n=3 < 20 - không đủ dữ liệu"})
        self.assertEqual(out, "Không ước lượng được: n=3 < 20 - không đủ dữ liệu")


if __name__ == "__main__":
    unittest.main()
