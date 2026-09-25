"""Unit tests for verify_numeric_claims() / _extract_numeric_claims() in
src/retrieval/rag_pipeline.py - added 2026-09-12, P1 roadmap item (see
docs/FinEx_rebrand_and_Q2_roadmap_2026-09-07.md §6c and docs/PROGRESS.md
same-date entry for the design + the explicitly-flagged-as-unconfirmed unit
normalization assumptions this test locks in).

Pure logic (regex + arithmetic), no DB/Ollama needed for the tests
themselves - but importing rag_pipeline.py pulls in
src.causal.market_regime -> src.causal.market_factors -> sqlalchemy at
module level (same as every other test that touches rag_pipeline), so this
still needs the project's normal Python env (sqlalchemy installed) to run,
same "cannot be exercised from the no-DB/no-Ollama sandbox that wrote it"
situation as rag_pipeline.py itself - see that module's docstring.

    python3 -m unittest tests.test_retrieval.test_rag_pipeline_numeric_verification -v
"""
import unittest

from src.retrieval.rag_pipeline import _extract_numeric_claims, verify_numeric_claims

# 2026-09-12 (same day, added after Quang asked "Fact Verification phải
# giải thích được?"): fact-checking pipeline literature treats explanation/
# justification generation as a required stage, not optional polish -
# Eldifrawi et al. (2024), "Automated Justification Production for Claim
# Veracity in Fact Checking: A Survey on Architectures and Approaches", ACL
# 2024. verify_numeric_claims() now reports "closest_source_value"/
# "diff_pct"/"diff_points" alongside "matched" so a mismatch says WHAT it
# was compared against and BY HOW MUCH, not just True/False - the tests
# below lock in that behavior.


class TestExtractNumericClaims(unittest.TestCase):
    def test_vietnamese_thousands_separator_with_unit(self):
        claims = _extract_numeric_claims("Doanh thu đạt 12.500 tỷ đồng trong năm 2024.")
        self.assertEqual(claims, [(12500 * 1e9, False)])

    def test_bare_year_is_not_a_claim(self):
        """A bare 4-digit year (no unit, no separator) must NOT be treated
        as a numeric claim to verify - see the module comment above
        _NUMERIC_CLAIM_RE for why this is deliberately excluded."""
        self.assertEqual(_extract_numeric_claims("Năm 2024 công ty tăng trưởng tốt."), [])

    def test_vietnamese_decimal_percent(self):
        self.assertEqual(_extract_numeric_claims("Biên lợi nhuận ròng là 12,5%."), [(12.5, True)])

    def test_english_decimal_percent(self):
        self.assertEqual(_extract_numeric_claims("Net margin was 12.5% in 2024."), [(12.5, True)])

    def test_bare_number_with_percent_unit(self):
        self.assertEqual(_extract_numeric_claims("Lãi tăng 50%."), [(50.0, True)])

    def test_multiple_thousands_separators(self):
        self.assertEqual(_extract_numeric_claims("Tổng tài sản 1.234.567 triệu đồng."), [(1234567 * 1e6, False)])

    def test_known_risk_three_digit_ambiguity_documented_assumption(self):
        """Documented, accepted risk (see module comment): a bare number
        with exactly one separator followed by exactly 3 digits is always
        read as a thousands separator, never a 3-decimal-place value - "500"
        below is treated as a thousands group, not "12.500" = 12.5."""
        self.assertEqual(_extract_numeric_claims("Giá trị 12.500 đơn vị."), [(12500.0, False)])


class TestVerifyNumericClaims(unittest.TestCase):
    def setUp(self):
        self.chunks = [{
            "chunk_text": "Theo BCTC, doanh thu năm 2024 đạt 12.500 tỷ đồng, tăng 15% so với cùng kỳ."
        }]

    def test_all_claims_match_is_verified(self):
        answer = "Doanh thu năm 2024 đạt khoảng 12.500 tỷ đồng, tăng 15% so với cùng kỳ."
        result = verify_numeric_claims(answer, self.chunks)
        self.assertEqual(result["label"], "verified")
        self.assertTrue(all(c["matched"] for c in result["claims"]))
        # exact match -> diff should be exactly 0, not None (closest found)
        self.assertEqual(result["claims"][0]["diff_pct"], 0.0)
        self.assertEqual(result["claims"][0]["closest_source_value"], 12500 * 1e9)

    def test_mismatched_number_flags_whole_answer(self):
        answer = "Doanh thu năm 2024 đạt khoảng 20.000 tỷ đồng."
        result = verify_numeric_claims(answer, self.chunks)
        self.assertEqual(result["label"], "unverified_claim_mismatch")
        claim = result["claims"][0]
        self.assertFalse(claim["matched"])
        # explainability (2026-09-12): a mismatch must say WHAT it was
        # compared against and BY HOW MUCH, not just False.
        self.assertEqual(claim["closest_source_value"], 12500 * 1e9)
        self.assertAlmostEqual(claim["diff_pct"], 60.0)
        self.assertIsNone(claim["diff_points"])  # not a percent claim

    def test_no_numeric_claim_label_for_qualitative_answer(self):
        result = verify_numeric_claims("Công ty hoạt động trong ngành thép.", self.chunks)
        self.assertEqual(result, {"label": "no_numeric_claim", "claims": []})

    def test_small_rounding_difference_within_2pct_tolerance_still_verified(self):
        """12.500 vs 12.520 tỷ = 0.16% relative difference - within the
        documented (Quang-unconfirmed) 2% tolerance, so still "verified"."""
        chunks = [{"chunk_text": "Doanh thu 12.520 tỷ đồng."}]
        result = verify_numeric_claims("Doanh thu khoảng 12.500 tỷ đồng.", chunks)
        self.assertEqual(result["label"], "verified")

    def test_percent_tolerance_uses_absolute_0_2_point_threshold(self):
        """15.1% vs 15.0% = 0.1 percentage point, within the 0.2-point
        absolute tolerance for percentages even though the relative
        difference (0.67%) would also pass - either path should verify."""
        chunks = [{"chunk_text": "Tăng trưởng 15,0% so với cùng kỳ."}]
        result = verify_numeric_claims("Tăng trưởng 15,1% so với cùng kỳ.", chunks)
        self.assertEqual(result["label"], "verified")
        claim = result["claims"][0]
        self.assertEqual(claim["closest_source_value"], 15.0)
        self.assertAlmostEqual(claim["diff_points"], 0.1)
        self.assertIsNone(claim["diff_pct"])  # a percent claim, not an absolute one

    def test_no_source_claim_of_same_type_has_no_closest(self):
        """A percent claim with NO percent numbers anywhere in the source
        text has nothing to compare against - closest_source_value/diff must
        be None (not 0, not a crash), and matched must be False (can't be
        verified against nothing)."""
        chunks = [{"chunk_text": "Doanh thu 12.500 tỷ đồng."}]  # no % in source
        result = verify_numeric_claims("Tăng trưởng 15%.", chunks)
        claim = result["claims"][0]
        self.assertFalse(claim["matched"])
        self.assertIsNone(claim["closest_source_value"])
        self.assertIsNone(claim["diff_pct"])
        self.assertIsNone(claim["diff_points"])

    def test_missing_chunk_text_key_does_not_crash(self):
        """web_fallback_answer()'s chunks and disclosure chunks both always
        carry "chunk_text", but this must fail safe (no numeric match, not a
        KeyError/crash) if a caller ever passes a chunk dict without it."""
        result = verify_numeric_claims("Doanh thu 12.500 tỷ đồng.", [{}])
        self.assertEqual(result["label"], "unverified_claim_mismatch")


if __name__ == "__main__":
    unittest.main()
