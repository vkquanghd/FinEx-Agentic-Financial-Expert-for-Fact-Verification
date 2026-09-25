"""Unit tests for src/router/scope_notes.py's bilingual fix (2026-09-08).

Regression test for a REAL bug found live: an English question could get a
Vietnamese scope_hint() suffix appended, because polish_not_found() only
checked Vietnamese "not found" markers and had no `lang` parameter at all -
see scope_notes.py's module docstring for the full incident. All tests here
use ticker=None so nothing touches the DB (scope_hint()'s ticker branch is
fail-soft to DB errors anyway via _ticker_data_status()'s try/except, but
avoiding it keeps this suite pure/offline per the project's test convention).

    python3 -m unittest tests.test_router.test_scope_notes -v
"""
import unittest

from src.router.scope_notes import polish_not_found, scope_hint


class TestPolishNotFound(unittest.TestCase):
    def test_vietnamese_answer_default_lang_gets_vietnamese_hint(self):
        """Pre-existing behavior (lang defaults to "vi") must stay unchanged
        for every call site that hasn't been updated to pass lang="""
        answer = "Không tìm thấy thông tin trong tài liệu được cung cấp."
        result = polish_not_found(answer, ticker=None)
        self.assertIn("Câu hỏi này nằm ngoài phạm vi dữ liệu demo", result)
        self.assertNotIn("out of scope", result)

    def test_english_question_english_answer_gets_english_hint(self):
        """The "everything worked as instructed" case: EN question, LLM
        correctly answered in English -> EN hint appended, no Vietnamese."""
        answer = "No information found in the provided documents."
        result = polish_not_found(answer, ticker=None, lang="en")
        self.assertIn("out of scope for the demo's data", result)
        self.assertNotIn("Câu hỏi này nằm ngoài phạm vi", result)

    def test_english_question_but_model_answered_in_vietnamese_still_gets_english_hint(self):
        """THE REGRESSION CASE (2026-09-08): question is English (lang="en"
        passed by query_router.py, detected from the QUESTION, not the
        answer), but the local LLM slipped into Vietnamese anyway and wrote
        the Vietnamese "not found" phrase. Before the fix, polish_not_found()
        had no `lang` param and always appended the Vietnamese scope_hint,
        regardless of the question's language - producing a fully-Vietnamese
        suffix under an English question. After the fix: the Vietnamese
        marker is still recognized (checks both marker sets), but the
        appended hint follows `lang` (the question's language), not the
        answer's actual (possibly-wrong) language."""
        answer = "Không tìm thấy thông tin trong tài liệu được cung cấp."
        result = polish_not_found(answer, ticker=None, lang="en")
        self.assertIn("out of scope for the demo's data", result)

    def test_answer_with_content_is_returned_unchanged(self):
        """No "not found" marker in either language -> no hint appended,
        answer passed through verbatim regardless of `lang`."""
        answer = "Tổng tài sản của VNM quý 1 năm 2026 là 50.000 tỷ đồng."
        self.assertEqual(polish_not_found(answer, ticker=None, lang="vi"), answer)
        self.assertEqual(polish_not_found(answer, ticker="VNM", lang="en"), answer)

    def test_english_marker_variant_no_relevant_excerpt(self):
        """rag_pipeline.NO_CONTEXT_MSG_EN's opening phrase must also be
        recognized as a "not found" marker in English."""
        answer = "No relevant excerpt was found in the loaded disclosure documents."
        result = polish_not_found(answer, ticker=None, lang="en")
        self.assertIn("out of scope for the demo's data", result)


class TestScopeHint(unittest.TestCase):
    def test_vietnamese_default(self):
        hint = scope_hint(ticker=None)
        self.assertIn("Hiện demo trả lời tốt nhất các dạng câu", hint)

    def test_english(self):
        hint = scope_hint(ticker=None, lang="en")
        self.assertIn("The demo currently answers these question types best", hint)
        # Must not silently fall back to the Vietnamese text.
        self.assertNotIn("Hiện demo trả lời tốt nhất", hint)


if __name__ == "__main__":
    unittest.main()
