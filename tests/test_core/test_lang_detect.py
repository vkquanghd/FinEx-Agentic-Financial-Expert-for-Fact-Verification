"""Unit tests for src/core/lang_detect.py - added 2026-09-08 alongside the
bilingual (VI/EN) support feature (see docs/PROGRESS.md same-date entry).
Pure logic, zero dependencies - safe anywhere.

    python3 -m unittest tests.test_core.test_lang_detect -v
"""
import unittest

from src.core.lang_detect import detect_language


class TestDetectLanguage(unittest.TestCase):
    def test_vietnamese_with_diacritics(self):
        self.assertEqual(detect_language("Rủi ro pháp lý của VNM là gì?"), "vi")

    def test_english_question(self):
        self.assertEqual(detect_language("What are VNM's legal risks?"), "en")

    def test_vietnamese_numeric_question(self):
        self.assertEqual(detect_language("Tổng tài sản của CTD năm 2025 là bao nhiêu?"), "vi")

    def test_english_numeric_question(self):
        self.assertEqual(detect_language("What was CTD's total assets in 2025?"), "en")

    def test_bare_ticker_defaults_to_english(self):
        """No Vietnamese diacritics present -> defaults to "en" (see
        detect_language()'s docstring for why this is a low-stakes default,
        not a real failure mode)."""
        self.assertEqual(detect_language("VNM"), "en")

    def test_empty_string_defaults_to_english(self):
        self.assertEqual(detect_language(""), "en")

    def test_mixed_case_diacritics_both_detected(self):
        self.assertEqual(detect_language("HỘI ĐỒNG QUẢN TRỊ PNJ"), "vi")
        self.assertEqual(detect_language("hội đồng quản trị pnj"), "vi")


if __name__ == "__main__":
    unittest.main()
