"""Unit tests for scripts/web_fallback_lookup.py's pure-logic helpers only
(build_search_query/strip_html) - the actual network calls (ddgs search,
requests page fetch, Ollama generate) are intentionally NOT exercised here,
same reasoning as tests/test_dialogue/test_finetune_context_resolver.py:
this script is an out-of-scope-for-RQ1-RQ7 convenience tool (see its module
docstring), not part of anything the thesis evaluates, so it gets a much
lighter test than e.g. src/evaluation/metrics.py.

    python3 -m unittest tests.test_dialogue.test_web_fallback_lookup -v
"""
import unittest

from scripts.web_fallback_lookup import DEFAULT_SITES, build_search_query, strip_html
from scripts.web_news_ingest import has_material_query_term, is_relevant_result


class TestBuildSearchQuery(unittest.TestCase):
    def test_no_sites_returns_query_unchanged(self):
        self.assertEqual(build_search_query("VNM báo cáo tài chính", sites=None),
                         "VNM báo cáo tài chính")

    def test_empty_sites_list_returns_query_unchanged(self):
        self.assertEqual(build_search_query("VNM báo cáo tài chính", sites=[]),
                         "VNM báo cáo tài chính")

    def test_single_site_appends_filter(self):
        result = build_search_query("VNM", sites=["cafef.vn"])
        self.assertEqual(result, "VNM (site:cafef.vn)")

    def test_multiple_sites_joined_with_or(self):
        result = build_search_query("VNM", sites=["cafef.vn", "vietstock.vn"])
        self.assertEqual(result, "VNM (site:cafef.vn OR site:vietstock.vn)")

    def test_default_sites_constant_used_by_cli_is_nonempty(self):
        self.assertGreater(len(DEFAULT_SITES), 0)


class TestStripHtml(unittest.TestCase):
    def test_strips_tags(self):
        html = "<html><body><p>Xin chào <b>thế giới</b></p></body></html>"
        self.assertEqual(strip_html(html), "Xin chào thế giới")

    def test_removes_script_and_style_content_entirely(self):
        html = "<html><head><style>.a{color:red}</style></head><body>" \
               "<script>alert('x')</script>Nội dung thật</body></html>"
        result = strip_html(html)
        self.assertNotIn("alert", result)
        self.assertNotIn("color:red", result)
        self.assertIn("Nội dung thật", result)

    def test_collapses_whitespace(self):
        html = "<p>A</p>\n\n\n<p>   B   </p>"
        self.assertEqual(strip_html(html), "A B")

    def test_respects_max_chars(self):
        html = "<p>" + ("a" * 100) + "</p>"
        result = strip_html(html, max_chars=10)
        self.assertEqual(len(result), 10)

    def test_prefers_article_body_over_navigation(self):
        html = (
            "<html><body><nav>Trang chủ Thị trường Doanh nghiệp</nav>"
            "<article><h1>VNM update</h1><p>Legal risk detail &amp; timeline.</p>"
            "<p>Second relevant paragraph.</p></article><footer>Liên hệ</footer></body></html>"
        )
        result = strip_html(html)
        self.assertIn("VNM update Legal risk detail & timeline.", result)
        self.assertNotIn("Trang chủ", result)
        self.assertNotIn("Liên hệ", result)


class TestWebIngestRelevance(unittest.TestCase):
    def test_requires_ticker_and_question_term(self):
        self.assertTrue(is_relevant_result(
            "VNM", "legal risk", "VNM legal update", "", "The legal risk is described here."
        ))
        self.assertFalse(is_relevant_result(
            "VNM", "legal risk", "VNM company profile", "", "Share-price information only."
        ))
        self.assertFalse(is_relevant_result(
            "VNM", "legal risk", "General legal risk", "", "No company is named."
        ))

    def test_navigation_only_body_is_not_material_evidence(self):
        self.assertFalse(has_material_query_term(
            "Trang chủ Chứng khoán Cổ phiếu Doanh nghiệp Đăng nhập", "legal risk", "VNM"
        ))
        self.assertTrue(has_material_query_term(
            "The company disclosed its legal risk in the report.", "legal risk", "VNM"
        ))


if __name__ == "__main__":
    unittest.main()
