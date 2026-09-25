"""Unit tests for src/dialogue/context_resolver.py - covers the 3
non-model baselines (NoHistory/RawConcat/HumanOracle) and extract_entities()
directly, without needing Ollama/torch/peft running anywhere. The other 2
baselines (LLMPromptOnlyResolver -> Ollama, FineTunedContextResolver -> a
local mT5+LoRA checkpoint) are exercised only for their lazy-import/
clear-error behavior, never for a real generation (that needs the project
owner's own machine - see scripts/finetune_context_resolver.py's docstring).

    python3 -m unittest tests.test_dialogue.test_context_resolver -v
"""
import unittest

from src.dialogue.context_resolver import (
    ConversationTurn, FineTunedContextResolver, HumanOracleResolver,
    NoHistoryResolver, RawConcatResolver, ResolvedQuery, SessionMemory,
    extract_entities,
)


class TestExtractEntities(unittest.TestCase):
    def test_extracts_ticker_and_year(self):
        result = extract_entities("Tổng tài sản của VNM quý 1 năm 2026 là bao nhiêu?")
        self.assertEqual(result["ticker"], "VNM")
        self.assertEqual(result["year"], 2026)

    def test_ignores_non_ticker_abbreviations(self):
        result = extract_entities("EPS của PNJ năm 2025 là bao nhiêu?")
        self.assertEqual(result["ticker"], "PNJ")

    def test_no_ticker_present(self):
        result = extract_entities("Rủi ro pháp lý là gì?")
        self.assertIsNone(result["ticker"])
        self.assertIsNone(result["year"])

    def test_multiple_years_takes_last(self):
        result = extract_entities("So sánh doanh thu VNM giữa 2024 và 2025")
        self.assertEqual(result["year"], 2025)


class TestNoHistoryResolver(unittest.TestCase):
    def test_passes_current_utterance_through_unchanged(self):
        resolver = NoHistoryResolver()
        history = [ConversationTurn(0, "user", "Công ty kiểm toán của CTD năm 2025 là ai?")]
        result = resolver.resolve(history, "nó thế nào?")
        self.assertEqual(result.standalone_question, "nó thế nào?")
        self.assertIsNone(result.ticker)  # "nó" carries no extractable ticker - the whole point of this baseline failing here
        self.assertEqual(result.resolver_name, "no_history")

    def test_extracts_entities_from_self_contained_current_turn(self):
        resolver = NoHistoryResolver()
        result = resolver.resolve([], "Tổng tài sản của HPG năm 2025 là bao nhiêu?")
        self.assertEqual(result.ticker, "HPG")
        self.assertEqual(result.fiscal_year, 2025)


class TestRawConcatResolver(unittest.TestCase):
    def test_concatenates_last_k_turns_plus_current(self):
        resolver = RawConcatResolver(k=2)
        history = [
            ConversationTurn(0, "user", "Công ty kiểm toán của CTD năm 2025 là ai?"),
            ConversationTurn(1, "user", "Còn doanh thu thì sao?"),
        ]
        result = resolver.resolve(history, "nó thế nào?")
        self.assertIn("CTD", result.standalone_question)
        self.assertIn("nó thế nào?", result.standalone_question)
        self.assertEqual(result.ticker, "CTD")  # carried over from the concatenated window

    def test_respects_window_size_k(self):
        resolver = RawConcatResolver(k=1)
        history = [
            ConversationTurn(0, "user", "Công ty kiểm toán của CTD năm 2025 là ai?"),
            ConversationTurn(1, "user", "Còn doanh thu PNJ thì sao?"),
        ]
        result = resolver.resolve(history, "nó thế nào?")
        # k=1 -> only the immediately preceding turn (PNJ one) is in the window, not CTD's
        self.assertNotIn("CTD", result.standalone_question)


class TestHumanOracleResolver(unittest.TestCase):
    def test_returns_gold_standalone_question_from_item(self):
        resolver = HumanOracleResolver()
        item = {
            "gold_standalone_question": "Công ty kiểm toán của CTD năm 2025 là ai?",
            "gold_ticker": "CTD",
        }
        result = resolver.resolve_from_item(item)
        self.assertEqual(result.standalone_question, item["gold_standalone_question"])
        self.assertEqual(result.ticker, "CTD")
        self.assertEqual(result.resolver_name, "human_oracle")


class TestSessionMemory(unittest.TestCase):
    def test_update_from_resolved_query_sets_fields(self):
        memory = SessionMemory(session_id="s1")
        resolved = ResolvedQuery(standalone_question="q", ticker="HPG", fiscal_year=2025, indicator="revenue")
        memory.update_from(resolved)
        self.assertEqual(memory.resolved_ticker, "HPG")
        self.assertEqual(memory.resolved_year, 2025)
        self.assertEqual(memory.resolved_indicator, "revenue")

    def test_update_from_does_not_overwrite_with_none(self):
        memory = SessionMemory(session_id="s1", resolved_ticker="HPG")
        resolved = ResolvedQuery(standalone_question="q", ticker=None)
        memory.update_from(resolved)
        self.assertEqual(memory.resolved_ticker, "HPG")  # unchanged, not clobbered by a None


class TestFineTunedContextResolverStub(unittest.TestCase):
    def test_raises_clear_error_when_no_checkpoint_exists(self):
        """No checkpoint has ever been trained inside this sandbox (needs
        network + GPU, see scripts/finetune_context_resolver.py) - confirms
        the failure mode is a clear, actionable FileNotFoundError, not a
        silent wrong-answer or an import crash."""
        resolver = FineTunedContextResolver(checkpoint_dir="/nonexistent/path/for/testing")
        with self.assertRaises(FileNotFoundError):
            resolver.resolve([], "nó thế nào?")


if __name__ == "__main__":
    unittest.main()
