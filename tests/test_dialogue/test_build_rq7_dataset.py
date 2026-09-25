"""Unit tests for scripts/build_rq7_dataset.py - most importantly, that
every gold_* value in the OUTPUT dataset is copied verbatim from a real item
in the pre-existing data/eval/benchmark.jsonl (the "no invented ground
truth" principle applied to RQ7 - see that script's module docstring).

    python3 -m unittest tests.test_dialogue.test_build_rq7_dataset -v
"""
import json
import unittest
from pathlib import Path

from scripts.build_rq7_dataset import (
    BENCHMARK_PATH, OUTPUT_PATH, build_dataset, load_source_items,
)

EXPECTED_CATEGORIES = {
    "pronoun_reference", "ellipsis", "result_reference", "subject_switch",
    "return_to_old_topic", "self_contained", "noisy_irrelevant_history",
    "self_correction",
}


class TestBuildRQ7Dataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source_items = load_source_items()
        cls.source_by_id = {item["id"]: item for item in cls.source_items}
        cls.conversations = build_dataset(cls.source_items)

    def test_produces_at_least_one_conversation_per_error_category(self):
        categories_seen = {c["error_category"] for c in self.conversations}
        self.assertEqual(categories_seen, EXPECTED_CATEGORIES)

    def test_every_gold_value_traces_to_a_real_benchmark_item(self):
        """The core integrity check: gold_standalone_question/gold_route/
        gold_ticker for every conversation must be an EXACT, unmodified
        copy of some real item in benchmark.jsonl - never fabricated."""
        for conv in self.conversations:
            source_id = conv["gold_source_benchmark_id"]
            self.assertIn(source_id, self.source_by_id,
                          f"{conv['conv_id']}'s gold_source_benchmark_id {source_id!r} not in benchmark.jsonl")
            real_item = self.source_by_id[source_id]
            self.assertEqual(conv["gold_standalone_question"], real_item["question"])
            self.assertEqual(conv["gold_route"], real_item["expected_branch"])
            self.assertEqual(conv["gold_ticker"], real_item["ticker"])

    def test_conv_ids_are_unique(self):
        ids = [c["conv_id"] for c in self.conversations]
        self.assertEqual(len(ids), len(set(ids)))

    def test_every_conversation_has_at_least_two_turns(self):
        for conv in self.conversations:
            self.assertGreaterEqual(len(conv["turns"]), 2, conv["conv_id"])

    def test_last_turn_utterance_is_the_thing_to_resolve(self):
        """Sanity check: the last turn should generally NOT be byte-identical
        to the gold standalone question for the reduced categories (that's
        the whole point - it needs resolving) - self_contained/
        noisy_irrelevant_history are the deliberate exception (already
        standalone by design)."""
        for conv in self.conversations:
            if conv["error_category"] in ("self_contained", "noisy_irrelevant_history"):
                continue
            last_utterance = conv["turns"][-1]["utterance"]
            self.assertNotEqual(last_utterance, conv["gold_standalone_question"],
                                f"{conv['conv_id']} ({conv['error_category']}) - reduced turn should differ from gold")

    def test_pronoun_reference_uses_no_pronoun_word(self):
        for conv in self.conversations:
            if conv["error_category"] != "pronoun_reference":
                continue
            self.assertIn("nó", conv["turns"][-1]["utterance"])

    def test_output_file_matches_freshly_built_dataset(self):
        """Regression guard: the committed data/eval/rq7_conversational_benchmark.jsonl
        must actually be the output of the current build_dataset() logic, not
        a stale file from before an edit to this script."""
        self.assertTrue(OUTPUT_PATH.exists(), f"{OUTPUT_PATH} not found - run scripts/build_rq7_dataset.py")
        with OUTPUT_PATH.open(encoding="utf-8") as f:
            on_disk = [json.loads(line) for line in f if line.strip()]
        self.assertEqual(len(on_disk), len(self.conversations))
        on_disk_ids = {c["conv_id"] for c in on_disk}
        fresh_ids = {c["conv_id"] for c in self.conversations}
        self.assertEqual(on_disk_ids, fresh_ids)


if __name__ == "__main__":
    unittest.main()
