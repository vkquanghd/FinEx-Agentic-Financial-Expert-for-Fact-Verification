"""Unit test for scripts/finetune_context_resolver.py's pure-Python part
only (load_training_pairs()) - the actual training loop needs
torch/transformers/peft + network + GPU, none of which this sandbox has
(see that script's docstring), so it is intentionally NOT exercised here.

    python3 -m unittest tests.test_dialogue.test_finetune_context_resolver -v
"""
import unittest

from scripts.finetune_context_resolver import load_training_pairs


class TestLoadTrainingPairs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pairs = load_training_pairs()

    def test_produces_one_pair_per_conversation(self):
        self.assertEqual(len(self.pairs), 64)  # 8 categories x 8 conversations, as of 2026-07-28

    def test_every_pair_has_required_keys(self):
        required = {"conv_id", "error_category", "input_text", "target_text"}
        for pair in self.pairs:
            self.assertEqual(required, required & pair.keys())

    def test_input_text_contains_resolve_prefix_and_current_utterance(self):
        for pair in self.pairs:
            self.assertTrue(pair["input_text"].startswith("resolve:"))

    def test_target_text_is_never_empty(self):
        for pair in self.pairs:
            self.assertTrue(pair["target_text"].strip())


if __name__ == "__main__":
    unittest.main()
