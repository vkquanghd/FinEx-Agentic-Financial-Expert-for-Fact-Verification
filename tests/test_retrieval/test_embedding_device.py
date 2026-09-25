"""Device selection remains automatic unless a demo override is set."""
import unittest
from unittest.mock import patch

from src.retrieval.embeddings import _pick_device


class TestEmbeddingDevice(unittest.TestCase):
    def test_explicit_cpu_override_does_not_probe_accelerators(self):
        with patch.dict("os.environ", {"FINEX_EMBEDDING_DEVICE": "cpu"}, clear=False):
            self.assertEqual(_pick_device(), "cpu")

    def test_unknown_override_falls_back_to_auto(self):
        with patch.dict("os.environ", {"FINEX_EMBEDDING_DEVICE": "invalid"}, clear=False), \
             patch("torch.cuda.is_available", return_value=False), \
             patch("torch.backends.mps.is_available", return_value=False):
            self.assertEqual(_pick_device(), "cpu")


if __name__ == "__main__":
    unittest.main()
