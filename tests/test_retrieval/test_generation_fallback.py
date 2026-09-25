"""RAG must retain retrieved disclosure sources if local generation fails."""
import unittest
from unittest.mock import patch

from src.models import generation, llm_factory


class TestGenerationFallback(unittest.TestCase):
    def test_ollama_failure_returns_sources_not_a_server_error(self):
        chunks = [{
            "ticker": "VNM", "doctype": "annual_report", "source_file": "vnm_2024.pdf",
            "page": 12, "year": 2024, "similarity": 0.91, "chunk_text": "Revenue disclosure.",
        }]
        with patch.object(
            generation.llm_factory, "generate",
            side_effect=llm_factory.OllamaNotReachable("offline"),
        ):
            result = generation.answer_with_citation("What was VNM revenue in 2024?", chunks)

        self.assertEqual(result["generation_fallback"], "ollama_unavailable")
        self.assertIn("official disclosure pages", result["answer"])
        self.assertEqual(result["sources"][0]["page"], 12)


if __name__ == "__main__":
    unittest.main()
