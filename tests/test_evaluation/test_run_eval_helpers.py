"""Unit tests for the pure-logic helpers in src/evaluation/run_eval.py -
`load_benchmark()`, `_vn_number_variants()`, and the `WHATIF_ANSWER_FIELD`
constant. Deliberately does NOT import/test the runner functions
themselves (run_routing/run_whatif_exact_match/run_baseline_dag_only/etc.)
since those lazily import src.router.rules / src.structured.query_builder /
src.retrieval.rag_pipeline, which pull in sqlalchemy/sentence-transformers -
not available in every environment (see docs/PROGRESS.md - this sandbox has
neither). The helpers tested here have no such dependency and are safe to
run anywhere:

    python3 -m unittest tests.test_evaluation.test_run_eval_helpers -v
"""
import json
import tempfile
import unittest
from pathlib import Path

from src.evaluation.run_eval import WHATIF_ANSWER_FIELD, _vn_number_variants, load_benchmark


class TestVnNumberVariants(unittest.TestCase):
    def test_positive_integer_like_float(self):
        variants = _vn_number_variants(815142288987.0)
        self.assertIn("815,142,288,987", variants)
        self.assertIn("815.142.288.987", variants)
        self.assertIn("815142288987", variants)

    def test_negative_number(self):
        variants = _vn_number_variants(-388327151185)
        self.assertIn("-388,327,151,185", variants)
        self.assertIn("-388.327.151.185", variants)

    def test_none_returns_empty(self):
        self.assertEqual(_vn_number_variants(None), [])

    def test_non_numeric_string_returns_empty(self):
        self.assertEqual(_vn_number_variants("garbage"), [])

    def test_rounds_to_nearest_whole_unit(self):
        variants = _vn_number_variants(100.4)
        self.assertIn("100", variants)


class TestWhatifAnswerField(unittest.TestCase):
    def test_is_gross_profit_after(self):
        """WHATIF_ANSWER_FIELD must stay "gross_profit_after" - it encodes
        a real finding (all 26 whatif_exact_match questions share one fixed
        template asking specifically for the post-scenario gross profit,
        confirmed 2026-07-28 by masking tickers and diffing all 26
        questions down to one template) that a future benchmark.jsonl
        change could silently invalidate. If this constant ever needs to
        change, re-verify the template assumption in benchmark.jsonl first -
        see docs/PROGRESS.md's baseline #1/#2 scoring-fix entry."""
        self.assertEqual(WHATIF_ANSWER_FIELD, "gross_profit_after")


class TestLoadBenchmark(unittest.TestCase):
    def test_loads_jsonl_skipping_blank_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bench.jsonl"
            path.write_text(
                '{"id": "a", "category": "routing"}\n'
                "\n"
                '{"id": "b", "category": "whatif_exact_match"}\n',
                encoding="utf-8",
            )
            items = load_benchmark(path)
        self.assertEqual(len(items), 2)
        self.assertEqual(items[0]["id"], "a")
        self.assertEqual(items[1]["id"], "b")

    def test_real_benchmark_file_loads_and_has_176_items(self):
        """Regression check against the real project benchmark file - fails
        loudly if data/eval/benchmark.jsonl is ever accidentally truncated
        or corrupted (176 items as of 2026-09-10 - was 177 after the
        2026-09-08 expansion, -1 on 2026-09-10 after a real run_eval.py
        showed VCB fails routing since it's not a known ticker in the live
        `companies` table - see docs/PROGRESS.md's 2026-09-10 entry - update
        this count deliberately if new items are added, not by accident)."""
        path = Path(__file__).resolve().parents[2] / "data" / "eval" / "benchmark.jsonl"
        if not path.exists():
            self.skipTest(f"{path} not found - skipping repo-relative check")
        items = load_benchmark(path)
        self.assertEqual(len(items), 176)
        ids = [item["id"] for item in items]
        self.assertEqual(len(ids), len(set(ids)), "duplicate ids in benchmark.jsonl")
        for item in items:
            self.assertIn("id", item)
            self.assertIn("category", item)


if __name__ == "__main__":
    unittest.main()
