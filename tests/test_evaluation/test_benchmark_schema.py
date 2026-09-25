"""Structural validation of data/eval/benchmark.jsonl - catches an item
missing a field a runner will KeyError/silently-mishandle on, without
needing Postgres/Ollama. Formalizes the ad-hoc `json.loads()`/schema checks
done by hand while building the Week 8 additions (2026-07-27/28) into a
real, re-runnable test.

    python3 -m unittest tests.test_evaluation.test_benchmark_schema -v
"""
import json
import unittest
from pathlib import Path

BENCHMARK_PATH = Path(__file__).resolve().parents[2] / "data" / "eval" / "benchmark.jsonl"

# Keys every item must have regardless of category.
COMMON_REQUIRED = {"id", "category"}

# Category-specific required keys, beyond COMMON_REQUIRED. `nhanh_a_vs_b`'s
# "expected_value" is deliberately allowed to be present-but-None (the
# ab_hoaphat_fullname_totalassets_2025 qualitative-only item) - checked for
# KEY PRESENCE only, not non-null, unlike the others.
CATEGORY_REQUIRED = {
    "routing": {"question", "expected_branch"},
    "whatif_exact_match": {"question", "ticker", "expected_branch", "expected_fields", "tolerance_pct"},
    "rag_faithfulness": {"question", "ticker", "expected_branch"},
    "rag_temporal": {"question", "ticker", "expected_branch", "expected_year"},
    "nhanh_a_vs_b": {"question", "ticker", "expected_value", "tolerance_pct"},
    "regime_ablation": {"question", "ticker"},
    "causal_comparison": {"treatment_col", "outcome_col", "covariates"},
    "absa_eval": {"ticker"},
    # 2026-09-08 - see metrics.sparse_coverage_summary()/run_eval.run_sparse_coverage():
    # deliberately no "expected_*" key - this category has no gold text, it
    # grades a structural behavior (admit-missing-data vs hallucinate), not
    # content correctness, computed live from a real answer() call.
    "sparse_coverage_no_hallucination": {"question", "ticker"},
}

EXPECTED_CATEGORY_COUNTS = {
    # +8 2026-08-07: dag_trend (3) + auditor_lookup (3) + causal_unsupported
    # (1) + absa_unsupported (1) routing coverage for the two new branches
    # added that day (see docs/PROGRESS.md).
    # +10 2026-09-08: route_group_full_* routing items for the "đủ dữ liệu"
    # ticker group (see docs/eval_methodology_reference.md). -1 2026-09-10:
    # route_group_full_vcb_total_assets REMOVED - a real live run_eval.py
    # showed VCB fails routing to "dag" (predicted "rag" instead) because
    # VCB is NOT a known ticker in the live `companies` table at all -
    # scripts/seed_companies.py's CORE_16 docstring explicitly documents
    # VCB/BID/ACB/VPB were dropped in favor of MWG/PNJ/DGW/FRT (banks use a
    # different financial-statement template, out of this project's DAG
    # schema scope) - see scripts/verify_ticker_groups.py and docs/PROGRESS.md
    # 2026-09-10 entry. This was a bad test item (VCB was never a valid
    # ticker to test against), not a router bug - removed rather than "fixed".
    "routing": 26,  # was 17, briefly 27
    "whatif_exact_match": 26,
    # +5 2026-09-08: tinix_group_* PLACEHOLDER items (TiniX-confirmed
    # ticker group) - expected_keywords: [] deliberately, excluded from
    # avg_keyword_faithfulness until filled in with a real run's ground truth.
    "rag_faithfulness": 73,  # was 68
    "rag_temporal": 2,
    "nhanh_a_vs_b": 27,
    "regime_ablation": 4,  # was 2 (PNJ, GAS) - +2 2026-08-07: HPG, VNM
    "causal_comparison": 2,  # was 1 - +1 2026-08-07: causal_leverage_current_ratio
    "absa_eval": 5,
    # 2026-09-08 - new category, see docs/eval_methodology_reference.md.
    "sparse_coverage_no_hallucination": 11,
}


def _load_items():
    items = []
    with BENCHMARK_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


@unittest.skipUnless(BENCHMARK_PATH.exists(), f"{BENCHMARK_PATH} not found")
class TestBenchmarkSchema(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.items = _load_items()

    def test_every_line_is_valid_json_with_common_keys(self):
        for item in self.items:
            missing = COMMON_REQUIRED - item.keys()
            self.assertFalse(missing, f"item {item.get('id', '<no id>')} missing common keys: {missing}")

    def test_ids_are_unique(self):
        ids = [item["id"] for item in self.items]
        dupes = {i for i in ids if ids.count(i) > 1}
        self.assertFalse(dupes, f"duplicate ids: {dupes}")

    def test_category_specific_required_keys_present(self):
        for item in self.items:
            cat = item["category"]
            required = CATEGORY_REQUIRED.get(cat)
            if required is None:
                continue  # unknown category - covered by test_only_known_categories
            missing = required - item.keys()
            self.assertFalse(missing, f"item {item['id']} (category={cat}) missing keys: {missing}")

    def test_only_known_categories(self):
        unknown = {item["category"] for item in self.items} - set(CATEGORY_REQUIRED)
        self.assertFalse(unknown, f"unknown categories found: {unknown}")

    def test_category_counts_match_expected(self):
        """Deliberately hardcoded counts (not just 'count > 0') - catches an
        accidental duplicate/truncated append the same way
        test_run_eval_helpers.py's 140-total check does, but per-category so
        a bad edit to one category doesn't hide behind another category's
        surplus items."""
        counts = {}
        for item in self.items:
            counts[item["category"]] = counts.get(item["category"], 0) + 1
        self.assertEqual(counts, EXPECTED_CATEGORY_COUNTS)

    def test_whatif_exact_match_all_five_expected_fields_present(self):
        """Every whatif_exact_match item must carry all 5 fields
        run_baseline_dag_only()/run_baseline_rag_only_whatif() and
        run_whatif_exact_match() all read from - missing one would silently
        under-count instead of erroring."""
        required_fields = {"revenue_before", "cogs_before", "cogs_pct_change",
                            "gross_profit_before", "gross_profit_after"}
        for item in self.items:
            if item["category"] != "whatif_exact_match":
                continue
            missing = required_fields - item["expected_fields"].keys()
            self.assertFalse(missing, f"{item['id']} missing expected_fields: {missing}")

    def test_whatif_exact_match_follows_single_fixed_template(self):
        """The WHATIF_ANSWER_FIELD scoring fix (2026-07-28, see
        docs/PROGRESS.md) depends on EVERY whatif_exact_match question
        following the exact same template with only the ticker varying -
        this test fails loudly if a future benchmark edit breaks that
        assumption, since run_eval.py's WHATIF_ANSWER_FIELD constant would
        then silently score the wrong thing for the new item."""
        templates = set()
        for item in self.items:
            if item["category"] != "whatif_exact_match":
                continue
            templates.add(item["question"].replace(item["ticker"], "<TICKER>"))
        self.assertEqual(len(templates), 1, f"expected 1 template, found {len(templates)}: {templates}")

    def test_nhanh_a_vs_b_qualitative_only_item_has_null_expected_value(self):
        """Confirms exactly one nhanh_a_vs_b item is the deliberate
        qualitative-only case (expected_value: null) - if this count
        changes, nhanh_a_vs_b_summary()'s accuracy-exclusion logic and this
        test's assumption both need re-checking together."""
        null_items = [item for item in self.items
                      if item["category"] == "nhanh_a_vs_b" and item.get("expected_value") is None]
        self.assertEqual(len(null_items), 1)


if __name__ == "__main__":
    unittest.main()
