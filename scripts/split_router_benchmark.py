"""Splits data/eval/benchmark.jsonl into router_development.jsonl (tuning)
and router_test.jsonl (held-out, what gets reported for RQ1/RQ2/RQ6 routing
accuracy) - by TEMPLATE FAMILY, not by row, so paraphrases of the same
underlying question never appear on both sides of the split.

WHY this matters: src/evaluation/run_eval.py already documents (see its
2026-07-28 comment near WHATIF_ANSWER_FIELD) that all 26 whatif_exact_match
rows in benchmark.jsonl "follow ONE fixed Vietnamese template ... confirmed
by diffing every question with its ticker masked out" - i.e. the benchmark
already silently contains near-duplicate paraphrases (same template,
different ticker/number filled in). A RANDOM row-level split could put one
paraphrase of a template in router_development.jsonl and a near-identical
paraphrase of the SAME template in router_test.jsonl - any routing-rule
tuning done against the dev set would then get an inflated, leaked test
score on that template family, not a real held-out estimate. Splitting by
family instead of by row is the fix.

Template family = (category, normalized_question). Normalization
(_normalize_question below) masks out the parts of a question that vary
row-to-row for the SAME underlying template - ticker symbols (a generic
2-5 uppercase-ASCII-letter heuristic, not a curated ticker list, since this
script is meant to run standalone on just benchmark.jsonl with no DB
dependency - good enough to cluster templates correctly for this purpose,
even if it occasionally also masks a fixed all-caps term like "COGS"
consistently across every row of the same template, which does not break
grouping), years, and numbers/percentages - leaving the fixed template
skeleton behind. Rows whose normalized skeletons match exactly are treated
as the same family.

Split algorithm (per category, so both splits keep coverage across
categories/branches where possible): families are sorted deterministically
by template_id, then distributed using the classic "evenly spread k items
across n slots" (Bresenham-style) allocation - see _bresenham_split().
Categories with only ONE template family (e.g. causal_comparison: 1 row,
regime_ablation: 2 rows sharing 1 family) cannot literally be split without
duplicating a family across both sides - these are deliberately assigned
ENTIRELY to router_test.jsonl (not dev), so every category the thesis
reports a routing-accuracy number for still has real held-out coverage;
router_development.jsonl is only for informal rule-tuning and losing a
singleton category there is the much smaller cost.

A hard invariant is checked before writing anything: no template_id may
appear in BOTH output files. This should be impossible by construction
(every row of a family is assigned the same split), but is checked
explicitly and raises RuntimeError if violated, rather than trusting the
logic silently - same "verify, don't just assume" stance used throughout
this project's other builder scripts.

Usage:
    python3 -m scripts.split_router_benchmark
    python3 -m scripts.split_router_benchmark --test-fraction 0.3
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

DEFAULT_INPUT = Path("data/eval/benchmark.jsonl")
DEFAULT_DEV_OUTPUT = Path("data/eval/router_development.jsonl")
DEFAULT_TEST_OUTPUT = Path("data/eval/router_test.jsonl")

_TICKER_RE = re.compile(r"\b[A-Z]{2,5}\b")
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_NUMBER_RE = re.compile(r"\d+([.,]\d+)?%?")
_PUNCT_RE = re.compile(r"[^\w\sÀ-ỹ<>]", re.UNICODE)
_WS_RE = re.compile(r"\s+")


def _normalize_question(question: str) -> str:
    """Masks ticker-like/year/number tokens (the parts that vary between
    paraphrases of the SAME template) so paraphrases collapse to the same
    normalized skeleton. See this module's docstring for why a generic
    heuristic (not a curated ticker list) is used, and why that's fine for
    this specific purpose (clustering, not entity extraction)."""
    masked = _TICKER_RE.sub("<TICK>", question)
    masked = _YEAR_RE.sub("<YEAR>", masked)
    masked = _NUMBER_RE.sub("<NUM>", masked)
    masked = masked.lower()
    masked = _PUNCT_RE.sub(" ", masked)
    return _WS_RE.sub(" ", masked).strip()


def _template_key(item: dict) -> str:
    """Normalized template-clustering key for one benchmark row. Most
    categories have a natural-language 'question' field - normalized via
    _normalize_question(). A few categories (causal_comparison, absa_eval)
    have NO 'question' at all (they specify a ticker + structural params
    directly, e.g. {"treatment_col": "leverage", "outcome_col": "net_margin",
    ...} or just {"ticker": "HPG"}) - for those, fall back to the row's
    remaining structural shape with the per-row identifying value (ticker)
    excluded, so e.g. all 5 absa_eval rows (same shape, different ticker)
    still collapse into ONE family instead of each becoming its own
    singleton (which would send every one of them entirely to
    router_test.jsonl per the singleton rule, needlessly starving
    router_development.jsonl of ABSA coverage for no real leakage reason -
    they're paraphrases of "run ABSA on ticker X", exactly the kind of
    same-template variation this script exists to group)."""
    if "question" in item:
        return _normalize_question(item["question"])
    fallback = {k: v for k, v in item.items()
                if k not in ("id", "notes", "question", "ticker", "expected_year", "category")}
    return json.dumps(fallback, sort_keys=True, ensure_ascii=False)


def assign_template_ids(items: list) -> list:
    """Returns a NEW list of items (shallow copies), each with a
    'template_id' field added: f"{category}__t{NNN}", stable/deterministic
    given the same input order (sorted by (category, template_key) before
    numbering, not by first-seen order, so re-running this script against
    an unchanged benchmark.jsonl always assigns the same ids)."""
    keyed = [(item["category"], _template_key(item)) for item in items]
    unique_keys = sorted(set(keyed))
    template_id_by_key = {}
    counters = defaultdict(int)
    for category, norm_q in unique_keys:
        counters[category] += 1
        template_id_by_key[(category, norm_q)] = f"{category}__t{counters[category]:03d}"

    out = []
    for item, key in zip(items, keyed):
        new_item = dict(item)
        new_item["template_id"] = template_id_by_key[key]
        out.append(new_item)
    return out


def _bresenham_split(n: int, test_fraction: float) -> list:
    """Returns a list of n booleans (True = assigned to test).

    n <= 1: entirely test (singleton-category rule, see module docstring).
    n >= 2: at least 1 family is ALWAYS assigned to test - a plain
    round(n * test_fraction) can floor to 0 for small n (e.g. n=2,
    test_fraction=0.3 -> 0.6 -> naive flooring gives 0 test families),
    which would leave a whole category with zero held-out coverage despite
    having more than one template to draw from - defeats the purpose of a
    per-category-aware split. n_test = max(1, round(n * test_fraction))
    guards against that. The n_test slots are then spread as EVENLY as
    possible across the n sorted template_ids (evenly-spaced index
    selection, same "distribute k items across n slots with minimal
    clumping" idea as Bresenham's line algorithm) rather than a contiguous
    block, which would bias toward whichever end of the sorted order
    happens to land in the block."""
    if n <= 1:
        return [True] * n
    n_test = min(max(1, round(n * test_fraction)), n)
    flags = [False] * n
    for i in range(n_test):
        idx = min(round((i + 0.5) * n / n_test), n - 1)
        flags[idx] = True
    # Rounding can occasionally collide two i's onto the same idx for small
    # n/n_test combinations - top up with the next available False slot so
    # exactly n_test flags end up True regardless.
    while sum(flags) < n_test:
        for i in range(n):
            if not flags[i]:
                flags[i] = True
                break
    return flags


def split_benchmark(items: list, test_fraction: float = 0.3) -> tuple:
    """Returns (dev_items, test_items, family_summary) - items already
    carry 'template_id' (see assign_template_ids()). family_summary is
    {template_id: {"category", "n_rows", "split"}} for the printed report."""
    tagged = assign_template_ids(items)

    by_category_families = defaultdict(list)
    for item in tagged:
        fams = by_category_families[item["category"]]
        if item["template_id"] not in fams:
            fams.append(item["template_id"])

    split_by_template_id = {}
    for category, family_ids in by_category_families.items():
        family_ids_sorted = sorted(family_ids)
        flags = _bresenham_split(len(family_ids_sorted), test_fraction)
        for tid, is_test in zip(family_ids_sorted, flags):
            split_by_template_id[tid] = "test" if is_test else "dev"

    dev_items, test_items = [], []
    family_summary = defaultdict(lambda: {"category": None, "n_rows": 0, "split": None})
    for item in tagged:
        tid = item["template_id"]
        family_summary[tid]["category"] = item["category"]
        family_summary[tid]["n_rows"] += 1
        family_summary[tid]["split"] = split_by_template_id[tid]
        if split_by_template_id[tid] == "test":
            test_items.append(item)
        else:
            dev_items.append(item)

    return dev_items, test_items, dict(family_summary)


def _write_jsonl(path: Path, items: list) -> None:
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--dev-output", type=Path, default=DEFAULT_DEV_OUTPUT)
    ap.add_argument("--test-output", type=Path, default=DEFAULT_TEST_OUTPUT)
    ap.add_argument("--test-fraction", type=float, default=0.3,
                     help="Target fraction of TEMPLATE FAMILIES (not rows) per category assigned to "
                          "router_test.jsonl (default 0.3). Categories with only 1 family always go "
                          "entirely to test - see module docstring.")
    args = ap.parse_args()

    items = []
    with args.input.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))

    dev_items, test_items, family_summary = split_benchmark(items, test_fraction=args.test_fraction)

    # Hard invariant: no template_id in both outputs.
    dev_families = {i["template_id"] for i in dev_items}
    test_families = {i["template_id"] for i in test_items}
    overlap = dev_families & test_families
    if overlap:
        raise RuntimeError(
            f"LEAKAGE DETECTED: {len(overlap)} template_id(s) present in BOTH router_development.jsonl "
            f"and router_test.jsonl - refusing to write output. This should be impossible by "
            f"construction, so this indicates a real bug in split_benchmark(): {sorted(overlap)}"
        )

    _write_jsonl(args.dev_output, dev_items)
    _write_jsonl(args.test_output, test_items)

    n_families = len(family_summary)
    n_test_families = sum(1 for f in family_summary.values() if f["split"] == "test")
    print(f"benchmark.jsonl: {len(items)} row(s) -> {n_families} template famil(y/ies) "
          f"({n_test_families} test, {n_families - n_test_families} dev)")
    print(f"  {args.dev_output}: {len(dev_items)} row(s)")
    print(f"  {args.test_output}: {len(test_items)} row(s)")

    by_category = defaultdict(lambda: {"dev": 0, "test": 0})
    for tid, info in family_summary.items():
        by_category[info["category"]][info["split"]] += info["n_rows"]
    print("  Per-category row counts (dev/test):")
    for category in sorted(by_category):
        counts = by_category[category]
        print(f"    {category}: dev={counts['dev']}, test={counts['test']}")


if __name__ == "__main__":
    main()
