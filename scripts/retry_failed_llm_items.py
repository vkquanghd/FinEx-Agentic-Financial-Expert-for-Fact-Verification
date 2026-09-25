"""Re-run ONLY the benchmark items that failed with a transient error in the
last full `run_eval.py` pass, instead of re-running all 176 items (which
costs 5-8+ hours of real Ollama generation - see docs/PROGRESS.md's
2026-09-11 entry).

Real incident this targets (found 2026-09-11, from the first full non-
routing-only run since the VCB cleanup): after ~7-8 hours of sustained local
Ollama load, `OllamaNotReachable` started appearing for real (not the
transient 5xx `_generate_raw()` already retries - this is the read-timeout
case, i.e. Ollama itself stopped responding within the 180s window even
after 3 attempts). It hit 3 categories:
    rag                        9 / 75 items
    baseline_rag_only_whatif  20 / 26 items
    absa_eval                  4 / 5  items
(sparse_coverage_no_hallucination failed for a DIFFERENT, non-transient
reason - migration 014_web_chunks.sql was written but never applied to the
live Postgres DB, so every item hit `UndefinedTable: relation "web_chunks"
does not exist`. Re-running those items won't help until the migration is
applied - this script does NOT retry that category, see README/DOCKER.md.)

nhanh_a_vs_b's nhanh_b_accuracy=0.0 is NOT a transient failure either - it's
the documented "27/27 success but 0 rows" finding (Nhánh B, given no
ms_code hint, guesses a plausible-but-wrong `item_name` string and the exact
WHERE match silently returns 0 rows - see text_to_sql_agent.py's docstring,
2026-08-07). Retrying won't change that; if you want to see how much an
explicit ms_code hint would help, run
`python3 -m src.evaluation.run_eval` isn't the tool for that ablation -
see the with_ms_code_hint parameter in text_to_sql_agent.answer()/
run_eval.run_nhanh_a_vs_b() instead.

BEFORE running this: make sure Ollama is actually back up
(`curl http://localhost:11434/api/tags` should return JSON, not hang/error).
The whole point of retrying is that the last run's failures were the SERVER
being unreachable, not a code bug - retrying against a still-dead Ollama
will just reproduce the same errors.

Usage:
    python3 -m scripts.retry_failed_llm_items
    python3 -m scripts.retry_failed_llm_items --report data/eval/report.json
"""
import argparse
import json
from pathlib import Path

from src.evaluation.metrics import baseline_rag_only_summary
from src.evaluation.run_eval import (
    _rag_metrics,
    load_benchmark,
    run_absa_eval,
    run_baseline_rag_only_whatif,
    run_rag,
)

RETRYABLE_CATEGORIES = {
    # report key -> (benchmark categories this runner filters on, runner fn)
    "rag": (("rag_faithfulness", "rag_temporal"), None),  # handled specially (run_rag signature differs)
    "absa_eval": (("absa_eval",), run_absa_eval),
    "baseline_rag_only_whatif": (("whatif_exact_match",), run_baseline_rag_only_whatif),
}


def _failed_ids(report: dict, report_key: str) -> set:
    detail = report.get(report_key, {}).get("detail", [])
    return {d["id"] for d in detail if "error" in d}


def _replace_by_id(old_detail: list, new_detail: list) -> list:
    """Merge new_detail into old_detail, matching by "id" - keeps every item
    that wasn't retried untouched, replaces exactly the ones that were."""
    new_by_id = {d["id"]: d for d in new_detail}
    merged = [new_by_id.get(d["id"], d) for d in old_detail]
    # in case a retried id somehow isn't in old_detail (shouldn't happen)
    seen = {d["id"] for d in merged}
    merged += [d for d in new_detail if d["id"] not in seen]
    return merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="data/eval/benchmark.jsonl")
    ap.add_argument("--report", default="data/eval/report.json")
    args = ap.parse_args()

    report_path = Path(args.report)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    all_items = load_benchmark(Path(args.benchmark))
    by_id = {it["id"]: it for it in all_items}

    total_retried = 0

    # --- rag (rag_faithfulness + rag_temporal) ---
    failed = _failed_ids(report, "rag")
    if failed:
        print(f"[rag] retrying {len(failed)} failed item(s): {sorted(failed)}")
        subset = [by_id[i] for i in failed if i in by_id]
        result = run_rag(subset, with_temporal_anchor=True)
        merged_detail = _replace_by_id(report["rag"]["detail"], result["detail"])
        report["rag"] = {"metrics": _rag_metrics(merged_detail), "detail": merged_detail}
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        total_retried += len(failed)
        print(f"    -> new avg_keyword_faithfulness={report['rag']['metrics']['avg_keyword_faithfulness']} "
              f"n_errors={report['rag']['metrics']['n_errors']}")
    else:
        print("[rag] no failed items to retry.")

    # --- absa_eval ---
    failed = _failed_ids(report, "absa_eval")
    if failed:
        print(f"[absa_eval] retrying {len(failed)} failed item(s): {sorted(failed)}")
        subset = [by_id[i] for i in failed if i in by_id]
        result = run_absa_eval(subset)
        merged_detail = _replace_by_id(report["absa_eval"]["detail"], result["detail"])
        n = len(merged_detail)
        n_absa_ok = sum(1 for d in merged_detail if d.get("absa_ok"))
        n_valid = sum(1 for d in merged_detail if d.get("structurally_valid"))
        report["absa_eval"] = {
            "metrics": {"n": n, "n_absa_ok": n_absa_ok, "n_structurally_valid": n_valid,
                        "structurally_valid_rate": round(n_valid / n, 4) if n else None},
            "detail": merged_detail,
        }
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        total_retried += len(failed)
        print(f"    -> new structurally_valid_rate={report['absa_eval']['metrics']['structurally_valid_rate']}")
    else:
        print("[absa_eval] no failed items to retry.")

    # --- baseline_rag_only_whatif ---
    failed = _failed_ids(report, "baseline_rag_only_whatif")
    if failed:
        print(f"[baseline_rag_only_whatif] retrying {len(failed)} failed item(s): {sorted(failed)}")
        subset = [by_id[i] for i in failed if i in by_id]
        result = run_baseline_rag_only_whatif(subset)
        merged_detail = _replace_by_id(report["baseline_rag_only_whatif"]["detail"], result["detail"])
        report["baseline_rag_only_whatif"] = {
            "metrics": baseline_rag_only_summary(merged_detail), "detail": merged_detail,
        }
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        total_retried += len(failed)
        print(f"    -> new any_expected_field_found_rate="
              f"{report['baseline_rag_only_whatif']['metrics'].get('any_expected_field_found_rate')}")
    else:
        print("[baseline_rag_only_whatif] no failed items to retry.")

    print(f"\nDone. Retried {total_retried} item(s) total. report.json updated in place.")
    print("NOT retried (need a different fix, not a re-run - see this script's docstring):")
    print("  - sparse_coverage_no_hallucination: apply migration 014_web_chunks.sql first")
    print("  - nhanh_a_vs_b: nhanh_b_accuracy=0.0 is a documented finding, not a failure")


if __name__ == "__main__":
    main()
