"""Task #23, roadmap item 6 (2026-07-30): evaluates the Router against the
FROZEN 39-question data/eval/router_test.jsonl - "the test set must never
again be used to fix/tune rules; develop only against the 101-question
router_development.jsonl; run the test only once a version is locked;
report confusion matrix and macro-F1."

**This script reads ONLY data/eval/router_test.jsonl. It must NEVER import
or read data/eval/router_development.jsonl** - that file is for iterating
on src/router/rules.py while developing; router_test.jsonl exists
specifically so there is ALWAYS one held-out set the rules were never
tuned against. If you are debugging a classify_intent() change, use
src/evaluation/run_eval.py's run_routing() against router_development.jsonl
(or benchmark.jsonl) instead of this script.

Uses classify_intent() only (src/router/rules.py), not the full route()
dispatch - same convention as src.evaluation.run_eval.run_routing(), pure
rule-based classification with no DB/LLM call, matching what the roadmap's
"routing rules" actually refers to.

Scope: evaluates every row in router_test.jsonl that carries a non-null
`expected_branch` - NOT filtered to category=="routing" only (unlike
run_eval.run_routing()'s narrower filter). router_test.jsonl's 39 rows span
several categories (routing, whatif_exact_match, rag_faithfulness,
rag_temporal, causal_comparison, absa_eval, regime_ablation,
nhanh_a_vs_b); of these, whatif_exact_match/rag_faithfulness/rag_temporal/
routing rows ALSO carry a real expected_branch (e.g. a whatif_exact_match
question is legitimate ground truth for "did the Router send this to the
whatif branch") - restricting to category=="routing" alone would only use
2 of the 39 rows, which does not match the roadmap's own "39-question"
framing of this file as the Router's frozen test set.

LOCKING DISCIPLINE: running this script WITHOUT --i-am-locking-this-version
is a dry run - it prints results but does NOT append to reports/router_
test_runs.jsonl. Only --i-am-locking-this-version writes an official,
timestamped, git-commit-stamped row to that append-only log. This is
deliberate friction, not a technical enforcement mechanism (nothing stops
someone from re-running this file's logic by hand) - the roadmap's actual
concern is a PROCESS discipline ("don't repeatedly peek at the test set
while iterating"), so the friction here is a print statement + an explicit
flag someone has to consciously type, not a database lock. Every locked
run is still logged (never overwritten), so "how many times has this test
actually been officially run" is answerable from the log itself, same
append-only-log philosophy as causal_experiment_runs (Task #11).

Usage:
    python3 -m scripts.router.evaluate_frozen_test                       # dry run, not logged
    python3 -m scripts.router.evaluate_frozen_test --i-am-locking-this-version
"""
import argparse
import json
import subprocess
import time
from pathlib import Path

ROUTER_TEST_PATH = Path("data/eval/router_test.jsonl")
RUNS_LOG_PATH = Path("reports/router_test_runs.jsonl")

_FORBIDDEN_PATH_FRAGMENT = "router_development"  # defense-in-depth guard, see load_frozen_test()


def _git_commit() -> str:
    """Best-effort short git SHA - same pattern as scripts/run_causal_
    experiment.py's _git_commit(), None if unavailable."""
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=5, check=True)
        return out.stdout.strip() or None
    except Exception:
        return None


def load_frozen_test(path: Path = ROUTER_TEST_PATH) -> list:
    """Loads router_test.jsonl and returns only rows with a non-null
    expected_branch (see module docstring for why this isn't restricted to
    category=="routing"). Refuses outright (raises, does not just warn) if
    ever pointed at a path containing "router_development" - a defense-in-
    depth guard against this frozen-test script being accidentally invoked
    against the dev set by a copy-paste path mistake."""
    if _FORBIDDEN_PATH_FRAGMENT in str(path):
        raise ValueError(
            f"Refusing to load {path} - this script must NEVER read router_development.jsonl "
            f"(see this module's docstring). Use src.evaluation.run_eval.run_routing() for dev-set iteration."
        )
    items = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return [it for it in items if it.get("expected_branch") is not None]


def evaluate() -> dict:
    from src.evaluation.metrics import confusion_matrix, routing_accuracy
    from src.router.rules import classify_intent

    items = load_frozen_test()
    if not items:
        raise SystemExit(f"0 row(s) with a non-null expected_branch in {ROUTER_TEST_PATH} - nothing to evaluate.")

    predicted, expected, detail = [], [], []
    for item in items:
        intent = classify_intent(item["question"])
        predicted.append(intent.branch)
        expected.append(item["expected_branch"])
        detail.append({
            "id": item["id"], "category": item.get("category"), "question": item["question"],
            "expected_branch": item["expected_branch"], "predicted_branch": intent.branch,
            "correct": intent.branch == item["expected_branch"], "reason": getattr(intent, "reason", None),
        })

    metrics = routing_accuracy(predicted, expected)
    cm = confusion_matrix(predicted, expected)

    return {
        "n": len(items),
        "source_file": str(ROUTER_TEST_PATH),
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "per_class": metrics["per_class"],
        "confusion_matrix": cm,
        "detail": detail,
        "n_misclassified": sum(1 for d in detail if not d["correct"]),
    }


def print_confusion_matrix(cm: dict):
    labels = cm["labels"]
    matrix = cm["matrix"]
    col_width = max(len(l) for l in labels + ["actual\\pred"]) + 2
    header = "actual\\pred".ljust(col_width) + "".join(l.rjust(col_width) for l in labels)
    print(header)
    for i, label in enumerate(labels):
        row = label.ljust(col_width) + "".join(str(matrix[i][j]).rjust(col_width) for j in range(len(labels)))
        print(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--i-am-locking-this-version", action="store_true",
                     help="Writes this run as an OFFICIAL, timestamped, git-commit-stamped row to "
                          f"{RUNS_LOG_PATH} (append-only). Without this flag, results print but are NOT logged - "
                          "a plain run is a dry run, not an official evaluation of a locked rules version.")
    ap.add_argument("--report", type=Path, default=Path("reports/router_frozen_test_latest.json"))
    args = ap.parse_args()

    result = evaluate()

    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n=== Router frozen-test evaluation: {result['source_file']} ({result['n']} row(s)) ===")
    print(f"Accuracy={result['accuracy']:.4f}  Macro-F1={result['macro_f1']:.4f}  "
          f"Misclassified={result['n_misclassified']}/{result['n']}")
    print("\nPer-class:")
    for label, info in result["per_class"].items():
        print(f"  {label:24s} precision={info['precision']:.3f}  recall={info['recall']:.3f}  "
              f"f1={info['f1']:.3f}  support={info['support']}")
    print("\nConfusion matrix (rows=actual, columns=predicted):")
    print_confusion_matrix(result["confusion_matrix"])

    if result["n_misclassified"]:
        print("\nMisclassified items:")
        for d in result["detail"]:
            if not d["correct"]:
                print(f"  [{d['id']}] expected={d['expected_branch']!r} predicted={d['predicted_branch']!r} "
                      f"- {d['question']!r}")

    if args.i_am_locking_this_version:
        RUNS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        log_row = {
            "logged_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "code_commit": _git_commit(),
            "n": result["n"], "accuracy": result["accuracy"], "macro_f1": result["macro_f1"],
            "n_misclassified": result["n_misclassified"],
        }
        with RUNS_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(log_row, ensure_ascii=False) + "\n")
        print(f"\n*** LOCKED RUN LOGGED to {RUNS_LOG_PATH}: {log_row} ***")
    else:
        print(f"\n(DRY RUN - not logged. Pass --i-am-locking-this-version to record this as an "
              f"official run in {RUNS_LOG_PATH}.)")

    print(f"\nReport written: {args.report}")


if __name__ == "__main__":
    main()
