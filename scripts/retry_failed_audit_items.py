"""Retry ONLY the items in data/eval/answer_quality_audit.jsonl that failed
with a transient error (ok=false), instead of re-running the whole 52-
question audit again.

Why this script exists (2026-09-13): the full run Quang did on 2026-09-12
produced 12 failures, ALL of them the exact same error -
`OllamaNotReachable ... after 3 attempt(s) ... read timeout=180` (see
docs/PROGRESS.md, "2026-09-13 - phân tích nguyên nhân 12 câu lỗi" entry, for
the full analysis). scripts/audit_answer_quality.py's own "resume" logic
(see its docstring/main()) skips ANY question id already present in the
output file, success or failure alike - so simply re-running
`python -m scripts.audit_answer_quality` again would SKIP the 12 failed
items FOREVER, not retry them. This script is the fix, following the exact
same pattern the project already uses for a different, earlier evaluation
harness (see scripts/retry_failed_llm_items.py's docstring for that
precedent): find the failed ids, remove exactly those rows from the JSONL,
then re-run the underlying question generator for exactly those ids and
append fresh results.

BEFORE running this: confirm Ollama is actually responsive again -
    curl http://localhost:11434/api/tags
should return JSON immediately, not hang for several seconds. If it hangs,
restart Ollama first (quit Ollama.app / `killall ollama`, then
`ollama serve` again, or reopen Ollama.app) - retrying against a still-
degraded Ollama process will just reproduce the same 12 timeouts. Setting
OLLAMA_KEEP_ALIVE (e.g. `export OLLAMA_KEEP_ALIVE=30m` before `ollama
serve`) is also worth doing before a long retry batch, so the model isn't
evicted from memory and reloaded from disk between questions - a slow
reload under memory pressure is the leading suspect for why every failure
timed out at the full 180s x3, not a quick connection error (see
PROGRESS.md entry for the reasoning).

Usage:
    python -m scripts.retry_failed_audit_items
    python -m scripts.retry_failed_audit_items --file data/eval/answer_quality_audit.jsonl
"""
import argparse
import json
from pathlib import Path

from scripts.audit_answer_quality import QUESTIONS, run_one


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="data/eval/answer_quality_audit.jsonl")
    args = ap.parse_args()

    path = Path(args.file)
    if not path.exists():
        raise SystemExit(f"{path} not found - run scripts.audit_answer_quality first.")

    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    failed_ids = sorted({r["id"] for r in records if not r.get("ok", False)})
    if not failed_ids:
        print("No failed (ok=false) items found in the file - nothing to retry.")
        return

    print(f"Found {len(failed_ids)} failed item(s) to retry: {failed_ids}")
    kept = [r for r in records if r["id"] not in failed_ids]

    # Write back only the already-successful rows FIRST (before attempting
    # any retry) - if this script itself gets interrupted mid-retry, the
    # file is left in a clean, non-duplicated state rather than half old/
    # half new for the same id.
    with path.open("w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Kept {len(kept)} already-successful row(s); removed {len(failed_ids)} failed row(s) - now retrying.")

    by_id = {q["id"]: q for q in QUESTIONS}
    with path.open("a", encoding="utf-8") as f:
        for i, qid in enumerate(failed_ids, 1):
            q = by_id.get(qid)
            if q is None:
                print(f"  [{i}/{len(failed_ids)}] id={qid}: not found in QUESTIONS (was it removed?) - skipping")
                continue
            print(f"[{i}/{len(failed_ids)}] id={qid} ({q['category']}): {q['question'][:70]}...")
            record = run_one(q)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            status = "ERROR" if not record.get("ok") else (
                "REFUSAL?" if record.get("looks_like_refusal") else "answered"
            )
            print(f"    -> {status} ({record['elapsed_s']}s)")

    print(f"\nDone. Re-run `python -m scripts.audit_results_to_excel` to refresh the Excel/CSV.")


if __name__ == "__main__":
    main()
