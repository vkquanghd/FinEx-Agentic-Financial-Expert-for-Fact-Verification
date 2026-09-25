#!/usr/bin/env python3
"""Batch-test S1/S2/S3 on a fixed set of English questions, without clicking
through the UI one question at a time.

Usage
-----
Native / dev.sh mode (qa on 127.0.0.1:8101):
    python scripts/test_router_batch.py

Docker mode (qa's port is NOT published to the host - run it inside the
qa container instead, where 127.0.0.1:8101 IS reachable, since ./scripts
is bind-mounted read-only into every API container):
    docker compose exec qa python scripts/test_router_batch.py

Only Part 1 (single-turn) or only Part 2 (RQ3 memory) instead of both:
    python scripts/test_router_batch.py --part 1
    python scripts/test_router_batch.py --part 2

Custom questions instead of the built-in set (one question per line):
    python scripts/test_router_batch.py --questions my_questions.txt

What it does
------------
Part 1: sends each question through S1 (rag-only), S2 (dag-only) and S3
(router), and prints branch + latency + a short answer excerpt for each -
so you compare all three at a glance instead of three separate manual runs.

Part 2 (RQ3): sends multi-turn conversations through S3 ONLY (memory only
exists in the router), turn by turn with a REAL wait between turns and a
REAL history array - the thing the earlier manual test got wrong by
pasting a whole conversation as one message. Prints resolved_question and
context_resolved for each turn so you can see memory firing (or correctly
NOT firing on self-contained turns) without reading the full answer text.

Every answer is also scanned for Vietnamese diacritics as an automatic
"still leaking Vietnamese into an English answer?" check (this is the bug
fixed on 2026-09-25 in display_labels.py) - so you get a LEAK flag instead
of having to read every answer by eye. S1 is a frozen Vietnamese-only
baseline (see dag_pipeline.py's docstring) so a LEAK flag there is
EXPECTED, not a bug - only look for it on S2/S3.

Results are also written to a CSV next to this script so you have a
record to paste into the thesis or compare against a later run.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
import time
import uuid
from pathlib import Path

try:
    import requests
except ImportError:
    print("Missing dependency: pip install requests --break-system-packages", file=sys.stderr)
    sys.exit(1)

# Vietnamese-specific diacritic characters. Plain ASCII + "đ"/"Đ" only shows
# up here, never in ordinary English prose, so any hit is a real leak, not a
# false positive from a ticker name or a number.
_VN_CHARS = re.compile(
    r"[àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợ"
    r"ùúủũụưừứửữựỳýỷỹỵđÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊ"
    r"ÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴĐ]"
)

# --- Part 1: single-turn questions, run through S1/S2/S3 ------------------
# English, one per structured branch (see api/shared/schemas.py's Branch
# Literal) plus a couple of RAG/out_of_scope cases. Tickers reused from
# earlier real test runs (CTD/VNM/HPG/VNZ/HSG/KTS), so they should already
# have data loaded.
DEFAULT_QUESTIONS = [
    "What was CTD's total assets in 2024?",                                   # dag
    "How did CTD's total assets change over the last 3 years?",               # dag_trend
    "What is VNM's current ratio in 2024?",                                   # ratio_lookup
    "What if the interest rate increased by 2% for CTD - how would pretax profit be affected?",  # whatif
    "If steel prices rose 10%, how would HPG's gross margin change?",         # whatif (macro)
    "Is the current ratio affected by asset turnover for CTD?",               # causal
    "Compare the current ratio of CTD and HPG in 2024.",                     # compare
    "What is the market sentiment on VNM's profitability?",                  # absa
    "Who is the auditor for VNZ in 2025?",                                    # auditor_lookup (identity)
    "What did the auditor say about VNZ's 2025 financial statements?",       # auditor_lookup opinion -> should be rag
    "Which bank offers the highest 12-month deposit rate?",                   # bank_rate_lookup
    "What is CTD's stock price today?",                                      # price_snapshot
    "What was HPG's stock price over the last week?",                        # price range
    "What were the total assets of both CTD and KTS in 2024?",               # dag_multi
    "What contingent liabilities does VNZ disclose?",                        # rag + numeric verification
    "What are VNM's main legal risks?",                                      # rag
    "Should I buy VNM stock now?",                                           # out_of_scope
    "What will HPG's stock price be next month?",                           # out_of_scope
]

# --- Part 2: RQ3 multi-turn conversations, S3 only -------------------------
# Each conversation is a list of turns (question strings), sent one at a
# time with history = the raw previous user turns (mirrors app.js's
# `.slice(-2)`: only the last 2 user turns are sent, never the resolved
# ones). Labeled with the RQ3 benchmark category it targets
# (data/eval/rq7_conversational_benchmark.jsonl).
CONVERSATIONS = [
    ("pronoun_reference", [
        "What was CTD's total assets in 2024?",
        "What about its total liabilities?",
    ]),
    ("ellipsis", [
        "What is VNM's current ratio in 2024?",
        "And in 2023?",
    ]),
    ("subject_switch", [
        "What was CTD's total assets in 2024?",
        "What about HPG?",
    ]),
    ("self_contained (memory must NOT rewrite this)", [
        "What is CTD's total assets in 2024?",
        "What if the interest rate changed for CTD - how would pretax profit be affected?",
    ]),
    ("return_to_old_topic", [
        "What was CTD's total assets in 2024?",
        "What is HPG's current ratio in 2024?",
        "Going back to CTD, what about its total liabilities?",
    ]),
]


def check_health(base_url: str) -> None:
    try:
        r = requests.get(f"{base_url}/health", timeout=5)
        r.raise_for_status()
        data = r.json()
        print(f"[health] status={data['status']}  {data.get('detail', '')}")
        if data["status"] != "ok":
            print("  -> not fully healthy. Structured branches (dag/ratio/price/"
                  "auditor/bank_rate) still work; rag/absa/whatif/causal need Ollama.")
    except Exception as exc:
        print(f"[health] could not reach {base_url}: {exc}")
        print("  -> is the qa service actually running/reachable from here? See the "
              "docstring at the top of this script for dev.sh vs Docker mode.")
    print()


def ask(base_url: str, question: str, session_id: str, strategy: str,
        history: list[str] | None = None, top_k: int = 5) -> dict:
    payload = {
        "question": question,
        "session_id": session_id,
        "top_k": top_k,
        "strategy": strategy,
        "history": history or [],
        "ui_lang": "en",
    }
    t0 = time.perf_counter()
    r = requests.post(f"{base_url}/ask", json=payload, timeout=120)
    wall_ms = int((time.perf_counter() - t0) * 1000)
    r.raise_for_status()
    data = r.json()
    data["_wall_ms"] = wall_ms
    return data


# absa's "evidence" quote is BY DESIGN a verbatim excerpt from the (Vietnamese)
# source disclosure - src/absa/absa_pipeline.py's SYSTEM_PROMPT explicitly asks
# for "trích dẫn... nguyên văn" and is not lang-branched at all, same rationale
# as RAG quoting the original filing verbatim. So a LEAK on an absa answer is
# expected, not the display_labels.py class of bug - only the aspect LABELS
# ("Profitability / business results: ...") need to be English, and they are.
EXPECTED_QUOTE_BRANCHES = {"absa"}


def leak_flag(answer: str) -> str:
    return "LEAK" if _VN_CHARS.search(answer or "") else "clean"


def short(text: str, n: int = 90) -> str:
    text = (text or "").replace("\n", " ").strip()
    return text if len(text) <= n else text[: n - 1] + "…"


def run_part1(base_url: str, questions: list[str], rows: list[dict]) -> None:
    print("=" * 100)
    print("PART 1 - S1 / S2 / S3 side by side")
    print("=" * 100)
    for q in questions:
        print(f"\nQ: {q}")
        for strategy in ("s1", "s2", "s3"):
            session_id = uuid.uuid4().hex
            try:
                data = ask(base_url, q, session_id, strategy)
            except Exception as exc:
                print(f"  {strategy.upper():<3} ERROR: {exc}")
                rows.append({"part": 1, "strategy": strategy, "question": q, "error": str(exc)})
                continue
            flag = leak_flag(data["answer"])
            expected = strategy == "s1" or data["branch"] in EXPECTED_QUOTE_BRANCHES
            note = "" if flag == "clean" or expected else "  <-- check this"
            print(f"  {strategy.upper():<3} branch={data['branch']:<16} "
                  f"latency={data['latency_ms']:>6}ms  [{flag}]{note}")
            print(f"        {short(data['answer'])}")
            rows.append({
                "part": 1, "strategy": strategy, "question": q,
                "branch": data["branch"], "fallback_from": data.get("fallback_from"),
                "latency_ms": data["latency_ms"], "leak": flag,
                "answer_excerpt": short(data["answer"], 200), "error": "",
            })


def run_part2(base_url: str, rows: list[dict]) -> None:
    print("\n" + "=" * 100)
    print("PART 2 - RQ3 conversational memory (S3 only, real turn-by-turn)")
    print("=" * 100)
    for category, turns in CONVERSATIONS:
        print(f"\n--- {category} ---")
        session_id = uuid.uuid4().hex
        sent_questions: list[str] = []
        for i, q in enumerate(turns, start=1):
            history = sent_questions[-2:]  # mirrors app.js's .slice(-2)
            try:
                data = ask(base_url, q, session_id, "s3", history=history)
            except Exception as exc:
                print(f"  T{i} ERROR: {exc}")
                rows.append({"part": 2, "category": category, "turn": i, "question": q, "error": str(exc)})
                sent_questions.append(q)
                continue
            resolved = data.get("resolved_question")
            flag = leak_flag(data["answer"])
            print(f"  T{i}: {q}")
            if data.get("context_resolved"):
                print(f"       -> memory rewrote to: {resolved}")
            else:
                print(f"       -> sent as-is (memory did not rewrite)")
            print(f"       branch={data['branch']:<16} latency={data['latency_ms']:>6}ms  [{flag}]")
            print(f"       {short(data['answer'])}")
            rows.append({
                "part": 2, "category": category, "turn": i, "question": q,
                "resolved_question": resolved, "context_resolved": data.get("context_resolved"),
                "branch": data["branch"], "latency_ms": data["latency_ms"], "leak": flag,
                "answer_excerpt": short(data["answer"], 200), "error": "",
            })
            sent_questions.append(q)
            time.sleep(0.3)  # small real gap between turns, not required but mirrors a live demo


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://127.0.0.1:8101",
                     help="qa service base URL (default: %(default)s)")
    ap.add_argument("--part", choices=["1", "2", "both"], default="both")
    ap.add_argument("--questions", type=Path, default=None,
                     help="Optional text file, one English question per line, replaces the built-in Part 1 list")
    ap.add_argument(
        "--out", type=Path, default=None,
        help="Default: data/eval/router_batch_results.csv under the repo root. NOT next to "
             "this script - in Docker mode ./scripts is bind-mounted read-only (:ro), while "
             "./data is writable, so a default under scripts/ would crash with 'Read-only "
             "file system' the moment this ran inside the qa container.")
    args = ap.parse_args()

    if args.out is None:
        repo_root = Path(__file__).resolve().parent.parent
        args.out = repo_root / "data" / "eval" / "router_batch_results.csv"
    args.out.parent.mkdir(parents=True, exist_ok=True)

    questions = DEFAULT_QUESTIONS
    if args.questions:
        questions = [ln.strip() for ln in args.questions.read_text(encoding="utf-8").splitlines() if ln.strip()]

    check_health(args.base_url)

    rows: list[dict] = []
    if args.part in ("1", "both"):
        run_part1(args.base_url, questions, rows)
    if args.part in ("2", "both"):
        run_part2(args.base_url, rows)

    fieldnames = sorted({k for row in rows for k in row.keys()})
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    n_leak = sum(
        1 for r in rows
        if r.get("leak") == "LEAK" and r.get("strategy") != "s1"
        and r.get("branch") not in EXPECTED_QUOTE_BRANCHES
    )
    n_err = sum(1 for r in rows if r.get("error"))
    print("\n" + "=" * 100)
    print(f"Done. {len(rows)} calls -> {args.out}")
    print(f"  Vietnamese-leak flags worth checking (excludes S1 baseline + absa quoted evidence): {n_leak}")
    print(f"  Errors: {n_err}")
    print("=" * 100)


if __name__ == "__main__":
    main()
