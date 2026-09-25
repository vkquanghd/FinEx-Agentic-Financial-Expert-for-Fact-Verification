"""Interest-rate WHATIF + bank-deposit-rate-lookup addendum - 2026-09-15.
Continues the id numbering of the final_benchmark_*.py series (ids
121-126) - see final_benchmark_macro_en.py's docstring for why each of
these stays its own file.

WHY: Quang specifically asked (2026-09-15) for interest-rate-driven WHATIF
scenarios, ideally tied to "the bank the company works with, or the best
bank based on that bank's own characteristics". Investigation found real,
per-bank deposit-rate data in the 47-file macro batch (Vietstock "LÃI SUẤT
TIỀN GỬI NGÂN HÀNG", category DEPOSIT_RATES - TCB/VIB/VCB/CTG/... by day),
but NO data anywhere in this project records which bank any given ticker
actually borrows from or banks with - inventing that mapping would violate
the project's "no invented ground truth" rule everywhere else in this
router. So two SEPARATE, both real, capabilities were built instead (see
docs/PROGRESS.md's 2026-09-15 entry for the full design):
  1. `interest_rate_sensitivity` WHATIF (src/structured/calculator.py's
     new `sensitivity_what_if_interest_rate()`) - shocks a ticker's real
     "Chi phí tài chính" using VNIBOR's real overnight-rate move,
     economy-wide (not bank-specific, since no such data exists).
  2. `bank_rate_lookup` branch (src/causal/macro_lookup.py's new
     `best_deposit_rate_banks()`) - ticker-INDEPENDENT ranking of which
     bank currently reports the best deposit rate, a real market fact.

TICKER SELECTION: unlike gold/oil/industry/trade, `interest_rate_sensitivity`
has NO ticker-eligibility restriction in rules.py (see that branch's own
comment - "Chi phí tài chính" is a real line for every PANEL ticker, no
allowlist needed). This file's original 4 questions only used PANEL tickers
(CTD/HSG/BSR/KTS); the 2026-09-15 (tiếp 4) expansion below DELIBERATELY
also includes all 4 CORE_16 tickers (HPG/VNM/MWG/VIC) to exercise the
routing-correct-but-data-absent path: classify_intent() correctly returns
"whatif" for both tiers (routing has no notion of PANEL/CORE_16), but on a
live DB a CORE_16 ticker will honestly refuse with "no financial_line_items
data" (same accepted limitation as GAS in OIL_SENSITIVE_TICKERS, documented
in final_benchmark_trade_en.py). bank_rate_lookup needs no ticker at all.

CATEGORY: `interest_rate_whatif` (Group 2) and `bank_rate_lookup` (new,
ticker-independent - closer in shape to `auditor_lookup`, Group 1).

2026-09-15 (tiếp 4): EXPANDED 6 -> 30, ids re-numbered 161-190 (was
121-126). Composition: 16 `interest_rate_whatif` (all 8 benchmark tickers x
2 languages, 4 phrasing templates rotated), 10 `bank_rate_lookup` (5 VI + 5
EN phrasing variants, all empirically verified via classify_intent() before
being added here - one candidate phrasing that looked plausible by eye
("What-if scenario for MWG in 2024: how does the interest rate affect
pretax profit?") turned out to route to `causal_unsupported` instead,
because "affect"/"affects" is ALSO a registered CAUSAL marker
(`\baffects?\b` in rules.py) and CAUSAL is checked before WHATIF in
classify_intent()'s priority order - caught by testing, not assumed;
reworded to avoid the word "affect" entirely), and 4 refusal-control
questions confirming `bank_rate_lookup`'s ticker guard actually blocks a
ticker-bearing "which bank is best for TICKER" question (these correctly
fall through to RAG, not `bank_rate_lookup` - also empirically verified,
not assumed, before being written down as `expected_branch`).

METHODOLOGY: reused byte-for-byte from final_benchmark_100_en.py.

Usage:
    python -m scripts.final_benchmark_rate_en
    python -m scripts.score_audit_results --input data/eval/final_benchmark_rate_en.jsonl --chart \\
        --chart-output data/eval/runs/final_benchmark_rate_score.png
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

QUESTIONS = [
    # ============================================================
    # interest_rate_whatif - 16 Qs: all 8 benchmark tickers x 2 languages.
    # PANEL (CTD/HSG/BSR/KTS) expected to compute a real number on a live
    # DB; CORE_16 (HPG/VNM/MWG/VIC) expected to route correctly ("whatif")
    # but honestly refuse at execution time (no financial_line_items) -
    # same accepted limitation as GAS/OIL_SENSITIVE_TICKERS.
    # ============================================================
    dict(id=161, category="interest_rate_whatif", lang="vi", tickers=["CTD"],
         expected_branch="whatif", expected_outcome="answer",
         question="Nếu lãi suất tăng thì lợi nhuận trước thuế của CTD thay đổi ra sao?"),
    dict(id=162, category="interest_rate_whatif", lang="en", tickers=["CTD"],
         expected_branch="whatif", expected_outcome="answer",
         question="What if the interest rate changed for CTD - how would pretax profit be affected?"),
    dict(id=163, category="interest_rate_whatif", lang="vi", tickers=["HSG"],
         expected_branch="whatif", expected_outcome="answer",
         question="Nếu lãi suất biến động thì HSG bị ảnh hưởng thế nào?"),
    dict(id=164, category="interest_rate_whatif", lang="en", tickers=["HSG"],
         expected_branch="whatif", expected_outcome="answer",
         question="Suppose the interest rate shifted - how would HSG be affected?"),
    dict(id=165, category="interest_rate_whatif", lang="vi", tickers=["BSR"],
         expected_branch="whatif", expected_outcome="answer",
         question="Kịch bản lãi suất cho BSR năm 2024 sẽ ra sao?"),
    dict(id=166, category="interest_rate_whatif", lang="en", tickers=["BSR"],
         expected_branch="whatif", expected_outcome="answer",
         question="What-if scenario for BSR in 2024: how would pretax profit change under a real interest rate move?"),
    dict(id=167, category="interest_rate_whatif", lang="vi", tickers=["KTS"],
         expected_branch="whatif", expected_outcome="answer",
         question="Giả sử lãi suất thay đổi, KTS sẽ ra sao?"),
    dict(id=168, category="interest_rate_whatif", lang="en", tickers=["KTS"],
         expected_branch="whatif", expected_outcome="answer",
         question="What if the interest rate changed for KTS - how would pretax profit be affected?"),
    dict(id=169, category="interest_rate_whatif", lang="vi", tickers=["HPG"],
         expected_branch="whatif", expected_outcome="refuse",
         question="Nếu lãi suất tăng thì lợi nhuận trước thuế của HPG thay đổi ra sao?",
         note="HPG=CORE_16 (zero financial_line_items) - routing correctly "
              "resolves to 'whatif', but a live-DB run should honestly "
              "refuse with 'no financial data', not fabricate a number."),
    dict(id=170, category="interest_rate_whatif", lang="en", tickers=["HPG"],
         expected_branch="whatif", expected_outcome="refuse",
         question="What if the interest rate changed for HPG - how would pretax profit be affected?"),
    dict(id=171, category="interest_rate_whatif", lang="vi", tickers=["VNM"],
         expected_branch="whatif", expected_outcome="refuse",
         question="Nếu lãi suất biến động thì VNM bị ảnh hưởng thế nào?",
         note="VNM=CORE_16 - same honest-refusal expectation as id=169."),
    dict(id=172, category="interest_rate_whatif", lang="en", tickers=["VNM"],
         expected_branch="whatif", expected_outcome="refuse",
         question="What if the interest rate changed for VNM - how would pretax profit be affected?"),
    dict(id=173, category="interest_rate_whatif", lang="vi", tickers=["MWG"],
         expected_branch="whatif", expected_outcome="refuse",
         question="Kịch bản lãi suất cho MWG năm 2024 sẽ ra sao?",
         note="MWG=CORE_16 - same honest-refusal expectation as id=169."),
    dict(id=174, category="interest_rate_whatif", lang="en", tickers=["MWG"],
         expected_branch="whatif", expected_outcome="refuse",
         question="What-if scenario for MWG in 2024: how would pretax profit change under a real interest rate move?"),
    dict(id=175, category="interest_rate_whatif", lang="vi", tickers=["VIC"],
         expected_branch="whatif", expected_outcome="refuse",
         question="Giả sử lãi suất thay đổi, VIC sẽ ra sao?",
         note="VIC=CORE_16 - same honest-refusal expectation as id=169. "
              "Completes all 8 benchmark tickers across this category."),
    dict(id=176, category="interest_rate_whatif", lang="en", tickers=["VIC"],
         expected_branch="whatif", expected_outcome="refuse",
         question="Suppose the interest rate shifted - how would VIC be affected?"),
    # ============================================================
    # bank_rate_lookup - 10 Qs: 5 VI + 5 EN phrasing variants, all
    # ticker-independent, all empirically verified to route correctly
    # before being added (see module docstring).
    # ============================================================
    dict(id=177, category="bank_rate_lookup", lang="vi", tickers=[],
         expected_branch="bank_rate_lookup", expected_outcome="answer",
         question="Ngân hàng nào có lãi suất tiền gửi tốt nhất hiện nay?"),
    dict(id=178, category="bank_rate_lookup", lang="vi", tickers=[],
         expected_branch="bank_rate_lookup", expected_outcome="answer",
         question="Ngân hàng nào có lãi suất tốt nhất?"),
    dict(id=179, category="bank_rate_lookup", lang="vi", tickers=[],
         expected_branch="bank_rate_lookup", expected_outcome="answer",
         question="Đâu là ngân hàng có lãi suất tiền gửi tốt nhất?"),
    dict(id=180, category="bank_rate_lookup", lang="vi", tickers=[],
         expected_branch="bank_rate_lookup", expected_outcome="answer",
         question="Ngân hàng nào đang trả lãi suất tiền gửi cao nhất?"),
    dict(id=181, category="bank_rate_lookup", lang="vi", tickers=[],
         expected_branch="bank_rate_lookup", expected_outcome="answer",
         question="Lãi suất tiền gửi ngân hàng nào tốt nhất hiện nay?"),
    dict(id=182, category="bank_rate_lookup", lang="en", tickers=[],
         expected_branch="bank_rate_lookup", expected_outcome="answer",
         question="Which bank has the best deposit rate right now?"),
    dict(id=183, category="bank_rate_lookup", lang="en", tickers=[],
         expected_branch="bank_rate_lookup", expected_outcome="answer",
         question="What bank currently offers the highest deposit rate?"),
    dict(id=184, category="bank_rate_lookup", lang="en", tickers=[],
         expected_branch="bank_rate_lookup", expected_outcome="answer",
         question="What is the best bank for deposit rates today?"),
    dict(id=185, category="bank_rate_lookup", lang="en", tickers=[],
         expected_branch="bank_rate_lookup", expected_outcome="answer",
         question="Which bank offers the best deposit rate currently?"),
    dict(id=186, category="bank_rate_lookup", lang="en", tickers=[],
         expected_branch="bank_rate_lookup", expected_outcome="answer",
         question="What is the highest deposit rate bank right now?"),
    # ============================================================
    # bank_rate_lookup_control - 4 Qs: confirms the branch's ticker guard
    # (`not ticker_here`) actually blocks a ticker-bearing question from
    # hitting bank_rate_lookup - these fall through to RAG instead (no
    # WHATIF trigger word either), which will honestly say it has no such
    # company-bank-relationship data rather than silently answering as if
    # it were a plain deposit-rate ranking question. Verified empirically
    # via classify_intent() (branch="rag"), not assumed.
    # ============================================================
    dict(id=187, category="bank_rate_lookup_control", lang="vi", tickers=["CTD"],
         expected_branch="rag", expected_outcome="refuse",
         question="CTD có lãi suất tiền gửi ngân hàng tốt nhất không?",
         note="Ticker present -> bank_rate_lookup's 'not ticker_here' guard "
              "correctly blocks it; no WHATIF trigger word either, so it "
              "falls to RAG, which has no company-bank-relationship data "
              "to answer from and should honestly say so."),
    dict(id=188, category="bank_rate_lookup_control", lang="vi", tickers=["HSG"],
         expected_branch="rag", expected_outcome="refuse",
         question="Ngân hàng nào cho HSG vay với lãi suất tốt nhất?"),
    dict(id=189, category="bank_rate_lookup_control", lang="en", tickers=["HSG"],
         expected_branch="rag", expected_outcome="refuse",
         question="Which bank offers the best interest rate for HSG?"),
    dict(id=190, category="bank_rate_lookup_control", lang="en", tickers=["CTD"],
         expected_branch="rag", expected_outcome="refuse",
         question="Which bank does CTD work with for the best interest rate?"),
]

_REFUSAL_PATTERNS = re.compile(
    r"không tìm thấy|không tính được|không ước lượng được|không phân tích được|"
    r"no information found|not found in|không xác định được|không đủ dữ liệu|"
    r"không xếp hạng được",
    re.IGNORECASE,
)


def _looks_like_refusal(answer_text: str) -> bool:
    return bool(_REFUSAL_PATTERNS.search(answer_text or ""))


def _branch_matches_expected(actual_branch: str, expected_branch: str) -> bool:
    if not actual_branch or not expected_branch:
        return False
    allowed = {b.strip() for b in expected_branch.split(" or ")}
    return actual_branch in allowed


def run_one(q: dict) -> dict:
    from src.router.query_router import route

    t0 = time.time()
    record = {
        "id": q["id"], "category": q["category"], "lang": q["lang"],
        "tickers": q["tickers"], "question": q["question"],
        "expected_branch": q["expected_branch"], "expected_outcome": q["expected_outcome"],
    }
    try:
        result = route(q["question"], with_rationale=False)
        elapsed = round(time.time() - t0, 1)
        answer = result.get("answer", "")
        actual_branch = result.get("branch")
        refused = _looks_like_refusal(answer)
        record.update({
            "ok": True,
            "actual_branch": actual_branch,
            "branch_match": _branch_matches_expected(actual_branch, q["expected_branch"]),
            "looks_like_refusal": refused,
            "outcome_match": (("refuse" if refused else "answer") == q["expected_outcome"]),
            "intent_reason": result.get("intent_reason"),
            "answer": answer,
            "elapsed_s": elapsed,
        })
    except Exception as e:
        elapsed = round(time.time() - t0, 1)
        record.update({"ok": False, "error": f"{type(e).__name__}: {e}", "elapsed_s": elapsed})
    return record


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/eval/final_benchmark_rate_en.jsonl")
    ap.add_argument("--only", default=None)
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done_ids = set()
    if out_path.exists():
        with out_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    done_ids.add(json.loads(line)["id"])

    only_ids = {int(x) for x in args.only.split(",")} if args.only else None
    todo = [q for q in QUESTIONS if q["id"] not in done_ids and (only_ids is None or q["id"] in only_ids)]
    print(f"{len(QUESTIONS)} total, {len(done_ids)} already done, {len(todo)} to run.")

    with out_path.open("a", encoding="utf-8") as f:
        for i, q in enumerate(todo, 1):
            print(f"\n[{i}/{len(todo)}] id={q['id']} ({q['category']}, {q['lang']}): {q['question'][:70]}...")
            record = run_one(q)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            status = "OK" if record.get("ok") else "ERROR"
            print(f"  -> {status} branch={record.get('actual_branch', '?')} ({record.get('elapsed_s')}s)")

    print(f"\nDone. Results in {out_path}")


if __name__ == "__main__":
    main()
