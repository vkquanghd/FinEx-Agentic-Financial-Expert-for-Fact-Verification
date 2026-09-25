"""BCTC alias-expansion addendum - 2026-09-15. Continues the id numbering
of the final_benchmark_*.py series (ids 136-151) - see
final_benchmark_macro_en.py's docstring for why each addendum stays its
own file rather than being appended into final_benchmark_100_en.py.

WHY: the 4th (last) item Quang selected via AskUserQuestion this session -
"Mở rộng alias 45 dòng BCTC". Auditing `src/structured/schema.py` against
`src/core/constants.py`'s full 45-item `CORE_MS_CODES` found that only ~20
codes had a natural VI/EN alias beyond the OFFICIAL line-item name that
step 1 of that file auto-registers (which still carries its Roman-numeral/
letter prefix - e.g. code 120's official name normalizes to "ii au tu tai
chinh ngan han", a string nobody would actually type). The other 25 codes
were reachable ONLY by typing that exact unwieldy official string - in
practice unmatchable from a real question. Full audit + the fix (adding a
natural VI alias, minus the prefix, plus VI/EN variants, per unaliased
code) is in `src/structured/schema.py`'s new section "4)" and
docs/PROGRESS.md's 2026-09-15 entry.

REAL BUG CAUGHT WHILE BUILDING THIS (not a question-design issue - a bug
in my own edit, caught before it shipped): the first version of that
schema.py edit accidentally deleted the pre-existing
`for _alias, _stype, _code in _ENGLISH_ALIASES: _register(...)` loop while
inserting the new section right after it - which would have silently
un-registered EVERY English alias in the file (revenue, cogs, total
assets, all of it), not just failed to add anything new. Caught by testing
`resolve_indicator("What is the net revenue of HSG in 2023?")` directly
(returned None instead of code "10") before running this benchmark -
fixed by restoring the loop. Documented here because it's exactly the kind
of collateral-damage-from-an-edit this project's regression discipline
exists to catch, and it very nearly shipped silently.

DELIBERATE AMBIGUITY CALL: code "01" ("Doanh thu bán hàng và cung cấp dịch
vụ", GROSS revenue) got an ENGLISH-ONLY alias ("gross revenue"/"gross
sales"), no Vietnamese one. Its official name differs from code "10"'s
("Doanh thu THUẦN về bán hàng và cung cấp dịch vụ", NET revenue) by only
the word "thuần" - "doanh thu bán hàng" is used loosely by many Vietnamese
speakers to mean NET revenue in casual speech, so registering it as a
short VI alias for "01" risked silently mis-resolving a real net-revenue
question to the gross-revenue line. Left unaliased in VI rather than
guess; see id=151 below, which is a control case confirming a bare
"Doanh thu" VI question still resolves to code "10" (net), unaffected by
the new "01" alias.

TICKER SELECTION: all questions use PANEL tickers (CTD, HSG, BSR, KTS) -
same tier restriction as every structured_lookup question in this series,
since only PANEL tickers have real financial_line_items rows to actually
answer a DAG lookup on a live DB.

CATEGORY: `alias_expansion_lookup` (Group 1, DAG family) - all expected to
route to "dag", same shape as final_benchmark_100_en.py's own
structured_lookup category.

2026-09-15 (tiếp 4): EXPANDED 16 -> 30, ids re-numbered 221-250 (was
136-151). Now ALL 25 previously-unaliased codes get exactly one question
each (was 14/25 before - 11 codes had no coverage at all), tickers cycling
CTD/HSG/BSR/KTS and language strictly alternating VI/EN (code "01" always
lands on an EN slot, consistent with the VI-ambiguity call above). 3 more
questions added beyond the 25+2-control base: a second instance of code 25
("Chi phí bán hàng") with an explicit QUARTER period instead of full-year
(exercises the quarter-period parser on a newly-aliased code, not just the
year path), a second instance of code 26 using the SHORTER registered
alias "chi phí quản lý" (without "doanh nghiệp") to confirm both VI
variants for that code resolve correctly, and a second instance of code 31
in English with a quarter period. Every one of these 30 questions was
verified directly against `resolve_indicator()` (not just
`classify_intent()`) before being written down, since alias resolution
itself - not just branch routing - is exactly what's under test here.

METHODOLOGY: reused byte-for-byte from final_benchmark_100_en.py.

Usage:
    python -m scripts.final_benchmark_alias_en
    python -m scripts.score_audit_results --input data/eval/final_benchmark_alias_en.jsonl --chart \\
        --chart-output data/eval/runs/final_benchmark_alias_score.png
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
    # alias_expansion_lookup - 25 Qs, one per previously-unaliased code
    # (was 14/25 covered before this expansion). Tickers cycle CTD/HSG/
    # BSR/KTS, language strictly alternates VI/EN.
    # ============================================================
    dict(id=221, category="alias_expansion_lookup", lang="vi", tickers=["CTD"],
         expected_branch="dag", expected_outcome="answer",
         question="Đầu tư tài chính ngắn hạn năm 2024 của CTD là bao nhiêu?"),
    dict(id=222, category="alias_expansion_lookup", lang="en", tickers=["HSG"],
         expected_branch="dag", expected_outcome="answer",
         question="What was HSG's short term receivables in 2024?"),
    dict(id=223, category="alias_expansion_lookup", lang="vi", tickers=["BSR"],
         expected_branch="dag", expected_outcome="answer",
         question="Tài sản sinh học ngắn hạn năm 2024 của BSR là bao nhiêu?"),
    dict(id=224, category="alias_expansion_lookup", lang="en", tickers=["KTS"],
         expected_branch="dag", expected_outcome="answer",
         question="What was KTS's other current assets in 2024?"),
    dict(id=225, category="alias_expansion_lookup", lang="vi", tickers=["CTD"],
         expected_branch="dag", expected_outcome="answer",
         question="Phải thu dài hạn năm 2024 của CTD là bao nhiêu?"),
    dict(id=226, category="alias_expansion_lookup", lang="en", tickers=["HSG"],
         expected_branch="dag", expected_outcome="answer",
         question="What was HSG's fixed assets in 2024?"),
    dict(id=227, category="alias_expansion_lookup", lang="vi", tickers=["BSR"],
         expected_branch="dag", expected_outcome="answer",
         question="Tài sản sinh học dài hạn năm 2024 của BSR là bao nhiêu?"),
    dict(id=228, category="alias_expansion_lookup", lang="en", tickers=["KTS"],
         expected_branch="dag", expected_outcome="answer",
         question="What was KTS's investment property in 2024?"),
    dict(id=229, category="alias_expansion_lookup", lang="vi", tickers=["CTD"],
         expected_branch="dag", expected_outcome="answer",
         question="Tài sản dở dang dài hạn năm 2024 của CTD là bao nhiêu?"),
    dict(id=230, category="alias_expansion_lookup", lang="en", tickers=["HSG"],
         expected_branch="dag", expected_outcome="answer",
         question="What was HSG's long term financial investments in 2024?"),
    dict(id=231, category="alias_expansion_lookup", lang="vi", tickers=["BSR"],
         expected_branch="dag", expected_outcome="answer",
         question="Tài sản dài hạn khác năm 2024 của BSR là bao nhiêu?"),
    dict(id=232, category="alias_expansion_lookup", lang="en", tickers=["KTS"],
         expected_branch="dag", expected_outcome="answer",
         question="What was KTS's gross revenue in 2024?",
         note="Code income_statement/01 - EN-only alias by design, see "
              "module docstring's ambiguity-call note."),
    dict(id=233, category="alias_expansion_lookup", lang="vi", tickers=["CTD"],
         expected_branch="dag", expected_outcome="answer",
         question="Doanh thu hoạt động tài chính năm 2024 của CTD là bao nhiêu?"),
    dict(id=234, category="alias_expansion_lookup", lang="en", tickers=["HSG"],
         expected_branch="dag", expected_outcome="answer",
         question="What was HSG's financial expense in 2024?",
         note="Code income_statement/23 ('Chi phí tài chính') - same line "
              "sensitivity_what_if_interest_rate() shocks via direct SQL; "
              "now also plainly DAG-lookupable, independent of that WHATIF path."),
    dict(id=235, category="alias_expansion_lookup", lang="vi", tickers=["BSR"],
         expected_branch="dag", expected_outcome="answer",
         question="Chi phí bán hàng năm 2024 của BSR là bao nhiêu?"),
    dict(id=236, category="alias_expansion_lookup", lang="en", tickers=["KTS"],
         expected_branch="dag", expected_outcome="answer",
         question="What was KTS's general and administrative expenses in 2024?"),
    dict(id=237, category="alias_expansion_lookup", lang="vi", tickers=["CTD"],
         expected_branch="dag", expected_outcome="answer",
         question="Lợi nhuận thuần từ hoạt động kinh doanh năm 2024 của CTD là bao nhiêu?"),
    dict(id=238, category="alias_expansion_lookup", lang="en", tickers=["HSG"],
         expected_branch="dag", expected_outcome="answer",
         question="What was HSG's other income in 2024?"),
    dict(id=239, category="alias_expansion_lookup", lang="vi", tickers=["BSR"],
         expected_branch="dag", expected_outcome="answer",
         question="Chi phí khác năm 2024 của BSR là bao nhiêu?"),
    dict(id=240, category="alias_expansion_lookup", lang="en", tickers=["KTS"],
         expected_branch="dag", expected_outcome="answer",
         question="What was KTS's other profit in 2024?"),
    dict(id=241, category="alias_expansion_lookup", lang="vi", tickers=["CTD"],
         expected_branch="dag", expected_outcome="answer",
         question="Chi phí thuế TNDN hiện hành năm 2024 của CTD là bao nhiêu?"),
    dict(id=242, category="alias_expansion_lookup", lang="en", tickers=["HSG"],
         expected_branch="dag", expected_outcome="answer",
         question="What was HSG's deferred corporate income tax expense in 2024?"),
    dict(id=243, category="alias_expansion_lookup", lang="vi", tickers=["BSR"],
         expected_branch="dag", expected_outcome="answer",
         question="Lãi suy giảm trên cổ phiếu năm 2024 của BSR là bao nhiêu?"),
    dict(id=244, category="alias_expansion_lookup", lang="en", tickers=["KTS"],
         expected_branch="dag", expected_outcome="answer",
         question="What was KTS's net cash flow during the period in 2024?"),
    dict(id=245, category="alias_expansion_lookup", lang="vi", tickers=["CTD"],
         expected_branch="dag", expected_outcome="answer",
         question="Tiền đầu kỳ năm 2024 của CTD là bao nhiêu?",
         note="Short form deliberately avoids 'tương đương' - see "
              "src/structured/schema.py's diacritic-normalization note."),
    # ============================================================
    # Controls + depth variants (5 Qs)
    # ============================================================
    dict(id=246, category="alias_expansion_lookup", lang="en", tickers=["BSR"],
         expected_branch="dag", expected_outcome="answer",
         question="What was BSR's gross revenue in 2024?",
         note="Control: code '01' (gross, EN-only alias) - must resolve to "
              "'01', NOT '10' (net revenue), confirming the two stay distinct."),
    dict(id=247, category="alias_expansion_control", lang="vi", tickers=["KTS"],
         expected_branch="dag", expected_outcome="answer",
         question="Doanh thu năm 2024 của KTS là bao nhiêu?",
         note="Must still resolve to code '10' (net revenue) - regression "
              "control for the '01' vs '10' ambiguity call above."),
    dict(id=248, category="alias_expansion_lookup", lang="vi", tickers=["HSG"],
         expected_branch="dag", expected_outcome="answer",
         question="Chi phí bán hàng quý 2 năm 2024 của HSG là bao nhiêu?",
         note="Second instance of code 25, QUARTER period instead of "
              "full-year - exercises the quarter-period parser on a newly "
              "aliased code, not just the default year path."),
    dict(id=249, category="alias_expansion_lookup", lang="vi", tickers=["BSR"],
         expected_branch="dag", expected_outcome="answer",
         question="Chi phí quản lý năm 2024 của BSR là bao nhiêu?",
         note="Second instance of code 26, using the SHORTER registered "
              "alias 'chi phí quản lý' (without 'doanh nghiệp') - confirms "
              "both VI variants resolve to the same code."),
    dict(id=250, category="alias_expansion_lookup", lang="en", tickers=["CTD"],
         expected_branch="dag", expected_outcome="answer",
         question="What was CTD's other income in Q3 2024?",
         note="Second instance of code 31, EN + quarter period."),
]

_REFUSAL_PATTERNS = re.compile(
    r"không tìm thấy|không tính được|không ước lượng được|không phân tích được|"
    r"no information found|not found in|không xác định được|không đủ dữ liệu",
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
    ap.add_argument("--out", default="data/eval/final_benchmark_alias_en.jsonl")
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
