"""Score data/eval/answer_quality_audit.jsonl against explicit, documented
ground truth, and chart the result - 2026-09-14, requested by Quang:

    "bộ benchmark cần có số liệu biểu đồ test dựa vào cái gì câu trả lời
    dự kiến cần phải rõ ràng để đối chiếu với mô hình và benchmark dựa
    trên dữ kiện nào liệt kê rõ ràng ra"

Before this script, comparing "did the system do the right thing" against
"what it was supposed to do" meant a human reading all 60 rows by eye -
audit_answer_quality.py already recorded per-question ground truth
(`expected_branch`, and `numeric_verification` for DAG/whatif items with a
real number to check), but nothing rolled it up into a pass/fail rate or a
chart. This script does exactly that roll-up - it does NOT invent any new
ground truth; every check below cites exactly which existing, already-
documented field it reads.

THREE INDEPENDENT CHECKS, EACH WITH ITS OWN GROUND-TRUTH SOURCE (this is the
"dựa trên dữ kiện nào" the request asked for - spelled out explicitly, not
left implicit):

1. ROUTING CHECK ("branch_match", added 2026-09-14 to audit_answer_quality.py's
   run_one()) - ground truth = `expected_branch`, a field already present on
   EVERY one of the 60 QUESTIONS entries in audit_answer_quality.py, written
   BY QUANG'S REQUEST when each question was designed (see that file's own
   header comment: "expected_branch_hint (best guess for what SHOULD
   happen)"). Mechanical: did src.router.query_router.route() pick the
   branch the question was built to exercise? Says NOTHING about whether the
   answer text itself is good - a right-branch, wrong-answer case still
   scores as a routing pass.

2. NUMERIC CHECK ("numeric_verification.label") - ground truth = the actual
   retrieved source chunk's own number, NOT a hand-typed expected value (see
   src/evaluation/numeric_claims.py's verify_numeric_claims(), wired into
   rag_pipeline.py well before this script existed) - it re-parses every
   number the LLM's answer states and checks it against the closest number
   literally present in the chunks the LLM was given, flagging "verified"/
   "mismatch"/"unverifiable". Only present for questions whose branch
   produces a checkable number (dag/rag-with-a-figure/whatif) - qualitative
   questions (legal risks, board members, ABSA, ...) have no single number
   to check and are correctly skipped here, not force-scored.

3. REFUSAL-OUTCOME CHECK ("outcome_match") - ground truth = `expected_outcome`
   when the row carries one (every 2026-09-14+ file: final_benchmark_100_en/
   macro/ratio/rate/trade/alias - a per-QUESTION label written at design
   time), else CATEGORY_EXPECTED_OUTCOME below (a per-CATEGORY label, for the
   older 60-row audit file that predates per-question expected_outcome).
   Every category/row with neither is reported as "needs_human_review",
   the honest answer, not skipped silently - matches the project's existing
   3-label rubric (docs/answer_quality_audit_2026-09-12.md section 4:
   "Trả lời được / Không trả lời được / Trả lời mà như không trả lời" -
   that middle judgment call is NOT automatable and this script does not
   pretend otherwise).

   2026-09-15 CHANGE (real bug found and fixed, see docs/PROGRESS.md same-
   date entry for the full diagnosis): this check used to trust each row's
   own precomputed `outcome_match`/`looks_like_refusal` fields when present.
   Reading real answers by hand (data/eval/final_benchmark_macro_en.jsonl
   id=127, final_benchmark_100_en.jsonl ids 75/81/87...) showed the system
   was refusing CORRECTLY but each benchmark file's own keyword-based
   `_looks_like_refusal()` regex did not recognise the actual refusal
   phrasing used by the `out_of_scope`/`whatif_unsupported` branches (e.g.
   "Chưa thể trả lời tự động...", "Đây là câu hỏi dự đoán giá..." - neither
   contains any of "không tìm thấy"/"không tính được"/etc.), so a correct
   refusal was scored as a wrong outcome. This script now recomputes the
   refusal call itself (`_refused()` below) instead of trusting the row:
   a `actual_branch` that is DEFINITIONALLY a decline
   (`out_of_scope`/`whatif_unsupported`/`absa_unsupported`/
   `causal_unsupported` - the router only ever reaches these branches to
   decline, never to answer) counts as a refusal regardless of phrasing;
   every other branch falls back to the keyword patterns, since those
   branches (rag/whatif/dag/...) can legitimately either answer or refuse
   depending on data availability, which text alone must distinguish.
   Separately, `data/eval/final_benchmark_macro_en.jsonl` had 24 rows
   (ids 101,102,105-126) with a genuinely wrong `expected_outcome="answer"`
   for CORE_16-tier tickers (HPG/VNM/MSN/KDC/SAB/DPM/DCM/GAS/MWG/PNJ/DGW/FRT
   - zero `financial_line_items` rows per `scripts/seed_companies.py`'s
   CORE_16 list, so the WHATIF branch cannot compute a real shock for them)
   - only HSG (ids 103/104) among the file's 13 mapped tickers is PANEL
   tier and should truly expect "answer". Corrected directly in both
   `scripts/final_benchmark_macro_en.py` (source) and the already-run
   `.jsonl` (so the existing live-run data doesn't need to be re-run).

Usage:
    python -m scripts.score_audit_results
    python -m scripts.score_audit_results --input data/eval/answer_quality_audit.jsonl --chart
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

# Branches the router only ever reaches to DECLINE an answer, never to give
# one - see src/router/rules.py. Any row landing on one of these is a
# refusal by construction, regardless of the exact wording used, so this is
# checked BEFORE falling back to keyword matching (see docstring's
# "2026-09-15 CHANGE" note above for why keyword-only matching missed real,
# correct refusals).
_DEFINITELY_REFUSAL_BRANCHES = {
    "out_of_scope", "whatif_unsupported", "absa_unsupported", "causal_unsupported",
}

# Union of every _REFUSAL_PATTERNS variant across the 6 final_benchmark_*.py
# files (they drifted slightly - trade_en/rate_en added a couple of extra
# phrases the others don't have) plus the 2 new phrasings found by hand this
# session ("chưa thể trả lời", "cannot ... auto-answer") - kept as ONE
# superset here so this script's fallback check is at least as generous as
# any individual file's own, not narrower.
_KEYWORD_REFUSAL_PATTERNS = re.compile(
    r"không tìm thấy|không tính được|không ước lượng được|không phân tích được|"
    r"không xác định được|không đủ dữ liệu|không xếp hạng được|"
    r"chưa có ánh xạ|chưa được xác định|chưa thể trả lời|"
    r"no information found|not found in",
    re.IGNORECASE,
)


def _refused(record: dict) -> bool:
    if record.get("actual_branch") in _DEFINITELY_REFUSAL_BRANCHES:
        return True
    return bool(_KEYWORD_REFUSAL_PATTERNS.search(record.get("answer") or ""))

DEFAULT_INPUT = "data/eval/answer_quality_audit.jsonl"

# Per-CATEGORY expected outcome ("answer" | "refuse") - see module docstring
# check #3. Each entry's comment quotes the EXISTING note already written in
# audit_answer_quality.py's QUESTIONS list for that category, so this
# mapping is a roll-up of intent already on record, not a new invention.
CATEGORY_EXPECTED_OUTCOME = {
    # id=23/24 notes: "whatif_core16 (đối chứng)" - CORE_16 has zero
    # financial_line_items (confirmed 2026-09-12 by scanning all 213
    # Vietstock exports), so sensitivity_what_if() MUST report missing
    # revenue/cogs - a correct refusal, not a bug.
    "whatif_core16": "refuse",
    # id=50/51/58 notes literally say: "ky vong tu choi chu khong doan" (50),
    # "Hoan toan ngoai pham vi" (51), "ky vong he thong TU CHOI dua loi
    # khuyen" (58).
    "out_of_scope": "refuse",
    # id=52/59 notes: "ky vong tu choi ro rang, KHONG bia so" / "tu choi dung
    # vi VNMX khong ton tai".
    "unknown_ticker": "refuse",
    # id=13-15 notes: "DAG khong co du lieu HPG/GAS/DGW" -> falls back to
    # RAG, which SHOULD still find real disclosure text for these
    # well-covered CORE_16 tickers (RAG has PDFs for all 16 - see
    # docs/answer_quality_audit_2026-09-12.md section 1) - unlike
    # out_of_scope/unknown_ticker this is a data-presence fallback, not a
    # designed dead end, so the expected outcome is still a real answer.
    "dag_core16_fallback": "answer",
    # Every rag_*_panel/dag_*/trend_panel(non-core16)/whatif_panel/causal/
    # compare/absa/auditor_lookup/qualitative question was built around a
    # ticker+figure/document CONFIRMED present (see each category's own
    # header comment in audit_answer_quality.py, e.g. "PANEL co DAG - ky
    # vong route=dag" or "MSN la ticker CORE_16... da xac nhan co du lieu")
    # - these are "should answer" by construction. Categories deliberately
    # left OUT of this dict:
    #   - causal_negative_control: answer-shaped but its PASS condition is
    #     "shows a weak/near-zero effect", which refusal-keyword matching
    #     cannot judge.
    #   - trend_core16 (ids 18/19): docs/answer_quality_audit_2026-09-12.md's
    #     own table labels these "DAG_TREND (đối chứng)" - a NEGATIVE
    #     control like whatif_core16 (CORE_16 has zero financial_line_items,
    #     so dag_trend WILL fail) - but unlike whatif_core16, whether the
    #     RAG fallback then finds enough qualitative text to give a real
    #     answer or has to refuse is genuinely undetermined (CORE_16 DOES
    #     have RAG-indexed PDFs, just not for this specific "biến động qua
    #     các năm" framing) - asserting either outcome here would be a
    #     guess, so left as needs_human_review instead.
    #   - compound_p3_bug: expected_branch is literally "?" for 2 of its 3
    #     questions (see QUESTIONS - the known P3 multi-part-question router
    #     bug means there's no single well-defined expected branch yet).
    "rag_core16": "answer", "rag_panel": "answer", "dag_panel": "answer",
    "trend_panel": "answer", "trend_panel_quarterly": "answer",
    "whatif_panel": "answer", "causal": "answer", "compare": "answer",
    "compare_gold": "answer", "absa": "answer", "auditor_lookup": "answer",
    "qualitative": "answer",
}


def load_records(path: Path) -> list:
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def score(records: list) -> dict:
    per_category = defaultdict(lambda: {
        "n": 0, "branch_match": 0, "branch_checked": 0,
        "outcome_match": 0, "outcome_checked": 0, "needs_human_review": 0,
        "numeric_verified": 0, "numeric_checked": 0,
        "errors": 0,
    })
    rows = []
    for r in records:
        cat = r["category"]
        bucket = per_category[cat]
        bucket["n"] += 1

        if not r.get("ok"):
            bucket["errors"] += 1
            rows.append({**r, "branch_match": None, "outcome_match": None, "numeric_label": None})
            continue

        # 1. Routing check - prefer the field written directly by run_one()
        # (2026-09-14+ runs); recompute for older rows that predate it so
        # this script also scores the existing 60-row file untouched.
        if "branch_match" in r:
            branch_match = r["branch_match"]
        else:
            allowed = {b.strip() for b in r.get("expected_branch", "").split(" or ")}
            branch_match = r.get("actual_branch") in allowed
        bucket["branch_checked"] += 1
        bucket["branch_match"] += int(bool(branch_match))

        # 2. Numeric check (only when present - see docstring #2)
        num = r.get("numeric_verification")
        numeric_label = num.get("label") if isinstance(num, dict) else None
        if numeric_label:
            bucket["numeric_checked"] += 1
            bucket["numeric_verified"] += int(numeric_label == "verified")

        # 3. Outcome check - ALWAYS recomputed here via _refused() (2026-09-15,
        # see docstring's "2026-09-15 CHANGE" note): does NOT trust a row's
        # own precomputed `outcome_match`/`looks_like_refusal` fields, because
        # those were computed by each benchmark file's own narrower keyword
        # regex at run time and were shown (by reading real answers) to
        # mis-score correct out_of_scope/whatif_unsupported refusals as
        # wrong. Ground truth is `expected_outcome` on the row when present
        # (2026-09-14+ files - a real per-question label set at design time,
        # corrected for the 24 mislabelled macro rows on 2026-09-15 - see
        # note above), else CATEGORY_EXPECTED_OUTCOME below for the older
        # 60-row audit file.
        expected_outcome = r.get("expected_outcome") or CATEGORY_EXPECTED_OUTCOME.get(cat)
        outcome_match = None
        if expected_outcome is not None:
            actual_outcome = "refuse" if _refused(r) else "answer"
            outcome_match = (actual_outcome == expected_outcome)
            bucket["outcome_checked"] += 1
            bucket["outcome_match"] += int(outcome_match)
        else:
            bucket["needs_human_review"] += 1

        rows.append({**r, "branch_match": branch_match, "outcome_match": outcome_match,
                     "numeric_label": numeric_label})

    return {"per_category": dict(per_category), "rows": rows}


def print_summary(scored: dict):
    per_category = scored["per_category"]
    print("=" * 100)
    print(f"{'category':<26}{'n':<5}{'branch_match':<16}{'outcome_match':<16}{'numeric_verified':<18}{'errors':<8}")
    print("-" * 100)
    total = {"n": 0, "branch_ok": 0, "branch_n": 0, "outcome_ok": 0, "outcome_n": 0,
             "num_ok": 0, "num_n": 0, "errors": 0}
    for cat in sorted(per_category):
        b = per_category[cat]
        branch_str = f"{b['branch_match']}/{b['branch_checked']}" if b["branch_checked"] else "n/a"
        outcome_str = (f"{b['outcome_match']}/{b['outcome_checked']}"
                       if b["outcome_checked"] else "needs review")
        numeric_str = f"{b['numeric_verified']}/{b['numeric_checked']}" if b["numeric_checked"] else "n/a"
        print(f"{cat:<26}{b['n']:<5}{branch_str:<16}{outcome_str:<16}{numeric_str:<18}{b['errors']:<8}")
        total["n"] += b["n"]
        total["branch_ok"] += b["branch_match"]; total["branch_n"] += b["branch_checked"]
        total["outcome_ok"] += b["outcome_match"]; total["outcome_n"] += b["outcome_checked"]
        total["num_ok"] += b["numeric_verified"]; total["num_n"] += b["numeric_checked"]
        total["errors"] += b["errors"]
    branch_total_str = f"{total['branch_ok']}/{total['branch_n']}"
    outcome_total_str = f"{total['outcome_ok']}/{total['outcome_n']}"
    numeric_total_str = f"{total['num_ok']}/{total['num_n']}"
    print("-" * 100)
    print(f"{'TOTAL':<26}{total['n']:<5}{branch_total_str:<16}{outcome_total_str:<16}"
          f"{numeric_total_str:<18}{total['errors']:<8}")
    n_review = sum(b["needs_human_review"] for b in per_category.values())
    print(f"\n{n_review} question(s) have NO automated outcome check (category not in "
          f"CATEGORY_EXPECTED_OUTCOME) - these still need a human to read the answer text "
          f"and assign one of the 3 rubric labels per docs/answer_quality_audit_2026-09-12.md.")
    print("\nMismatches (branch or outcome):")
    for row in scored["rows"]:
        if row.get("branch_match") is False or row.get("outcome_match") is False:
            print(f"  id={row['id']:<3} cat={row['category']:<22} branch_match={row.get('branch_match')} "
                  f"outcome_match={row.get('outcome_match')} actual_branch={row.get('actual_branch')} "
                  f"expected_branch={row.get('expected_branch')}")


def maybe_render_chart(scored: dict, out_path: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[chart] matplotlib not installed - skipping PNG (pip install matplotlib --break-system-packages)")
        return

    per_category = scored["per_category"]
    cats = sorted(per_category)
    branch_rates = []
    outcome_rates = []
    for cat in cats:
        b = per_category[cat]
        branch_rates.append(100.0 * b["branch_match"] / b["branch_checked"] if b["branch_checked"] else 0)
        outcome_rates.append(100.0 * b["outcome_match"] / b["outcome_checked"] if b["outcome_checked"] else None)

    fig, ax = plt.subplots(figsize=(12, 6))
    x = range(len(cats))
    width = 0.35
    ax.bar([i - width / 2 for i in x], branch_rates, width, label="Routing match (%)", color="#2c7fb8")
    outcome_x = [i + width / 2 for i, r in zip(x, outcome_rates) if r is not None]
    outcome_y = [r for r in outcome_rates if r is not None]
    ax.bar(outcome_x, outcome_y, width, label="Refusal-outcome match (%)", color="#31a354")
    ax.set_xticks(list(x))
    ax.set_xticklabels(cats, rotation=45, ha="right")
    ax.set_ylabel("Match rate (%)")
    ax.set_ylim(0, 110)
    ax.set_title("Answer-quality audit: routing + refusal-outcome match rate by category\n"
                  "(ground truth = expected_branch + CATEGORY_EXPECTED_OUTCOME, see script docstring)")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"[chart] saved {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default=DEFAULT_INPUT)
    ap.add_argument("--chart", action="store_true")
    ap.add_argument("--chart-output", default="data/eval/runs/audit_score_by_category.png")
    args = ap.parse_args()

    records = load_records(Path(args.input))
    print(f"Loaded {len(records)} records from {args.input}\n")
    scored = score(records)
    print_summary(scored)
    if args.chart:
        maybe_render_chart(scored, Path(args.chart_output))


if __name__ == "__main__":
    main()
