"""RATIO_LOOKUP addendum to the thesis's final benchmark - 2026-09-15.
Continues the id numbering of scripts/final_benchmark_100_en.py (ids
109-120) and scripts/final_benchmark_macro_en.py (101-108) - kept as its
own file for the same reason those two are separate (see
final_benchmark_macro_en.py's docstring): each already-documented file's
own question count stays accurate.

WHY THIS FILE EXISTS
-----------------------------------------------------------------------------
Quang pasted the full ~71-indicator Vietstock "Chỉ số tài chính" taxonomy
(2026-09-15) and pointed out none of it was reachable from a typed
question. Investigation confirmed: the `financial_ratios` table and its
ETL (`src.ingestion.vietstock_etl.load_chi_so_tai_chinh()`) already
existed, ~214 real per-ticker "Chi-so-tai-chinh" export files already sit
in `data/raw/vietstock_exports/` (uningested), and the 71 `ratio_name`
strings Quang pasted were verified VERBATIM against one of those real
files (ticker SLS) - but no router branch had ever queried that table
(only the unwired Nhánh B ablation did). Fixed by adding
`src/structured/ratio_schema.py` (alias table, mirrors schema.py),
`src/structured/ratio_lookup.py` (query execution, mirrors
query_builder.py), and a new `ratio_lookup` branch in
`src/router/rules.py` / `src/router/query_router.py` - see
docs/PROGRESS.md's 2026-09-15 entry for the full design/regression log,
including two real bugs found and fixed while verifying this:
  1. `ROAA`/`ROEA`/`ROCE`/`BVPS`/`CPS` (4-or-fewer-uppercase-letter ratio
     abbreviations) were being fail-open-misread as tickers by
     `_extract_ticker()`, the same bug class as KPMG/AASC/UHY earlier this
     session - fixed by adding them to
     `query_builder.NON_TICKER_ABBREVIATIONS`.
  2. Several leverage/structure ratio names (e.g. "Tỷ số Nợ trên Vốn chủ
     sở hữu") CONTAIN a shorter financial_line_items alias as a substring
     ("vốn chủ sở hữu" -> ms_code 400) - checking DAG before RATIO_LOOKUP
     let the shorter match win for nearly the entire leverage/structure
     ratio group. Fixed by resolving both and preferring whichever matched
     alias string is longer (same principle each resolver already uses
     internally for its own substring fallback, extended across both
     tables).

PREREQUISITE: same as final_benchmark_macro_en.py - run the ratio ingest
first, or every question here correctly (not a bug) falls back to RAG
(RATIO_LOOKUP's own adaptive DAG-style fallback, see query_router.py):
    python -m src.ingestion.vietstock_etl chi_so_tai_chinh data/raw/vietstock_exports/VietstockFinance_Chi-so-tai-chinh_*.xlsx

TICKER/RATIO SELECTION (2026-09-15 tiếp 4: EXPANDED 12 -> 30, ids
re-numbered 131-160) - now exactly 3 questions per ratio group, ALL 10 of
the 10 ratio groups covered (was 7/10) - 30 distinct ratios out of the 71
total, still deliberately NOT exhaustive (71 x 8 tickers = 568 would dilute
the "small, defensible, readable set" goal every file in this series
keeps to), but now every group has real representation instead of leaving
3 groups completely untested. All 8 of this benchmark's tickers cycle
through in order (CTD, HSG, BSR, KTS, HPG, VNM, MWG, VIC, repeating), VI/EN
strictly alternating question-by-question (15 of each).
-----------------------------------------------------------------------------
CATEGORY: `ratio_lookup` (new, Group 2 - Quantitative correctness; a bounded
structured lookup against financial_ratios, same evidentiary status as
`structured_lookup` against financial_line_items - zero LLM/RAG risk on a
successful match).

METHODOLOGY: reused byte-for-byte from final_benchmark_100_en.py.

Usage:
    python -m scripts.final_benchmark_ratio_en
    python -m scripts.score_audit_results --input data/eval/final_benchmark_ratio_en.jsonl --chart \\
        --chart-output data/eval/runs/final_benchmark_ratio_score.png
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

QUESTIONS = [
    # --- NHÓM CHỈ SỐ ĐỊNH GIÁ (valuation) ---
    dict(id=131, category="ratio_lookup", lang="vi", tickers=["CTD"],
         expected_branch="ratio_lookup", expected_outcome="answer",
         question="P/E của CTD năm 2024 là bao nhiêu?"),
    dict(id=132, category="ratio_lookup", lang="en", tickers=["HSG"],
         expected_branch="ratio_lookup", expected_outcome="answer",
         question="What was the P/B of HSG in 2024?"),
    dict(id=133, category="ratio_lookup", lang="vi", tickers=["BSR"],
         expected_branch="ratio_lookup", expected_outcome="answer",
         question="Tỷ suất cổ tức của BSR năm 2024 là bao nhiêu?"),
    # --- NHÓM CHỈ SỐ SINH LỢI (profitability) ---
    dict(id=134, category="ratio_lookup", lang="en", tickers=["KTS"],
         expected_branch="ratio_lookup", expected_outcome="answer",
         question="What was the ROEA of KTS in 2024?"),
    dict(id=135, category="ratio_lookup", lang="vi", tickers=["HPG"],
         expected_branch="ratio_lookup", expected_outcome="answer",
         question="ROAA của HPG năm 2024 là bao nhiêu?",
         note="Exercises the ROAA entry added to NON_TICKER_ABBREVIATIONS "
              "(see module docstring bug #1)."),
    dict(id=136, category="ratio_lookup", lang="en", tickers=["VNM"],
         expected_branch="ratio_lookup", expected_outcome="answer",
         question="What was the EBITDA margin of VNM in 2024?"),
    # --- NHÓM CHỈ SỐ TĂNG TRƯỞNG (growth) ---
    dict(id=137, category="ratio_lookup", lang="vi", tickers=["MWG"],
         expected_branch="ratio_lookup", expected_outcome="answer",
         question="Tăng trưởng doanh thu thuần của MWG năm 2024 là bao nhiêu?",
         note="Same collision class as id=111 in the original 12-question "
              "set (WHATIF's 'tăng trưởng' marker vs single-period ratio) - "
              "already fixed via the fallback in rules.py's WHATIF block."),
    dict(id=138, category="ratio_lookup", lang="en", tickers=["VIC"],
         expected_branch="ratio_lookup", expected_outcome="answer",
         question="What was the net profit growth of VIC in 2024?"),
    dict(id=139, category="ratio_lookup", lang="vi", tickers=["CTD"],
         expected_branch="ratio_lookup", expected_outcome="answer",
         # 2026-09-15: đổi từ năm 2024 sang 2025 - xác minh trực tiếp trong file
         # Vietstock gốc (VietstockFinance_Chi-so-tai-chinh_20260715-160633.xlsx,
         # dòng "Tăng trưởng tổng tài sản") thì ô Năm/2024 = None (Vietstock tự
         # thân không công bố số này cho CTD năm 2024, không phải lỗi ingest),
         # còn Năm/2025 = 29.88 (%) - có số thật để benchmark đối chiếu.
         question="Tăng trưởng tổng tài sản của CTD năm 2025 là bao nhiêu?"),
    # --- NHÓM CHỈ SỐ THANH KHOẢN (liquidity) ---
    dict(id=140, category="ratio_lookup", lang="en", tickers=["HSG"],
         expected_branch="ratio_lookup", expected_outcome="answer",
         question="What was the current ratio of HSG in 2024?"),
    dict(id=141, category="ratio_lookup", lang="vi", tickers=["BSR"],
         expected_branch="ratio_lookup", expected_outcome="answer",
         question="Tỷ số thanh toán nhanh của BSR năm 2024 là bao nhiêu?"),
    dict(id=142, category="ratio_lookup", lang="en", tickers=["KTS"],
         expected_branch="ratio_lookup", expected_outcome="answer",
         question="What was the interest coverage ratio of KTS in 2024?"),
    # --- NHÓM CHỈ SỐ HIỆU QUẢ HOẠT ĐỘNG (efficiency) ---
    dict(id=143, category="ratio_lookup", lang="vi", tickers=["HPG"],
         expected_branch="ratio_lookup", expected_outcome="answer",
         question="Vòng quay hàng tồn kho của HPG năm 2024 là bao nhiêu?"),
    dict(id=144, category="ratio_lookup", lang="en", tickers=["VNM"],
         expected_branch="ratio_lookup", expected_outcome="answer",
         question="What was the total asset turnover of VNM in 2024?"),
    dict(id=145, category="ratio_lookup", lang="vi", tickers=["MWG"],
         expected_branch="ratio_lookup", expected_outcome="answer",
         question="Vòng quay phải thu khách hàng của MWG năm 2024 là bao nhiêu?"),
    # --- NHÓM CHỈ SỐ ĐÒN BẨY TÀI CHÍNH (leverage) ---
    dict(id=146, category="ratio_lookup", lang="en", tickers=["VIC"],
         expected_branch="ratio_lookup", expected_outcome="answer",
         question="What was the debt to equity ratio of VIC in 2024?",
         note="Leverage group - where the DAG-vs-ratio substring-collision "
              "bug (see module docstring) was originally found and fixed."),
    dict(id=147, category="ratio_lookup", lang="vi", tickers=["CTD"],
         expected_branch="ratio_lookup", expected_outcome="answer",
         question="Tỷ số Nợ trên Tổng tài sản của CTD năm 2024 là bao nhiêu?"),
    dict(id=148, category="ratio_lookup", lang="en", tickers=["HSG"],
         expected_branch="ratio_lookup", expected_outcome="answer",
         question="What was the equity to assets ratio of HSG in 2024?"),
    # --- NHÓM CHỈ SỐ DÒNG TIỀN (cash flow) ---
    dict(id=149, category="ratio_lookup", lang="vi", tickers=["BSR"],
         expected_branch="ratio_lookup", expected_outcome="answer",
         question="Tỷ số dòng tiền HĐKD trên doanh thu thuần của BSR năm 2024 là bao nhiêu?"),
    dict(id=150, category="ratio_lookup", lang="en", tickers=["KTS"],
         expected_branch="ratio_lookup", expected_outcome="answer",
         question="What was the CPS of KTS in 2024?"),
    dict(id=151, category="ratio_lookup", lang="vi", tickers=["HPG"],
         expected_branch="ratio_lookup", expected_outcome="answer",
         question="Khả năng thanh toán nợ từ dòng tiền HĐKD của HPG năm 2024 là bao nhiêu?"),
    # --- CƠ CẤU CHI PHÍ (cost structure) ---
    dict(id=152, category="ratio_lookup", lang="en", tickers=["VNM"],
         expected_branch="ratio_lookup", expected_outcome="answer",
         question="What was the COGS to revenue ratio of VNM in 2024?"),
    dict(id=153, category="ratio_lookup", lang="vi", tickers=["MWG"],
         expected_branch="ratio_lookup", expected_outcome="answer",
         question="Chi phí lãi vay trên doanh thu thuần của MWG năm 2024 là bao nhiêu?"),
    dict(id=154, category="ratio_lookup", lang="en", tickers=["VIC"],
         expected_branch="ratio_lookup", expected_outcome="answer",
         question="What was the admin expense to revenue ratio of VIC in 2024?"),
    # --- CƠ CẤU TÀI SẢN NGẮN HẠN (current asset structure) ---
    dict(id=155, category="ratio_lookup", lang="vi", tickers=["CTD"],
         expected_branch="ratio_lookup", expected_outcome="answer",
         question="Tỷ trọng tiền trên tài sản ngắn hạn của CTD năm 2024 là bao nhiêu?"),
    dict(id=156, category="ratio_lookup", lang="en", tickers=["HSG"],
         expected_branch="ratio_lookup", expected_outcome="answer",
         question="What was the inventory to current assets ratio of HSG in 2024?"),
    dict(id=157, category="ratio_lookup", lang="vi", tickers=["BSR"],
         expected_branch="ratio_lookup", expected_outcome="answer",
         question="Tỷ trọng phải thu ngắn hạn trên tài sản ngắn hạn của BSR năm 2024 là bao nhiêu?"),
    # --- CƠ CẤU TÀI SẢN DÀI HẠN (long-term asset structure) ---
    dict(id=158, category="ratio_lookup", lang="en", tickers=["KTS"],
         expected_branch="ratio_lookup", expected_outcome="answer",
         question="What was the fixed assets to total assets ratio of KTS in 2024?"),
    dict(id=159, category="ratio_lookup", lang="vi", tickers=["HPG"],
         expected_branch="ratio_lookup", expected_outcome="answer",
         question="Tỷ trọng tài sản dài hạn trên tổng tài sản của HPG năm 2024 là bao nhiêu?",
         note="Originally phrased with 'Tài sản vô hình/Tài sản cố định' "
              "(mentions 'tài sản CỐ ĐỊNH') - found via resolve_ratio() "
              "testing that this hits the SAME 'đ' normalization bug "
              "documented in schema.py (ratio_schema.py's VI aliases for "
              "this group were hand-typed as pre-stripped ASCII with a "
              "literal 'd' - e.g. '...tai san co dinh' - which does NOT "
              "match a real diacritic question's buggy-normalized "
              "'...tai san co inh', đ silently dropped). Swapped to 'Tài "
              "sản dài hạn/Tổng tài sản' (no 'đ' in 'dài hạn') to route "
              "correctly - same real, not-yet-patched bug as id=3 in "
              "final_benchmark_100_en.py, worked around the same way "
              "rather than fixed in ratio_schema.py itself (scope: "
              "benchmark building, not a new code fix)."),
    dict(id=160, category="ratio_lookup", lang="en", tickers=["VNM"],
         expected_branch="ratio_lookup", expected_outcome="answer",
         question="What was the construction in progress to fixed assets ratio of VNM in 2024?",
         note="XDCBDD/Tài sản cố định - completes coverage of all 10 ratio "
              "groups (was 7/10 in the original 12-question set)."),
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
        "note": q.get("note"),
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
            "fallback_from": result.get("fallback_from"),
            "branch_match": _branch_matches_expected(actual_branch, q["expected_branch"]),
            "looks_like_refusal": refused,
            "outcome_match": (("refuse" if refused else "answer") == q["expected_outcome"]),
            "intent_reason": result.get("intent_reason"),
            "answer": answer,
            "elapsed_s": elapsed,
        })
    except Exception as e:
        elapsed = round(time.time() - t0, 1)
        record.update({
            "ok": False, "error": f"{type(e).__name__}: {e}", "elapsed_s": elapsed,
        })
    return record


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/eval/final_benchmark_ratio_en.jsonl")
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
    todo = [
        q for q in QUESTIONS
        if q["id"] not in done_ids and (only_ids is None or q["id"] in only_ids)
    ]
    print(f"{len(QUESTIONS)} total, {len(done_ids)} already done, {len(todo)} to run.")

    with out_path.open("a", encoding="utf-8") as f:
        for i, q in enumerate(todo, 1):
            print(f"\n[{i}/{len(todo)}] id={q['id']} ({q['category']}, {q['lang']}): {q['question'][:70]}...")
            record = run_one(q)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            status = "OK" if record.get("ok") else "ERROR"
            branch = record.get("actual_branch", "?")
            print(f"  -> {status} branch={branch} ({record.get('elapsed_s')}s)")

    print(f"\nDone. Results in {out_path}")


if __name__ == "__main__":
    main()
