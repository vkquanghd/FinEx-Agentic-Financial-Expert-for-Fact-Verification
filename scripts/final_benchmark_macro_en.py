"""Macro-enriched WHATIF addendum to the thesis's final benchmark -
2026-09-15. Continues the id numbering of scripts/final_benchmark_100_en.py
(ids 101-108) - a separate file rather than appended into that one so the
already-documented "100 questions, 13 categories" claim in that file's own
docstring and in docs/PROGRESS.md stays accurate. Run/report alongside it.

WHY THIS FILE EXISTS
-----------------------------------------------------------------------------
Quang uploaded a 47-file Vietstock "Dữ liệu vĩ mô" (national macro data)
export (2026-09-15) and asked (1) whether this data already existed in the
project, and (2) that it actually be used to enrich the benchmark, not just
sit in a folder.

Investigation (2026-09-15) found two things:
  1. `data/raw/vietstock_exports/macro/` existed but was EMPTY (only
     `.gitkeep`) - the ETL for this exact file shape
     (`src/ingestion/macro_etl.py`) was already fully built, and its
     `TITLE_TO_CATEGORY` dict matches all 47 uploaded file titles 1:1
     (GDP, CPI, PPI, IPI, XPI, MPI, labor, population, unemployment,
     retail, PMI, IIP (incl. by-industry), trade, government debt,
     budget, credit, deposits, interest rates, FX - see that file for the
     full list), but nothing had ever been copied in or run.
  2. A SECOND, more interesting gap: `src/structured/calculator.py`'s
     `sensitivity_what_if_macro()` - a ceteris-paribus WHATIF scenario that
     uses the ticker's own REAL industry production/price index (from
     exactly this macro dataset's IIP_BY_INDUSTRY/RETAIL categories, via
     `src/causal/macro_lookup.TICKER_INDUSTRY_MAP`) as the COGS shock,
     instead of a user-typed hypothetical percentage - was already fully
     implemented and already wired into `app_streamlit.py`'s manual
     ticker/year dropdown, but had NO natural-language trigger anywhere in
     `src/router/rules.py`. A typed chat question could never reach it -
     only the Streamlit demo's manual picker could. This is the same bug
     class as the `dag_multi`/`out_of_scope` gaps found and fixed
     2026-09-14 (a real capability existed but the router never routed a
     typed question to it).

FIX (2026-09-15, both changes small, mirror the EXISTING gold/oil
sensitivity pattern exactly - see their `_GOLD_WORDS`/`_OIL_WORDS`/
`gold_sensitivity`/`oil_sensitivity` code for the template this follows)
-----------------------------------------------------------------------------
  - `src/router/rules.py`: added `_INDUSTRY_WORDS` marker set ("chỉ số
    ngành", "chỉ số sản xuất công nghiệp", "industry index", ...) and a new
    `classify_intent()` branch, checked in the same WHATIF block as
    gold/oil, right after the oil check. Returns `whatif_unsupported` with
    an explicit reason for any ticker NOT in `TICKER_INDUSTRY_MAP` (same
    "no invented ground truth" discipline as gold/oil), `whatif` +
    `whatif_kind="macro_sensitivity"` otherwise.
  - `src/router/query_router.py`: imported `sensitivity_what_if_macro`,
    added its dispatch branch in `_dispatch_whatif()` and its formatting
    branch in `_format_whatif()` (mirrors `gold_sensitivity`'s
    `gold_source` handling, using the function's own `macro_source` key).

Both changes verified via `classify_intent()` directly (sandboxed, no DB -
same technique used throughout this session) before being used here - see
scripts/verify_100_regression.py's sibling check output in
docs/PROGRESS.md's 2026-09-15 entry.

TICKER SELECTION (4 of this benchmark's existing 8 tickers - only the ones
with a REAL entry in TICKER_INDUSTRY_MAP, no new tickers introduced)
-----------------------------------------------------------------------------
  - HPG, HSG -> IIP_BY_INDUSTRY "Sản xuất kim loại" (steel) - both already
    PANEL-tier tickers in this benchmark.
  - VNM -> IIP_BY_INDUSTRY "Sản xuất, chế biến thực phẩm" (dairy/food
    processing) - already a CORE_16 ticker in this benchmark.
  - MWG -> RETAIL "Bán lẻ hàng hóa" - already a CORE_16 ticker in this
    benchmark.
Deliberately EXCLUDED (no entry in TICKER_INDUSTRY_MAP, would correctly
return whatif_unsupported - not tested here since that refusal path is
already covered by the general "no ticker/no mapping" cases elsewhere):
CTD, BSR, KTS, VIC (this benchmark's other 4 tickers).

DATA GRANULARITY CHECK (2026-09-15, confirmed by reading the actual
uploaded files, not assumed)
-----------------------------------------------------------------------------
`calculator.sensitivity_what_if_macro()` looks up `period_type="YEAR"` for
IIP_BY_INDUSTRY and `period_type="MONTH"` for every other category
(including RETAIL). Opened the two relevant files directly to confirm this
matches the real export shape: "CHỈ SỐ SẢN XUẤT CÔNG NGHIỆP PHÂN THEO NGÀNH
CÔNG NGHIỆP" (IIP_BY_INDUSTRY) has annual columns ('2021','2022','2023',
'2024'); "BÁN LẺ" (RETAIL) has monthly columns ('Tháng 7/2021', 'Tháng
8/2021', ...) - both match the code's assumption, so no mismatch to fix.

CATEGORY: `industry_index_whatif` (new, Group 2 - Quantitative correctness;
a ceteris-paribus scenario grounded in real macro data rather than a
user-typed %, same evidentiary status as the existing gold/oil sensitivity
questions).

2026-09-15 (tiếp 4) - EXPANDED 8 -> 30, all 13 real `TICKER_INDUSTRY_MAP`
tickers covered (not just the 4 originally in this file's ticker pool):
HPG/HSG (steel), VNM/MSN/KDC (food processing), SAB (beverages), DPM/DCM
(chemicals), GAS (coke/refined petroleum), MWG/PNJ/DGW/FRT (retail) - one
VI + one EN question per ticker (26 questions), phrasing rotated across 5
templates per language so no two adjacent questions read identically, PLUS
4 refusal-control questions (2 VI + 2 EN) using real estate tickers
(VIC/VHM/NVL/KDH) deliberately NOT in `TICKER_INDUSTRY_MAP` - these MUST
resolve to `whatif_unsupported`, confirming the router doesn't silently
borrow another ticker's industry mapping. ids re-numbered 101-130 (was
101-108) to make room; ratio/rate/trade/alias addendum files re-numbered
accordingly in the same pass - see docs/PROGRESS.md's 2026-09-15 (tiếp 4)
entry for the full renumbering table.

METHODOLOGY: reused byte-for-byte from final_benchmark_100_en.py (resumable
JSONL, same run_one()/`_looks_like_refusal()`/`_branch_matches_expected()`).

Usage:
    python -m scripts.final_benchmark_macro_en
    python -m scripts.score_audit_results --input data/eval/final_benchmark_macro_en.jsonl --chart \\
        --chart-output data/eval/runs/final_benchmark_macro_score.png

PREREQUISITE: run the macro ingest FIRST, or every question here will
correctly (not a bug) return `whatif_unsupported` with a "chưa đủ dữ liệu
macro... chạy src.ingestion.macro_etl trước" error, per
sensitivity_what_if_macro()'s own explicit guard:
    python -m src.ingestion.macro_etl data/raw/vietstock_exports/macro/*.xlsx
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

QUESTIONS = [
    dict(id=101, category="industry_index_whatif", lang="vi", tickers=["HPG"],
         expected_branch="whatif", expected_outcome="refuse",
         question="Nếu chỉ số ngành sản xuất của HPG biến động thì lợi nhuận gộp thay đổi thế nào?",
         note="HPG -> IIP_BY_INDUSTRY 'Sản xuất kim loại' (steel)."),
    dict(id=102, category="industry_index_whatif", lang="en", tickers=["HPG"],
         expected_branch="whatif", expected_outcome="refuse",
         question="What if the industry index for HPG changed - how would gross profit be affected?"),
    dict(id=103, category="industry_index_whatif", lang="vi", tickers=["HSG"],
         expected_branch="whatif", expected_outcome="answer",
         question="Giả sử chỉ số sản xuất công nghiệp ngành của HSG tăng, doanh thu công ty bị ảnh hưởng ra sao?",
         note="HSG -> same 'Sản xuất kim loại' index as HPG (both steel) - "
              "confirms the same real macro shock applies consistently "
              "across 2 companies in the same mapped industry."),
    dict(id=104, category="industry_index_whatif", lang="en", tickers=["HSG"],
         expected_branch="whatif", expected_outcome="answer",
         question="Suppose HSG's industry production index shifted - how would revenue be affected?"),
    dict(id=105, category="industry_index_whatif", lang="vi", tickers=["VNM"],
         expected_branch="whatif", expected_outcome="refuse",
         question="Kịch bản chỉ số ngành cho VNM trong năm 2024 sẽ tác động thế nào đến lợi nhuận gộp?",
         note="VNM -> IIP_BY_INDUSTRY 'Sản xuất, chế biến thực phẩm'."),
    dict(id=106, category="industry_index_whatif", lang="en", tickers=["VNM"],
         expected_branch="whatif", expected_outcome="refuse",
         question="What-if scenario for VNM in 2024: how would gross profit change under a real industry index move?",
         note="Reworded from an earlier draft that used 'how does the "
              "industry index affect gross profit' - present-tense "
              "'affect' (no 'ed') matches rules.py's CAUSAL marker "
              "\\baffects?\\b, which is checked BEFORE WHATIF, so that "
              "phrasing routed to causal_unsupported instead. Caught via "
              "the full-suite regression, same bug class already found "
              "and avoided in final_benchmark_rate_en.py - missed here "
              "the first time, fixed once the regression caught it."),
    dict(id=107, category="industry_index_whatif", lang="vi", tickers=["MSN"],
         expected_branch="whatif", expected_outcome="refuse",
         question="Nếu chỉ số ngành thay đổi, MSN sẽ ra sao?",
         note="MSN -> same 'Sản xuất, chế biến thực phẩm' index as VNM/KDC."),
    dict(id=108, category="industry_index_whatif", lang="en", tickers=["MSN"],
         expected_branch="whatif", expected_outcome="refuse",
         question="What if the industry index changed for MSN - what happens to the company?"),
    dict(id=109, category="industry_index_whatif", lang="vi", tickers=["KDC"],
         expected_branch="whatif", expected_outcome="refuse",
         question="Giả sử chỉ số sản xuất ngành biến động thực tế, KDC bị ảnh hưởng thế nào?",
         note="KDC -> same 'Sản xuất, chế biến thực phẩm' index as VNM/MSN."),
    dict(id=110, category="industry_index_whatif", lang="en", tickers=["KDC"],
         expected_branch="whatif", expected_outcome="refuse",
         question="Suppose the real industry production index moved - how would KDC be affected?"),
    dict(id=111, category="industry_index_whatif", lang="vi", tickers=["SAB"],
         expected_branch="whatif", expected_outcome="refuse",
         question="Nếu chỉ số ngành sản xuất của SAB biến động thì lợi nhuận gộp thay đổi thế nào?",
         note="SAB -> IIP_BY_INDUSTRY 'Sản xuất đồ uống' (beverages) - only "
              "ticker in this file mapped to this specific index."),
    dict(id=112, category="industry_index_whatif", lang="en", tickers=["SAB"],
         expected_branch="whatif", expected_outcome="refuse",
         question="What if the industry index for SAB changed - how would gross profit be affected?"),
    dict(id=113, category="industry_index_whatif", lang="vi", tickers=["DPM"],
         expected_branch="whatif", expected_outcome="refuse",
         question="Giả sử chỉ số sản xuất công nghiệp ngành của DPM tăng, doanh thu công ty bị ảnh hưởng ra sao?",
         note="DPM -> IIP_BY_INDUSTRY 'Sản xuất hoá chất và sản phẩm hoá chất'."),
    dict(id=114, category="industry_index_whatif", lang="en", tickers=["DPM"],
         expected_branch="whatif", expected_outcome="refuse",
         question="Suppose DPM's industry production index shifted - how would revenue be affected?"),
    dict(id=115, category="industry_index_whatif", lang="vi", tickers=["DCM"],
         expected_branch="whatif", expected_outcome="refuse",
         question="Kịch bản chỉ số ngành cho DCM trong năm 2024 sẽ tác động thế nào đến lợi nhuận gộp?",
         note="DCM -> same 'Sản xuất hoá chất...' index as DPM."),
    dict(id=116, category="industry_index_whatif", lang="en", tickers=["DCM"],
         expected_branch="whatif", expected_outcome="refuse",
         question="What-if scenario for DCM in 2024: how would gross profit change under a real industry index move?"),
    dict(id=117, category="industry_index_whatif", lang="vi", tickers=["GAS"],
         expected_branch="whatif", expected_outcome="refuse",
         question="Nếu chỉ số ngành thay đổi, GAS sẽ ra sao?",
         note="GAS -> IIP_BY_INDUSTRY 'Sản xuất than cốc, sản phẩm dầu mỏ "
              "tinh chế' - only ticker mapped to this index; GAS is also "
              "OIL_SENSITIVE_TICKERS' only entry, so this question and an "
              "oil-price WHATIF question about GAS are two DIFFERENT real "
              "mechanisms (coke/refined-petroleum production index vs. "
              "WTI price), not duplicates."),
    dict(id=118, category="industry_index_whatif", lang="en", tickers=["GAS"],
         expected_branch="whatif", expected_outcome="refuse",
         question="What if the industry index changed for GAS - what happens to the company?"),
    dict(id=119, category="industry_index_whatif", lang="vi", tickers=["MWG"],
         expected_branch="whatif", expected_outcome="refuse",
         question="Giả sử chỉ số sản xuất ngành biến động thực tế, MWG bị ảnh hưởng thế nào?",
         note="MWG -> RETAIL 'Bán lẻ hàng hóa' (MONTH granularity, unlike "
              "the IIP_BY_INDUSTRY tickers above which are YEAR) - "
              "exercises the other period_type branch inside "
              "sensitivity_what_if_macro()."),
    dict(id=120, category="industry_index_whatif", lang="en", tickers=["MWG"],
         expected_branch="whatif", expected_outcome="refuse",
         question="Suppose the real industry production index moved - how would MWG be affected?"),
    dict(id=121, category="industry_index_whatif", lang="vi", tickers=["PNJ"],
         expected_branch="whatif", expected_outcome="refuse",
         question="Nếu chỉ số ngành sản xuất của PNJ biến động thì lợi nhuận gộp thay đổi thế nào?",
         note="PNJ -> same RETAIL index as MWG/DGW/FRT. PNJ is ALSO the "
              "only GOLD_SENSITIVE_TICKERS entry - two different real "
              "mechanisms (retail-sales index vs. gold price), same "
              "distinction as GAS's industry-vs-oil pair above."),
    dict(id=122, category="industry_index_whatif", lang="en", tickers=["PNJ"],
         expected_branch="whatif", expected_outcome="refuse",
         question="What if the industry index for PNJ changed - how would gross profit be affected?"),
    dict(id=123, category="industry_index_whatif", lang="vi", tickers=["DGW"],
         expected_branch="whatif", expected_outcome="refuse",
         question="Giả sử chỉ số sản xuất công nghiệp ngành của DGW tăng, doanh thu công ty bị ảnh hưởng ra sao?",
         note="DGW -> same RETAIL index as MWG/PNJ/FRT."),
    dict(id=124, category="industry_index_whatif", lang="en", tickers=["DGW"],
         expected_branch="whatif", expected_outcome="refuse",
         question="Suppose DGW's industry production index shifted - how would revenue be affected?"),
    dict(id=125, category="industry_index_whatif", lang="vi", tickers=["FRT"],
         expected_branch="whatif", expected_outcome="refuse",
         question="Kịch bản chỉ số ngành cho FRT trong năm 2024 sẽ tác động thế nào đến lợi nhuận gộp?",
         note="FRT -> same RETAIL index as MWG/PNJ/DGW - completes all 4 "
              "RETAIL-mapped tickers in this file."),
    dict(id=126, category="industry_index_whatif", lang="en", tickers=["FRT"],
         expected_branch="whatif", expected_outcome="refuse",
         question="What-if scenario for FRT in 2024: how would gross profit change under a real industry index move?"),
    # Refusal-control (4 Qs, 2 VI + 2 EN) - real estate tickers deliberately
    # NOT in TICKER_INDUSTRY_MAP (VIC/VHM/NVL/KDH - see that dict's own
    # comment in macro_lookup.py: "not in IIP scope"). Must correctly
    # refuse, not silently borrow another ticker's industry mapping.
    dict(id=127, category="industry_index_whatif_control", lang="vi", tickers=["VIC"],
         expected_branch="whatif_unsupported", expected_outcome="refuse",
         question="Nếu chỉ số ngành sản xuất của VIC biến động thì lợi nhuận gộp ra sao?",
         note="VIC (real estate) has no TICKER_INDUSTRY_MAP entry - must refuse."),
    dict(id=128, category="industry_index_whatif_control", lang="vi", tickers=["VHM"],
         expected_branch="whatif_unsupported", expected_outcome="refuse",
         question="Giả sử chỉ số ngành của VHM thay đổi, doanh thu công ty ra sao?",
         note="VHM (real estate) has no TICKER_INDUSTRY_MAP entry - must refuse."),
    dict(id=129, category="industry_index_whatif_control", lang="en", tickers=["NVL"],
         expected_branch="whatif_unsupported", expected_outcome="refuse",
         question="What if the industry index for NVL changed - how would gross profit be affected?",
         note="NVL (real estate) has no TICKER_INDUSTRY_MAP entry - must refuse."),
    dict(id=130, category="industry_index_whatif_control", lang="en", tickers=["KDH"],
         expected_branch="whatif_unsupported", expected_outcome="refuse",
         question="Suppose KDH's industry production index shifted - how would revenue be affected?",
         note="KDH (real estate) has no TICKER_INDUSTRY_MAP entry - must refuse."),
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
        raw = result.get("raw") or {}
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
            "macro_source": raw.get("macro_source") if isinstance(raw, dict) else None,
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
    ap.add_argument("--out", default="data/eval/final_benchmark_macro_en.jsonl")
    ap.add_argument("--only", default=None, help="Comma-separated question ids, e.g. 101,102")
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
