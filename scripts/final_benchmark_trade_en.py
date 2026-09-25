"""Export/import-by-commodity WHATIF addendum - 2026-09-15. Continues the id
numbering of the final_benchmark_*.py series (ids 127-134) - see
final_benchmark_macro_en.py's docstring for why each of these stays its own
file rather than being appended into final_benchmark_100_en.py.

WHY: Quang's 2026-09-15 message explicitly asked to "tối ưu" (optimize) the
export/import-by-commodity data too, "vì các mã sẽ tương ứng với các ngành
và bộ vĩ mô riêng" (because tickers correspond to industries and their own
macro datasets) - the same TICKER_INDUSTRY_MAP-style pattern just built for
`sensitivity_what_if_macro()` (industry production index), applied to the
REAL monthly EXPORTS_BY_COMMODITY/IMPORTS_BY_COMMODITY rows found in the
47-file macro batch (confirmed real commodity row names: "Sắt thép" in both
the export and import files, "Dầu thô"/"Xăng dầu", "Phân bón" in the import
file - see docs/PROGRESS.md's 2026-09-15 entry for the full file-scan log).

FIX (mirrors the gold/oil/industry pattern exactly)
-----------------------------------------------------------------------------
  - `src/causal/macro_lookup.py`: added `TRADE_INDUSTRY_MAP` (ticker ->
    (category, indicator_name, direction)) and `trade_index_for_ticker()`.
    Only 4 tickers mapped, deliberately small ("no invented ground truth"):
      HPG, HSG -> EXPORTS_BY_COMMODITY "Sắt thép" (steel), direction=export
        (both are real steel producers/exporters - a rise in national steel
        export value is a genuine demand-side tailwind for their revenue).
      DPM, DCM -> IMPORTS_BY_COMMODITY "Phân bón" (fertilizer), direction=
        import (both are domestic fertilizer producers - a RISE in national
        fertilizer IMPORTS signals import competition against their own
        domestic sales, a headwind, NOT a tailwind - see below).
  - `src/structured/calculator.py`: added `sensitivity_what_if_trade()` -
    same ceteris-paribus REVENUE-shock model as `sensitivity_what_if_oil()`,
    but the pct_change source is `trade_index_for_ticker()`'s mapped
    commodity, and for `direction="import"` mappings the sign is FLIPPED
    before being applied (a bigger import figure is a headwind, not a
    tailwind - see that function's docstring for the full mechanism note,
    which is explicitly flagged as the LESS certain of the two mappings,
    same honesty caveat pattern as `OIL_SENSITIVE_TICKERS`).
  - `src/router/rules.py`: added `_TRADE_WORDS` marker set ("xuất khẩu",
    "nhập khẩu", "export", "import") and a new classify_intent() WHATIF
    branch, checked in the same block as gold/oil/industry/interest_rate,
    right after interest_rate. Ticker-restricted to `TRADE_INDUSTRY_MAP`
    (like gold/oil/industry - NOT unrestricted like interest_rate, since a
    commodity's trade volume is only a meaningful revenue proxy for a
    company that actually produces/competes in that exact commodity).
  - `src/router/query_router.py`: imported `sensitivity_what_if_trade`,
    added its dispatch branch in `_dispatch_whatif()` (whatif_kind=
    "trade_sensitivity") and its formatting branch in `_format_whatif()`
    (uses the function's own "trade_source" key). No `schemas.py`/migration
    014 change needed - reuses the existing "whatif" branch label, same as
    gold/oil/industry/interest_rate all do.

IMPORTANT HONESTY NOTE ON TICKER TIER (per this project's PANEL vs CORE_16
distinction - see final_benchmark_100_en.py's docstring for the full
definition): of the 4 mapped tickers, only HSG is PANEL-tier (real
financial_line_items rows, so sensitivity_what_if_trade() can actually
compute a real revenue/COGS/gross-profit scenario against a live DB). HPG,
DPM, and DCM are all CORE_16-tier (per scripts/seed_companies.py's CORE_16
list) - ZERO financial_line_items rows - so on a live DB run they will
correctly and honestly return `{"ok": False, "error": "...không có dữ liệu
doanh thu/giá vốn..."}` (sensitivity_what_if()'s own pre-existing no-data
guard), NOT a fabricated number. This is NOT a bug and NOT unique to this
feature - GAS (the only entry in OIL_SENSITIVE_TICKERS) is ALSO CORE_16 and
has had this exact same accepted limitation since before this session. The
mapping's job is to be an HONEST, industry-justified pairing between ticker
and commodity; whether the underlying financial_line_items rows happen to
be populated for a given ticker is a separate, live-DB-only fact that
cannot be invented or worked around here - flagged explicitly rather than
silently avoided by only testing HSG.

TICKER SELECTION: all 4 TRADE_INDUSTRY_MAP tickers included on purpose
(2 export-direction, 2 import-direction) specifically so the benchmark
exercises BOTH the tailwind (export) and headwind-via-sign-flip (import)
code paths, and so the PANEL-success-vs-CORE_16-honest-refusal contrast is
visible in the results table exactly like structured_lookup's PANEL/CORE_16
split already is.

CATEGORY: `trade_commodity_whatif` (Group 5, WHATIF family).

2026-09-15 (tiếp 4): EXPANDED 9 -> 30, ids re-numbered 191-220 (was
127-135). 20 questions across the 4 mapped tickers (5 phrasing templates
each, 3 VI + 2 EN, all empirically verified via classify_intent() before
being written down) + 10 refusal-control questions spanning 8 different
unmapped tickers across varied sectors (dairy, retail, construction,
petroleum refining, real estate, beverages, jewellery, electronics retail)
- including 2 tickers (GAS, PNJ) that ARE real entries in this project's
OTHER mapping tables (TICKER_INDUSTRY_MAP / GOLD_SENSITIVE_TICKERS
respectively) specifically to confirm the mapping tables are independent
of each other, not a case of "any ticker with some real-data mapping
anywhere works everywhere". expected_outcome also corrected in this pass:
the original 9-question file said "answer" for all trade_commodity_whatif
questions even though its own notes already flagged HPG/DPM/DCM as
CORE_16 (honest-refusal expected) - now explicit ("refuse" for HPG/DPM/DCM,
"answer" only for HSG), matching final_benchmark_rate_en.py's same-tier
convention.

METHODOLOGY: reused byte-for-byte from final_benchmark_100_en.py /
final_benchmark_macro_en.py / final_benchmark_rate_en.py.

Usage:
    python -m scripts.final_benchmark_trade_en
    python -m scripts.score_audit_results --input data/eval/final_benchmark_trade_en.jsonl --chart \\
        --chart-output data/eval/runs/final_benchmark_trade_score.png
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
    # trade_commodity_whatif - 20 Qs: 4 mapped tickers x 5 phrasing
    # templates each (3 VI + 2 EN). HSG is the only PANEL ticker (real
    # calc expected); HPG/DPM/DCM are all CORE_16 (honest-refusal expected
    # on a live DB, per scripts/seed_companies.py's CORE_16 list) -
    # expected_outcome reflects that split explicitly below (fixed from
    # the original 9-question version, which said "answer" uniformly even
    # though its own notes already flagged the CORE_16 refusal - now
    # consistent with final_benchmark_rate_en.py's same-tier convention).
    # All 5 templates verified empirically via classify_intent() before
    # being written down (see docs/PROGRESS.md's 2026-09-15 tiếp 4 entry).
    # ============================================================
    dict(id=191, category="trade_commodity_whatif", lang="vi", tickers=["HPG"],
         expected_branch="whatif", expected_outcome="refuse",
         note="HPG=CORE_16 (zero financial_line_items) - honest refusal expected live.",
         question="Nếu xuất khẩu sắt thép tăng thì doanh thu HPG thay đổi ra sao?"),
    dict(id=192, category="trade_commodity_whatif", lang="vi", tickers=["HPG"],
         expected_branch="whatif", expected_outcome="refuse",
         question="Giả sử xuất khẩu sắt thép biến động, HPG sẽ bị ảnh hưởng ra sao?"),
    dict(id=193, category="trade_commodity_whatif", lang="vi", tickers=["HPG"],
         expected_branch="whatif", expected_outcome="refuse",
         question="Kịch bản xuất khẩu sắt thép cho HPG năm 2024 sẽ ra sao?"),
    dict(id=194, category="trade_commodity_whatif", lang="en", tickers=["HPG"],
         expected_branch="whatif", expected_outcome="refuse",
         question="What if steel exports increased - how would HPG revenue be affected?"),
    dict(id=195, category="trade_commodity_whatif", lang="en", tickers=["HPG"],
         expected_branch="whatif", expected_outcome="refuse",
         question="Suppose steel exports shifted - how would HPG be affected?"),
    dict(id=196, category="trade_commodity_whatif", lang="vi", tickers=["HSG"],
         expected_branch="whatif", expected_outcome="answer",
         note="HSG=PANEL (real financial_line_items) - the only ticker in "
              "this category expected to compute a real number live.",
         question="Nếu xuất khẩu sắt thép tăng thì doanh thu HSG thay đổi ra sao?"),
    dict(id=197, category="trade_commodity_whatif", lang="vi", tickers=["HSG"],
         expected_branch="whatif", expected_outcome="answer",
         question="Giả sử xuất khẩu sắt thép biến động, HSG sẽ bị ảnh hưởng ra sao?"),
    dict(id=198, category="trade_commodity_whatif", lang="vi", tickers=["HSG"],
         expected_branch="whatif", expected_outcome="answer",
         question="Kịch bản xuất khẩu sắt thép cho HSG năm 2024 sẽ ra sao?"),
    dict(id=199, category="trade_commodity_whatif", lang="en", tickers=["HSG"],
         expected_branch="whatif", expected_outcome="answer",
         question="What if steel exports increased - how would HSG revenue be affected?"),
    dict(id=200, category="trade_commodity_whatif", lang="en", tickers=["HSG"],
         expected_branch="whatif", expected_outcome="answer",
         question="Suppose steel exports shifted - how would HSG be affected?"),
    dict(id=201, category="trade_commodity_whatif", lang="vi", tickers=["DPM"],
         expected_branch="whatif", expected_outcome="refuse",
         note="DPM=CORE_16 - honest refusal expected live. Also exercises "
              "the import-direction sign flip (see calculator.py docstring).",
         question="Nếu nhập khẩu phân bón tăng thì doanh thu DPM bị ảnh hưởng thế nào?"),
    dict(id=202, category="trade_commodity_whatif", lang="vi", tickers=["DPM"],
         expected_branch="whatif", expected_outcome="refuse",
         question="Giả sử nhập khẩu phân bón biến động, DPM sẽ ra sao?"),
    dict(id=203, category="trade_commodity_whatif", lang="vi", tickers=["DPM"],
         expected_branch="whatif", expected_outcome="refuse",
         question="Kịch bản nhập khẩu phân bón cho DPM năm 2024 sẽ ra sao?"),
    dict(id=204, category="trade_commodity_whatif", lang="en", tickers=["DPM"],
         expected_branch="whatif", expected_outcome="refuse",
         question="What if fertilizer imports increased - how would DPM revenue be affected?"),
    dict(id=205, category="trade_commodity_whatif", lang="en", tickers=["DPM"],
         expected_branch="whatif", expected_outcome="refuse",
         question="Suppose fertilizer imports shifted - how would DPM be affected?"),
    dict(id=206, category="trade_commodity_whatif", lang="vi", tickers=["DCM"],
         expected_branch="whatif", expected_outcome="refuse",
         note="DCM=CORE_16 - honest refusal expected live.",
         question="Nếu nhập khẩu phân bón tăng thì doanh thu DCM bị ảnh hưởng thế nào?"),
    dict(id=207, category="trade_commodity_whatif", lang="vi", tickers=["DCM"],
         expected_branch="whatif", expected_outcome="refuse",
         question="Giả sử nhập khẩu phân bón biến động, DCM sẽ ra sao?"),
    dict(id=208, category="trade_commodity_whatif", lang="vi", tickers=["DCM"],
         expected_branch="whatif", expected_outcome="refuse",
         question="Kịch bản nhập khẩu phân bón cho DCM năm 2024 sẽ ra sao?"),
    dict(id=209, category="trade_commodity_whatif", lang="en", tickers=["DCM"],
         expected_branch="whatif", expected_outcome="refuse",
         question="What if fertilizer imports increased - how would DCM revenue be affected?"),
    dict(id=210, category="trade_commodity_whatif", lang="en", tickers=["DCM"],
         expected_branch="whatif", expected_outcome="refuse",
         question="Suppose fertilizer imports shifted - how would DCM be affected?"),
    # ============================================================
    # trade_commodity_whatif_control - 10 Qs: mapped-word questions for
    # tickers NOT in TRADE_INDUSTRY_MAP - must honestly refuse
    # (whatif_unsupported), never silently borrow HPG/HSG's steel mapping
    # or DPM/DCM's fertilizer mapping. Spans a wide mix of sectors
    # (dairy, retail, construction, petroleum refining, real estate,
    # beverages, jewellery/gold, electronics retail) to show the refusal
    # isn't sector-specific - the guard is purely "not a
    # TRADE_INDUSTRY_MAP key", nothing more.
    # ============================================================
    dict(id=211, category="trade_commodity_whatif_control", lang="vi", tickers=["VNM"],
         expected_branch="whatif_unsupported", expected_outcome="refuse",
         note="VNM (dairy) has no TRADE_INDUSTRY_MAP entry.",
         question="Nếu xuất khẩu sắt thép tăng thì doanh thu VNM thay đổi ra sao?"),
    dict(id=212, category="trade_commodity_whatif_control", lang="vi", tickers=["MWG"],
         expected_branch="whatif_unsupported", expected_outcome="refuse",
         note="MWG (retail) has no TRADE_INDUSTRY_MAP entry.",
         question="Giả sử xuất khẩu sắt thép biến động, MWG sẽ bị ảnh hưởng ra sao?"),
    dict(id=213, category="trade_commodity_whatif_control", lang="vi", tickers=["CTD"],
         expected_branch="whatif_unsupported", expected_outcome="refuse",
         note="CTD (construction) has no TRADE_INDUSTRY_MAP entry, despite "
              "construction being a plausible steel END-USER - the mapping "
              "is deliberately restricted to real EXPORT/IMPORT producers, "
              "not every company that might use the commodity.",
         question="Kịch bản nhập khẩu phân bón cho CTD năm 2024 sẽ ra sao?"),
    dict(id=214, category="trade_commodity_whatif_control", lang="en", tickers=["BSR"],
         expected_branch="whatif_unsupported", expected_outcome="refuse",
         note="BSR (petroleum refining) has no TRADE_INDUSTRY_MAP entry.",
         question="What if fertilizer imports increased - how would BSR revenue be affected?"),
    dict(id=215, category="trade_commodity_whatif_control", lang="vi", tickers=["KTS"],
         expected_branch="whatif_unsupported", expected_outcome="refuse",
         question="Nếu nhập khẩu phân bón tăng thì doanh thu KTS bị ảnh hưởng thế nào?"),
    dict(id=216, category="trade_commodity_whatif_control", lang="en", tickers=["VIC"],
         expected_branch="whatif_unsupported", expected_outcome="refuse",
         note="VIC (real estate) has no TRADE_INDUSTRY_MAP entry.",
         question="Suppose steel exports shifted - how would VIC be affected?"),
    dict(id=217, category="trade_commodity_whatif_control", lang="vi", tickers=["GAS"],
         expected_branch="whatif_unsupported", expected_outcome="refuse",
         note="GAS has a REAL entry in TICKER_INDUSTRY_MAP (industry-index "
              "WHATIF, coke/refined petroleum) but NOT in TRADE_INDUSTRY_MAP "
              "- confirms the two mapping tables are independent, a ticker "
              "being real in one doesn't make it real in the other.",
         question="Nếu xuất khẩu sắt thép tăng thì doanh thu GAS thay đổi ra sao?"),
    dict(id=218, category="trade_commodity_whatif_control", lang="en", tickers=["SAB"],
         expected_branch="whatif_unsupported", expected_outcome="refuse",
         note="SAB (beverages) has no TRADE_INDUSTRY_MAP entry.",
         question="What if fertilizer imports increased - how would SAB revenue be affected?"),
    dict(id=219, category="trade_commodity_whatif_control", lang="vi", tickers=["PNJ"],
         expected_branch="whatif_unsupported", expected_outcome="refuse",
         note="PNJ has a REAL entry in GOLD_SENSITIVE_TICKERS (gold-price "
              "WHATIF) but NOT in TRADE_INDUSTRY_MAP - same "
              "independent-mapping-tables point as GAS (id=217).",
         question="Giả sử xuất khẩu sắt thép biến động, PNJ sẽ ra sao?"),
    dict(id=220, category="trade_commodity_whatif_control", lang="en", tickers=["FRT"],
         expected_branch="whatif_unsupported", expected_outcome="refuse",
         note="FRT (electronics retail) has no TRADE_INDUSTRY_MAP entry.",
         question="What if steel exports increased - how would FRT revenue be affected?"),
]

_REFUSAL_PATTERNS = re.compile(
    r"không tìm thấy|không tính được|không ước lượng được|không phân tích được|"
    r"no information found|not found in|không xác định được|không đủ dữ liệu|"
    r"chưa có ánh xạ|chưa được xác định",
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
    ap.add_argument("--out", default="data/eval/final_benchmark_trade_en.jsonl")
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
