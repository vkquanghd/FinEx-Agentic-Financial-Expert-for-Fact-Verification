"""Final, curated evaluation set for the thesis's Results chapter - 2026-09-14.

WHY THIS FILE EXISTS (do not merge back into audit_answer_quality.py)
-----------------------------------------------------------------------
scripts/audit_answer_quality.py (60 questions, 21 categories, mostly
Vietnamese) was a diagnostic tool: cast a wide net across every router
branch and every ticker-coverage tier to FIND bugs. It succeeded - it is
the reason src/router/rules.py's causal-marker regex, the new
OUT_OF_SCOPE branch, and the WHATIF sign/compound-percentage parsing were
fixed on 2026-09-14 (see docs/PROGRESS.md, same date).

With less than 24 hours left before the report is due, Quang asked for a
different, smaller artifact: a benchmark whose SCOPE, TICKER SELECTION,
and LANGUAGE MIX are each independently justified in writing, suitable to
paste directly into a Master's thesis Evaluation Methodology section - not
a diagnostic tool, a REPORTED RESULT. Concretely, requested:
  1. Everything documented in English (the thesis defense language).
  2. Underlying data stays Vietnamese (it is Vietnamese financial
     disclosure data - that is not something to hide, it is the subject
     of the thesis), but QUESTIONS deliberately mix Vietnamese and
     English phrasing against the SAME small ticker set, so language cost
     (see scripts/bilingual_latency_benchmark.py) and language-routing
     correctness are both exercised on identical underlying content.
  3. "We have a lot of data but need just a few good tickers" - a small,
     hand-picked ticker set, not the full 24-ticker spread
     audit_answer_quality.py used, each ticker's coverage claim traceable
     to a specific, already-recorded verification (see TICKER SELECTION
     below) - no new coverage claim is asserted here that wasn't already
     established and dated elsewhere in this project.
  4. Drop the PNJ gold-price what-if scenario entirely. Investigated
     2026-09-14 (see docs/PROGRESS.md "còn giá vàng thì sao?" entry): real
     domestic gold-price data DOES exist (macro_indicators, category
     MKT_GOLD_SJC, ~24 rows as of 2026-07-27), but PNJ itself - the ONLY
     ticker in calculator.GOLD_SENSITIVE_TICKERS - is CORE_16 tier and has
     ZERO rows in financial_line_items, so the scenario calculation always
     fails regardless of which % or gold-price source feeds it. Including
     it here would report a guaranteed, uninformative failure instead of
     testing anything about the system's actual capability - Quang's own
     instruction was "bỏ qua cái vàng đi, tập trung vào data có sẵn"
     (skip gold, focus on the data that is actually available).

TICKER SELECTION (5 tickers total - each claim below is a citation to an
already-recorded, dated finding elsewhere in this project, not a new
assertion)
-----------------------------------------------------------------------
PANEL tier (real DAG structured coverage, confirmed via a direct scan of
data/raw/vietstock_exports/*.xlsx on 2026-09-12 -
docs/answer_quality_audit_2026-09-12.md section 1):
  - CTD (Coteccons, construction) - the ONE ticker confirmed to have full
    4/4-modality coverage (RAG + DAG + TiniX OCR + Vietstock Excel) as of
    2026-07-20 (same source doc). Used as the primary "everything works"
    showcase ticker.
  - HSG (Hoa Sen Group, steel) - PANEL tier, already exercised
    successfully across DAG, RAG, and WHATIF branches earlier in this
    project's audit history (audit_answer_quality.py ids 6, 20-22).
  - BSR (Binh Son Refining, oil refining) - PANEL tier, exercised
    successfully in DAG (audit_answer_quality.py id 12) and is the ticker
    used for the DAG_TREND quarterly-breakdown fix investigated
    2026-09-14 (id 53).

CORE_16 tier (RAG coverage only, ZERO financial_line_items rows - see
docs/answer_quality_audit_2026-09-12.md section 1 for how this was
confirmed by scanning all 213 valid Vietstock export files and finding
none for any of the 16 tickers in scripts.seed_companies.CORE_16):
  - HPG (Hoa Phat Group, steel) - the ticker most heavily exercised in
    this project's own router-bug-fixing history (2026-09-14's causal
    regex fix and OUT_OF_SCOPE branch were both diagnosed against HPG
    questions).
  - VNM (Vinamilk, dairy) - the most internationally recognisable
    Vietnamese listed company, a deliberate choice given the thesis is
    defended in English: a foreign committee member is far more likely to
    already know "Vinamilk" than a smaller CORE_16 ticker, which matters
    when reading example Q&A pairs cold.

Deliberately EXCLUDED from this file: PNJ (see gold-price rationale
above), and every other CORE_16/PANEL ticker used in
audit_answer_quality.py's wider 24-ticker sweep - this file is not trying
to re-prove coverage breadth (that sweep already exists and is dated), it
is trying to report a small, defensible, readable set of results.

CATEGORY TAXONOMY (13 categories x 2 tickers each = 26 questions)
-----------------------------------------------------------------------
Each category maps to one specific claim in one of the thesis's 3 research
question groups (docs/thesis_latex/chap1.tex, "Research questions"):

  Group 1 (Adaptive routing):
    structured_lookup      - single ticker, single indicator, single
                              period -> DAG branch.
    multi_ticker_lookup    - the SAME indicator/period across >=2 tickers
                              in one question -> DAG_MULTI branch (added
                              2026-09-14, fixes the "silently answers only
                              the first ticker" bug - see docs/PROGRESS.md).
                              One PANEL+PANEL pair (both should succeed)
                              and one CORE_16+CORE_16 pair (both should
                              honestly fall back to RAG per ticker, never
                              silently dropping one) - deliberately
                              contrasted to show the fix handles BOTH
                              outcomes correctly, not just the easy case.
    multi_year_trend       - trend language, no specific year -> DAG_TREND.
    qualitative_disclosure - a question with no single numeric answer
                              (legal risk, governance) -> RAG branch. Asked
                              in BOTH languages against the SAME ticker
                              (HPG) specifically to pair with
                              scripts/bilingual_latency_benchmark.py's
                              token/latency comparison methodology.
    scenario_whatif         - a single ceteris-paribus % scenario ->
                              WHATIF branch, deterministic calculator.
    compound_whatif         - TWO simultaneous % scenarios (COGS AND
                              revenue) in one question -> exercises the
                              2026-09-14 fix for the "only the first %
                              was parsed" bug (see docs/PROGRESS.md,
                              "compound_p3_bug" entry, id 41).
    multi_ticker_compare    - price-performance comparison across >=2
                              tickers -> COMPARE branch.
    auditor_lookup          - reverse lookup (audit firm -> client list),
                              no ticker needed -> AUDITOR_LOOKUP branch.
    out_of_scope_prediction - future stock-price prediction -> the new
                              OUT_OF_SCOPE branch (2026-09-14).
    out_of_scope_opinion    - a named third party's personal opinion ->
                              OUT_OF_SCOPE branch. One instance uses Warren
                              Buffett (the one figure with a verified,
                              hand-written bio in
                              src/router/query_router.py's
                              _KNOWN_PUBLIC_FIGURES) and one uses a name
                              NOT in that lookup table, to report both the
                              enriched and the generic code path honestly.
    out_of_scope_advice     - a personal buy/sell recommendation request
                              -> OUT_OF_SCOPE branch (compliance boundary).

  Group 2 (Quantitative correctness):
    cross_sectional_causal   - the ONE treatment/outcome pair with a real
                                literature citation attached
                                (leverage -> net_margin, Myers 1984
                                pecking-order theory; see
                                src/router/query_router.py's
                                _format_causal()) -> CAUSAL branch.
    causal_negative_control  - a pair with NO theoretically expected
                                relationship (asset_turnover ->
                                current_ratio, already used as negative
                                control id 55 in audit_answer_quality.py)
                                -> CAUSAL branch. Reports whether the
                                system distinguishes a real theoretical
                                anchor from an arbitrary pair, rather than
                                asserting significance either way.

Group 3 (Conversational memory) has its OWN dedicated, already-executed
benchmark (data/eval/runs/rq7_resolvers_2026-08-14.json,
scripts/eval_rq7_resolvers.py) - not duplicated here, since this file's
questions are all single-turn by design (multi-turn context resolution is
a fundamentally different evaluation shape, not just "more categories").

LANGUAGE ASSIGNMENT
-----------------------------------------------------------------------
Each question is tagged `lang` ("vi" or "en") for reporting. Assignment is
alternated deliberately within each category (never all-VI or all-EN for
the same category) so the results table can report a routing/answer-
quality breakdown BY LANGUAGE as well as by category, directly answering
"does asking in English change routing correctness or answer quality,
given the corpus is Vietnamese-first" - the same question
scripts/bilingual_latency_benchmark.py answers for LATENCY, this answers
for CORRECTNESS.

METHODOLOGY, unchanged from audit_answer_quality.py (reused deliberately,
not reinvented)
-----------------------------------------------------------------------
- Resumable JSONL output (one line per question, flushed immediately) -
  same convention as every other real-run script in this project.
- `expected_branch`: written at design time (this file), same "documents
  intent, not a hard assertion" status as audit_answer_quality.py's own
  field of the same name.
- `expected_outcome` ("answer" | "refuse"): NEW field, written per
  question here (not inferred from category the way
  scripts/score_audit_results.py's CATEGORY_EXPECTED_OUTCOME does for the
  older, messier 60-question set) - every ticker/category combination in
  THIS file was hand-picked specifically because its expected outcome is
  unambiguous, so there is no "needs_human_review" bucket needed here.
- `branch_match` / `looks_like_refusal` / `numeric_verification`: reused
  directly from route()'s own result shape and
  audit_answer_quality.py's `_looks_like_refusal()` regex - identical
  ground-truth machinery, just applied to a smaller, cleaner set.

Usage:
    python -m scripts.final_benchmark_en
    python -m scripts.final_benchmark_en --only 1,2,3
    python -m scripts.score_audit_results --input data/eval/final_benchmark_en.jsonl --chart \\
        --chart-output data/eval/runs/final_benchmark_score.png
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# id, category, lang ("vi"|"en"), tickers (list, usually len 1),
# expected_branch, expected_outcome ("answer"|"refuse"), question, note
# (design rationale specific to this one item, beyond the category-level
# rationale already given in the module docstring above).
QUESTIONS = [
    # --- 1. structured_lookup (DAG) ---
    dict(id=1, category="structured_lookup", lang="vi", tickers=["CTD"],
         expected_branch="dag", expected_outcome="answer",
         question="Doanh thu thuần năm 2024 của CTD là bao nhiêu?"),
    dict(id=2, category="structured_lookup", lang="en", tickers=["HPG"],
         expected_branch="rag", expected_outcome="answer",
         question="What was HPG's total assets in 2024?",
         note="HPG is CORE_16 (RAG-only by design) - expected_branch is "
              "'rag', not 'dag', unlike the PANEL-tier CTD pair above. "
              "This contrast is intentional: it reports the SAME question "
              "shape resolving through two different branches depending "
              "on ticker coverage tier, which is itself part of the "
              "Adaptive Routing story."),

    # --- 2. multi_ticker_lookup (DAG_MULTI, 2026-09-14 fix) ---
    dict(id=3, category="multi_ticker_lookup", lang="vi", tickers=["CTD", "HSG"],
         expected_branch="dag_multi", expected_outcome="answer",
         question="Doanh thu năm 2024 của cả CTD và HSG là bao nhiêu?",
         note="Both PANEL tier - expects BOTH tickers to resolve via real "
              "DAG data, no silent drop (the exact bug fixed 2026-09-14)."),
    dict(id=4, category="multi_ticker_lookup", lang="en", tickers=["HPG", "VNM"],
         expected_branch="dag_multi", expected_outcome="answer",
         question="What were the total assets of both HPG and VNM in 2024?",
         note="Both CORE_16 tier - expects DAG_MULTI to correctly fall "
              "back to RAG for EACH ticker independently (both honestly "
              "degrade), not silently answer only one."),

    # --- 3. multi_year_trend (DAG_TREND) ---
    dict(id=5, category="multi_year_trend", lang="vi", tickers=["CTD"],
         expected_branch="dag_trend", expected_outcome="answer",
         question="Doanh thu của CTD biến động qua các năm gần đây thế nào?"),
    dict(id=6, category="multi_year_trend", lang="en", tickers=["BSR"],
         expected_branch="dag_trend", expected_outcome="answer",
         question="What is the trend in BSR's net profit over recent years?",
         note="Phrasing verified against src/router/rules.py's "
              "TREND_LANGUAGE_MARKERS, which only matches a bare "
              "'\\btrend\\b' (word-boundary) - an earlier draft phrased "
              "this as 'trended over recent years', which the regex does "
              "NOT match (no word boundary between 'trend' and 'ed') - a "
              "real, narrow EN-phrasing gap, noted here rather than "
              "silently avoided."),

    # --- 4. qualitative_disclosure (RAG) - same ticker, both languages ---
    dict(id=7, category="qualitative_disclosure", lang="vi", tickers=["HPG"],
         expected_branch="rag", expected_outcome="answer",
         question="HPG nêu rủi ro pháp lý gì trong báo cáo thường niên?"),
    dict(id=8, category="qualitative_disclosure", lang="en", tickers=["HPG"],
         expected_branch="rag", expected_outcome="answer",
         question="What legal risks does HPG's annual report mention?",
         note="Deliberately the SAME ticker and topic as id=7, only the "
              "language differs - pairs with "
              "scripts/bilingual_latency_benchmark.py's methodology "
              "(same underlying question in VI and EN) so this file can "
              "report a CORRECTNESS comparison alongside that file's "
              "LATENCY/token comparison."),

    # --- 5. scenario_whatif (WHATIF, single %, PANEL-only so it can "
    # actually succeed - see module docstring on excluding CORE_16/PNJ) ---
    dict(id=9, category="scenario_whatif", lang="vi", tickers=["HSG"],
         expected_branch="whatif", expected_outcome="answer",
         question="Nếu COGS của HSG tăng 5% năm 2025 thì lợi nhuận gộp thay đổi thế nào?"),
    dict(id=10, category="scenario_whatif", lang="en", tickers=["CTD"],
         expected_branch="whatif", expected_outcome="answer",
         question="What if CTD's revenue decreased by 3% in 2025 - how would gross profit change?",
         note="Deliberately phrased as a DECREASE - exercises the "
              "2026-09-14 sign-handling fix (previously 'giảm X%'/"
              "'decreased by X%' was silently computed as +X%, see "
              "docs/PROGRESS.md id=21 entry). Uses 'what if' rather than "
              "a bare 'if' - src/router/rules.py's WHATIF_MARKERS matches "
              "'what\\s*-?\\s*if' but has no standalone '\\bif\\b' marker "
              "(a bare 'if' would be too common a word to use as a "
              "reliable signal without false positives elsewhere) - "
              "verified against classify_intent() directly."),

    # --- 6. compound_whatif (WHATIF, two simultaneous %, 2026-09-14 fix) ---
    dict(id=11, category="compound_whatif", lang="vi", tickers=["HSG"],
         expected_branch="whatif", expected_outcome="answer",
         question="Nếu COGS của HSG tăng 5% và doanh thu giảm 3% cùng lúc năm 2025 thì lợi nhuận gộp ra sao?"),
    dict(id=12, category="compound_whatif", lang="en", tickers=["CTD"],
         expected_branch="whatif", expected_outcome="answer",
         question="What if CTD's COGS increased by 4% and revenue increased by 2% at the same time in 2025 - how would gross profit change?",
         note="Same 'what if' phrasing fix as id=10 above."),

    # --- 7. multi_ticker_compare (COMPARE) ---
    dict(id=13, category="multi_ticker_compare", lang="vi", tickers=["CTD", "HSG"],
         expected_branch="compare", expected_outcome="answer",
         question="So sánh hiệu suất giá của CTD và HSG trong năm qua"),
    dict(id=14, category="multi_ticker_compare", lang="en", tickers=["HPG", "VNM"],
         expected_branch="compare", expected_outcome="answer",
         question="Compare the price performance of HPG and VNM"),

    # --- 8. auditor_lookup (no ticker needed - reverse lookup) ---
    dict(id=15, category="auditor_lookup", lang="en", tickers=[],
         expected_branch="auditor_lookup", expected_outcome="answer",
         question="Which companies is KPMG the auditor for?",
         note="src/router/rules.py's AUDITOR_LOOKUP check requires the "
              "literal substring 'auditor' (or 'kiểm toán' for "
              "Vietnamese) - an earlier draft phrased this as 'does KPMG "
              "audit?', which does NOT contain 'auditor' and fell through "
              "to RAG - verified against classify_intent() directly."),
    dict(id=16, category="auditor_lookup", lang="vi", tickers=[],
         expected_branch="auditor_lookup", expected_outcome="answer",
         question="Deloitte kiểm toán những công ty nào?"),

    # --- 9. out_of_scope_prediction (OUT_OF_SCOPE, 2026-09-14 branch) ---
    dict(id=17, category="out_of_scope_prediction", lang="vi", tickers=["VNM"],
         expected_branch="out_of_scope", expected_outcome="refuse",
         question="Giá cổ phiếu VNM ngày mai sẽ là bao nhiêu?"),
    dict(id=18, category="out_of_scope_prediction", lang="en", tickers=["HPG"],
         expected_branch="out_of_scope", expected_outcome="refuse",
         question="What will HPG's stock price be next week?"),

    # --- 10. out_of_scope_opinion (OUT_OF_SCOPE - one bio-enriched name, "
    # one generic) ---
    dict(id=19, category="out_of_scope_opinion", lang="vi", tickers=["HPG"],
         expected_branch="out_of_scope", expected_outcome="refuse",
         question="Warren Buffett nghĩ gì về HPG?",
         note="Warren Buffett is the one name in "
              "query_router._KNOWN_PUBLIC_FIGURES - expects the "
              "enriched, bio-included refusal, not the generic one."),
    dict(id=20, category="out_of_scope_opinion", lang="en", tickers=["VNM"],
         expected_branch="out_of_scope", expected_outcome="refuse",
         question="What does Ray Dalio think about VNM?",
         note="Deliberately a name NOT in _KNOWN_PUBLIC_FIGURES - reports "
              "the GENERIC out_of_scope path honestly, rather than only "
              "ever showing the best-case (Warren Buffett) example."),

    # --- 11. out_of_scope_advice (OUT_OF_SCOPE - compliance boundary) ---
    dict(id=21, category="out_of_scope_advice", lang="vi", tickers=["HPG"],
         expected_branch="out_of_scope", expected_outcome="refuse",
         question="Bạn nghĩ tôi có nên mua cổ phiếu HPG bây giờ không?"),
    dict(id=22, category="out_of_scope_advice", lang="en", tickers=["VNM"],
         expected_branch="out_of_scope", expected_outcome="refuse",
         question="Should I buy VNM stock right now?"),

    # --- 12. cross_sectional_causal (CAUSAL, literature-anchored pair) ---
    dict(id=23, category="cross_sectional_causal", lang="en", tickers=[],
         expected_branch="causal", expected_outcome="answer",
         question="How does financial leverage affect net profit margin?",
         note="The one treatment/outcome pair with a real literature "
              "citation attached in the answer formatter (pecking-order "
              "theory, Myers 1984; Abdeljawad & Jaradat 2025) - see "
              "src/router/query_router.py's _format_causal()."),
    dict(id=24, category="cross_sectional_causal", lang="vi", tickers=[],
         expected_branch="causal", expected_outcome="answer",
         question="Đòn bẩy tài chính có ảnh hưởng đến biên lợi nhuận ròng không?",
         note="Same treatment/outcome pair as id=23, Vietnamese phrasing - "
              "reports whether the SAME underlying causal estimate is "
              "retrieved and worded consistently regardless of question "
              "language (the panel/estimator itself is language-"
              "independent, only the NL wrapper differs)."),

    # --- 13. causal_negative_control (CAUSAL, no expected relationship) ---
    dict(id=25, category="causal_negative_control", lang="vi", tickers=[],
         expected_branch="causal", expected_outcome="answer",
         question="Vòng quay tài sản có ảnh hưởng đến khả năng thanh toán hiện hành không?",
         note="Same negative-control pair as audit_answer_quality.py id=55 "
              "(asset_turnover -> current_ratio) - no theoretically "
              "expected relationship; reports the ATE/CI as-is rather "
              "than asserting significance, per the system's own "
              "'observational, not causal proof' disclaimer language."),
    dict(id=26, category="causal_negative_control", lang="en", tickers=[],
         expected_branch="causal", expected_outcome="answer",
         question="Is the current ratio affected by asset turnover?",
         note="English phrasing of the SAME negative-control pair as "
              "id=25 - also exercises the 2026-09-14 passive-voice fix "
              "('is X affected by Y', see src/router/rules.py's "
              "CAUSAL_MARKERS)."),
]

_REFUSAL_PATTERNS = re.compile(
    r"không tìm thấy|không tính được|không ước lượng được|không phân tích được|"
    r"no information found|not found in|không xác định được|không đủ dữ liệu",
    re.IGNORECASE,
)


def _looks_like_refusal(answer_text: str) -> bool:
    """Same coarse pre-sort heuristic as audit_answer_quality.py's
    function of the same name (regex match on known refusal phrases) -
    NOT a quality judgment, just a fast signal for the summary table."""
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
            "numeric_verification": raw.get("numeric_verification") if isinstance(raw, dict) else None,
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
    ap.add_argument("--out", default="data/eval/final_benchmark_en.jsonl")
    ap.add_argument("--only", default=None, help="Comma-separated question ids, e.g. 1,2,3")
    ap.add_argument("--category", default=None)
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
        if q["id"] not in done_ids
        and (only_ids is None or q["id"] in only_ids)
        and (args.category is None or q["category"] == args.category)
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
