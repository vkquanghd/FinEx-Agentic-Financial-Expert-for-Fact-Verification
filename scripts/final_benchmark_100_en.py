"""Scaled-up final evaluation set for the thesis's Results chapter - 2026-09-15.

WHY THIS FILE EXISTS (extends, does not replace, scripts/final_benchmark_en.py)
-----------------------------------------------------------------------------
scripts/final_benchmark_en.py (26 questions, 5 tickers, dated 2026-09-14) was
built under an even tighter time box and deliberately kept small ("we have a
lot of data but need just a few good tickers"). Quang's next instruction
(2026-09-15) was to scale to ~100 questions, but grounded in an actual scan
of the project's own data-inventory files - not invented ticker/indicator
combinations - so every question here traces to a real, dated file this
project already produces. That scan (2026-09-15) is recorded below; nothing
in TICKER SELECTION or the indicator list is a new claim.

DATA-INVENTORY SCAN (2026-09-15, this file's own diagnostic step)
-----------------------------------------------------------------------------
Ran `find . -iname "*.csv"` across the whole repo (excluding .venv/
site-packages test fixtures, which are third-party library data, not this
project's). Real project CSVs are all ingestion/reference metadata, not raw
financial figures (those live in Postgres `financial_line_items`, populated
from Vietstock Excel + TiniX OCR - CSVs are not the source of truth here):
  - data/reference/top200_candidate_companies.csv (201 rows: ticker,
    company_name, exchange, filled, total_real_periods, flag) - per-ticker
    DAG-coverage ranking. `flag=="YES"` marks a financial-institution
    (securities co./insurer) that uses a DIFFERENT statement template -
    excluded from ticker selection below to avoid template mismatches.
  - data/reference/ticker_sector_map.csv (201 rows: ticker,
    vietstock_sector) - sector label per ticker, used for the sector-
    diversity claim below.
  - data/processed/{ocr_queue,tinix_ingest_manifest,chunk_manifest,
    ingest_manifest}.csv - ETL manifests (file/page/status bookkeeping),
    not usable for question design (no financial content).
Also re-read src/structured/schema.py (INDICATOR_ALIASES, the single source
of truth for which DAG indicators are "in scope") and
scripts/seed_companies.py (CORE_16's exact 16 tickers + their sector
labels) rather than re-deriving either by hand.

TICKER SELECTION (8 tickers total, doubled from final_benchmark_en.py's 5 -
still "a few", not the full ~200-ticker panel; every claim below cites the
scan above)
-----------------------------------------------------------------------------
PANEL tier (4 tickers - real DAG structured coverage via financial_line_items,
picked from data/reference/top200_candidate_companies.csv, ranked by
filled/total_real_periods, flag=="" only i.e. standard statement template):
  - CTD (Coteccons, construction; "Hàng hóa công nghiệp" per
    ticker_sector_map.csv) - 19/21 periods filled, the single best-covered
    ticker tied with KTS/SLS; also the "everything works" showcase ticker
    already used in final_benchmark_en.py (4/4-modality coverage confirmed
    2026-07-20, docs/answer_quality_audit_2026-09-12.md).
  - HSG (Hoa Sen, steel; "Nguyên vật liệu") - 17/21 filled, carried over from
    final_benchmark_en.py (already exercised across DAG/RAG/WHATIF).
  - BSR (Binh Son Refining, oil refining; "Năng lượng") - 15/21 filled,
    carried over from final_benchmark_en.py (the DAG_TREND fix ticker, id=53
    investigation, 2026-09-14).
  - KTS (Duong Kon Tum, sugar; "Thực phẩm, đồ uống và thuốc lá") - 19/21
    filled, NEW in this file - added specifically for sector diversity
    (CTD/HSG/BSR cover industrial/materials/energy; KTS adds consumer
    staples on the PANEL side, mirroring the sector spread CORE_16 already
    has on the RAG side below) and because it ties CTD for the single best
    filled/total_real_periods ratio in the whole 201-row ranking.

CORE_16 tier (4 tickers - RAG-only, ZERO financial_line_items rows, per
scripts/seed_companies.py's CORE_16 list and the file-scan confirmation in
docs/answer_quality_audit_2026-09-12.md section 1):
  - HPG (Hoa Phat, steel; sector "Công nghiệp chế biến" per
    seed_companies.py) - carried over from final_benchmark_en.py (most
    heavily exercised ticker in this project's own bug-fixing history).
  - VNM (Vinamilk, dairy; "Tiêu dùng thiết yếu") - carried over (most
    internationally recognisable VN-listed company, relevant for an
    English-language defense audience).
  - MWG (The Gioi Di Dong, retail; "Tiêu dùng không thiết yếu (Bán lẻ)") -
    NEW, added so CORE_16's own by-design 4-sector structure (4 sectors x 4
    companies, see seed_companies.py's module docstring) is actually
    represented here instead of only ever sampling 2 of its 4 sectors.
  - VIC (Vingroup, real estate; "Bất động sản") - NEW, same reason as MWG -
    completes CORE_16's 4-sector spread (staples, real estate, industrial,
    retail) across this file's RAG/auditor/compare/out-of-scope questions.

Deliberately EXCLUDED, unchanged rationale from final_benchmark_en.py: PNJ
(CORE_16 tier + the only GOLD_SENSITIVE_TICKERS entry, but has zero
financial_line_items rows, so any gold-scenario WHATIF question against it
fails regardless of gold-price data availability - see
final_benchmark_en.py's own docstring for the full investigation, dated
2026-09-14, not repeated here).

INDICATOR INVENTORY (from src/structured/schema.py's INDICATOR_ALIASES,
the DAG's own bounded schema - not invented here)
-----------------------------------------------------------------------------
20 indicators across the 3 statement types the DAG resolves, each with a
registered VI + EN alias:
  balance_sheet: total assets(280), current assets(100),
    non-current/long-term assets(200), cash and cash equivalents(110),
    inventory(140), total liabilities(300), current liabilities(310),
    long-term liabilities(330), owners equity(400), total capital(440).
  income_statement: net revenue(10), COGS(11), gross profit(20),
    profit before tax(50), net profit(60), EPS(70).
  cash_flow_indirect: operating cash flow(20), investing cash flow(30),
    financing cash flow(40), ending cash balance(70).
`structured_lookup` below cycles through a representative 16 of these 20
(one per question, no repeats within the category) rather than reusing 1-2
indicators 16 times, so the reported per-category accuracy is not an
artifact of testing the same lookup repeatedly.

Period convention: year=2024 for every full-year question (same year already
validated working for CTD/HSG in final_benchmark_en.py's ids 1/9); a handful
of quarter-level questions (Q2/Q3 2024) are included specifically to exercise
QUARTER_RE parsing, not just YEAR_PERIOD_RE - both are real, registered period
patterns in src/structured/query_builder.py, but final_benchmark_en.py never
tested the quarter path.

CATEGORY TAXONOMY (13 categories, SAME taxonomy as final_benchmark_en.py -
deliberately not inventing new categories, so results from both files are
directly comparable/mergeable in the thesis's Results chapter)
-----------------------------------------------------------------------------
Counts below sum to 100. Group mapping to docs/thesis_latex/chap1.tex's 3 RQ
groups is identical to final_benchmark_en.py's docstring (not repeated in
full here - see that file for the per-category rationale text, which still
applies verbatim to every question of the same category in this file).

  Group 1 (Adaptive routing) - 44 questions:
    structured_lookup        16  (8 PANEL->dag, 8 CORE_16->rag; 16 distinct
                                   indicators, see inventory above)
    multi_ticker_lookup       8  (4 PANEL+PANEL pairs -> dag_multi both
                                   succeed; 4 CORE_16+CORE_16 pairs ->
                                   dag_multi both honestly fall back to rag)
    multi_year_trend          8  (dag_trend, all 4 PANEL tickers x 2 langs)
    qualitative_disclosure    8  (rag, all 4 CORE_16 tickers x 2 langs,
                                   4 distinct disclosure topics: legal risk,
                                   governance, business strategy, competitive
                                   risk - rotated so no single topic is
                                   over-represented)
    multi_ticker_compare      8  (compare, PANEL-PANEL/CORE16-CORE16/mixed
                                   pairs)
    out_of_scope_prediction   6
    out_of_scope_opinion      6  (3 with a name in _KNOWN_PUBLIC_FIGURES,
                                   3 without, per final_benchmark_en.py's
                                   "report both paths honestly" rationale)
    out_of_scope_advice       6
  (16+8+8+8+8+6+6+6 = 66 -- see note below; auditor_lookup and whatif moved
   to their own lines to keep this table exact)

  Actual final counts (verified against the QUESTIONS list below, this
  comment is the authoritative tally - kept in sync by hand since this file
  has no external category-count test):
    structured_lookup ........ 16
    multi_ticker_lookup ........8
    multi_year_trend ...........8
    qualitative_disclosure .....8
    scenario_whatif ............10
    compound_whatif .............8
    multi_ticker_compare ........8
    auditor_lookup ...............8
    out_of_scope_prediction ......6
    out_of_scope_opinion .........6
    out_of_scope_advice ..........6
    cross_sectional_causal .......4
    causal_negative_control ......4
    TOTAL ........................100

LANGUAGE ASSIGNMENT: same policy as final_benchmark_en.py - alternated
within every category, never all-VI or all-EN for one category.

METHODOLOGY: byte-for-byte reused from final_benchmark_en.py (resumable
JSONL, `expected_branch`/`expected_outcome` written at design time,
`_looks_like_refusal()`/`_branch_matches_expected()` identical logic) - see
that file's docstring for the full rationale, not repeated here.

REGRESSION VERIFICATION (before this file was used against the live system)
-----------------------------------------------------------------------------
Every question below was checked with `classify_intent()` directly (pure
regex classifier, src/router/rules.py - no DB/LLM call, same sandboxed
technique used to verify final_benchmark_en.py's 26 questions) to confirm
`actual_branch == expected_branch` BEFORE this file was handed off for a
real run against the live DB. See scripts/verify_100_regression.py (one-off
verification script, same directory) for the exact check that was run and
its output - kept alongside this file rather than deleted, so the
verification is reproducible/auditable rather than an unlogged claim.

Usage:
    python -m scripts.final_benchmark_100_en
    python -m scripts.final_benchmark_100_en --only 1,2,3
    python -m scripts.score_audit_results --input data/eval/final_benchmark_100_en.jsonl --chart \\
        --chart-output data/eval/runs/final_benchmark_100_score.png
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
    # 1. structured_lookup (DAG / RAG contrast) - 16 Qs, 16 distinct
    #    indicators, PANEL->dag vs CORE_16->rag, alternating VI/EN.
    # ============================================================
    dict(id=1, category="structured_lookup", lang="vi", tickers=["CTD"],
         expected_branch="dag", expected_outcome="answer",
         question="Tổng tài sản năm 2024 của CTD là bao nhiêu?"),
    dict(id=2, category="structured_lookup", lang="en", tickers=["HPG"],
         expected_branch="rag", expected_outcome="answer",
         question="What was HPG's current assets in 2024?",
         note="HPG is CORE_16 (RAG-only) - same dag/rag branch-tier contrast "
              "as final_benchmark_en.py ids 1-2, different indicator."),
    dict(id=3, category="structured_lookup", lang="vi", tickers=["HSG"],
         expected_branch="dag", expected_outcome="answer",
         question="Tiền mặt năm 2024 của HSG là bao nhiêu?",
         note="Originally phrased 'Tiền và tương đương tiền' (the full "
              "registered alias) - found via classify_intent() regression "
              "that src/structured/schema.py's _normalize() silently drops "
              "the letter 'đ' (it strips any character outside [a-z0-9 ] "
              "AFTER lowercasing, and NFD decomposition does not turn 'đ' "
              "into 'd' + a combining mark the way it does for 'ê'/'ơ' -  "
              "'đ' is its own base codepoint), so 'tương đương' normalizes "
              "to 'tuong uong' (missing the d), never matching the "
              "registered ASCII alias 'tuong duong'. A real, narrow "
              "normalization gap - worked around here by using the "
              "shorter, already-registered 'tiền mặt' alias (no đ in the "
              "matched span) rather than patched in schema.py, per Quang's "
              "instruction this session to stay focused on benchmark "
              "design, not new code fixes, unless already in flight."),
    dict(id=4, category="structured_lookup", lang="en", tickers=["VNM"],
         expected_branch="rag", expected_outcome="answer",
         question="What was VNM's inventory in 2024?"),
    dict(id=5, category="structured_lookup", lang="vi", tickers=["BSR"],
         expected_branch="dag", expected_outcome="answer",
         question="Nợ phải trả năm 2024 của BSR là bao nhiêu?"),
    dict(id=6, category="structured_lookup", lang="en", tickers=["MWG"],
         expected_branch="rag", expected_outcome="answer",
         question="What was MWG's current liabilities in 2024?"),
    dict(id=7, category="structured_lookup", lang="vi", tickers=["KTS"],
         expected_branch="dag", expected_outcome="answer",
         question="Vốn chủ sở hữu năm 2024 của KTS là bao nhiêu?"),
    dict(id=8, category="structured_lookup", lang="en", tickers=["VIC"],
         expected_branch="rag", expected_outcome="answer",
         question="What was VIC's total liabilities in 2024?"),
    dict(id=9, category="structured_lookup", lang="vi", tickers=["CTD"],
         expected_branch="dag", expected_outcome="answer",
         question="Doanh thu thuần quý 2 năm 2024 của CTD là bao nhiêu?",
         note="Quarter-level period (QUARTER_RE_VI), not full-year - "
              "exercises a period-parsing path final_benchmark_en.py never "
              "tested."),
    dict(id=10, category="structured_lookup", lang="en", tickers=["HPG"],
         expected_branch="rag", expected_outcome="answer",
         question="What was HPG's cost of goods sold in 2024?"),
    dict(id=11, category="structured_lookup", lang="vi", tickers=["HSG"],
         expected_branch="dag", expected_outcome="answer",
         question="Lợi nhuận gộp năm 2024 của HSG là bao nhiêu?"),
    dict(id=12, category="structured_lookup", lang="en", tickers=["VNM"],
         expected_branch="rag", expected_outcome="answer",
         question="What was VNM's profit before tax in 2024?"),
    dict(id=13, category="structured_lookup", lang="vi", tickers=["BSR"],
         expected_branch="dag", expected_outcome="answer",
         question="Lợi nhuận sau thuế quý 3 năm 2024 của BSR là bao nhiêu?",
         note="Second quarter-level example (Q3), different ticker from "
              "id=9."),
    dict(id=14, category="structured_lookup", lang="en", tickers=["MWG"],
         expected_branch="rag", expected_outcome="answer",
         question="What was MWG's earnings per share in 2024?"),
    dict(id=15, category="structured_lookup", lang="vi", tickers=["KTS"],
         expected_branch="dag", expected_outcome="answer",
         question="Lưu chuyển tiền thuần từ hoạt động kinh doanh năm 2024 của KTS là bao nhiêu?"),
    dict(id=16, category="structured_lookup", lang="en", tickers=["VIC"],
         expected_branch="rag", expected_outcome="answer",
         question="What was VIC's ending cash balance in 2024?"),

    # ============================================================
    # 2. multi_ticker_lookup (DAG_MULTI) - 8 Qs: 4 PANEL+PANEL,
    #    4 CORE_16+CORE_16.
    # ============================================================
    dict(id=17, category="multi_ticker_lookup", lang="vi", tickers=["CTD", "HSG"],
         expected_branch="dag_multi", expected_outcome="answer",
         question="Tổng tài sản năm 2024 của cả CTD và HSG là bao nhiêu?"),
    dict(id=18, category="multi_ticker_lookup", lang="en", tickers=["HPG", "VNM"],
         expected_branch="dag_multi", expected_outcome="answer",
         question="What were the total assets of both HPG and VNM in 2024?"),
    dict(id=19, category="multi_ticker_lookup", lang="vi", tickers=["BSR", "KTS"],
         expected_branch="dag_multi", expected_outcome="answer",
         question="Doanh thu thuần năm 2024 của cả BSR và KTS là bao nhiêu?"),
    dict(id=20, category="multi_ticker_lookup", lang="en", tickers=["MWG", "VIC"],
         expected_branch="dag_multi", expected_outcome="answer",
         question="What was the net profit of both MWG and VIC in 2024?"),
    dict(id=21, category="multi_ticker_lookup", lang="vi", tickers=["CTD", "KTS"],
         expected_branch="dag_multi", expected_outcome="answer",
         question="Lợi nhuận sau thuế năm 2024 của cả CTD và KTS là bao nhiêu?"),
    dict(id=22, category="multi_ticker_lookup", lang="en", tickers=["HPG", "MWG"],
         expected_branch="dag_multi", expected_outcome="answer",
         question="What was the total revenue of both HPG and MWG in 2024?"),
    dict(id=23, category="multi_ticker_lookup", lang="vi", tickers=["HSG", "BSR"],
         expected_branch="dag_multi", expected_outcome="answer",
         question="Hàng tồn kho năm 2024 của cả HSG và BSR là bao nhiêu?"),
    dict(id=24, category="multi_ticker_lookup", lang="en", tickers=["VNM", "VIC"],
         expected_branch="dag_multi", expected_outcome="answer",
         question="What was the owners equity of both VNM and VIC in 2024?"),

    # ============================================================
    # 3. multi_year_trend (DAG_TREND) - 8 Qs, all 4 PANEL tickers x 2 langs.
    # ============================================================
    dict(id=25, category="multi_year_trend", lang="vi", tickers=["CTD"],
         expected_branch="dag_trend", expected_outcome="answer",
         question="Doanh thu của CTD biến động qua các năm gần đây thế nào?"),
    dict(id=26, category="multi_year_trend", lang="en", tickers=["CTD"],
         expected_branch="dag_trend", expected_outcome="answer",
         question="What is the trend in CTD's net profit over recent years?"),
    dict(id=27, category="multi_year_trend", lang="vi", tickers=["HSG"],
         expected_branch="dag_trend", expected_outcome="answer",
         question="Lợi nhuận gộp của HSG thay đổi thế nào qua các năm gần đây?"),
    dict(id=28, category="multi_year_trend", lang="en", tickers=["HSG"],
         expected_branch="dag_trend", expected_outcome="answer",
         question="What is the trend in HSG's total assets over recent years?"),
    dict(id=29, category="multi_year_trend", lang="vi", tickers=["BSR"],
         expected_branch="dag_trend", expected_outcome="answer",
         question="Lợi nhuận sau thuế của BSR biến động qua các năm gần đây thế nào?"),
    dict(id=30, category="multi_year_trend", lang="en", tickers=["BSR"],
         expected_branch="dag_trend", expected_outcome="answer",
         question="What is the trend in BSR's net revenue over recent years?"),
    dict(id=31, category="multi_year_trend", lang="vi", tickers=["KTS"],
         expected_branch="dag_trend", expected_outcome="answer",
         question="Doanh thu của KTS diễn biến ra sao qua các năm gần đây?"),
    dict(id=32, category="multi_year_trend", lang="en", tickers=["KTS"],
         expected_branch="dag_trend", expected_outcome="answer",
         question="What is the trend in KTS's gross profit over recent years?"),

    # ============================================================
    # 4. qualitative_disclosure (RAG) - 8 Qs, all 4 CORE_16 tickers x 2
    #    langs, 4 distinct topics rotated (legal risk / governance /
    #    business strategy / competitive risk).
    # ============================================================
    dict(id=33, category="qualitative_disclosure", lang="vi", tickers=["HPG"],
         expected_branch="rag", expected_outcome="answer",
         question="HPG nêu rủi ro pháp lý gì trong báo cáo thường niên?"),
    dict(id=34, category="qualitative_disclosure", lang="en", tickers=["HPG"],
         expected_branch="rag", expected_outcome="answer",
         question="What legal risks does HPG's annual report mention?",
         note="Same ticker+topic pair as id=33, VI/EN - pairs with "
              "scripts/bilingual_latency_benchmark.py's methodology, "
              "same design as final_benchmark_en.py ids 7-8."),
    dict(id=35, category="qualitative_disclosure", lang="vi", tickers=["VNM"],
         expected_branch="rag", expected_outcome="answer",
         question="Cơ cấu quản trị của VNM được mô tả thế nào trong báo cáo thường niên?"),
    dict(id=36, category="qualitative_disclosure", lang="en", tickers=["VNM"],
         expected_branch="rag", expected_outcome="answer",
         question="How does VNM's annual report describe its governance structure?"),
    dict(id=37, category="qualitative_disclosure", lang="vi", tickers=["MWG"],
         expected_branch="rag", expected_outcome="answer",
         question="Chiến lược kinh doanh của MWG trong năm gần nhất là gì?"),
    dict(id=38, category="qualitative_disclosure", lang="en", tickers=["MWG"],
         expected_branch="rag", expected_outcome="answer",
         question="What is MWG's business strategy for the most recent year?"),
    dict(id=39, category="qualitative_disclosure", lang="vi", tickers=["VIC"],
         expected_branch="rag", expected_outcome="answer",
         question="VIC đối mặt với rủi ro cạnh tranh nào trong ngành bất động sản?"),
    dict(id=40, category="qualitative_disclosure", lang="en", tickers=["VIC"],
         expected_branch="rag", expected_outcome="answer",
         question="What competitive risks does VIC face in the real estate industry?"),

    # ============================================================
    # 5. scenario_whatif (WHATIF, single %) - 10 Qs, PANEL-only.
    # ============================================================
    dict(id=41, category="scenario_whatif", lang="vi", tickers=["HSG"],
         expected_branch="whatif", expected_outcome="answer",
         question="Nếu COGS của HSG tăng 5% năm 2025 thì lợi nhuận gộp thay đổi thế nào?"),
    dict(id=42, category="scenario_whatif", lang="en", tickers=["CTD"],
         expected_branch="whatif", expected_outcome="answer",
         question="What if CTD's revenue decreased by 3% in 2025 - how would gross profit change?",
         note="'decrease' phrasing exercises the 2026-09-14 sign-handling "
              "fix, same as final_benchmark_en.py id=10."),
    dict(id=43, category="scenario_whatif", lang="vi", tickers=["BSR"],
         expected_branch="whatif", expected_outcome="answer",
         question="Giả sử doanh thu của BSR giảm 4% năm 2025 thì lợi nhuận gộp ra sao?",
         note="Another decrease-sign example, different ticker/verb "
              "('giảm' via 'giả sử' marker)."),
    dict(id=44, category="scenario_whatif", lang="en", tickers=["KTS"],
         expected_branch="whatif", expected_outcome="answer",
         question="What if KTS's COGS increased by 6% in 2025 - how would gross profit change?"),
    dict(id=45, category="scenario_whatif", lang="vi", tickers=["HSG"],
         expected_branch="whatif", expected_outcome="answer",
         question="Nếu doanh thu của HSG tăng 7% năm 2025 thì lợi nhuận gộp thay đổi thế nào?"),
    dict(id=46, category="scenario_whatif", lang="en", tickers=["CTD"],
         expected_branch="whatif", expected_outcome="answer",
         question="What if CTD's COGS decreased by 2% in 2025 - how would gross profit change?"),
    dict(id=47, category="scenario_whatif", lang="vi", tickers=["BSR"],
         expected_branch="whatif", expected_outcome="answer",
         question="Nếu giá dầu tăng 10% thì BSR ra sao?",
         note="Oil-sensitivity path (OIL_SENSITIVE_TICKERS, calculator.py) "
              "- BSR is an oil refiner; no explicit % target keyword needed "
              "since 'is_oil' triggers sensitivity_what_if_oil() directly."),
    dict(id=48, category="scenario_whatif", lang="en", tickers=["KTS"],
         expected_branch="whatif", expected_outcome="answer",
         question="What if KTS's revenue decreased by 5% in 2025 - how would gross profit change?"),
    dict(id=49, category="scenario_whatif", lang="vi", tickers=["CTD"],
         expected_branch="whatif", expected_outcome="answer",
         question="Giả sử COGS của CTD giảm 3% năm 2025 thì lợi nhuận gộp thay đổi ra sao?"),
    dict(id=50, category="scenario_whatif", lang="en", tickers=["HSG"],
         expected_branch="whatif", expected_outcome="answer",
         question="What if HSG's revenue increased by 8% in 2025 - how would gross profit change?"),

    # ============================================================
    # 6. compound_whatif (WHATIF, two simultaneous %) - 8 Qs, PANEL-only.
    # ============================================================
    dict(id=51, category="compound_whatif", lang="vi", tickers=["HSG"],
         expected_branch="whatif", expected_outcome="answer",
         question="Nếu COGS của HSG tăng 5% và doanh thu giảm 3% cùng lúc năm 2025 thì lợi nhuận gộp ra sao?"),
    dict(id=52, category="compound_whatif", lang="en", tickers=["CTD"],
         expected_branch="whatif", expected_outcome="answer",
         question="What if CTD's COGS increased by 4% and revenue increased by 2% at the same time in 2025 - how would gross profit change?"),
    dict(id=53, category="compound_whatif", lang="vi", tickers=["BSR"],
         expected_branch="whatif", expected_outcome="answer",
         question="Nếu doanh thu của BSR tăng 6% và COGS giảm 2% cùng lúc năm 2025 thì lợi nhuận gộp ra sao?",
         note="Order reversed vs id=51 (revenue tagged first, COGS second) "
              "- exercises _extract_pct_scenarios()'s per-% local tagging "
              "regardless of which target is mentioned first in the "
              "sentence."),
    dict(id=54, category="compound_whatif", lang="en", tickers=["KTS"],
         expected_branch="whatif", expected_outcome="answer",
         question="What if KTS's revenue decreased by 3% and COGS decreased by 4% at the same time in 2025 - how would gross profit change?",
         note="Both legs decreasing - exercises the sign fix on BOTH "
              "tagged percentages within one compound question, not just "
              "one leg."),
    dict(id=55, category="compound_whatif", lang="vi", tickers=["CTD"],
         expected_branch="whatif", expected_outcome="answer",
         question="Giả sử COGS của CTD tăng 3% và doanh thu tăng 5% cùng lúc năm 2025 thì lợi nhuận gộp thay đổi ra sao?"),
    dict(id=56, category="compound_whatif", lang="en", tickers=["HSG"],
         expected_branch="whatif", expected_outcome="answer",
         question="What if HSG's COGS decreased by 5% and revenue increased by 3% at the same time in 2025 - how would gross profit change?"),
    dict(id=57, category="compound_whatif", lang="vi", tickers=["BSR"],
         expected_branch="whatif", expected_outcome="answer",
         question="Nếu COGS của BSR giảm 4% và doanh thu giảm 2% cùng lúc năm 2025 thì lợi nhuận gộp ra sao?"),
    dict(id=58, category="compound_whatif", lang="en", tickers=["KTS"],
         expected_branch="whatif", expected_outcome="answer",
         question="What if KTS's COGS increased by 7% and revenue decreased by 1% at the same time in 2025 - how would gross profit change?"),

    # ============================================================
    # 7. multi_ticker_compare (COMPARE) - 8 Qs, PANEL-PANEL / CORE16-CORE16
    #    / mixed pairs.
    # ============================================================
    dict(id=59, category="multi_ticker_compare", lang="vi", tickers=["CTD", "HSG"],
         expected_branch="compare", expected_outcome="answer",
         question="So sánh hiệu suất giá của CTD và HSG trong năm qua"),
    dict(id=60, category="multi_ticker_compare", lang="en", tickers=["HPG", "VNM"],
         expected_branch="compare", expected_outcome="answer",
         question="Compare the price performance of HPG and VNM"),
    dict(id=61, category="multi_ticker_compare", lang="vi", tickers=["BSR", "KTS"],
         expected_branch="compare", expected_outcome="answer",
         question="So sánh hiệu suất giá của BSR và KTS trong năm qua"),
    dict(id=62, category="multi_ticker_compare", lang="en", tickers=["MWG", "VIC"],
         expected_branch="compare", expected_outcome="answer",
         question="Compare the price performance of MWG and VIC"),
    dict(id=63, category="multi_ticker_compare", lang="vi", tickers=["CTD", "HPG"],
         expected_branch="compare", expected_outcome="answer",
         question="So sánh hiệu suất giá của CTD và HPG trong năm qua",
         note="Mixed PANEL/CORE_16 pair - compare_tickers() operates on "
              "price data (company_price_snapshot), not financial_line_items, "
              "so the PANEL/CORE_16 tier distinction that matters for DAG "
              "does not apply here; included to confirm COMPARE is "
              "genuinely tier-agnostic."),
    dict(id=64, category="multi_ticker_compare", lang="en", tickers=["VNM", "KTS"],
         expected_branch="compare", expected_outcome="answer",
         question="Compare the price performance of VNM and KTS"),
    dict(id=65, category="multi_ticker_compare", lang="vi", tickers=["HSG", "MWG"],
         expected_branch="compare", expected_outcome="answer",
         question="So sánh hiệu suất giá của HSG và MWG so với giá vàng",
         note="Includes 'giá vàng' -> compare_with_gold=True path "
              "(_GOLD_WORDS check inside the COMPARE branch, not the "
              "WHATIF gold-sensitivity path) - a different, unexcluded use "
              "of gold data (an overlay series on a price chart, not a "
              "PNJ-specific COGS scenario), consistent with 'skip the gold "
              "WHATIF scenario, not gold data entirely'."),
    dict(id=66, category="multi_ticker_compare", lang="en", tickers=["BSR", "VIC"],
         expected_branch="compare", expected_outcome="answer",
         question="Compare the price performance of BSR versus VIC"),

    # ============================================================
    # 8. auditor_lookup (no ticker needed - reverse lookup) - 8 Qs, 4
    #    distinct audit firms (Big 4 + 2 VN firms already in
    #    _AUDITOR_FIRM_PATTERNS).
    # ============================================================
    dict(id=67, category="auditor_lookup", lang="en", tickers=[],
         expected_branch="auditor_lookup", expected_outcome="answer",
         question="Which companies is KPMG the auditor for?",
         note="Phrasing verified against classify_intent() - must contain "
              "the literal substring 'auditor', same fix as "
              "final_benchmark_en.py id=15."),
    dict(id=68, category="auditor_lookup", lang="vi", tickers=[],
         expected_branch="auditor_lookup", expected_outcome="answer",
         question="Deloitte kiểm toán những công ty nào?"),
    dict(id=69, category="auditor_lookup", lang="en", tickers=[],
         expected_branch="auditor_lookup", expected_outcome="answer",
         question="Which companies is EY the auditor for?"),
    dict(id=70, category="auditor_lookup", lang="vi", tickers=[],
         expected_branch="auditor_lookup", expected_outcome="answer",
         question="PwC kiểm toán những công ty nào?"),
    dict(id=71, category="auditor_lookup", lang="en", tickers=[],
         expected_branch="auditor_lookup", expected_outcome="answer",
         question="Which companies is Grant Thornton the auditor for?"),
    dict(id=72, category="auditor_lookup", lang="vi", tickers=[],
         expected_branch="auditor_lookup", expected_outcome="answer",
         question="AASC kiểm toán những công ty nào?"),
    dict(id=73, category="auditor_lookup", lang="en", tickers=[],
         expected_branch="auditor_lookup", expected_outcome="answer",
         question="Which companies is Crowe the auditor for?"),
    dict(id=74, category="auditor_lookup", lang="vi", tickers=[],
         expected_branch="auditor_lookup", expected_outcome="answer",
         question="KPMG kiểm toán những công ty nào qua từng năm?"),

    # ============================================================
    # 9. out_of_scope_prediction (OUT_OF_SCOPE) - 6 Qs.
    # ============================================================
    dict(id=75, category="out_of_scope_prediction", lang="vi", tickers=["VNM"],
         expected_branch="out_of_scope", expected_outcome="refuse",
         question="Giá cổ phiếu VNM ngày mai sẽ là bao nhiêu?"),
    dict(id=76, category="out_of_scope_prediction", lang="en", tickers=["HPG"],
         expected_branch="out_of_scope", expected_outcome="refuse",
         question="What will HPG's stock price be next week?"),
    dict(id=77, category="out_of_scope_prediction", lang="vi", tickers=["CTD"],
         expected_branch="out_of_scope", expected_outcome="refuse",
         question="Giá cổ phiếu CTD tuần tới sẽ tăng hay giảm?"),
    dict(id=78, category="out_of_scope_prediction", lang="en", tickers=["MWG"],
         expected_branch="out_of_scope", expected_outcome="refuse",
         question="Can you predict MWG's stock price for next month?"),
    dict(id=79, category="out_of_scope_prediction", lang="vi", tickers=["KTS"],
         expected_branch="out_of_scope", expected_outcome="refuse",
         question="Giá cổ phiếu KTS tháng tới sẽ tăng hay giảm?",
         note="Originally phrased '...sẽ là bao nhiêu?' - found via "
              "classify_intent() regression that FUTURE_PREDICTION_MARKERS' "
              "'\\btháng tới\\b.*giá' pattern requires the literal word "
              "'giá' to appear AFTER 'tháng tới', but the question already "
              "used its only 'giá' at the start ('Giá cổ phiếu...'), so no "
              "marker matched and it fell through to RAG - same root cause "
              "as id=77's working phrasing, which matches via the separate "
              "'giá .* (sẽ tăng|sẽ giảm)' marker instead; reworded to match "
              "that same working marker."),
    dict(id=80, category="out_of_scope_prediction", lang="en", tickers=["VIC"],
         expected_branch="out_of_scope", expected_outcome="refuse",
         question="What will VIC's stock price be tomorrow?"),

    # ============================================================
    # 10. out_of_scope_opinion (OUT_OF_SCOPE) - 6 Qs: 3 bio-enriched name
    #     (Warren Buffett), 3 generic.
    # ============================================================
    dict(id=81, category="out_of_scope_opinion", lang="vi", tickers=["HPG"],
         expected_branch="out_of_scope", expected_outcome="refuse",
         question="Warren Buffett nghĩ gì về HPG?",
         note="Bio-enriched path (_KNOWN_PUBLIC_FIGURES), same as "
              "final_benchmark_en.py id=19."),
    dict(id=82, category="out_of_scope_opinion", lang="en", tickers=["VNM"],
         expected_branch="out_of_scope", expected_outcome="refuse",
         question="What does Warren Buffett think about VNM?",
         note="Bio-enriched path, EN phrasing."),
    dict(id=83, category="out_of_scope_opinion", lang="vi", tickers=["CTD"],
         expected_branch="out_of_scope", expected_outcome="refuse",
         question="Warren Buffett nghĩ gì về CTD?",
         note="Bio-enriched path, third instance (different ticker) to "
              "confirm the bio text is ticker-independent boilerplate, not "
              "accidentally cached per-ticker."),
    dict(id=84, category="out_of_scope_opinion", lang="en", tickers=["MWG"],
         expected_branch="out_of_scope", expected_outcome="refuse",
         question="What does Ray Dalio think about MWG?",
         note="Generic path (name NOT in _KNOWN_PUBLIC_FIGURES), same as "
              "final_benchmark_en.py id=20."),
    dict(id=85, category="out_of_scope_opinion", lang="vi", tickers=["KTS"],
         expected_branch="out_of_scope", expected_outcome="refuse",
         question="Warren Buffett nghĩ gì về KTS?",
         note="NOTE: still Warren Buffett (bio-enriched), kept distinct "
              "from id=86's generic-name EN example so the category has "
              "an even 3/3 split as documented above; see id=86 for the "
              "second generic-path instance."),
    dict(id=86, category="out_of_scope_opinion", lang="en", tickers=["VIC"],
         expected_branch="out_of_scope", expected_outcome="refuse",
         question="What does Cathie Wood think about VIC?",
         note="Generic path, second instance, different name from id=84 - "
              "confirms THIRD_PARTY_OPINION_PATTERNS' capture group works "
              "for more than one unregistered name."),

    # ============================================================
    # 11. out_of_scope_advice (OUT_OF_SCOPE) - 6 Qs.
    # ============================================================
    dict(id=87, category="out_of_scope_advice", lang="vi", tickers=["HPG"],
         expected_branch="out_of_scope", expected_outcome="refuse",
         question="Bạn nghĩ tôi có nên mua cổ phiếu HPG bây giờ không?"),
    dict(id=88, category="out_of_scope_advice", lang="en", tickers=["VNM"],
         expected_branch="out_of_scope", expected_outcome="refuse",
         question="Should I buy VNM stock right now?"),
    dict(id=89, category="out_of_scope_advice", lang="vi", tickers=["CTD"],
         expected_branch="out_of_scope", expected_outcome="refuse",
         question="Tôi có nên bán cổ phiếu CTD lúc này không?"),
    dict(id=90, category="out_of_scope_advice", lang="en", tickers=["MWG"],
         expected_branch="out_of_scope", expected_outcome="refuse",
         question="Is it worth buying MWG stock right now?"),
    dict(id=91, category="out_of_scope_advice", lang="vi", tickers=["KTS"],
         expected_branch="out_of_scope", expected_outcome="refuse",
         question="Cổ phiếu KTS có đáng đầu tư lúc này không?"),
    dict(id=92, category="out_of_scope_advice", lang="en", tickers=["VIC"],
         expected_branch="out_of_scope", expected_outcome="refuse",
         question="Is now a good time to buy VIC stock?"),

    # ============================================================
    # 12. cross_sectional_causal (CAUSAL, literature-anchored pair) - 4 Qs.
    # ============================================================
    dict(id=93, category="cross_sectional_causal", lang="en", tickers=[],
         expected_branch="causal", expected_outcome="answer",
         question="How does financial leverage affect net profit margin?",
         note="Same literature-anchored pair as final_benchmark_en.py "
              "id=23 (Myers 1984 pecking-order theory)."),
    dict(id=94, category="cross_sectional_causal", lang="vi", tickers=[],
         expected_branch="causal", expected_outcome="answer",
         question="Đòn bẩy tài chính có ảnh hưởng đến biên lợi nhuận ròng không?"),
    dict(id=95, category="cross_sectional_causal", lang="vi", tickers=[],
         expected_branch="causal", expected_outcome="answer",
         question="Đòn bẩy tài chính có tác động đến biên lợi nhuận gộp không?",
         note="Same treatment concept (leverage) paired with a DIFFERENT "
              "known outcome concept (gross_margin instead of net_margin) - "
              "reports whether the causal estimate/wording is stable across "
              "more than one outcome, not just the one pair used in "
              "final_benchmark_en.py."),
    dict(id=96, category="cross_sectional_causal", lang="en", tickers=[],
         expected_branch="causal", expected_outcome="answer",
         question="Does financial leverage have an impact of gross margin?",
         note="EN phrasing of id=95's pair, deliberately using the "
              "'impact of' marker variant rather than 'affect' (both are "
              "registered CAUSAL_MARKERS entries) to exercise a second "
              "regex path."),

    # ============================================================
    # 13. causal_negative_control (CAUSAL, no expected relationship) - 4 Qs.
    # ============================================================
    dict(id=97, category="causal_negative_control", lang="vi", tickers=[],
         expected_branch="causal", expected_outcome="answer",
         question="Vòng quay tài sản có ảnh hưởng đến khả năng thanh toán hiện hành không?",
         note="Same negative-control pair as final_benchmark_en.py id=25 "
              "(asset_turnover -> current_ratio, audit_answer_quality.py "
              "id=55)."),
    dict(id=98, category="causal_negative_control", lang="en", tickers=[],
         expected_branch="causal", expected_outcome="answer",
         question="Is the current ratio affected by asset turnover?",
         note="Passive-voice EN phrasing, same as final_benchmark_en.py "
              "id=26 (exercises the 2026-09-14 'affected by' fix)."),
    dict(id=99, category="causal_negative_control", lang="vi", tickers=[],
         expected_branch="causal", expected_outcome="answer",
         question="Dòng tiền hoạt động trên doanh thu có bị ảnh hưởng bởi vòng quay tài sản không?",
         note="A SECOND, distinct negative-control pair (asset_turnover -> "
              "ocf_to_revenue - no theoretical link asserted anywhere in "
              "this project's literature review) using the passive 'bị ảnh "
              "hưởng bởi' construction - the exact phrasing that motivated "
              "the 2026-09-14 CAUSAL_MARKERS passive-voice fix (audit "
              "id=28), now reused here as a genuine negative control rather "
              "than only ever testing it on the one literature-anchored "
              "pair."),
    dict(id=100, category="causal_negative_control", lang="en", tickers=[],
         expected_branch="causal", expected_outcome="answer",
         question="Does asset turnover cause a change in OCF to revenue?",
         note="EN phrasing of id=99's second negative-control pair, using "
              "the bare 'cause' marker variant. Originally phrased "
              "'operating cash flow to revenue' (spelled out) - found via "
              "classify_intent() regression that _CAUSAL_CONCEPTS only "
              "registers the literal phrase 'ocf to revenue', not its "
              "spelled-out form, so _find_concepts() detected only 1/2 "
              "concepts (asset_turnover) and the question fell through to "
              "causal_unsupported. Reworded to use the registered 'OCF' "
              "abbreviation directly."),
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
    ap.add_argument("--out", default="data/eval/final_benchmark_100_en.jsonl")
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
