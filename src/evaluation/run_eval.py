"""Evaluation runner - loads data/eval/benchmark.jsonl, runs each question
through the real Router/RAG/What-if pipeline, and reports the metrics fixed
in docs/Thesis_Direction_Redesign.pdf section 7.

Two modes, since "routing" questions are pure classify_intent() logic (no
DB/LLM needed) but every other category needs the real Postgres/Ollama
stack this sandbox doesn't have:

    python3 -m src.evaluation.run_eval --routing-only
        Runs ONLY the "routing" category - classify_intent() is pure logic,
        this works with no DB/LLM at all. Safe to run anywhere, including
        the sandbox that wrote this file.

    python3 -m src.evaluation.run_eval
        Runs the FULL benchmark (all categories) - needs a real
        Postgres/Ollama stack. Run this on the project owner's
        machine, not in a sandbox with no such stack.

Week 8 additions (2026-07-27) - added 4 more categories, one per remaining
open RQ that had zero eval coverage before this:
  - "nhanh_a_vs_b"      (RQ2) - Nhánh A (DAG) vs Nhánh B (Text-to-SQL Agent)
    on the same question/gold value, run_nhanh_a_vs_b().
  - "regime_ablation"   (RQ3) - use_regime_signal on/off, descriptive only
    (no fabricated "better" judgment), run_regime_ablation().
  - "causal_comparison" (RQ6) - T-learner vs R-learner/DML on the same
    treatment/outcome pair, run_causal_comparison().
  - "absa_eval"         (RQ5) - structural/grounding checks only (no human-
    labeled sentiment ground truth exists), run_absa_eval().
See each function's docstring and src/evaluation/metrics.py's matching
*_summary()/*_comparison()/*_check() function for why each one deliberately
stops short of claiming an "accuracy" it has no real ground truth for.

Week 8 additions (2026-07-28) - completes the 6-baseline/ablation design in
docs/research_scope_map.html (see README Step 15's "6-baseline mapping"
table for how all 6 map to code): 3 more categories needed new code
(the other 3 baselines reuse nhanh_a_vs_b/regime_ablation/routing above):
  - "baseline_dag_only"          (RQ1/RQ2) - run_baseline_dag_only().
  - "baseline_rag_only_whatif"   (RQ1/RQ2) - run_baseline_rag_only_whatif().
  - "synthetic_causal_calibration" (RQ6)   - run_synthetic_causal_calibration(),
    the one category that deliberately uses SYNTHETIC, not real, data (see
    that function's own docstring for why).

Output: prints a summary table to stdout AND writes a full JSON report to
--output (default data/eval/report.json) with per-question detail, so a
failing case can be inspected instead of just seeing an aggregate number.

Unit tests for every pure-logic piece (all of metrics.py, plus run_eval.py's
_vn_number_variants()/load_benchmark()/WHATIF_ANSWER_FIELD, plus
benchmark.jsonl's own schema) live under tests/test_evaluation/ - run with
`python3 -m unittest discover -s tests/test_evaluation -v`, zero external
dependencies needed, safe anywhere including this sandbox.
"""
import argparse
import json
from pathlib import Path

from src.evaluation.metrics import (
    absa_structural_check,
    baseline_dag_only_summary,
    baseline_rag_only_summary,
    causal_estimator_comparison,
    exact_match,
    keyword_faithfulness,
    nhanh_a_vs_b_summary,
    regime_ablation_summary,
    routing_accuracy,
    sparse_coverage_summary,
    synthetic_causal_calibration,
    temporal_anchor_error_rate,
)


def load_benchmark(path: Path) -> list:
    items = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def run_routing(items: list) -> dict:
    """RQ1/RQ2/RQ6 - classify_intent() only, no dispatch. Pure logic, no
    DB/LLM required."""
    from src.router.rules import classify_intent

    predicted, expected, detail = [], [], []
    for item in items:
        if item["category"] != "routing":
            continue
        intent = classify_intent(item["question"])
        predicted.append(intent.branch)
        expected.append(item["expected_branch"])
        detail.append({
            "id": item["id"], "question": item["question"],
            "expected_branch": item["expected_branch"], "predicted_branch": intent.branch,
            "correct": intent.branch == item["expected_branch"], "reason": intent.reason,
        })
    metrics = routing_accuracy(predicted, expected) if predicted else {"n": 0}
    return {"metrics": metrics, "detail": detail}


def run_sparse_coverage(items: list, on_progress=None) -> dict:
    """"sparse_coverage_no_hallucination" category (added 2026-09-08) - see
    metrics.sparse_coverage_summary()'s docstring for what this tests and
    why. Needs live Postgres+Ollama (calls the real rag_pipeline.answer()).
    Same per-item try/except + on_progress checkpoint pattern as
    run_whatif_exact_match()/run_rag() below, for the same reason (one bad
    item shouldn't discard an otherwise-completed run)."""
    from src.retrieval.rag_pipeline import NO_CONTEXT_MSG, NO_CONTEXT_MSG_EN, answer

    detail = []
    for item in items:
        if item["category"] != "sparse_coverage_no_hallucination":
            continue
        try:
            result = answer(item["question"], ticker=item.get("ticker"))
            degraded = (
                result.get("answer") in (NO_CONTEXT_MSG, NO_CONTEXT_MSG_EN)
                or result.get("source_tier") == "web_fallback"
            )
            detail.append({
                "id": item["id"], "ticker": item.get("ticker"), "question": item["question"],
                "degraded_correctly": degraded,
                "actual_source_tier": result.get("source_tier"),
                "actual_answer_preview": (result.get("answer") or "")[:200],
            })
        except Exception as e:
            detail.append({
                "id": item["id"], "ticker": item.get("ticker"), "question": item["question"],
                "degraded_correctly": False,
                "error": f"{type(e).__name__}: {e}",
            })
        if on_progress:
            on_progress(detail)
    metrics = sparse_coverage_summary(detail) if detail else {"n": 0}
    return {"metrics": metrics, "detail": detail}


def run_whatif_exact_match(items: list, on_progress=None) -> dict:
    """RQ2 - full route() dispatch (needs Postgres) + exact_match() against
    known-good numbers hardcoded in the benchmark from a real prior run.

    Each item's route() call is wrapped individually (added 2026-07-22,
    same incident as run_rag() below) - one bad dispatch (DB hiccup, etc.)
    is recorded as an "error" item and the run continues, instead of an
    unhandled exception discarding every already-computed result in this
    category."""
    from src.router.query_router import route

    detail = []
    for item in items:
        if item["category"] != "whatif_exact_match":
            continue
        try:
            result = route(item["question"])
            raw = result.get("raw") or {}
            tol = item.get("tolerance_pct", 0.5)
            field_results = {}
            for field, expected_val in item.get("expected_fields", {}).items():
                actual_val = raw.get(field)
                field_results[field] = {
                    "expected": expected_val, "actual": actual_val,
                    "match": exact_match(actual_val, expected_val, tolerance_pct=tol),
                }
            all_match = bool(field_results) and all(f["match"] for f in field_results.values())
            detail.append({
                "id": item["id"], "question": item["question"],
                "branch_ok": result["branch"] == item["expected_branch"],
                "fields": field_results, "all_fields_match": all_match,
            })
        except Exception as e:
            detail.append({
                "id": item["id"], "question": item["question"],
                "branch_ok": False, "fields": {}, "all_fields_match": False,
                "error": f"{type(e).__name__}: {e}",
            })
        if on_progress:
            on_progress(detail)
    n = len(detail)
    n_pass = sum(1 for d in detail if d["all_fields_match"] and d["branch_ok"])
    return {"metrics": {"n": n, "pass": n_pass, "pass_rate": round(n_pass / n, 4) if n else None},
            "detail": detail}


def _rag_metrics(detail: list) -> dict:
    """Shared by run_rag()'s final return and its checkpoint callback, so
    the partial report written mid-run has the same live-updating
    aggregate metrics as the final one, computed over whatever items have
    completed so far (errored items are excluded from avg_keyword_faithfulness
    since they have no keyword_faithfulness score, same as a None score)."""
    temporal_items = [d for d in detail if d.get("category") == "rag_temporal" and "error" not in d]
    tae = temporal_anchor_error_rate(temporal_items)
    kf_scores = [d["keyword_faithfulness"]["score"] for d in detail
                 if "error" not in d and d.get("keyword_faithfulness", {}).get("score") is not None]
    avg_kf = round(sum(kf_scores) / len(kf_scores), 4) if kf_scores else None
    n_errors = sum(1 for d in detail if "error" in d)
    return {"temporal_anchor_error_rate": tae, "avg_keyword_faithfulness": avg_kf, "n_errors": n_errors}


def run_rag(items: list, with_temporal_anchor: bool = True, on_progress=None) -> dict:
    """RQ1/RQ4 - full rag_pipeline.answer() (needs pgvector + Ollama).
    Covers both "rag_faithfulness" (keyword check) and "rag_temporal"
    (RQ4 year-anchoring check) categories in one pass, since both need the
    same live call. Pass with_temporal_anchor=False to reproduce the RQ4
    "before" condition for the ablation comparison the design doc asks for
    (temporarily monkeypatches apply_temporal_anchor to a no-op).

    Two resilience fixes added 2026-07-22, after a real run over 70 RAG
    items hit an unhandled ReadTimeout on item ~51 (Ollama stalled after
    ~25 min of sustained local inference) that killed the whole process -
    since main() only wrote report.json ONCE at the very end (same failure
    shape as the earlier vector_store.py single-transaction-commit bug),
    all 51 completed generations plus the already-finished 26-item whatif
    category were lost with nothing persisted:
    1. Each item's rag_pipeline.answer() call is now wrapped individually -
       one bad/timed-out generation (even after llm_factory's own retries
       are exhausted) is recorded as an "error" item, not a crash.
    2. `on_progress(detail)` is called after every item so the caller
       (run_eval.py's main()) can checkpoint report.json incrementally -
       a later crash (OOM, killed process, etc. - not even catchable by
       the try/except above) now loses at most the batch since the last
       checkpoint, not the entire run."""
    from src.retrieval import rag_pipeline

    detail = []
    patched = None
    if not with_temporal_anchor:
        patched = rag_pipeline.apply_temporal_anchor
        rag_pipeline.apply_temporal_anchor = lambda chunks, question, top_k: (chunks[:top_k], None)

    try:
        for item in items:
            if item["category"] not in ("rag_faithfulness", "rag_temporal"):
                continue
            try:
                result = rag_pipeline.answer(item["question"], ticker=item.get("ticker"))
                answer_text = result.get("answer", "")
                sources = result.get("sources", [])
                actual_year = sources[0]["year"] if sources else None
                kf = keyword_faithfulness(answer_text, item.get("expected_keywords", []))
                detail.append({
                    "id": item["id"], "question": item["question"], "category": item["category"],
                    "answer": answer_text, "expected_year": item.get("expected_year"),
                    "actual_year": actual_year, "keyword_faithfulness": kf,
                    "temporal_note": result.get("temporal_note"),
                })
            except Exception as e:
                detail.append({
                    "id": item["id"], "question": item["question"], "category": item["category"],
                    "error": f"{type(e).__name__}: {e}",
                })
            if on_progress:
                on_progress(detail)
    finally:
        if patched is not None:
            rag_pipeline.apply_temporal_anchor = patched

    return {"metrics": _rag_metrics(detail), "detail": detail}


def run_nhanh_a_vs_b(items: list, on_progress=None, with_ms_code_hint: bool = False) -> dict:
    """RQ2 (Week 8) - Nhánh A (src/structured/query_builder.py, DAG
    schema-constrained lookup, không LLM) vs Nhánh B (src/structured/
    text_to_sql_agent.py, agentic Text-to-SQL, LLM tự sinh SQL) trên CÙNG
    câu hỏi, CÙNG gold value ("nhanh_a_vs_b" category trong benchmark.jsonl -
    mỗi item's expected_value tái sử dụng nguyên revenue_before đã real-
    verify từ whatif_exact_match, không có số nào tự bịa ra ở đây).

    Nhánh B trả về SQL bất kỳ (không schema cố định như Nhánh A) - lấy giá
    trị đầu tiên của dòng đầu tiên làm "actual value" để so exact_match
    (heuristic hợp lý cho câu hỏi 1-giá-trị kiểu "doanh thu X năm Y là bao
    nhiêu", không tổng quát cho câu hỏi nhiều cột/nhiều dòng).

    `with_ms_code_hint=False` mặc định (giữ nguyên hành vi/kết quả eval run
    #1, docs/eval_run1_results.md). True = ablation (2026-08-07): cấp thêm
    bảng ms_code->item_name chính thức vào prompt Nhánh B - xem
    text_to_sql_agent.answer()'s docstring. Chỉ scripts/
    ablation_nhanh_b_ms_code.py gọi với True; run_eval() (hàm main) không
    đổi để không làm sai lệch số liệu đã thu thập."""
    from src.structured.query_builder import execute, parse_question
    from src.structured.text_to_sql_agent import answer as answer_b

    detail = []
    for item in items:
        if item["category"] != "nhanh_a_vs_b":
            continue
        try:
            tol = item.get("tolerance_pct", 0.5)
            expected = item.get("expected_value")
            # Items with no verified gold number (expected_value is null in
            # benchmark.jsonl - e.g. the "Tập đoàn Hòa Phát" full-name case)
            # are run for real but excluded from accuracy scoring - see
            # nhanh_a_vs_b_summary()'s docstring for why.
            qualitative_only = expected is None

            parsed = parse_question(item["question"])
            result_a = execute(parsed)
            a_value = result_a.get("value") if result_a.get("ok") else None
            a_correct = None if qualitative_only else exact_match(a_value, expected, tolerance_pct=tol)

            result_b = answer_b(item["question"], with_ms_code_hint=with_ms_code_hint)
            gate2b = result_b.get("gate2b_ast_check")
            if not gate2b or not gate2b.get("available"):
                gate2b_would_flag = None
            elif gate2b.get("parse_error"):
                gate2b_would_flag = True
            else:
                gate2b_would_flag = bool(gate2b.get("problems"))
            b_value = None
            if result_b.get("ok") and result_b.get("rows"):
                b_value = next(iter(result_b["rows"][0].values()), None)
            if qualitative_only:
                b_correct = None
            else:
                b_correct = exact_match(b_value, expected, tolerance_pct=tol) if b_value is not None else False

            detail.append({
                "id": item["id"], "question": item["question"], "expected_value": expected,
                "qualitative_only": qualitative_only,
                "nhanh_a_ok": result_a.get("ok"), "nhanh_a_value": a_value, "nhanh_a_correct": a_correct,
                "nhanh_a_error": result_a.get("error"),
                "nhanh_b_value": b_value, "nhanh_b_correct": b_correct,
                "nhanh_b_gate_failed": result_b.get("gate_failed"),
                "nhanh_b_gate2b_would_flag": gate2b_would_flag,
                # row_count added 2026-07-28, real run finding: a gate_failed=None
                # ("success") case can still return 0 rows (WHERE clause with an
                # exact-string item_name mismatch is syntactically/schema valid
                # SQL, not an execution error) - no existing gate catches this,
                # so gate_failed_distribution's "success" bucket alone is
                # misleading without also checking row_count. See
                # nhanh_a_vs_b_summary()'s "n_success_zero_rows".
                "nhanh_b_row_count": result_b.get("row_count"),
                "nhanh_b_generated_sql": result_b.get("generated_sql"),
                "nhanh_b_problems": result_b.get("problems"),
                "nhanh_b_with_ms_code_hint": result_b.get("with_ms_code_hint"),
            })
        except Exception as e:
            detail.append({"id": item["id"], "question": item["question"], "error": f"{type(e).__name__}: {e}"})
        if on_progress:
            on_progress(detail)
    return {"metrics": nhanh_a_vs_b_summary(detail), "detail": detail}


def run_regime_ablation(items: list, on_progress=None) -> dict:
    """RQ3 (Week 8) - chạy MỖI câu hỏi trong "regime_ablation" category 2
    lần qua rag_pipeline.answer(): use_regime_signal=True rồi False, để đo
    mức độ rerank thực sự thay đổi thứ tự nguồn khi regime hiện tại là
    bear/volatile - KHÔNG chấm đúng/sai (xem regime_ablation_summary()'s
    docstring cho lý do). Câu hỏi nên là các ticker có cơ chế nhạy cảm với
    chu kỳ thị trường đã biết thật (PNJ/vàng, GAS/dầu - xem
    src/structured/calculator.py's GOLD_SENSITIVE_TICKERS/
    OIL_SENSITIVE_TICKERS), tái sử dụng câu hỏi RAG đã có trong
    rag_faithfulness cho cùng ticker khi có thể."""
    from src.retrieval import rag_pipeline

    with_signal, without_signal, detail = [], [], []
    for item in items:
        if item["category"] != "regime_ablation":
            continue
        try:
            r_on = rag_pipeline.answer(item["question"], ticker=item.get("ticker"), use_regime_signal=True)
            r_off = rag_pipeline.answer(item["question"], ticker=item.get("ticker"), use_regime_signal=False)
            with_signal.append(r_on)
            without_signal.append(r_off)
            detail.append({
                "id": item["id"], "question": item["question"], "ticker": item.get("ticker"),
                "market_regime": r_on.get("market_regime"),
                "sources_with_signal": [s.get("source_file") for s in r_on.get("sources", [])],
                "sources_without_signal": [s.get("source_file") for s in r_off.get("sources", [])],
            })
        except Exception as e:
            detail.append({"id": item["id"], "question": item["question"], "error": f"{type(e).__name__}: {e}"})
        if on_progress:
            on_progress(detail)
    return {"metrics": regime_ablation_summary(with_signal, without_signal), "detail": detail}


def run_causal_comparison(items: list) -> dict:
    """RQ6 (Week 8) - T-learner (estimate_ate) vs R-learner/DML
    (estimate_ate_dml) trên cùng cặp treatment/outcome/covariates mỗi
    "causal_comparison" benchmark item (mặc định leverage/net_margin - cặp
    dùng xuyên suốt RQ6 trong dự án này). build_panel() chỉ gọi 1 LẦN mỗi
    item (không phải 2 lần cho 2 estimator) vì nó query lại toàn bộ company
    panel từ Postgres, không rẻ. Không có on_progress/checkpoint per-item
    ở đây vì category này thường chỉ có 1-2 item (khác với whatif/rag có
    hàng chục câu)."""
    from src.causal.panel_builder import build_panel
    from src.causal.treatment_effect import estimate_ate, estimate_ate_dml

    detail = []
    for item in items:
        if item["category"] != "causal_comparison":
            continue
        try:
            treatment = item.get("treatment_col", "leverage")
            outcome = item.get("outcome_col", "net_margin")
            covariates = tuple(item.get("covariates", ["sector", "fiscal_year"]))
            panel = build_panel()
            t_result = estimate_ate(treatment_col=treatment, outcome_col=outcome,
                                     covariates=covariates, panel=panel)
            dml_result = estimate_ate_dml(treatment_col=treatment, outcome_col=outcome,
                                           covariates=covariates, panel=panel)
            comparison = causal_estimator_comparison(t_result, dml_result)
            detail.append({"id": item["id"], "treatment": treatment, "outcome": outcome,
                            "covariates": list(covariates), **comparison})
        except Exception as e:
            detail.append({"id": item["id"], "error": f"{type(e).__name__}: {e}"})
    n_ok = sum(1 for d in detail if d.get("ok"))
    return {"metrics": {"n": len(detail), "n_both_estimators_ok": n_ok}, "detail": detail}


def run_absa_eval(items: list, on_progress=None) -> dict:
    """RQ5 (Week 8) - analyze_ticker() (src/absa/absa_pipeline.py) cho mỗi
    ticker trong "absa_eval" category + absa_structural_check() - kiểm tra
    CẤU TRÚC (đủ 5 khía cạnh, sentiment hợp lệ, evidence khi cần), KHÔNG
    chấm sentiment đúng/sai (chưa có bộ gán nhãn con người - xem
    absa_structural_check()'s docstring). Nên dùng các ticker đã thật-test
    ở Week 7 (HPG/CTD/GAS/PNJ - xem docs/PROGRESS.md's "Test thật diện
    rộng Week 6 + Week 7" entry)."""
    from src.absa.absa_pipeline import analyze_ticker

    detail = []
    for item in items:
        if item["category"] != "absa_eval":
            continue
        try:
            result = analyze_ticker(item["ticker"], top_k=item.get("top_k", 10))
            check = absa_structural_check(result)
            detail.append({
                "id": item["id"], "ticker": item["ticker"], "absa_ok": result.get("ok"),
                "aspects": result.get("aspects"), "problems": result.get("problems"), **check,
            })
        except Exception as e:
            detail.append({"id": item["id"], "ticker": item.get("ticker"), "error": f"{type(e).__name__}: {e}"})
        if on_progress:
            on_progress(detail)
    n = len(detail)
    n_absa_ok = sum(1 for d in detail if d.get("absa_ok"))
    n_valid = sum(1 for d in detail if d.get("structurally_valid"))
    return {"metrics": {"n": n, "n_absa_ok": n_absa_ok, "n_structurally_valid": n_valid,
                        "structurally_valid_rate": round(n_valid / n, 4) if n else None},
            "detail": detail}



# All 26 whatif_exact_match questions follow ONE fixed Vietnamese template
# (confirmed 2026-07-28 by diffing every question with its ticker masked out -
# see docs/PROGRESS.md): "Nếu COGS của <TICKER> tăng 5% trong 2025, lợi
# nhuận gộp sẽ ra sao?" - i.e. the question is ALWAYS asking for the
# POST-scenario gross profit, which is exactly `expected_fields
# ["gross_profit_after"]`. Scoring "correct" against ANY of the 5
# expected_fields (revenue_before/cogs_before/gross_profit_before are all
# real CURRENT-state values, not what the question asks) is what a first
# version of this baseline did on 2026-07-28's real run - it produced a
# misleadingly high `correct_rate=0.9615`, because `resolve_indicator()`
# correctly maps "lợi nhuận gộp" to the gross-profit ms_code and DAG-only
# faithfully fetches the REAL reported (pre-scenario) gross profit, which
# happens to equal `gross_profit_before` - a real value, but not an answer
# to "what will it be AFTER the shock". Fixed to score against
# `gross_profit_after` specifically, the one field that actually answers
# the question as phrased.
WHATIF_ANSWER_FIELD = "gross_profit_after"


def run_baseline_dag_only(items: list, on_progress=None) -> dict:
    """Week 8 baseline #1 (docs/research_scope_map.html's "base-dagonly",
    RQ1/RQ2) - simulates "what if this system ONLY had Nhánh A (schema-
    constrained DAG lookup), no Router/What-if calculator/RAG at all" by
    running the RAW question text (whatif_exact_match's scenario phrasing,
    rag_faithfulness/rag_temporal's qualitative phrasing - NOT rewritten to
    be DAG-friendly, since a real single-source system would receive
    exactly these questions as-is) through `query_builder.parse_question()`
    + `execute()` directly.

    Expected finding (this IS the point of the baseline, not a bug if it
    shows near-0): `rag_faithfulness`/`rag_temporal` questions are
    qualitative ("rủi ro pháp lý của X là gì") - `resolve_indicator()` has
    no alias for that phrasing, so `parse_question()` can't resolve an
    `ms_code`, `execute()` returns `ok=False` - `answerable_rate` should be
    near 0 for these categories. `whatif_exact_match` questions describe a
    HYPOTHETICAL scenario ("nếu COGS tăng 5%..., lợi nhuận gộp sẽ ra sao?") -
    `resolve_indicator()` DOES correctly map "lợi nhuận gộp" to the gross-
    profit indicator (so `answerable_rate` is actually high - DAG-only IS
    schema-capable of finding the relevant line item), but the value it
    fetches is the REAL, CURRENT gross profit (`gross_profit_before`), not
    the scenario-adjusted `gross_profit_after` the question actually asks
    for - see `WHATIF_ANSWER_FIELD` above for why `correct` is scored
    against that one specific field, not "any of the 5". `correct_rate`
    should be near 0 there, contrasting with the real What-if branch's
    26/26 - the gap being ENTIRELY explained by "found the right line item,
    but can't apply the hypothetical shock to it", not by failing to find
    anything at all."""
    from src.structured.query_builder import execute, parse_question

    detail = []
    for item in items:
        if item["category"] not in ("whatif_exact_match", "rag_faithfulness", "rag_temporal"):
            continue
        try:
            parsed = parse_question(item["question"])
            result = execute(parsed)
            answerable = bool(result.get("ok"))
            correct = None
            matched_field = None
            if item["category"] == "whatif_exact_match" and answerable:
                value = result.get("value")
                tol = item.get("tolerance_pct", 0.5)
                expected_fields = item.get("expected_fields", {})
                answer_field_expected = expected_fields.get(WHATIF_ANSWER_FIELD)
                correct = exact_match(value, answer_field_expected, tolerance_pct=tol)
                # Informational only (not used for scoring) - which of the 5
                # fields, if any, DAG-only's fetched value happens to equal;
                # explains WHY answerable-but-wrong cases look the way they do.
                for field, expected_val in expected_fields.items():
                    if exact_match(value, expected_val, tolerance_pct=tol):
                        matched_field = field
                        break
            elif item["category"] == "whatif_exact_match":
                correct = False
            detail.append({
                "id": item["id"], "category": item["category"], "question": item["question"],
                "matched_field": matched_field,
                "answerable": answerable, "dag_only_value": result.get("value"),
                "dag_only_error": result.get("error"), "correct": correct,
            })
        except Exception as e:
            detail.append({"id": item["id"], "category": item["category"], "error": f"{type(e).__name__}: {e}"})
        if on_progress:
            on_progress(detail)
    return {"metrics": baseline_dag_only_summary(detail), "detail": detail}


def _vn_number_variants(value) -> list:
    """Vietnamese-disclosure-style renderings of a number, used to check
    whether a RAG-generated answer happens to contain it verbatim -
    comma-thousands (dag_pipeline.py's own `f'{value:,}'` convention),
    dot-thousands (common in Vietnamese financial text), and plain digits
    with no separator at all, all rounded to the nearest whole unit since
    financial_line_items values here are already whole VND."""
    try:
        n = int(round(float(value)))
    except (TypeError, ValueError):
        return []
    comma = f"{n:,}"
    return [comma, comma.replace(",", "."), str(n)]


def run_baseline_rag_only_whatif(items: list, on_progress=None) -> dict:
    """Week 8 baseline #2 (base-ragonly, RQ1/RQ2), whatif_exact_match slice
    only - simulates "what if this system ONLY had RAG (no temporal anchor,
    no What-if calculator)" by asking `rag_pipeline.answer()` the SAME raw
    scenario question and checking whether the SCENARIO answer
    (`expected_fields[WHATIF_ANSWER_FIELD]` - see that constant's comment
    above `run_baseline_dag_only()` for why it's specifically
    `gross_profit_after`, not "any of the 5") shows up verbatim (in a few
    plausible Vietnamese numeric formats) anywhere in the generated answer
    text. Also records `any_field_found` (informational only, not scored) -
    whether one of the OTHER 4 (real, current-state) fields happens to
    appear, which would mean RAG genuinely quoted a real disclosed figure,
    just not an answer to the scenario question asked - same distinction
    `run_baseline_dag_only()` makes via `matched_field`, kept consistent
    after the 2026-07-28 scoring fix there.

    Expected finding: near-0 `scenario_answer_found_rate`, since
    `gross_profit_after` is a number COMPUTED from a hypothetical (COGS
    +5%), not present in any real disclosure document for RAG to retrieve -
    contrasting with the real What-if branch's 26/26 (deterministic
    arithmetic, not retrieval-dependent). The `rag_faithfulness`/
    `rag_temporal` half of this baseline needs no new code - it's exactly
    `run_rag(with_temporal_anchor=False)` / `--no-temporal-anchor`, already
    built for the RQ4 ablation (see README Step 15)."""
    from src.retrieval import rag_pipeline

    detail = []
    for item in items:
        if item["category"] != "whatif_exact_match":
            continue
        try:
            result = rag_pipeline.answer(item["question"], ticker=item.get("ticker"), use_regime_signal=False)
            answer_text = result.get("answer", "")
            expected_fields = item.get("expected_fields", {})
            scenario_expected = expected_fields.get(WHATIF_ANSWER_FIELD)
            scenario_found = any(
                variant in answer_text for variant in _vn_number_variants(scenario_expected)
            )
            any_field_found = any(
                any(variant in answer_text for variant in _vn_number_variants(v))
                for v in expected_fields.values()
            )
            detail.append({
                "id": item["id"], "question": item["question"],
                "any_expected_field_found": scenario_found,  # kept name for baseline_rag_only_summary()
                "any_field_found_informational": any_field_found, "answer": answer_text,
            })
        except Exception as e:
            detail.append({"id": item["id"], "question": item["question"], "error": f"{type(e).__name__}: {e}"})
        if on_progress:
            on_progress(detail)
    return {"metrics": baseline_rag_only_summary(detail), "detail": detail}


def run_synthetic_causal_calibration() -> dict:
    """Week 8 baseline #6b (base-causalonly's "đo sai số ATE so với giá trị
    mô phỏng đã biết", RQ6) - the ONE piece of the evaluation harness that
    deliberately uses SYNTHETIC (not real-company) data, because real
    observational data never has a known-true causal effect to check bias
    against. Generates a panel from a linear DGP with a known constant
    marginal effect `true_beta`, runs both `estimate_ate()` (T-learner) and
    `estimate_ate_dml()` (R-learner/DML) against it, and reports each
    estimator's bias vs the analytically-known truth (computed directly
    from this same synthetic panel, not a separate infinite-sample oracle -
    a standard calibration-check design, not a real-world result; never
    reported as if it were a finding about actual Vietnamese companies).

    The 6-baseline design (docs/research_scope_map.html) also calls for
    checking what fraction of counterfactual/what-if questions get ROUTED
    correctly (vs DAG/RAG intercepting them) - that needs NO new code here,
    it's already the "whatif"/"whatif_unsupported"/"causal" rows of
    `run_routing()`'s per-class P/R/F1 breakdown (see `report["routing"]
    ["metrics"]["per_class"]`)."""
    import numpy as np
    import pandas as pd
    from src.causal.treatment_effect import estimate_ate, estimate_ate_dml

    rng = np.random.default_rng(42)
    n = 500
    true_beta = -0.08  # known constant marginal effect: net_margin per +1.0 unit of leverage
    sector = rng.choice(["A", "B", "C"], size=n)
    fiscal_year = rng.choice([2023, 2024, 2025], size=n)
    leverage = rng.uniform(0.1, 0.9, size=n)
    sector_effect = {"A": 0.01, "B": -0.02, "C": 0.03}
    year_effect = {2023: -0.01, 2024: 0.0, 2025: 0.01}
    noise = rng.normal(0, 0.03, size=n)
    net_margin = (true_beta * leverage
                  + np.array([sector_effect[s] for s in sector])
                  + np.array([year_effect[y] for y in fiscal_year])
                  + noise)
    panel = pd.DataFrame({"leverage": leverage, "net_margin": net_margin,
                           "sector": sector, "fiscal_year": fiscal_year})

    median = panel["leverage"].median()
    treated_mean = panel.loc[panel["leverage"] > median, "leverage"].mean()
    control_mean = panel.loc[panel["leverage"] <= median, "leverage"].mean()
    true_group_contrast = true_beta * (treated_mean - control_mean)

    try:
        t_result = estimate_ate(treatment_col="leverage", outcome_col="net_margin",
                                 covariates=("sector", "fiscal_year"), panel=panel)
        dml_result = estimate_ate_dml(treatment_col="leverage", outcome_col="net_margin",
                                       covariates=("sector", "fiscal_year"), panel=panel)
        calibration = synthetic_causal_calibration(t_result, dml_result, true_beta, true_group_contrast)
    except Exception as e:
        calibration = {"ok": False, "error": f"{type(e).__name__}: {e}"}

    return {"metrics": calibration,
            "detail": [{"n": n, "true_beta": true_beta, "true_group_contrast": round(true_group_contrast, 6)}]}


def print_summary(report: dict):
    print("=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    if "routing" in report:
        m = report["routing"]["metrics"]
        print(f"\n[Routing]  n={m.get('n')}  accuracy={m.get('accuracy')}  macro_f1={m.get('macro_f1')}")
        for label, stats in m.get("per_class", {}).items():
            print(f"    {label:22s} P={stats['precision']:.2f} R={stats['recall']:.2f} "
                  f"F1={stats['f1']:.2f} (n={stats['support']})")
    if "whatif_exact_match" in report:
        m = report["whatif_exact_match"]["metrics"]
        print(f"\n[What-if exact match]  n={m.get('n')}  pass={m.get('pass')}  "
              f"pass_rate={m.get('pass_rate')}")
    if "rag" in report:
        m = report["rag"]["metrics"]
        print(f"\n[RAG]  avg_keyword_faithfulness={m.get('avg_keyword_faithfulness')}  "
              f"temporal_anchor_error_rate={m.get('temporal_anchor_error_rate')}  "
              f"n_errors={m.get('n_errors', 0)}")
    if "nhanh_a_vs_b" in report:
        m = report["nhanh_a_vs_b"]["metrics"]
        print(f"\n[RQ2 - Nhánh A vs Nhánh B]  n={m.get('n')}  "
              f"nhanh_a_accuracy={m.get('nhanh_a_accuracy')}  nhanh_b_accuracy={m.get('nhanh_b_accuracy')}  "
              f"gate2b_would_have_caught_rate={m.get('gate2b_would_have_caught_rate')}")
        print(f"    gate_failed_distribution={m.get('gate_failed_distribution')}  "
              f"n_success_zero_rows={m.get('n_success_zero_rows')}  "
              f"success_zero_rows_rate={m.get('success_zero_rows_rate')}")
    if "regime_ablation" in report:
        m = report["regime_ablation"]["metrics"]
        print(f"\n[RQ3 - Regime ablation]  n={m.get('n')}  regime_active_rate={m.get('regime_active_rate')}  "
              f"rerank_order_changed_rate={m.get('rerank_order_changed_rate')}")
    if "causal_comparison" in report:
        m = report["causal_comparison"]["metrics"]
        print(f"\n[RQ6 - T-learner vs DML]  n={m.get('n')}  n_both_estimators_ok={m.get('n_both_estimators_ok')}")
    if "absa_eval" in report:
        m = report["absa_eval"]["metrics"]
        print(f"\n[RQ5 - ABSA structural check]  n={m.get('n')}  n_absa_ok={m.get('n_absa_ok')}  "
              f"structurally_valid_rate={m.get('structurally_valid_rate')}")
    if "baseline_dag_only" in report:
        m = report["baseline_dag_only"]["metrics"]
        print("\n[Baseline #1 - DAG-only]")
        for cat, stats in m.items():
            print(f"    {cat:20s} n={stats.get('n')}  answerable_rate={stats.get('answerable_rate')}  "
                  f"correct_rate={stats.get('correct_rate')}")
    if "baseline_rag_only_whatif" in report:
        m = report["baseline_rag_only_whatif"]["metrics"]
        print(f"\n[Baseline #2 - RAG-only, whatif slice]  n={m.get('n')}  "
              f"any_expected_field_found_rate={m.get('any_expected_field_found_rate')}")
        print("    (whatif branch itself: pass_rate above; RAG-only rag_faithfulness/rag_temporal slice "
              "= run with --no-temporal-anchor, see README Step 15)")
    if "synthetic_causal_calibration" in report:
        m = report["synthetic_causal_calibration"]["metrics"]
        print(f"\n[Baseline #6b - Synthetic causal calibration]  ok={m.get('ok')}  "
              f"t_learner_bias={m.get('t_learner_bias')}  dml_translated_bias={m.get('dml_translated_bias')}  "
              f"dml_marginal_bias={m.get('dml_marginal_bias')}")
    if "sparse_coverage_no_hallucination" in report:
        m = report["sparse_coverage_no_hallucination"]["metrics"]
        print(f"\n[Sparse-coverage no-hallucination check]  n={m.get('n')}  "
              f"degraded_correctly_rate={m.get('degraded_correctly_rate')}  "
              f"n_possible_hallucinations={m.get('n_possible_hallucinations')}")


def save_report(report: dict, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="data/eval/benchmark.jsonl")
    ap.add_argument("--output", default="data/eval/report.json")
    ap.add_argument("--routing-only", action="store_true",
                     help="Only run the 'routing' category (pure classify_intent() logic, no DB/LLM needed)")
    ap.add_argument("--no-temporal-anchor", action="store_true",
                     help="RQ4 ablation: disable Temporal Anchor Filtering to measure the 'before' error rate")
    args = ap.parse_args()

    items = load_benchmark(Path(args.benchmark))
    print(f"Loaded {len(items)} benchmark item(s) from {args.benchmark}")

    out_path = Path(args.output)

    # Checkpoint after every category, and after every single item within
    # whatif/rag (added 2026-07-22 - see run_rag()'s docstring for the real
    # incident that motivated this) - a crash anywhere after this point
    # loses at most the current in-flight item, never the whole run.
    report = {"routing": run_routing(items)}
    save_report(report, out_path)

    if not args.routing_only:
        def checkpoint_whatif(detail):
            n = len(detail)
            n_pass = sum(1 for d in detail if d.get("all_fields_match") and d.get("branch_ok"))
            report["whatif_exact_match"] = {
                "metrics": {"n": n, "pass": n_pass, "pass_rate": round(n_pass / n, 4) if n else None},
                "detail": detail,
            }
            save_report(report, out_path)

        report["whatif_exact_match"] = run_whatif_exact_match(items, on_progress=checkpoint_whatif)
        save_report(report, out_path)

        def checkpoint_rag(detail):
            report["rag"] = {"metrics": _rag_metrics(detail), "detail": detail}
            save_report(report, out_path)

        report["rag"] = run_rag(items, with_temporal_anchor=not args.no_temporal_anchor,
                                 on_progress=checkpoint_rag)
        save_report(report, out_path)

        # Week 8 additions (RQ2/RQ3/RQ5/RQ6) - same per-item checkpoint
        # pattern as whatif/rag above, each wrapped so one bad item/category
        # (e.g. econml not installed for causal_comparison) doesn't lose the
        # rest of the run.
        def checkpoint_nhanh_a_vs_b(detail):
            report["nhanh_a_vs_b"] = {"metrics": nhanh_a_vs_b_summary(detail), "detail": detail}
            save_report(report, out_path)

        report["nhanh_a_vs_b"] = run_nhanh_a_vs_b(items, on_progress=checkpoint_nhanh_a_vs_b)
        save_report(report, out_path)

        def checkpoint_regime_ablation(detail):
            report["regime_ablation"] = {"metrics": {"n": len(detail)}, "detail": detail}
            save_report(report, out_path)

        report["regime_ablation"] = run_regime_ablation(items, on_progress=checkpoint_regime_ablation)
        save_report(report, out_path)

        try:
            report["causal_comparison"] = run_causal_comparison(items)
        except Exception as e:
            report["causal_comparison"] = {"metrics": {"n": 0}, "detail": [],
                                            "error": f"{type(e).__name__}: {e}"}
        save_report(report, out_path)

        def checkpoint_absa(detail):
            n = len(detail)
            n_valid = sum(1 for d in detail if d.get("structurally_valid"))
            report["absa_eval"] = {
                "metrics": {"n": n, "n_structurally_valid": n_valid,
                            "structurally_valid_rate": round(n_valid / n, 4) if n else None},
                "detail": detail,
            }
            save_report(report, out_path)

        report["absa_eval"] = run_absa_eval(items, on_progress=checkpoint_absa)
        save_report(report, out_path)

        # Week 8 baseline/ablation comparison (2026-07-28) - completes the
        # 6-baseline design in docs/research_scope_map.html ("chưa bắt đầu
        # đầy đủ" note on Tuần 8). Baselines #3 (Text-to-SQL-only),
        # #4 (Router tĩnh/no-regime), and #6a (causal routing %) reuse
        # nhanh_a_vs_b/regime_ablation/routing above - no new runners for
        # those, see this section's PROGRESS.md entry for the mapping.
        def checkpoint_baseline_dag(detail):
            report["baseline_dag_only"] = {"metrics": baseline_dag_only_summary(detail), "detail": detail}
            save_report(report, out_path)

        report["baseline_dag_only"] = run_baseline_dag_only(items, on_progress=checkpoint_baseline_dag)
        save_report(report, out_path)

        def checkpoint_baseline_rag(detail):
            report["baseline_rag_only_whatif"] = {"metrics": baseline_rag_only_summary(detail), "detail": detail}
            save_report(report, out_path)

        report["baseline_rag_only_whatif"] = run_baseline_rag_only_whatif(items, on_progress=checkpoint_baseline_rag)
        save_report(report, out_path)

        def checkpoint_sparse_coverage(detail):
            report["sparse_coverage_no_hallucination"] = {
                "metrics": sparse_coverage_summary(detail), "detail": detail,
            }
            save_report(report, out_path)

        report["sparse_coverage_no_hallucination"] = run_sparse_coverage(
            items, on_progress=checkpoint_sparse_coverage
        )
        save_report(report, out_path)

        try:
            report["synthetic_causal_calibration"] = run_synthetic_causal_calibration()
        except Exception as e:
            report["synthetic_causal_calibration"] = {"metrics": {"ok": False, "error": f"{type(e).__name__}: {e}"},
                                                        "detail": []}
        save_report(report, out_path)

    print_summary(report)
    print(f"\nFull report written to {out_path}")


if __name__ == "__main__":
    main()
