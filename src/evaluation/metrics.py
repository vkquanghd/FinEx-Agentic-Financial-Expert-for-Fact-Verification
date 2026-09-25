"""Metric functions used by src/evaluation/run_eval.py. Each function is
pure (no DB/LLM/network calls of its own except llm_judge_faithfulness,
clearly marked) - takes predictions/references in, returns a score out, so
each one is independently unit-testable without a live stack. See
docs/Thesis_Direction_Redesign.pdf section 7 for which metric answers which RQ.
"""
import re


def routing_accuracy(predicted: list, expected: list) -> dict:
    """RQ1/RQ2/RQ6 - Routing Accuracy + Macro F1 over branch labels.
    `predicted`/`expected` are parallel lists of branch-label strings (e.g.
    "dag", "rag", "whatif", "whatif_unsupported", "causal", "compare").
    Macro F1 = unweighted mean of per-class F1 - matters here because
    branches are naturally imbalanced in any real question mix (far more
    RAG/DAG questions than causal/compare ones), and macro F1 doesn't let
    the frequent classes hide poor performance on the rare ones, unlike
    plain accuracy or micro F1."""
    if len(predicted) != len(expected):
        raise ValueError(f"length mismatch: {len(predicted)} predicted vs {len(expected)} expected")
    n = len(predicted)
    if n == 0:
        return {"accuracy": None, "macro_f1": None, "n": 0, "per_class": {}}

    correct = sum(1 for p, e in zip(predicted, expected) if p == e)
    labels = sorted(set(expected) | set(predicted))

    per_class = {}
    f1s = []
    for label in labels:
        tp = sum(1 for p, e in zip(predicted, expected) if p == label and e == label)
        fp = sum(1 for p, e in zip(predicted, expected) if p == label and e != label)
        fn = sum(1 for p, e in zip(predicted, expected) if p != label and e == label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class[label] = {"precision": round(precision, 4), "recall": round(recall, 4),
                             "f1": round(f1, 4), "support": tp + fn}
        f1s.append(f1)

    return {
        "accuracy": round(correct / n, 4),
        "macro_f1": round(sum(f1s) / len(f1s), 4) if f1s else None,
        "n": n,
        "per_class": per_class,
    }


def confusion_matrix(predicted: list, expected: list) -> dict:
    """Added 2026-07-30 (Task #23, roadmap item 6 - "run the [Router] test "
    "only once a version is locked; report confusion matrix and macro-F1")
    alongside routing_accuracy() above, which already computes macro-F1 but
    not a confusion matrix. Returns {"labels": [...], "matrix": [[...]]}
    where matrix[i][j] = count of items with true label labels[i] predicted
    as labels[j] - the standard row=actual/column=predicted convention.
    Kept separate from routing_accuracy() (not merged into its return dict)
    so a caller that only wants accuracy/macro-F1 (e.g. an ablation loop
    that runs this many times) doesn't pay for building a matrix it
    discards every time."""
    if len(predicted) != len(expected):
        raise ValueError(f"length mismatch: {len(predicted)} predicted vs {len(expected)} expected")
    labels = sorted(set(expected) | set(predicted))
    index = {label: i for i, label in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]
    for p, e in zip(predicted, expected):
        matrix[index[e]][index[p]] += 1
    return {"labels": labels, "matrix": matrix}


def exact_match(actual, expected, tolerance_pct: float = 0.5) -> bool:
    """RQ2 - Exact Match cho câu hỏi số liệu. `tolerance_pct` allows tiny
    float/rounding drift (default 0.5%) without counting a correct DAG/
    What-if lookup as a miss - the underlying value must still come from
    the same deterministic source, this isn't a fuzzy-match escape hatch.

    Coerces both `actual`/`expected` to float before comparing (real bug
    found 2026-07-28 running run_nhanh_a_vs_b() for real: `financial_line_items
    .value` is a Postgres NUMERIC column, so query_builder.execute()'s raw
    `value` comes back as `decimal.Decimal`, not `float` -
    `Decimal.__sub__(float)` raises `TypeError` outright rather than
    returning a wrong answer, which crashed 25/26 nhanh_a_vs_b items before
    Nhánh B even got called - see docs/PROGRESS.md. calculator.py's
    `_get_value()` already does this same cast at its own DB boundary; this
    makes exact_match() itself defensive so every caller gets it for free,
    not just callers that happen to cast first. An uncoercible value (e.g. a
    stray string) is treated as "no match", not a crash."""
    if actual is None or expected is None:
        return False
    try:
        actual = float(actual)
        expected = float(expected)
    except (TypeError, ValueError):
        return False
    if expected == 0:
        return actual == 0
    return abs(actual - expected) / abs(expected) * 100 <= tolerance_pct


_SQUASH_RE = re.compile(r"[^0-9a-zà-ỹ]+")

# Dưới ngưỡng này thì KHÔNG dùng khớp bỏ-qua-dấu-cách: một từ khoá 1-2 ký
# tự sau khi bỏ hết dấu cách/ký hiệu rất dễ khớp bừa vào giữa một từ khác
# ("EY" lọt vào "...ey..." nào đó). Từ khoá ngắn để bảng alias hãng kiểm
# toán lo (canonical_auditors), nơi có ngữ cảnh rõ ràng hơn.
_MIN_SQUASH_LEN = 3


def _squash(s: str) -> str:
    """Hạ chữ thường + bỏ mọi ký tự không phải chữ/số. Xử lý đúng các ca
    THẬT gặp trong eval run 1: 'i CPA' vs 'iCPA', 'An Viet' vs 'AnViet',
    và 'Pw C' (OCR tách chữ) vs 'PWC'."""
    return _SQUASH_RE.sub("", (s or "").lower())


def _keyword_found(keyword: str, answer: str) -> bool:
    """Ba lớp khớp, từ chặt tới lỏng - dừng ở lớp đầu tiên khớp được.

    Lớp 1 giữ NGUYÊN hành vi cũ (khớp chuỗi con), nên mọi kết quả trước đây
    đúng thì vẫn đúng; lớp 2 và 3 chỉ có thể làm điểm TĂNG, không bao giờ
    giảm.
    """
    if not keyword:
        return False
    # Lớp 1: khớp chuỗi con nguyên bản (hành vi gốc).
    if keyword.lower() in (answer or "").lower():
        return True
    # Lớp 2: bỏ qua dấu cách/dấu câu.
    if len(_squash(keyword)) >= _MIN_SQUASH_LEN and _squash(keyword) in _squash(answer):
        return True
    # Lớp 3: alias hãng kiểm toán - ground truth (disclosures.auditor, từ
    # Vietstock) lưu VIẾT TẮT còn câu trả lời trích TÊN PHÁP LÝ ĐẦY ĐỦ từ
    # tài liệu. Chuẩn hoá cả hai phía rồi mới so. Import cục bộ để module
    # metrics vẫn thuần/offline khi không có nhánh nào cần tới nó.
    try:
        from src.structured.auditor_lookup import canonical_auditors

        kw_canon = canonical_auditors(keyword) or {keyword.strip().upper()}
        if kw_canon & canonical_auditors(answer):
            return True
    except Exception:
        pass
    return False


def keyword_faithfulness(answer: str, expected_keywords: list) -> dict:
    """Cheap, fast proxy for RAGAS's faithfulness metric - does the
    generated answer actually contain the expected grounded facts, or did
    the LLM hedge/hallucinate/miss them? NOT a substitute for
    llm_judge_faithfulness() below (RAGAS itself is LLM-judge-based, not
    keyword-based) - use this for a first-pass, zero-cost sanity check
    across the whole benchmark, and the LLM-judge version for the actual
    thesis-reported faithfulness numbers.

    2026-08-07 - khớp qua _keyword_found() (3 lớp) thay vì `in` thô, sau khi
    lần chạy eval đầy đủ đầu tiên cho thấy metric này chấm SAI 11/69 câu
    TRẢ LỜI ĐÚNG thành 0 điểm, kéo faithfulness từ ~0.82 xuống 0.66. Nguyên
    nhân: ground truth và câu trả lời dùng HAI QUY ƯỚC ĐẶT TÊN khác nhau
    (viết tắt vs tên pháp lý đầy đủ), không phải hệ thống trả lời sai. Xem
    docs/eval_run1_faithfulness_analysis.md để có bảng phân loại từng câu.
    """
    if not expected_keywords:
        return {"score": None, "found": [], "missing": []}
    found = [kw for kw in expected_keywords if _keyword_found(kw, answer)]
    missing = [kw for kw in expected_keywords if kw not in found]
    return {"score": round(len(found) / len(expected_keywords), 4), "found": found, "missing": missing}


LLM_JUDGE_SYSTEM_PROMPT = (
    "Bạn là giám khảo đánh giá độ trung thực (faithfulness) của một câu trả "
    "lời tài chính so với các đoạn trích nguồn được cung cấp. Faithfulness "
    "nghĩa là MỌI khẳng định trong câu trả lời đều có thể truy nguyên trực "
    "tiếp về nội dung trong đoạn trích - không thêm số liệu, sự kiện, hay "
    "suy diễn nào không có trong đoạn trích.\n"
    "Chấm điểm 0.0 đến 1.0 (có thể dùng số thập phân), theo thang:\n"
    "1.0 = mọi khẳng định đều có căn cứ trực tiếp trong đoạn trích.\n"
    "0.5 = phần lớn có căn cứ nhưng có ít nhất một chi tiết không rõ nguồn.\n"
    "0.0 = câu trả lời chứa thông tin bịa đặt hoặc mâu thuẫn với đoạn trích.\n"
    "CHỈ trả lời bằng một số duy nhất (vd: 0.8), không giải thích thêm."
)


def llm_judge_faithfulness(question: str, answer: str, context: str) -> dict:
    """RAGAS-style faithfulness (RQ1, RQ4), using the project's own local
    Qwen3 (via src/models/llm_factory.py) as the judge instead of a paid
    API judge - same principle as RAGAS (Es et al., cited in
    Thesis_Direction_Redesign.pdf section 6.6), adapted to a zero-cost local
    model since that's this project's only LLM budget. Requires Ollama
    running - this is the one function in this module that makes a real
    network call, everything else here is pure/offline. Returns
    {"score": float|None, "raw": str} - `score` is None if the judge's
    output couldn't be parsed as a number (rare but possible with an 8B
    model; caller should treat that as "needs manual review", not silently
    drop it as 0)."""
    from src.models import llm_factory

    prompt = (
        f"Câu hỏi: {question}\n\n"
        f"Đoạn trích nguồn:\n{context}\n\n"
        f"Câu trả lời cần chấm:\n{answer}\n\n"
        f"Điểm faithfulness (0.0-1.0):"
    )
    raw = llm_factory.generate(prompt, system=LLM_JUDGE_SYSTEM_PROMPT, temperature=0.0)
    try:
        score = float(raw.strip().split()[0].rstrip("."))
        score = max(0.0, min(1.0, score))
    except (ValueError, IndexError):
        score = None
    return {"score": score, "raw": raw}


def temporal_anchor_error_rate(results: list) -> dict:
    """RQ4 - "tỷ lệ nhầm kỳ báo cáo trước/sau temporal anchor". `results` is
    a list of {"expected_year": int, "actual_year": int|None} dicts - one
    per rag_temporal benchmark question, comparing the year of the
    top-ranked source actually used against the year the benchmark expects.
    Run this twice (once with temporal anchoring on, once with it forced
    off - see run_eval.py's --no-temporal-anchor flag) and compare the two
    error rates for the RQ4 before/after ablation the design doc specifies."""
    scored = [r for r in results if r.get("expected_year") is not None]
    if not scored:
        return {"error_rate": None, "n": 0}
    errors = sum(1 for r in scored if r.get("actual_year") != r["expected_year"])
    return {"error_rate": round(errors / len(scored), 4), "n": len(scored), "errors": errors}


def nhanh_a_vs_b_summary(detail: list) -> dict:
    """RQ2 (Week 8) - tổng hợp so sánh Nhánh A (DAG, src/structured/query_builder.py -
    schema-constrained, không dùng LLM để sinh SQL) vs Nhánh B (Agentic
    Text-to-SQL, src/structured/text_to_sql_agent.py - LLM tự sinh SQL) trên
    CÙNG một câu hỏi/gold value mỗi item. `detail` là list dict do
    run_eval.run_nhanh_a_vs_b() sinh ra, mỗi phần tử có `nhanh_a_correct`
    (bool), `nhanh_b_correct` (bool), `nhanh_b_gate_failed` (None nếu Nhánh B
    thành công, hoặc một trong {"not_answerable","unsafe_statement",
    "schema_alignment","execution"} - xem text_to_sql_agent.answer()'s
    docstring), `nhanh_b_gate2b_would_flag` (bool|None - None nếu sqlglot
    chưa cài hoặc Nhánh B thành công/không tới Gate 2b).

    Ngoài accuracy 2 nhánh, trả về:
    - gate_failed_distribution: đếm theo từng gate_failed value của Nhánh B
      ("success" cho case Nhánh B ok=True - LƯU Ý "success" ở đây chỉ có
      nghĩa "SQL chạy được, không lỗi cú pháp/schema", KHÔNG có nghĩa "trả
      về đúng dữ liệu" - xem n_success_zero_rows ngay dưới).
    - n_success_zero_rows / success_zero_rows_rate: trong các case
      gate_failed=None ("success"), bao nhiêu case Nhánh B trả về 0 dòng
      (`row_count=0`) - PHÁT HIỆN THẬT 2026-07-28 (26 ticker thật): khi LLM
      viết `WHERE item_name = 'Doanh thu thuần'` (so khớp CHUỖI TUYỆT ĐỐI)
      thay vì `LIKE '%Doanh thu thuần%'`, câu SQL vẫn hợp lệ và CHẠY ĐƯỢC
      (không gate nào bắt được, vì đây không phải lỗi cú pháp/schema/thực
      thi - chỉ đơn giản là WHERE không khớp dòng nào trong dữ liệu thật,
      có thể do cách ghi `item_name` thật khác chuỗi LLM đoán) - đây là
      failure mode CHÍNH của Nhánh B trên bộ 26 ticker thật, không phải lỗi
      Postgres/execution như Gate 2b nhắm tới. `gate_failed_distribution`
      một mình sẽ đánh lừa người đọc là "hầu hết chạy thành công" nếu không
      đọc kèm 2 con số này.
    - gate2b_would_have_caught_rate: trong các case Nhánh B thất bại đúng ở
      gate "execution" (nghĩa là Cổng 1 an toàn + Cổng 2 regex ĐỀU pass,
      chỉ lộ ra khi chạy SQL thật trên Postgres), bao nhiêu % trong số đó
      Cổng 2b (AST, sqlglot) LẼ RA đã gắn cờ vấn đề TRƯỚC khi thực thi. Đây
      chính là con số "Cổng 2b giúp được bao nhiêu %" mà mục Gate 2b trong
      docs/PROGRESS.md ghi là "chưa đo được vì chưa chạy Nhánh B trên
      Postgres thật" - giờ đo được khi harness này chạy thật (2026-07-28:
      chưa có case "execution" nào trong lần chạy thật đầu tiên - toàn bộ
      thất bại của Nhánh B là zero-rows kiểu trên, không phải lỗi thực thi,
      nên gate2b_would_have_caught_rate=None trong lần chạy này - đúng, vì
      Gate 2b không nhắm tới loại lỗi "0 dòng" này)."""
    scored = [d for d in detail if "error" not in d]
    n = len(scored)
    if n == 0:
        return {"n": 0}
    # Items with no real expected_value (e.g. ab_hoaphat_fullname_totalassets_2025 -
    # kept purely to observe gate_failed/full-name-resolution behavior, not
    # scored for correctness since there's no verified gold number for it -
    # see its "notes" field in benchmark.jsonl) are excluded from the
    # accuracy denominator so a deliberately-ungraded qualitative item can't
    # silently drag nhanh_a_accuracy/nhanh_b_accuracy down.
    accuracy_scored = [d for d in scored if not d.get("qualitative_only")]
    n_acc = len(accuracy_scored)
    a_correct = sum(1 for d in accuracy_scored if d.get("nhanh_a_correct"))
    b_correct = sum(1 for d in accuracy_scored if d.get("nhanh_b_correct"))
    gate_dist: dict = {}
    for d in scored:
        g = d.get("nhanh_b_gate_failed") or "success"
        gate_dist[g] = gate_dist.get(g, 0) + 1
    success_items = [d for d in scored if d.get("nhanh_b_gate_failed") is None]
    zero_row_items = [d for d in success_items if d.get("nhanh_b_row_count") == 0]
    execution_failures = [d for d in scored if d.get("nhanh_b_gate_failed") == "execution"]
    gate2b_flagged = [d for d in execution_failures if d.get("nhanh_b_gate2b_would_flag") is True]
    gate2b_measurable = [d for d in execution_failures if d.get("nhanh_b_gate2b_would_flag") is not None]
    gate2b_rate = (round(len(gate2b_flagged) / len(gate2b_measurable), 4)
                   if gate2b_measurable else None)
    return {
        "n": n,
        "n_scored_for_accuracy": n_acc,
        "n_qualitative_only": n - n_acc,
        "nhanh_a_accuracy": round(a_correct / n_acc, 4) if n_acc else None,
        "nhanh_b_accuracy": round(b_correct / n_acc, 4) if n_acc else None,
        "gate_failed_distribution": gate_dist,
        "n_success_zero_rows": len(zero_row_items),
        "success_zero_rows_rate": (round(len(zero_row_items) / len(success_items), 4)
                                    if success_items else None),
        "n_execution_failures": len(execution_failures),
        "n_execution_failures_gate2b_measurable": len(gate2b_measurable),
        "gate2b_would_have_caught_rate": gate2b_rate,
    }


def sparse_coverage_summary(detail: list) -> dict:
    """Sparse-coverage / no-hallucination check (added 2026-09-08) for the
    "sparse_coverage_no_hallucination" benchmark category - 11 tickers
    confirmed (2026-09-07, from a real 222-row coverage query the project
    owner ran) to have near-zero disclosure_chunks/financial_line_items
    coverage. For a ticker in this group, the CORRECT system behavior is to
    say so - rag_pipeline.NO_CONTEXT_MSG/NO_CONTEXT_MSG_EN, or a clearly
    source_tier="web_fallback" answer - NOT a confident disclosure-tier
    answer, since there is essentially nothing in disclosure_chunks for the
    model to have faithfully grounded that answer in. This is a STRUCTURAL
    check (does the system admit missing data vs fabricate), not a
    content-correctness check - there is deliberately no gold answer text
    to compare against, same reasoning as absa_structural_check()'s
    "consistent with its own rules" framing above.

    `detail` items: {"id", "ticker", "question", "degraded_correctly" (bool)}
    - see run_eval.run_sparse_coverage() for how `degraded_correctly` is
    computed from a real rag_pipeline.answer() call."""
    n = len(detail)
    if n == 0:
        return {"n": 0}
    n_ok = sum(1 for d in detail if d.get("degraded_correctly"))
    return {
        "n": n,
        "degraded_correctly_rate": round(n_ok / n, 4),
        "n_possible_hallucinations": n - n_ok,
        "note": ("degraded_correctly_rate = tỷ lệ hệ thống THỪA NHẬN thiếu dữ liệu "
                 "(NO_CONTEXT_MSG/_EN hoặc web_fallback có gắn nhãn rõ) thay vì trả lời "
                 "tự tin không có căn cứ, trên các mã đã xác nhận gần như không có "
                 "disclosure_chunks. KHÔNG đo 'trả lời đúng nội dung' - không có gold "
                 "value định tính cho nhóm câu hỏi này, chỉ đo tính trung thực khi thiếu "
                 "dữ liệu (faithfulness dưới điều kiện xấu nhất)."),
    }


def regime_ablation_summary(with_signal: list, without_signal: list) -> dict:
    """RQ3 (Week 8) - so sánh MÔ TẢ (không có nhãn đúng/sai) giữa bật/tắt
    use_regime_signal (src/retrieval/rag_pipeline.answer()) trên cùng bộ
    câu hỏi, cùng thứ tự. `with_signal`/`without_signal` là 2 list kết quả
    rag_pipeline.answer(...) - mỗi phần tử có `sources` (list nguồn, dùng
    `source_file` để so thứ tự) và `market_regime` (regime thật tại thời
    điểm chạy, ví dụ "bear"/"volatile"/"bull"/"stable").

    KHÔNG tính "accuracy"/"đúng hơn" ở đây - RQ3 hiện chưa có ground truth
    relevance (xem docs/Ghi_chu_thao_luan_phan_bien_2026-07-26.md mục 3.2:
    "RQ3 chỉ nên giữ nếu có ground truth relevance, ablation có/không
    regime"). Phần ablation này chỉ đo MỨC ĐỘ rerank có làm thay đổi thứ tự
    nguồn hay không khi regime là bear/volatile - không khẳng định thay đổi
    đó tốt hơn hay tệ hơn."""
    n = min(len(with_signal), len(without_signal))
    if n == 0:
        return {"n": 0}
    regime_active = 0
    order_changed = 0
    order_changed_when_regime_active = 0
    for w, wo in zip(with_signal[:n], without_signal[:n]):
        regime = w.get("market_regime")
        is_active = regime in ("bear", "volatile")
        if is_active:
            regime_active += 1
        w_order = [s.get("source_file") for s in w.get("sources", [])]
        wo_order = [s.get("source_file") for s in wo.get("sources", [])]
        changed = w_order != wo_order
        if changed:
            order_changed += 1
        if changed and is_active:
            order_changed_when_regime_active += 1
    return {
        "n": n,
        "regime_active_rate": round(regime_active / n, 4),
        "rerank_order_changed_rate": round(order_changed / n, 4),
        "rerank_order_changed_when_regime_active": order_changed_when_regime_active,
        "note": ("Mô tả, không phải accuracy - cần ground truth relevance để "
                  "đánh giá tốt/xấu (xem Ghi_chu_thao_luan_phan_bien_2026-07-26.md mục 3.2)."),
    }


def causal_estimator_comparison(t_result: dict, dml_result: dict) -> dict:
    """RQ6 (Week 8) - đối chiếu T-learner (src/causal/treatment_effect.py's
    estimate_ate(), trả về `ate_t_learner`/`ate_95ci`) và R-learner/DML
    (estimate_ate_dml(), trả về `ate_dml_marginal`/`ate_dml_marginal_95ci`/
    `translated_group_contrast`) trên CÙNG cặp treatment/outcome/covariates.

    KHÔNG có "đúng/sai" ở đây - không có ground-truth causal effect thật
    trên dữ liệu quan sát thật (observational, không phải RCT). Chỉ báo
    cáo 2 ước lượng cạnh nhau và liệu khoảng tin cậy 95% của 2 phương pháp
    có overlap hay không - 2 phương pháp lệch giả định khác nhau (T-learner:
    median-split nhóm rời rạc; DML: hiệu ứng biên tuyến tính liên tục), nên
    overlap không tự động nghĩa là 'đúng cả hai', chỉ là 'nhất quán với
    nhau' - dùng `translated_group_contrast` (đã quy đổi cùng đơn vị với
    T-learner) để so, không so trực tiếp `ate_dml_marginal` với
    `ate_t_learner` (khác đơn vị, xem estimate_ate_dml()'s docstring)."""
    if not t_result.get("ok") or not dml_result.get("ok"):
        return {"ok": False, "t_learner": t_result, "dml": dml_result}
    t_ci = t_result.get("ate_95ci", [None, None])
    d_ci_marginal = dml_result.get("ate_dml_marginal_95ci", [None, None])
    translated = dml_result.get("translated_group_contrast")
    # Overlap check is done on the TRANSLATED (group-contrast-unit) DML
    # estimate, but ate_dml_marginal_95ci is in marginal-effect units, not
    # translated units - there is no exact translated CI (translation is a
    # point-estimate scalar multiply, not distribution-preserving), so we
    # only compare point estimates being on the same side / same order of
    # magnitude, and separately report whether T-learner's CI contains the
    # translated DML point estimate (a weaker, but honest, check).
    ci_contains_translated = None
    if t_ci[0] is not None and t_ci[1] is not None and translated is not None:
        ci_contains_translated = t_ci[0] <= translated <= t_ci[1]
    return {
        "ok": True,
        "t_learner_ate": t_result.get("ate_t_learner"),
        "t_learner_ci_95": t_ci,
        "dml_marginal_ate": dml_result.get("ate_dml_marginal"),
        "dml_marginal_ci_95": d_ci_marginal,
        "dml_translated_group_contrast": translated,
        "t_learner_ci_contains_dml_translated_point": ci_contains_translated,
        "note": ("dml_marginal_ate là hiệu ứng biên/1 đơn vị treatment (liên tục), KHÔNG cùng "
                 "đơn vị với t_learner_ate (median-split nhóm cao/thấp) - so sánh dùng "
                 "dml_translated_group_contrast, xem estimate_ate_dml()'s docstring."),
    }


ABSA_ASPECTS = ("profitability", "liquidity_leverage", "growth_outlook", "governance", "risk_legal")
ABSA_VALID_SENTIMENTS = {"positive", "negative", "neutral", "not_mentioned"}


def absa_structural_check(result: dict) -> dict:
    """RQ5 (Week 8) - kiểm tra CẤU TRÚC/tính nhất quán nội tại của kết quả
    ABSA (src/absa/absa_pipeline.py's analyze_ticker(), trả về `ok`/
    `aspects` dict keyed bởi 5 khía cạnh, mỗi khía cạnh {"sentiment",
    "evidence"}). KHÔNG chấm đúng/sai sentiment - chưa có bộ gán nhãn con
    người (xem docs/Ghi_chu_thao_luan_phan_bien_2026-07-26.md mục 3.3:
    "nếu không có bộ gán nhãn ABSA nhỏ do con người, RQ5 chỉ nên là
    demo/exploratory, không nên báo cáo 'accuracy'").

    Kiểm tra 3 điều, tất cả đều suy được từ chính output mà không cần nhãn
    đúng-sai bên ngoài: (1) đủ 5 khía cạnh có mặt, (2) sentiment mỗi khía
    cạnh nằm trong 4 giá trị hợp lệ, (3) khía cạnh không phải not_mentioned
    có kèm evidence (không rỗng) - tức là kiểm tra pipeline "tự nhất quán
    với chính quy tắc nó phải tuân theo" (SYSTEM_PROMPT's rule 2/3), không
    phải "trả lời đúng nội dung tài chính"."""
    if not result.get("ok"):
        return {"ok": False, "structurally_valid": False, "problems": result.get("problems")}
    aspects = result.get("aspects") or {}
    missing = [a for a in ABSA_ASPECTS if a not in aspects]
    invalid_sentiment = [a for a, v in aspects.items()
                          if a in ABSA_ASPECTS and v.get("sentiment") not in ABSA_VALID_SENTIMENTS]
    missing_evidence = [a for a, v in aspects.items()
                         if a in ABSA_ASPECTS and v.get("sentiment") not in (None, "not_mentioned")
                         and not v.get("evidence")]
    return {
        "ok": True,
        "all_5_aspects_present": not missing, "missing_aspects": missing,
        "all_sentiments_valid": not invalid_sentiment, "invalid_sentiment_aspects": invalid_sentiment,
        "evidence_present_when_needed": not missing_evidence, "missing_evidence_aspects": missing_evidence,
        "structurally_valid": not missing and not invalid_sentiment and not missing_evidence,
    }


def baseline_dag_only_summary(detail: list) -> dict:
    """RQ1/RQ2 (Week 8 baseline #1, docs/research_scope_map.html's
    "base-dagonly") - đo NẾU HỆ THỐNG CHỈ CÓ Nhánh A (schema-constrained
    DAG lookup, không Router, không What-if calculator, không RAG/PDF) thì
    trả lời được bao nhiêu % trong bộ câu hỏi hỗn hợp thật (whatif_exact_match
    + rag_faithfulness + rag_temporal - những câu KHÔNG phải lúc nào cũng
    hợp với DAG). `detail` do run_eval.run_baseline_dag_only() sinh ra, mỗi
    phần tử có `category`, `answerable` (bool - execute() trả `ok=True`),
    `correct` (bool|None - None cho câu rag_* vì không có "giá trị đúng"
    dạng số để so, chỉ có whatif_exact_match mới so được).

    Tách riêng theo category vì đây CHÍNH LÀ điểm của baseline này: DAG-only
    được kỳ vọng gần như KHÔNG trả lời được câu rag_* (không parse ra được
    ms_code từ câu hỏi định tính "rủi ro pháp lý là gì" - resolve_indicator()
    chỉ khớp alias chỉ tiêu tài chính, không khớp câu hỏi định tính), và với
    whatif_exact_match, DAG-only nhiều nhất chỉ lấy được giá trị TRẠNG THÁI
    HIỆN TẠI (vd revenue hiện tại) chứ không tính được giá trị SAU KỊCH BẢN
    giả định (vd gross_profit_after nếu COGS tăng 5%) vì đó là phép tính
    What-if, không phải lookup - nên `correct` gần như luôn False cho
    whatif_exact_match dù `answerable` có thể True (lấy nhầm giá trị hiện
    tại thay vì giá trị kịch bản)."""
    by_cat: dict = {}
    for d in detail:
        if "error" in d:
            continue
        cat = d.get("category", "unknown")
        by_cat.setdefault(cat, {"n": 0, "n_answerable": 0, "n_correct": 0, "n_scored_correct": 0})
        by_cat[cat]["n"] += 1
        if d.get("answerable"):
            by_cat[cat]["n_answerable"] += 1
        if d.get("correct") is not None:
            by_cat[cat]["n_scored_correct"] += 1
            if d.get("correct"):
                by_cat[cat]["n_correct"] += 1
    result = {}
    for cat, c in by_cat.items():
        result[cat] = {
            "n": c["n"],
            "answerable_rate": round(c["n_answerable"] / c["n"], 4) if c["n"] else None,
            "correct_rate": (round(c["n_correct"] / c["n_scored_correct"], 4)
                              if c["n_scored_correct"] else None),
        }
    return result


def baseline_rag_only_summary(detail: list) -> dict:
    """RQ1/RQ2 (Week 8 baseline #2, "base-ragonly") - đo NẾU HỆ THỐNG CHỈ
    CÓ RAG (không neo thời gian, không What-if calculator/DAG) thì có tự
    "mò" ra được giá trị số CHÍNH XÁC của một câu hỏi what-if hay không.
    `detail` do run_eval.run_baseline_rag_only_whatif() sinh ra trên các
    câu whatif_exact_match, mỗi phần tử có `any_expected_field_found` (bool)
    - có ít nhất 1 trong 5 expected_fields (đã format kiểu số Việt Nam)
    xuất hiện nguyên văn trong câu trả lời RAG hay không.

    Kỳ vọng: gần như luôn False, vì `gross_profit_after` là giá trị TÍNH
    RA từ kịch bản giả định (COGS +5%...), không tồn tại sẵn trong bất kỳ
    tài liệu công bố thông tin nào để RAG "tìm thấy" - đối lập với nhánh
    What-if (phép tính xác định, không đoán) đã đạt 26/26. Baseline này
    hoàn thiện bộ 3 so sánh cho RQ2: DAG-only (không tính được kịch bản),
    RAG-only (không tính được kịch bản), What-if branch (tính đúng 26/26)."""
    scored = [d for d in detail if "error" not in d]
    n = len(scored)
    if n == 0:
        return {"n": 0}
    n_found = sum(1 for d in scored if d.get("any_expected_field_found"))
    return {
        "n": n,
        "any_expected_field_found_rate": round(n_found / n, 4),
        "note": ("Tỷ lệ RAG-only 'mò trúng' được ít nhất 1 số liệu kịch bản - kỳ vọng "
                  "gần 0 vì đây là giá trị tính ra từ giả định, không có sẵn trong tài liệu "
                  "nào để truy xuất. Đối chứng với What-if branch (deterministic) đạt 26/26."),
    }


def synthetic_causal_calibration(t_result: dict, dml_result: dict, true_marginal_effect: float,
                                  true_group_contrast: float) -> dict:
    """RQ6 (Week 8 baseline #6b, "base-causalonly"'s "đo sai số ATE so với
    giá trị mô phỏng đã biết") - đo BIAS của T-learner/DML trên một panel
    TỔNG HỢP (synthetic, KHÔNG phải dữ liệu công ty thật - xem
    run_eval.run_synthetic_causal_calibration() để thấy DGP tự tạo) có hiệu
    ứng nhân quả biết trước CHÍNH XÁC (vì tự sinh dữ liệu theo công thức đã
    biết `true_marginal_effect`, không phải ước lượng trên dữ liệu quan sát
    thật - dữ liệu công ty thật trong `treatment_effect.py` KHÔNG BAO GIỜ
    có "giá trị đúng đã biết" để so, đây là lý do MUST dùng synthetic data
    cho riêng phép kiểm định hiệu chuẩn này, không vi phạm nguyên tắc
    "không bịa gold value" của dự án vì được gắn nhãn RÕ RÀNG là kiểm định
    tổng hợp, không trình bày như bằng chứng tài chính thật).

    So `t_result["ate_t_learner"]` và `dml_result["translated_group_contrast"]`
    với `true_group_contrast` (cùng đơn vị - "group contrast" dưới median
    split, tính trực tiếp từ chính panel tổng hợp bằng công thức DGP đã
    biết), và `dml_result["ate_dml_marginal"]` với `true_marginal_effect`
    (cùng đơn vị - hiệu ứng biên/1 đơn vị treatment, vì DGP tuyến tính hằng
    số theo thiết kế nên đây chính là con số DML ước lượng đúng nghĩa)."""
    if not t_result.get("ok") or not dml_result.get("ok"):
        return {"ok": False, "t_learner": t_result, "dml": dml_result}
    t_ate = t_result.get("ate_t_learner")
    dml_marginal = dml_result.get("ate_dml_marginal")
    dml_translated = dml_result.get("translated_group_contrast")
    return {
        "ok": True,
        "true_group_contrast": true_group_contrast,
        "t_learner_ate": t_ate,
        "t_learner_bias": round(t_ate - true_group_contrast, 6) if t_ate is not None else None,
        "dml_translated_group_contrast": dml_translated,
        "dml_translated_bias": (round(dml_translated - true_group_contrast, 6)
                                 if dml_translated is not None else None),
        "true_marginal_effect": true_marginal_effect,
        "dml_marginal_ate": dml_marginal,
        "dml_marginal_bias": (round(dml_marginal - true_marginal_effect, 6)
                               if dml_marginal is not None else None),
    }
