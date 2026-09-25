"""RQ5 (Week 7 of the locked 10-week plan) - zero-shot LLM Aspect-Based
Sentiment Analysis (ABSA) over Vietnamese corporate disclosure text. This
is the system's 7th classify_intent() branch (src/router/rules.py) - per
docs/architecture_diagram.html's own framing: "ABSA sẽ là nhánh
classify_intent() thứ 7... điểm neo tự nhiên đã có sẵn trong kiến trúc
Router hiện tại."

RQ5: "ABSA bằng zero-shot LLM prompting có cung cấp sắc thái cảm xúc theo
khía cạnh tài chính hữu ích, ở mức chấp nhận được, mà không cần dữ liệu
annotate/fine-tune không?" Per Thesis_Direction_Redesign.pdf section 2:
"ABSA (zero-shot LLM) - LÀM - tối giản - Giữ RQ về khía cạnh cảm xúc bằng
prompt LLM thay vì fine-tune PhoBERT+LoRA; giữ khả năng mở rộng sau
thesis." Fine-tuning ABSA (PhoBERT+LoRA) is explicitly deferred to future
work (section 9), not in the 10-week scope.

5 khía cạnh tài chính (defined here - "5 khía cạnh" is referenced
repeatedly across docs/PROGRESS.md, docs/research_scope_map.html and
docs/architecture_diagram.html but never actually enumerated anywhere
else in the project). Chosen to match standard dimensions used in
financial-disclosure sentiment analysis, and deliberately chosen to line
up with this project's own existing document-type taxonomy
(rag_pipeline.classify_doctype()'s 6 labels - kqkd_explanation/
governance_report/annual_report map naturally onto aspects 1/4/3 below):
  1. profitability      - Kết quả kinh doanh / Lợi nhuận
  2. liquidity_leverage  - Khả năng thanh toán & Đòn bẩy tài chính
  3. growth_outlook      - Tăng trưởng & Triển vọng
  4. governance          - Quản trị doanh nghiệp & Ban điều hành
  5. risk_legal          - Rủi ro hoạt động & Pháp lý

Design (zero-shot, no fine-tune, no annotated corpus - per section 9's
locked decision): ONE LLM call per company, given REAL RAG-retrieved
disclosure text as context - never analyzes "in a vacuum" off the LLM's
own prior knowledge of the company. Reuses
src/retrieval/vector_store.similarity_search() (same retrieval
infrastructure RAG uses, no new retrieval code). For EACH of the 5
aspects, the LLM must output a sentiment in {"positive","negative",
"neutral","mixed","not_mentioned"} plus a short evidence quote -
"not_mentioned" is a first-class, EXPECTED value (same "don't guess when
the text doesn't cover it" philosophy as src/models/generation.py's
SYSTEM_PROMPT rule 2), since a handful of retrieved chunks won't always
discuss all 5 aspects.

"mixed" added 2026-07-29 evening (Task #12 from the round-2 architecture
review, docs/PROGRESS.md has the full writeup): real disclosure text
routinely praises and criticizes the SAME aspect in the same passage (e.g.
"lợi nhuận quý này tăng so với cùng kỳ, nhưng vẫn thấp hơn kế hoạch đề ra"
- profitability is both up YoY and behind plan in one sentence). Forcing
that into a single positive/negative/neutral bucket either loses the
negative signal, loses the positive signal, or - if forced into "neutral"
- looks indistinguishable from a passage that said nothing evaluative at
all, which is a genuinely different case from "praised and criticized".
"mixed" is a first-class fourth non-not_mentioned value, not a fallback:
the LLM is instructed to use it specifically when genuine positive AND
negative signal both appear for the same aspect, not as a hedge for
uncertainty (that's what "neutral" and "not_mentioned" already cover).

Academic basis (Thesis_Direction_Redesign.pdf section 6.5): "Evaluating
Zero-Shot Multilingual Aspect-Based Sentiment Analysis with Large Language
Models" - zero-shot LLM ABSA reaches acceptable (if lower than
fine-tuned-specialist) quality without needing an annotated corpus, which
is the whole reason RQ5 chose this over fine-tuning PhoBERT+LoRA within
the 10-week scope.

NOT run end-to-end against a real Ollama/pgvector stack by the environment
that wrote this (same situation as every other DB/LLM-backed module in
this project) - verified here via mocked LLM output + mocked retrieval
only (see docs/PROGRESS.md's "Week 7" entry for what was and wasn't
tested).
"""
import json
import re
import time

from src.models import llm_factory
from src.retrieval.vector_store import similarity_search

ASPECTS = {
    "profitability": "Kết quả kinh doanh / Lợi nhuận",
    "liquidity_leverage": "Khả năng thanh toán & Đòn bẩy tài chính",
    "growth_outlook": "Tăng trưởng & Triển vọng",
    "governance": "Quản trị doanh nghiệp & Ban điều hành",
    "risk_legal": "Rủi ro hoạt động & Pháp lý",
}

VALID_SENTIMENTS = {"positive", "negative", "neutral", "mixed", "not_mentioned"}

# Broad composite retrieval query (2026-07-24) - deliberately covers all 5
# aspect areas in ONE query/retrieval pass rather than 5 separate ones (5x
# the embedding + LLM calls for a Week-7, zero-shot-only scope). Accepted
# trade-off: a single retrieval pass may not surface the single best chunk
# for every aspect equally well, but is far cheaper and still grounds the
# analysis in real disclosure text rather than the LLM's own prior.
_RETRIEVAL_QUERY = (
    "kết quả kinh doanh lợi nhuận doanh thu khả năng thanh toán đòn bẩy nợ vay "
    "tăng trưởng triển vọng chiến lược quản trị doanh nghiệp hội đồng quản trị "
    "rủi ro pháp lý hoạt động"
)

_MAX_CONTEXT_CHARS = 6000  # keep the prompt bounded - local Qwen3:8b inference, not an unbounded-context hosted API

SYSTEM_PROMPT = (
    "Bạn là công cụ phân tích cảm xúc theo khía cạnh (Aspect-Based Sentiment "
    "Analysis) cho tài liệu công bố thông tin tài chính doanh nghiệp Việt Nam. "
    "QUY TẮC BẮT BUỘC:\n"
    "1. CHỈ đánh giá dựa trên đoạn trích được cung cấp - KHÔNG dùng kiến thức "
    "nền để suy đoán tình hình công ty.\n"
    "2. Với MỖI khía cạnh trong 5 khía cạnh sau, phải trả về sentiment là "
    "CHÍNH XÁC một trong 5 giá trị: \"positive\", \"negative\", \"neutral\", "
    "\"mixed\", hoặc \"not_mentioned\".\n"
    "   - \"not_mentioned\": đoạn trích KHÔNG đề cập gì tới khía cạnh đó - "
    "KHÔNG được đoán bừa sentiment khi không có căn cứ trong đoạn trích.\n"
    "   - \"mixed\": đoạn trích có CẢ tín hiệu tích cực LẪN tiêu cực cho "
    "CÙNG khía cạnh đó (ví dụ: lợi nhuận tăng so với cùng kỳ NHƯNG vẫn thấp "
    "hơn kế hoạch) - chỉ dùng khi thực sự có cả hai chiều, KHÔNG dùng như "
    "một cách né tránh khi không chắc chắn (trường hợp không chắc chắn/ "
    "trung tính thật sự thì dùng \"neutral\").\n"
    "3. Với mỗi khía cạnh KHÔNG phải not_mentioned (bao gồm cả mixed), phải "
    "kèm evidence là một cụm trích dẫn ngắn (tối đa khoảng 20 từ) lấy nguyên "
    "văn từ đoạn trích, cho thấy rõ căn cứ. Nếu not_mentioned thì evidence = "
    "null. Với mixed, evidence nên bao gồm (hoặc tóm tắt) cả phần tích cực "
    "và phần tiêu cực nếu có thể.\n"
    "4. CHỈ trả về ĐÚNG một object JSON hợp lệ theo schema bên dưới - KHÔNG "
    "giải thích, KHÔNG dùng markdown/code fence, KHÔNG viết chữ nào khác "
    "ngoài chính JSON.\n\n"
    "Schema JSON bắt buộc (5 khía cạnh, đúng các key sau):\n"
    "{\n"
    '  "profitability": {"sentiment": "...", "evidence": "..." hoặc null},\n'
    '  "liquidity_leverage": {"sentiment": "...", "evidence": "..." hoặc null},\n'
    '  "growth_outlook": {"sentiment": "...", "evidence": "..." hoặc null},\n'
    '  "governance": {"sentiment": "...", "evidence": "..." hoặc null},\n'
    '  "risk_legal": {"sentiment": "...", "evidence": "..." hoặc null}\n'
    "}\n"
)

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(raw: str) -> dict:
    """Best-effort JSON extraction: try a direct parse first, then fall
    back to grabbing the first {...} block if the LLM wrapped it in
    markdown/stray commentary despite rule 4. Lets json.JSONDecodeError
    propagate if even that fails - caller (analyze_text()) turns it into
    an ok=False result, never crashes the whole request."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    match = _JSON_BLOCK_RE.search(raw)
    if not match:
        raise ValueError("Không tìm thấy khối JSON nào trong output của LLM.")
    return json.loads(match.group(0))


def _validate_result(parsed: dict) -> list:
    """Returns a list of problem strings (empty = valid). Checks all 5
    expected aspect keys are present with a valid sentiment value - does
    NOT try to repair a malformed result, just reports what's wrong (same
    "report, don't guess-fix" philosophy as text_to_sql_agent.py's gates)."""
    if not isinstance(parsed, dict):
        return [f"Kết quả JSON không phải object: {type(parsed).__name__}."]
    problems = []
    for key in ASPECTS:
        if key not in parsed:
            problems.append(f"Thiếu khía cạnh '{key}' trong kết quả JSON.")
            continue
        entry = parsed[key]
        if not isinstance(entry, dict) or "sentiment" not in entry:
            problems.append(f"Khía cạnh '{key}' thiếu trường 'sentiment'.")
            continue
        if entry["sentiment"] not in VALID_SENTIMENTS:
            problems.append(f"Khía cạnh '{key}' có sentiment không hợp lệ: '{entry['sentiment']}'.")
    return problems


def analyze_text(text: str) -> dict:
    """Core zero-shot ABSA call over an already-retrieved block of text (no
    retrieval here - see analyze_ticker() for the retrieval-integrated
    entry point the Router actually uses). Returns:
    {"ok", "aspects" (dict keyed by ASPECTS, or the raw-but-invalid parse
    on failure, or None), "problems", "raw_llm_output", "elapsed_s"}"""
    start = time.time()
    raw = llm_factory.generate(text[:_MAX_CONTEXT_CHARS], system=SYSTEM_PROMPT)
    elapsed = time.time() - start

    try:
        parsed = _extract_json(raw)
    except (ValueError, json.JSONDecodeError) as e:
        return {"ok": False, "aspects": None, "problems": [f"Không parse được JSON: {e}"],
                "raw_llm_output": raw, "elapsed_s": round(elapsed, 2)}

    problems = _validate_result(parsed)
    if problems:
        return {"ok": False, "aspects": parsed, "problems": problems,
                "raw_llm_output": raw, "elapsed_s": round(elapsed, 2)}

    return {"ok": True, "aspects": parsed, "problems": [], "raw_llm_output": raw,
            "elapsed_s": round(elapsed, 2)}


def analyze_ticker(ticker: str, top_k: int = 10) -> dict:
    """Retrieval-integrated entry point - what the Router's ABSA branch
    actually calls. Fetches real disclosure chunks for `ticker` (reusing
    vector_store.similarity_search(), same infra as RAG), concatenates
    them (bounded by _MAX_CONTEXT_CHARS), and runs analyze_text() over that
    real, grounded context. Returns the same dict as analyze_text() plus
    "sources" (the chunks actually used, for citation/transparency - same
    pattern as rag_pipeline.answer())."""
    chunks = similarity_search(_RETRIEVAL_QUERY, top_k=top_k, ticker=ticker)
    if not chunks:
        return {"ok": False, "aspects": None,
                "problems": [f"Không tìm thấy đoạn trích nào cho mã {ticker} trong pgvector."],
                "raw_llm_output": None, "elapsed_s": 0.0, "sources": []}

    combined = "\n\n".join(c["chunk_text"] for c in chunks)
    result = analyze_text(combined)
    result["sources"] = [
        {"ticker": c["ticker"], "doctype": c["doctype"], "source_file": c["source_file"],
         "page": c["page"], "similarity": round(c["similarity"], 4)}
        for c in chunks
    ]
    return result


def format_result(result: dict) -> str:
    """CLI/demo-friendly rendering - matches rag_pipeline.py/dag_pipeline.py's style."""
    if not result["ok"]:
        return f"[ABSA THẤT BẠI] {result['problems']}"
    lines = []
    for key, label in ASPECTS.items():
        entry = result["aspects"].get(key, {})
        sentiment = entry.get("sentiment", "?")
        evidence = entry.get("evidence")
        line = f"- {label}: {sentiment}"
        if evidence:
            line += f' ("{evidence}")'
        lines.append(line)
    if result.get("sources"):
        lines.append("\nNguồn:")
        for i, s in enumerate(result["sources"], start=1):
            lines.append(
                f"  [{i}] {s['ticker']} - {s['doctype']} - {s['source_file']} "
                f"trang {s['page']} (similarity={s['similarity']})"
            )
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    ticker = sys.argv[1] if len(sys.argv) > 1 else "HPG"
    print(f"ABSA (5 khía cạnh tài chính) cho mã: {ticker}")
    res = analyze_ticker(ticker)
    print(format_result(res))
    print(f"\n(elapsed: {res['elapsed_s']}s)")
