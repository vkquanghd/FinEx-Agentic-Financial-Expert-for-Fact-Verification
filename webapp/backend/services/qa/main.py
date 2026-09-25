"""QA service (mặt NGƯỜI DÙNG) - bọc router S3 của thesis thành API.

POST /ask: câu hỏi -> khai triển viết tắt (deterministic, trước LLM) ->
resolve ngữ cảnh hội thoại (RQ7 LLMPromptOnlyResolver, tắt được) ->
route() (DAG/RAG/what-if/causal/compare) -> log answer_run (HITL layer 1)
+ auto-enqueue nếu retrieval similarity thấp (uncertainty sampling).

Service này KHÔNG nhận feedback/rating - đó là việc của feedback service
(ranh giới microservice: QA trả lời, feedback thu tín hiệu người dùng).
"""
import time
import uuid

from pydantic import BaseModel, Field

from webapp.backend.shared.bootstrap import make_app  # noqa: F401  (sys.path side effect)

from src.core.feedback_store import (  # noqa: E402
    extract_evidence,
    log_answer_run,
    maybe_enqueue_low_similarity,
)
from src.dialogue.context_resolver import ConversationTurn, LLMPromptOnlyResolver  # noqa: E402
from src.router.query_router import route  # noqa: E402
from src.structured.abbreviations import expand_display  # noqa: E402

app = make_app(
    "qa-service",
    "Mặt người dùng: hỏi đáp tài chính qua Router S3 (DAG/RAG/what-if/causal/compare).",
)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    session_id: str | None = None
    # các câu hỏi user TRƯỚC ĐÓ trong phiên (cũ -> mới) cho resolver ngữ cảnh
    history: list[str] = []
    use_context_resolver: bool = True


class AskResponse(BaseModel):
    answer: str
    branch: str
    fallback_from: str | None
    intent_reason: str
    resolved_question: str
    original_question: str
    sources: list[dict] = []
    latency_ms: int
    answer_run_id: int | None
    session_id: str


def _resolve_context(history: list[str], question: str) -> str:
    if not history:
        return question
    try:
        turns = [ConversationTurn(i, "user", h) for i, h in enumerate(history)]
        return LLMPromptOnlyResolver().resolve(turns, question).standalone_question
    except Exception:
        return question  # resolver lỗi -> dùng nguyên câu, không vỡ lượt hỏi


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    session_id = req.session_id or f"web_{uuid.uuid4().hex[:12]}"
    expanded = expand_display(req.question)
    resolved = _resolve_context(req.history, expanded) if req.use_context_resolver else expanded

    t0 = time.perf_counter()
    result = route(resolved)
    latency_ms = int((time.perf_counter() - t0) * 1000)

    sources = []
    raw = result.get("raw") or {}
    if isinstance(raw, dict):
        for s in raw.get("sources") or []:
            sources.append({k: s.get(k) for k in ("ticker", "doctype", "source_file", "page", "year", "similarity")})

    answer_run_id = None
    try:
        evidence = extract_evidence(raw if isinstance(raw, dict) else None, result["branch"])
        answer_run_id = log_answer_run(
            session_id=session_id,
            question_text=req.question,
            resolved_question=resolved,
            branch=result["branch"],
            fallback_from=result.get("fallback_from"),
            answer_text=result["answer"],
            evidence=evidence,
            latency_ms=latency_ms,
        )
        maybe_enqueue_low_similarity(answer_run_id=answer_run_id, evidence=evidence)
    except Exception:
        pass  # HITL logging không bao giờ làm vỡ câu trả lời

    return AskResponse(
        answer=result["answer"],
        branch=result["branch"],
        fallback_from=result.get("fallback_from"),
        intent_reason=result.get("intent_reason", ""),
        resolved_question=resolved,
        original_question=req.question,
        sources=sources,
        latency_ms=latency_ms,
        answer_run_id=answer_run_id,
        session_id=session_id,
    )
