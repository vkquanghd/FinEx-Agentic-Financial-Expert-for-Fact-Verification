"""QA service (user-plane) - port 8101.

Bọc `src.router.query_router.route()` thành HTTP. KHÔNG chứa logic truy
xuất/suy luận nào của riêng nó: mọi thay đổi về chất lượng trả lời vẫn
diễn ra ở src/, service này chỉ lo vận chuyển và ghi nhật ký lượt trả lời.

Hai cách gọi cùng một việc:
  POST /ask         - đồng bộ, trả kết quả cuối. Dùng cho script/benchmark.
  GET  /ask/stream  - SSE, phát từng chặng (phân loại -> truy xuất -> sinh
                      câu trả lời). Dùng cho UI: người dùng thấy router
                      đang đi nhánh nào NGAY khi nó quyết định, thay vì
                      nhìn spinner câm trong 10-30 giây.

Ghi chú kỹ thuật: route() là hàm blocking (gọi Ollama, Postgres, pgvector).
Trong endpoint async, nó được đẩy sang threadpool bằng anyio.to_thread để
không chặn event loop - đúng khuyến cáo "đừng chạy code blocking trong
async def".
"""
import time
import uuid
from collections.abc import AsyncIterable
from contextlib import asynccontextmanager
from typing import Annotated, Any, Literal

import anyio
from fastapi import FastAPI, Query, Request
from fastapi.sse import EventSourceResponse, ServerSentEvent

from api.shared.config import ensure_repo_on_path, get_settings
from api.shared.schemas import AskRequest, AskResponse, CompanyRef, ServiceHealth

ensure_repo_on_path()

from src.router.query_router import route  # noqa: E402
from src.structured.abbreviations import expand_display  # noqa: E402
from src.dialogue.context_resolver import ConversationTurn, LLMPromptOnlyResolver  # noqa: E402
from src.core.lang_detect import detect_language, resolve_answer_language  # noqa: E402


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Nạp sẵn bảng alias chỉ tiêu để request đầu tiên không phải trả giá
    # import (~vài trăm ms). Không nạp trước model embedding: nó nặng và
    # chỉ nhánh RAG cần, để lazy-load đúng lúc hợp lý hơn.
    from src.structured.schema import INDICATOR_ALIASES

    app.state.n_aliases = len(INDICATOR_ALIASES)
    yield


app = FastAPI(
    title="Financial QA - QA Service",
    description="User-plane: hỏi đáp tài chính trên công bố thông tin doanh nghiệp VN.",
    version="1.0.0",
    lifespan=lifespan,
)

HistoryTurn = Annotated[str, Query(max_length=1200)]


def _to_sources(raw: dict | None) -> list[dict]:
    if not raw or not raw.get("sources"):
        return []
    # web_fallback sources (2026-09-07, see src/retrieval/rag_pipeline.py's
    # web_fallback_answer()) have a DIFFERENT shape - domain/source_url/
    # title/published_date instead of ticker/doctype/source_file/page - see
    # SourceRef's comment in api/shared/schemas.py.
    if raw.get("source_tier") == "web_fallback":
        return [
            {
                "domain": s.get("domain", ""),
                "source_url": s.get("source_url", ""),
                "title": s.get("title"),
                "published_date": s.get("published_date"),
                "similarity": s.get("similarity"),
            }
            for s in raw["sources"]
        ]
    return [
        {
            "ticker": s.get("ticker", ""),
            "doctype": s.get("doctype", ""),
            "source_file": s.get("source_file", ""),
            "page": s.get("page", 0),
            "year": s.get("year"),
            "similarity": s.get("similarity"),
            "retrieval": s.get("retrieval"),
        }
        for s in raw["sources"]
    ]


def _to_signals(raw: dict | None) -> dict[str, Any]:
    """Tín hiệu router đã dùng để quyết định - tách khỏi `sources` vì đây
    là siêu dữ liệu về CÁCH trả lời, không phải bằng chứng CHO câu trả lời."""
    if not raw:
        return {}
    # "numeric_verification" thêm 2026-09-25. rag_pipeline.verify_numeric_claims()
    # đã CHẠY trên mọi câu trả lời RAG từ lâu - so từng con số trong câu trả
    # lời với con số thật trong đoạn trích - nhưng kết quả dừng lại ở đây,
    # vì khoá đó không nằm trong danh sách này. Tức là cơ chế chống bịa số
    # của chính luận văn (slide 9: "numeric-claim checking") tính ra đúng
    # rồi bị vứt đi. Phát hiện khi một câu trả lời ghi 1,597,803,538,910
    # trong khi đoạn trích nó vừa dẫn ghi 159,780,353,891 - lệch 10 lần,
    # và không có gì cảnh báo.
    keys = ("market_regime", "temporal_note", "doctype_filter",
            "regime_rerank_note", "elapsed_s", "source_tier",
            "numeric_verification")
    return {k: raw[k] for k in keys if k in raw and raw[k] is not None}


def _run_route(question: str, top_k: int, strategy: str = "s3",
               ui_lang: str | None = None, answer_lang: str | None = None) -> dict:
    """Run one of the three demo systems using the same response contract.

    `ui_lang` is the interface toggle, forwarded to route() as a tie-breaker
    only (see AskRequest.ui_lang and lang_detect.resolve_answer_language()).
    S1/S2 are the thesis's ablation baselines and are deliberately left
    untouched: they must keep behaving exactly as they did when their
    benchmark numbers were produced, so a language change here would make
    Chapter 4's S1/S2 rows non-reproducible. Only S3 (the router, the system
    actually demoed) resolves the answer language.
    """
    if strategy == "s1":
        from src.retrieval.rag_pipeline import answer as rag_answer
        from src.router.rules import classify_intent
        intent = classify_intent(question)
        raw = rag_answer(question, top_k=top_k, ticker=getattr(intent, "ticker", None))
        return {"question": question, "branch": "rag", "intent_reason": "S1: retrieval-only",
                "raw": raw, "answer": raw["answer"], "fallback_from": None}
    if strategy == "s2":
        from src.structured.dag_pipeline import answer as dag_answer
        # 2026-09-25: S2 now gets the resolved answer language too. The
        # baseline stays reproducible because dag_pipeline.answer() defaults
        # to Vietnamese when no lang is passed - see its docstring.
        return {"question": question, "branch": "dag", "intent_reason": "S2: structured-lookup-only",
                "raw": {}, "answer": dag_answer(question, lang=answer_lang), "fallback_from": None}
    return route(question, top_k=top_k, ui_lang=ui_lang, answer_lang=answer_lang)


def _resolve_query(question: str, history: list[str], lang: str | None = None) -> str:
    """Resolve follow-up context, but reject destructive LLM rewrites.

    Long financial questions often contain several tickers/years.  A rewrite
    that drops any explicit entity changes the task, so the original expanded
    question is safer than the shortened rewrite.

    `lang` (2026-09-24) chỉ chọn NGÔN NGỮ của prompt viết lại, không đổi
    logic. Không truyền thì giữ nguyên prompt tiếng Việt như cũ, nên các
    baseline RQ3 đã chạy vẫn tái lập được.

    LƯU Ý cho demo RQ3: hàm này gọi Ollama. Ollama chết -> except -> trả về
    câu hỏi gốc, tức là BỘ NHỚ IM LẶNG KHÔNG HOẠT ĐỘNG mà không báo lỗi gì.
    Muốn quay demo phần bộ nhớ thì bắt buộc Ollama phải sống.
    """
    if not history:
        return question

    # 2026-09-25 - CỔNG: chỉ gọi bộ nhớ khi câu hỏi TỰ NÓ chưa trả lời được.
    #
    # Trước đây resolver chạy trên MỌI câu có lịch sử, kể cả câu đã đủ nghĩa.
    # Nó diễn đạt lại (paraphrase) cả những câu đó, và paraphrase có thể đổi
    # nhánh định tuyến vì router dựa trên từ khoá. Hai ca thật trong lần chạy
    # test 25/09, cả hai đều là câu hoàn chỉnh:
    #   "What if the interest rate changed for CTD - how would pretax profit
    #    be affected?"  (whatif)  ->  viết lại thành "What is the impact of a
    #    change in interest rates on CTD's pretax profit?"  -> "impact of" là
    #    từ khoá nhân quả -> causal_unsupported.
    #   "Is the current ratio affected by asset turnover?"  (causal)  -> viết
    #    lại thành "...influenced by..." -> mất từ khoá -> rag, và model trả
    #    lời bằng kiến thức chung, không có nguồn.
    # Đây chính là loại lỗi "self_contained" trong benchmark RQ3
    # (data/eval/rq7_conversational_benchmark.jsonl): bộ nhớ làm hỏng một
    # câu vốn không cần nó.
    #
    # Quy tắc: nếu câu hỏi ĐỨNG MỘT MÌNH đã đi vào một nhánh có cấu trúc
    # (dag, ratio_lookup, whatif, causal, compare, price_snapshot, ...), thì
    # nó đã đủ thông tin - giữ nguyên, không viết lại. Chỉ khi nó rơi vào
    # rag hoặc *_unsupported (thiếu công ty, thiếu kỳ, thiếu chỉ tiêu) mới
    # là lúc cần ngữ cảnh. Kiểm tra lại cả 8 hội thoại RQ3: 6 câu cần bộ nhớ
    # đều đứng một mình rơi vào rag/whatif_unsupported nên vẫn được viết
    # lại; 2 câu self_contained/noisy_history đứng một mình đã đúng nhánh
    # nên được giữ nguyên - đúng kỳ vọng của cả hai loại.
    #
    # Lợi ích phụ: câu có cấu trúc không còn tốn một lượt gọi Ollama trước
    # khi trả lời.
    try:
        from src.router.rules import classify_intent

        standalone_branch = classify_intent(question, lang or "vi").branch
        if standalone_branch != "rag" and not standalone_branch.endswith("_unsupported"):
            return question
    except Exception:
        pass  # không phân loại được thì cứ để resolver thử, như hành vi cũ

    try:
        turns = [ConversationTurn(i, "user", text) for i, text in enumerate(history)]
        rewritten = LLMPromptOnlyResolver().resolve(
            turns, question, lang=lang).standalone_question.strip()
        import re
        explicit = set(re.findall(r"\b[A-Z]{3,4}\b", question))
        years = set(re.findall(r"\b20\d{2}\b", question))
        if any(x not in rewritten for x in explicit | years):
            return question
        if len(rewritten) < max(20, int(len(question) * 0.25)):
            return question
        return rewritten
    except Exception:
        return question


def _log_run(session_id: str, question: str, resolved: str, result: dict,
             latency_ms: int, persona: str | None = None,
             username: str | None = None) -> int | None:
    """Ghi lượt trả lời vào feedback.answer_runs. Không bao giờ để lỗi ghi
    log làm hỏng câu trả lời đã tính xong - người dùng vẫn phải nhận được
    câu trả lời kể cả khi tầng HITL đang chết.

    CẢNH BÁO đã học được 2026-08-07: chính cái `except Exception: return
    None` này từng che giấu một bug thật suốt nhiều ngày - CHECK constraint
    của migration 013 thiếu nhãn 'auditor_lookup'/'dag_trend'/'*_unsupported'
    nên INSERT vi phạm CHECK, bị nuốt ở đây, khiến các nhánh đó KHÔNG BAO
    GIỜ có answer_run_id -> không hiện ô đánh giá -> 0 dữ liệu HITL, hoàn
    toàn im lặng. Migration 014 sửa CHECK. Nếu sau này thêm nhánh router
    mới, PHẢI cập nhật cả CHECK của 014 lẫn `Branch` Literal trong
    api/shared/schemas.py - nếu không sẽ tái diễn đúng kịch bản im lặng này.
    """
    try:
        from src.core.feedback_store import extract_evidence, log_answer_run

        return log_answer_run(
            session_id=session_id,
            question_text=question,
            resolved_question=resolved,
            branch=result["branch"],
            fallback_from=result.get("fallback_from"),
            answer_text=result["answer"],
            evidence=extract_evidence(result.get("raw"), result["branch"]),
            latency_ms=latency_ms,
            persona=persona,
            username=username,
        )
    except Exception:
        return None


def _post_log_hooks(answer_run_id: int | None, result: dict) -> None:
    """Uncertainty sampling (Settles 2009): tự đưa vào hàng đợi review các
    lượt mà hệ thống KÉM CHẮC CHẮN nhất - fallback nhánh, hoặc similarity
    truy xuất thấp. Đây là nửa tự động của HITL; nửa còn lại là rating."""
    if not answer_run_id:
        return
    try:
        from src.core.feedback_store import enqueue_review, maybe_enqueue_low_similarity

        if result.get("fallback_from"):
            enqueue_review(answer_run_id=answer_run_id, source="fallback", priority=2)
        raw = result.get("raw") or {}
        if raw.get("sources"):
            from src.core.feedback_store import extract_evidence

            maybe_enqueue_low_similarity(
                answer_run_id=answer_run_id,
                evidence=extract_evidence(raw, result["branch"]),
            )
    except Exception:
        pass


def _identity_from_headers(request: Request) -> tuple[str | None, str | None]:
    """(persona, username) từ header GATEWAY gắn vào - xem
    api/services/gateway/main.py's _forward_headers().

    Service này KHÔNG tự xác thực gì: gateway đã lọc bỏ header cùng tên do
    client gửi lên (_HOP_BY_HOP) rồi mới gắn giá trị thật từ cookie phiên
    đã ký HMAC. Ở đây chỉ đọc. Trả (None, None) khi gọi trực tiếp không qua
    gateway (script/benchmark/curl nội bộ) - NULL nghĩa là "không biết",
    không bịa giá trị mặc định.

    Vẫn kiểm tra persona có hợp lệ không (personas.is_valid) thay vì tin
    mù: nếu một ngày nào đó ai đó gọi thẳng cổng 8101, cột phân tích không
    nên nhận rác.
    """
    persona = request.headers.get("X-User-Persona")
    username = request.headers.get("X-User-Name")
    if persona:
        try:
            from src.core.personas import is_valid
            if not is_valid(persona):
                persona = None
        except Exception:
            persona = None
    return persona, username


# 2026-09-24 - trạng thái Ollama, có nhớ đệm ngắn.
#
# Vì sao cần: /health trước đây LUÔN trả "ok" kể cả khi Ollama chết, nên
# cách duy nhất phát hiện là hỏi một câu RAG rồi ngồi đợi - đúng ba lần
# trong phiên làm việc vừa rồi, lần lâu nhất 56 GIÂY mới hiện thông báo
# fallback. Trên sân khấu bảo vệ, 56 giây im lặng rồi báo lỗi là tình huống
# tệ nhất có thể xảy ra. Biết TRƯỚC khi hỏi thì còn kịp xử lý.
#
# Nhớ đệm 15 giây: /health bị poll định kỳ, mà mỗi lần kiểm tra là một lượt
# HTTP sang Ollama - không đệm thì chính cái health check lại thành tải.
_OLLAMA_CACHE: dict[str, Any] = {"at": 0.0, "ok": None, "detail": None}
_OLLAMA_CACHE_TTL_S = 15


def _ollama_state() -> tuple[bool, str]:
    """(reachable, mô tả ngắn). KHÔNG dùng llm_factory.check_ollama() vì hàm
    đó in ra stdout (viết cho CLI) - ở đây cần giá trị trả về sạch."""
    now = time.time()
    if _OLLAMA_CACHE["ok"] is not None and now - _OLLAMA_CACHE["at"] < _OLLAMA_CACHE_TTL_S:
        return _OLLAMA_CACHE["ok"], _OLLAMA_CACHE["detail"]
    try:
        import requests
        from src.models.llm_factory import OLLAMA_HOST, OLLAMA_MODEL

        resp = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=3)
        resp.raise_for_status()
        names = [m.get("name", "") for m in resp.json().get("models", [])]
        want = OLLAMA_MODEL.split(":")[0]
        if not any(n.startswith(want) for n in names):
            ok, detail = False, f"Ollama đang chạy nhưng chưa có model '{OLLAMA_MODEL}' (ollama pull {OLLAMA_MODEL})"
        else:
            ok, detail = True, f"Ollama OK ({OLLAMA_MODEL})"
    except Exception as exc:
        ok, detail = False, f"Ollama không phản hồi: {type(exc).__name__}"
    _OLLAMA_CACHE.update(at=now, ok=ok, detail=detail)
    return ok, detail


@app.get("/health")
def health() -> ServiceHealth:
    """Trạng thái QA service, CÓ tính tới Ollama.

    "degraded" chứ không phải "down" là có chủ ý: Ollama chết thì các nhánh
    có cấu trúc (dag, ratio_lookup, price_snapshot, compare, auditor_lookup,
    bank_rate_lookup) VẪN chạy hoàn hảo - chúng không gọi model nào. Chỉ
    RAG/ABSA/rationale là hỏng. Báo "down" sẽ nói quá mức độ hỏng và làm
    người dùng tưởng cả hệ thống chết.
    """
    ollama_ok, ollama_detail = _ollama_state()
    base = f"{app.state.n_aliases} alias chỉ tiêu đã nạp"
    if ollama_ok:
        return ServiceHealth(service="qa", status="ok", detail=f"{base} · {ollama_detail}")
    return ServiceHealth(
        service="qa", status="degraded",
        detail=(f"{base} · {ollama_detail}. Nhánh có cấu trúc (giá, tra cứu chỉ tiêu, "
                f"chỉ số, kiểm toán) vẫn trả lời bình thường; RAG/ABSA thì không."),
    )


@app.post("/ask")
async def ask(payload: AskRequest, request: Request) -> AskResponse:
    """Hỏi đáp đồng bộ. Trả kết quả cuối cùng trong một response."""
    persona, username = _identity_from_headers(request)
    # Resolve the answer language from what the USER typed, before the
    # context resolver rewrites it - see route()'s answer_lang docstring.
    lang = resolve_answer_language(payload.question, payload.ui_lang)
    expanded = expand_display(payload.question)
    resolved = await anyio.to_thread.run_sync(_resolve_query, expanded, payload.history, lang)
    t0 = time.perf_counter()
    result = await anyio.to_thread.run_sync(
        _run_route, resolved, payload.top_k, payload.strategy, payload.ui_lang, lang
    )
    latency_ms = int((time.perf_counter() - t0) * 1000)

    answer_run_id = await anyio.to_thread.run_sync(
        _log_run, payload.session_id, payload.question, resolved, result, latency_ms,
        persona, username
    )
    await anyio.to_thread.run_sync(_post_log_hooks, answer_run_id, result)

    return AskResponse(
        answer_run_id=answer_run_id,
        question=payload.question,
        resolved_question=resolved if resolved != payload.question else None,
        # Bộ nhớ hội thoại có thật sự làm gì không: so với câu ĐÃ khai triển
        # viết tắt, không phải câu gốc - nếu so với câu gốc thì một lượt chỉ
        # khai triển viết tắt cũng bị tính là "bộ nhớ đã hoạt động".
        context_resolved=(resolved != expanded),
        answer=result["answer"],
        branch=result["branch"],
        fallback_from=result.get("fallback_from"),
        intent_reason=result["intent_reason"],
        sources=_to_sources(result.get("raw")),
        signals=_to_signals(result.get("raw")),
        latency_ms=latency_ms,
    )


@app.get("/ask/stream", response_class=EventSourceResponse)
async def ask_stream(
    request: Request,
    question: Annotated[str, Query(min_length=1, max_length=3000)],
    session_id: Annotated[str, Query(min_length=1, max_length=64)],
    top_k: Annotated[int, Query(ge=1, le=20)] = 5,
    strategy: Literal["s1", "s2", "s3"] = "s3",
    history: Annotated[list[HistoryTurn] | None, Query()] = None,
    ui_lang: Annotated[Literal["vi", "en"] | None, Query()] = None,
) -> AsyncIterable[ServerSentEvent]:
    """Phát tiến trình trả lời theo từng chặng qua SSE.

    Chặng phát ra: `expand` -> `classify` -> `working` -> `answer` | `error`.
    `classify` chạy riêng (rẻ, thuần regex/từ điển, không gọi LLM) nên UI
    biết nhánh nào được chọn TRƯỚC khi phần tốn thời gian bắt đầu - đây
    chính là thứ làm thanh routing rail có ý nghĩa thay vì chỉ là hiệu ứng.

    `ui_lang` (2026-09-22) mirrors AskRequest.ui_lang on the POST endpoint -
    a tie-breaker for the answer language, not an override. Every stage note
    emitted below is localized with the SAME resolved language as the answer,
    so the UI never shows a Vietnamese progress line above an English answer.
    """
    persona, username = _identity_from_headers(request)
    lang = resolve_answer_language(question, ui_lang)
    en = lang == "en"
    expanded = expand_display(question)
    resolved = await anyio.to_thread.run_sync(_resolve_query, expanded, history or [], lang)
    if expanded != question:
        yield ServerSentEvent(
            event="expand",
            data={"original": question, "expanded": expanded,
                  "note": ("Expanded financial abbreviation from the dictionary." if en
                           else "Đã khai triển viết tắt tài chính bằng từ điển.")},
        )

    try:
        from src.router.rules import classify_intent

        intent = await anyio.to_thread.run_sync(classify_intent, resolved, lang)
        selected_branch = {"s1": "rag", "s2": "dag"}.get(strategy, intent.branch)
        selected_reason = ({
            "s1": "S1 selected: document retrieval (RAG-only).",
            "s2": "S2 selected: structured data lookup (DAG-only).",
        } if en else {
            "s1": "S1 đã được chọn: truy xuất tài liệu (RAG-only).",
            "s2": "S2 đã được chọn: truy vấn dữ liệu có cấu trúc (DAG-only).",
        }).get(strategy, intent.reason)
        yield ServerSentEvent(
            event="classify",
            data={"branch": selected_branch, "reason": selected_reason,
                  "ticker": getattr(intent, "ticker", None), "strategy": strategy},
        )
    except Exception as exc:
        yield ServerSentEvent(event="error", data={
            "detail": (f"Classification failed: {exc}" if en else f"Phân loại thất bại: {exc}")})
        return

    yield ServerSentEvent(
        event="working",
        data={"stage": selected_branch,
              "note": ("Retrieving and composing the answer." if en
                       else "Đang truy xuất và tổng hợp câu trả lời.")},
    )

    t0 = time.perf_counter()
    try:
        # `lang` (not ui_lang) is pinned here: it was resolved from the
        # ORIGINAL question above, while `resolved` may be an LLM rewrite -
        # see route()'s answer_lang docstring.
        result = await anyio.to_thread.run_sync(_run_route, resolved, top_k, strategy, ui_lang, lang)
    except Exception as exc:
        yield ServerSentEvent(
            event="error",
            data={"detail": (f"Could not process the question: {type(exc).__name__}: {exc}" if en
                             else f"Không xử lý được câu hỏi: {type(exc).__name__}: {exc}")},
        )
        return
    latency_ms = int((time.perf_counter() - t0) * 1000)

    answer_run_id = await anyio.to_thread.run_sync(
        _log_run, session_id, question, resolved, result, latency_ms,
        persona, username
    )
    await anyio.to_thread.run_sync(_post_log_hooks, answer_run_id, result)

    payload = AskResponse(
        answer_run_id=answer_run_id,
        question=question,
        resolved_question=resolved if resolved != question else None,
        context_resolved=(resolved != expanded),
        answer=result["answer"],
        branch=result["branch"],
        fallback_from=result.get("fallback_from"),
        intent_reason=result["intent_reason"],
        sources=_to_sources(result.get("raw")),
        signals=_to_signals(result.get("raw")),
        latency_ms=latency_ms,
    )
    yield ServerSentEvent(event="answer", data=payload.model_dump(mode="json"))


@app.get("/companies")
def list_companies(
    q: Annotated[str | None, Query(max_length=64)] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> list[CompanyRef]:
    """Danh mục công ty kèm cờ có dữ liệu nào - để UI gợi ý mã hỏi được,
    thay vì người dùng phải đoán mã nào đã nạp."""
    from sqlalchemy import text

    from src.core.db import get_session

    sql = text("""
        SELECT c.ticker, c.company_name, c.exchange, c.sector,
               EXISTS (SELECT 1 FROM financial_line_items f WHERE f.ticker = c.ticker) AS has_structured
        FROM companies c
        WHERE (:q IS NULL OR c.ticker ILIKE :like OR c.company_name ILIKE :like)
        ORDER BY has_structured DESC, c.ticker
        LIMIT :limit
    """)
    with get_session() as session:
        rows = session.execute(
            sql, {"q": q, "like": f"%{q}%" if q else None, "limit": limit}
        ).mappings().all()
    return [
        CompanyRef(
            ticker=r["ticker"], company_name=r["company_name"],
            exchange=r["exchange"], sector=r["sector"],
            has_structured=bool(r["has_structured"]), has_documents=False,
        )
        for r in rows
    ]


@app.get("/session/new")
def new_session() -> dict[str, str]:
    return {"session_id": uuid.uuid4().hex}
