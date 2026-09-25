"""Feedback service (user-plane, HITL) - port 8102.

Vì sao tách khỏi QA service dù cùng là user-plane: hai vòng đời khác nhau.
QA service phục vụ một câu hỏi rồi quên; feedback service tích lũy trạng
thái qua thời gian (hàng đợi review, regression candidates). Tách ra từ bây
giờ để lúc dựng warehouse không phải mổ QA service.

Ba tầng HITL, đúng theo thiết kế đã đối chiếu tài liệu (DSA v6 §7):
  1. Thu tín hiệu     - POST /feedback (rating 1-5 | skip | reason_code)
  2. Hàng đợi review  - GET /review/queue, POST /review/action
                        (uncertainty sampling: Settles 2009)
  3. Regression set   - reviewer viết expected answer + nguồn cần có, rồi
                        promote item thành test candidate tái chạy được.
"""
from datetime import datetime, timedelta
from typing import Annotated

from fastapi import FastAPI, HTTPException, Query, Request
from sqlalchemy import text

from api.shared.config import ensure_repo_on_path, get_settings
from api.shared.schemas import (
    BranchStat,
    FeedbackRequest,
    FeedbackResponse,
    GateStatus,
    RegressionCandidate,
    RegressionDecisionRequest,
    RegressionCandidateRequest,
    ReviewActionRequest,
    ReviewItem,
    ServiceHealth,
)

ensure_repo_on_path()

from src.core.db import get_session  # noqa: E402

app = FastAPI(
    title="Financial QA - Feedback Service",
    description="User-plane: vòng phản hồi có người trong vòng lặp (HITL).",
    version="1.0.0",
)

# Diễn giải vì sao một mục lọt vào hàng đợi - hiển thị thẳng cho người
# review, vì "biết vì sao hệ thống hỏi mình" là điều kiện để người ta trả
# lời có ích (Amershi et al. 2014).
SOURCE_LABELS = {
    "low_rating": "Người dùng chấm điểm thấp (≤2)",
    "fallback": "Router phải chuyển nhánh dự phòng",
    "low_similarity": "Độ tương đồng truy xuất thấp",
    "manual": "Được đưa vào thủ công",
}


def _identity_from_headers(request: Request) -> tuple[str | None, str | None]:
    """(persona, username) từ header GATEWAY gắn vào - bản sao có chủ đích
    của api/services/qa/main.py's hàm cùng tên (2026-09-22).

    Vì sao service này cũng cần: hai route curation
    (/review/promote, /review/regression/{id}/decision) sinh ra GROUND
    TRUTH - dòng 'accepted' được export thành fixture rồi dùng để kết luận
    hệ thống đúng/sai. Ghi lại ai viết và ai duyệt là điều kiện để (a) lọc
    bỏ đóng góp của một người review nhập ẩu mà không phải vứt cả bộ, và
    (b) ép tách vai. Xem phần provenance cuối migration 015.

    Service này KHÔNG tự xác thực: gateway đã lọc bỏ header cùng tên do
    client gửi (_HOP_BY_HOP) rồi mới gắn giá trị thật từ cookie phiên đã ký
    HMAC. Ở đây chỉ đọc. Trả (None, None) khi gọi trực tiếp không qua
    gateway - NULL = "không biết", không bịa mặc định.
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


@app.get("/health")
def health() -> ServiceHealth:
    try:
        with get_session() as session:
            session.execute(text("SELECT 1 FROM feedback.answer_runs LIMIT 1"))
        return ServiceHealth(service="feedback", status="ok")
    except Exception as exc:
        return ServiceHealth(service="feedback", status="degraded",
                             detail=f"{type(exc).__name__}: {exc}")


@app.post("/feedback")
def submit_feedback(payload: FeedbackRequest) -> FeedbackResponse:
    """Nhận một tín hiệu đánh giá.

    Bỏ qua (skip) CŨNG là dữ liệu: nó cho biết mẫu số của tỷ lệ phản hồi,
    nên được lưu thành bản ghi thật (is_skip=true) chứ không phải "không
    làm gì cả". Không có nó thì mọi thống kê rating đều là mẫu tự chọn.
    """
    if payload.rating is None and not payload.is_skip:
        raise HTTPException(status_code=422,
                            detail="Cần rating 1-5 hoặc đánh dấu bỏ qua.")
    from src.core.feedback_store import log_feedback

    try:
        feedback_id = log_feedback(
            answer_run_id=payload.answer_run_id,
            rating=payload.rating,
            is_skip=payload.is_skip,
            reason_code=payload.reason_code,
            free_text=payload.free_text,
        )
    except Exception as exc:
        raise HTTPException(status_code=502,
                            detail=f"Không ghi được đánh giá: {exc}") from exc

    enqueued = payload.rating is not None and payload.rating <= 2
    return FeedbackResponse(
        feedback_id=feedback_id,
        enqueued_for_review=enqueued,
        enqueue_reason=SOURCE_LABELS["low_rating"] if enqueued else None,
    )


@app.get("/review/queue")
def review_queue(
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
) -> list[ReviewItem]:
    sql = text("""
        SELECT q.id AS review_id, q.answer_run_id, q.source, q.priority,
               q.status, q.created_at,
               r.question_text, r.answer_text, r.branch
        FROM feedback.review_queue_lite q
        JOIN feedback.answer_runs r ON r.id = q.answer_run_id
        WHERE q.status = 'pending'
        ORDER BY q.priority, q.created_at
        LIMIT :limit
    """)
    with get_session() as session:
        rows = session.execute(sql, {"limit": limit}).mappings().all()
    return [
        ReviewItem(
            review_id=r["review_id"], answer_run_id=r["answer_run_id"],
            question_text=r["question_text"], answer_text=r["answer_text"],
            branch=r["branch"], source=r["source"],
            source_label=SOURCE_LABELS.get(r["source"], r["source"]),
            priority=r["priority"], status=r["status"], created_at=r["created_at"],
        )
        for r in rows
    ]


@app.post("/review/action")
def review_action(payload: ReviewActionRequest) -> dict[str, str]:
    """Kết luận một mục review.

    `review_queue_lite` là bảng DUY NHẤT được sửa trạng thái trong schema
    feedback - mọi bảng còn lại chỉ ghi thêm (append-only), để dấu vết
    đánh giá không bao giờ bị viết đè. Ghi chú xử lý là bắt buộc về mặt
    tinh thần (không ép ở tầng API) vì nó là phần "giải trình được" của
    quyết định con người.
    """
    from src.core.feedback_store import mark_review_status

    try:
        mark_review_status(review_id=payload.review_id, status=payload.status,
                           resolution_note=payload.resolution_note)
    except Exception as exc:
        raise HTTPException(status_code=502,
                            detail=f"Không cập nhật được: {exc}") from exc
    return {"status": payload.status}


@app.post("/review/promote", response_model=RegressionCandidate)
def promote_review(payload: RegressionCandidateRequest, request: Request) -> RegressionCandidate:
    """Promote a human-reviewed item into a regression-test candidate.

    This creates a durable expected result. It never retrains Qwen or edits
    the thesis benchmark automatically.

    2026-09-22: records WHO wrote the expected answer (see migration 015's
    provenance section). That identity is also what makes the separation-of-
    duties check on the decision route below possible at all.
    """
    from src.core.feedback_store import promote_review_to_regression

    persona, username = _identity_from_headers(request)
    try:
        candidate_id = promote_review_to_regression(
            review_id=payload.review_id,
            expected_answer=payload.expected_answer.strip(),
            expected_sources=payload.expected_sources.strip() if payload.expected_sources else None,
            reviewer_note=payload.reviewer_note.strip() if payload.reviewer_note else None,
            promoted_by=username,
            promoted_persona=persona,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502,
                            detail=f"Không tạo được regression candidate: {exc}") from exc

    with get_session() as session:
        row = session.execute(
            text("SELECT * FROM feedback.regression_candidates WHERE id = :id"),
            {"id": candidate_id},
        ).mappings().one()
    return RegressionCandidate(**dict(row))


@app.get("/review/regression", response_model=list[RegressionCandidate])
def regression_candidates(
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
) -> list[RegressionCandidate]:
    """List curated cases awaiting acceptance into a dedicated regression set."""
    with get_session() as session:
        rows = session.execute(
            text("""
                SELECT id, review_id, answer_run_id, question_text, expected_answer,
                       expected_sources, reviewer_note, status, created_at,
                       promoted_by, promoted_persona, decided_by, decided_persona
                FROM feedback.regression_candidates
                ORDER BY created_at DESC
                LIMIT :limit
            """),
            {"limit": limit},
        ).mappings().all()
    return [RegressionCandidate(**dict(row)) for row in rows]


@app.post("/review/regression/{candidate_id}/decision", response_model=RegressionCandidate)
def decide_regression(
    candidate_id: int, payload: RegressionDecisionRequest, request: Request,
) -> RegressionCandidate:
    """Accept a candidate for export, or reject it with an auditable note.

    2026-09-22: enforces separation of duties - whoever wrote the expected
    answer cannot approve it themselves (403). See
    feedback_store.decide_regression_candidate() for why this lives in the
    application layer rather than a DB CHECK.
    """
    from src.core.feedback_store import decide_regression_candidate

    persona, username = _identity_from_headers(request)
    try:
        decide_regression_candidate(
            candidate_id=candidate_id,
            status=payload.status,
            reviewer_note=payload.reviewer_note.strip() if payload.reviewer_note else None,
            decided_by=username,
            decided_persona=persona,
            allow_reversal=payload.allow_reversal,
        )
    except PermissionError as exc:
        # 403, KHÔNG phải 409: đây không phải xung đột trạng thái mà là
        # "tài khoản này không được phép làm việc này" - phân biệt rõ để
        # frontend nói đúng lý do cho người dùng.
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502,
                            detail=f"Không quyết định được regression candidate: {exc}") from exc

    with get_session() as session:
        row = session.execute(
            text("SELECT * FROM feedback.regression_candidates WHERE id = :id"),
            {"id": candidate_id},
        ).mappings().one()
    return RegressionCandidate(**dict(row))


@app.get("/gate")
def gate_status(
    window_days: Annotated[int, Query(ge=1, le=365)] = 30,
    branch: Annotated[str | None, Query(max_length=32)] = None,
) -> GateStatus:
    """Khoảng cách tới ngưỡng retrain, tính trên cửa sổ thời gian.

    Hai điều kiện ĐỘC LẬP và phải thỏa CẢ HAI:
      - đủ lượng: n_feedback >= min_feedback_required
      - có vấn đề chất lượng: mean_rating < mean_rating_threshold
    Đủ lượng nhưng chất lượng tốt thì KHÔNG retrain - đó là cổng hoạt động
    đúng, không phải cổng hỏng. Frontend phải trình bày rõ ý này.
    """
    settings = get_settings()
    window_end = datetime.now()
    window_start = window_end - timedelta(days=window_days)
    sql = text("""
        SELECT COUNT(*) FILTER (WHERE NOT f.is_skip)            AS n_feedback,
               COUNT(*) FILTER (WHERE f.is_skip)                AS n_skip,
               AVG(f.rating) FILTER (WHERE NOT f.is_skip)       AS mean_rating
        FROM feedback.answer_feedback f
        JOIN feedback.answer_runs r ON r.id = f.answer_run_id
        WHERE f.created_at >= :start
          AND (:branch IS NULL OR r.branch = :branch)
    """)
    with get_session() as session:
        row = session.execute(sql, {"start": window_start, "branch": branch}).mappings().one()

    n_feedback = int(row["n_feedback"] or 0)
    n_skip = int(row["n_skip"] or 0)
    mean_rating = float(row["mean_rating"]) if row["mean_rating"] is not None else None
    total = n_feedback + n_skip
    volume_met = n_feedback >= settings.retrain_gate_min_feedback
    quality_met = mean_rating is not None and mean_rating < settings.retrain_gate_mean_rating

    return GateStatus(
        n_feedback=n_feedback,
        n_skip=n_skip,
        mean_rating=round(mean_rating, 2) if mean_rating is not None else None,
        skip_rate=round(n_skip / total, 3) if total else None,
        min_feedback_required=settings.retrain_gate_min_feedback,
        mean_rating_threshold=settings.retrain_gate_mean_rating,
        condition_volume_met=volume_met,
        condition_quality_met=quality_met,
        gate_triggered=volume_met and quality_met,
        window_start=window_start,
        window_end=window_end,
    )


@app.get("/stats/branches")
def branch_stats(
    window_days: Annotated[int, Query(ge=1, le=365)] = 30,
) -> list[BranchStat]:
    """Rating theo nhánh router - bảng đánh giá con người trong luận văn.

    Đây là nguồn số liệu ĐỘC LẬP với benchmark tự động: benchmark đo đúng/
    sai theo đáp án dựng sẵn, bảng này đo cảm nhận hữu ích của người dùng
    thật trên câu hỏi thật.
    """
    window_start = datetime.now() - timedelta(days=window_days)
    sql = text("""
        SELECT r.branch,
               COUNT(DISTINCT r.id)                              AS n_answers,
               COUNT(f.id) FILTER (WHERE NOT f.is_skip)          AS n_rated,
               AVG(f.rating) FILTER (WHERE NOT f.is_skip)        AS mean_rating,
               COUNT(f.id) FILTER (WHERE f.is_skip)              AS n_skip
        FROM feedback.answer_runs r
        LEFT JOIN feedback.answer_feedback f ON f.answer_run_id = r.id
        WHERE r.created_at >= :start
        GROUP BY r.branch
        ORDER BY n_answers DESC
    """)
    with get_session() as session:
        rows = session.execute(sql, {"start": window_start}).mappings().all()
    out = []
    for r in rows:
        n_rated, n_skip = int(r["n_rated"] or 0), int(r["n_skip"] or 0)
        total = n_rated + n_skip
        out.append(BranchStat(
            branch=r["branch"], n_answers=int(r["n_answers"] or 0), n_rated=n_rated,
            mean_rating=round(float(r["mean_rating"]), 2) if r["mean_rating"] is not None else None,
            skip_rate=round(n_skip / total, 3) if total else None,
        ))
    return out


@app.get("/stats/reasons")
def reason_stats(
    window_days: Annotated[int, Query(ge=1, le=365)] = 30,
) -> list[dict]:
    """Phân bố lý do chấm điểm thấp - chỉ ra NÊN SỬA GÌ TRƯỚC.

    Rating trung bình cho biết "có vấn đề"; phân bố reason_code cho biết
    vấn đề nằm ở đâu (sai số liệu? sai thực thể? sai nhánh?). Không có
    bảng này thì feedback chỉ là tiếng ồn không hành động được.
    """
    window_start = datetime.now() - timedelta(days=window_days)
    sql = text("""
        SELECT reason_code, COUNT(*) AS n
        FROM feedback.answer_feedback
        WHERE created_at >= :start AND reason_code IS NOT NULL
        GROUP BY reason_code
        ORDER BY n DESC
    """)
    with get_session() as session:
        rows = session.execute(sql, {"start": window_start}).mappings().all()
    return [{"reason_code": r["reason_code"], "n": int(r["n"])} for r in rows]
