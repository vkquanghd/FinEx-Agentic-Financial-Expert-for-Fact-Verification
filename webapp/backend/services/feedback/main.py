"""Feedback service - trái tim HITL, tách khỏi luồng chat theo đúng bản chất.

Bản chất HITL của hệ này (DSA v6 §7, batch-feedback với approved-data
boundary - KHÔNG phải realtime supervision kiểu mahilo):

  Layer 1 - THU TÍN HIỆU: mỗi câu trả lời là một answer_run; người dùng
    rate 1-5 sao + reason_code, hoặc skip (skip cũng là data - mẫu số
    response-rate). [Wu et al. 2022]
  Layer 2 - UNCERTAINTY SAMPLING: rating <=2, fallback nhánh, hay
    similarity thấp -> tự vào review_queue_lite để NGƯỜI xem xét - dồn sức
    người vào chỗ hệ thống kém chắc nhất. [Settles 2009]
  Layer 3 - BATCH GATE: chỉ cân nhắc retrain khi N>=50 feedback và mean
    rating <3.5 - không bao giờ chỉnh model theo từng label. [Christiano
    et al. 2017]

API user-facing: POST /rate. API vận hành (review queue, gate stats) để
prefix /admin - sau thesis sẽ tách hẳn sang admin service + auth; giữ ở
đây tạm thời vì panel HITL của frontend cần chúng cho demo.
"""
from typing import Literal

from pydantic import BaseModel, Field
from sqlalchemy import text

from webapp.backend.shared.bootstrap import make_app  # noqa: F401

from src.core.db import get_session  # noqa: E402
from src.core.feedback_store import (  # noqa: E402
    LOW_RATING_THRESHOLD,
    list_pending_review,
    log_feedback,
    mark_review_status,
)

app = make_app(
    "feedback-service",
    "HITL: thu rating/skip (user), review queue + gate stats (admin-lite, tách sau thesis).",
)

REASON_CODES = ["good", "wrong_number", "wrong_branch", "missing_evidence",
                "hallucination", "outdated", "unclear", "too_slow", "other"]


class RateRequest(BaseModel):
    answer_run_id: int
    rating: int | None = Field(default=None, ge=1, le=5)
    is_skip: bool = False
    reason_code: Literal[tuple(REASON_CODES)] | None = None  # type: ignore[valid-type]
    free_text: str | None = Field(default=None, max_length=1000)


@app.post("/rate")
def rate(req: RateRequest):
    if req.rating is None and not req.is_skip:
        return {"ok": False, "error": "Cần rating (1-5) hoặc is_skip=true."}
    feedback_id = log_feedback(
        answer_run_id=req.answer_run_id,
        rating=req.rating,
        is_skip=req.is_skip,
        reason_code=req.reason_code,
        free_text=req.free_text,
    )
    return {
        "ok": True,
        "feedback_id": feedback_id,
        # minh bạch cơ chế uncertainty sampling với người dùng cuối
        "enqueued_for_review": bool(req.rating is not None and req.rating <= LOW_RATING_THRESHOLD),
    }


class ReviewAction(BaseModel):
    status: Literal["resolved", "reviewed", "dismissed"]
    resolution_note: str | None = Field(default=None, max_length=1000)


@app.get("/admin/review-queue")
def review_queue(limit: int = 20):
    return {"items": list_pending_review(limit=limit)}


@app.post("/admin/review-queue/{review_id}")
def review_action(review_id: int, action: ReviewAction):
    mark_review_status(review_id=review_id, status=action.status,
                       resolution_note=action.resolution_note)
    return {"ok": True}


@app.get("/admin/hitl-stats")
def hitl_stats():
    """Số liệu cho panel HITL: tiến độ batch gate + phân bố tín hiệu.
    Gate rule cố định của thesis: N>=50 và mean<3.5 (DSA v6 §7)."""
    with get_session() as session:
        row = session.execute(text("""
            SELECT COUNT(*) FILTER (WHERE NOT f.is_skip)      AS n_rated,
                   COUNT(*) FILTER (WHERE f.is_skip)          AS n_skip,
                   AVG(f.rating) FILTER (WHERE NOT f.is_skip) AS mean_rating
            FROM feedback.answer_feedback f
        """)).one()
        per_branch = session.execute(text("""
            SELECT r.branch,
                   COUNT(*) FILTER (WHERE NOT f.is_skip)      AS n_rated,
                   AVG(f.rating) FILTER (WHERE NOT f.is_skip) AS mean_rating
            FROM feedback.answer_feedback f
            JOIN feedback.answer_runs r ON r.id = f.answer_run_id
            GROUP BY r.branch ORDER BY r.branch
        """)).mappings().all()
        n_pending = session.execute(text(
            "SELECT COUNT(*) FROM feedback.review_queue_lite WHERE status = 'pending'"
        )).scalar()
        n_runs = session.execute(text("SELECT COUNT(*) FROM feedback.answer_runs")).scalar()

    n_rated = row.n_rated or 0
    mean_rating = float(row.mean_rating) if row.mean_rating is not None else None
    return {
        "n_answer_runs": n_runs,
        "n_rated": n_rated,
        "n_skip": row.n_skip or 0,
        "mean_rating": mean_rating,
        "pending_reviews": n_pending,
        "gate": {
            "rule": "N>=50 và mean<3.5 mới cân nhắc retrain (batch, không bao giờ per-label)",
            "n_required": 50,
            "n_progress": n_rated,
            "would_trigger": bool(n_rated >= 50 and mean_rating is not None and mean_rating < 3.5),
        },
        "per_branch": [dict(b) for b in per_branch],
    }
