"""Write/read helpers for the `feedback` schema (infra/postgres/migrations/013_feedback_hitl.sql).

This is the thesis-now HITL loop: chat demo -> app_streamlit.py tab_chat calls
log_answer_run() right after Router (S3) answers, then log_feedback() when the
user rates/skips. Design intent and citations live in the migration's SQL
comments and docs/DSA_v6_Bo_sung_2026-08-04.md section 7 - this module is
deliberately thin (no ORM models, plain SQL via get_session()) so it stays a
1:1 mirror of the schema and is easy to re-verify against it.

Kept append-only per the schema's own discipline: log_answer_run/log_feedback
only INSERT. review_queue_lite is the one mutable table (status transitions
happen via mark_review_status(), not here - that's an Airflow/manual-review
action, not something a chat turn does automatically).
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Any

from sqlalchemy import text

from src.core.db import get_session

# Rules-based router (src/router/rules.py) has no internal version constant
# today; this literal is bumped by hand whenever rules.py's intent patterns
# change materially, so answer_runs.router_version stays a meaningful label
# for "which router logic produced this answer" without adding version
# machinery to rules.py just for logging.
ROUTER_VERSION_LABEL = "s3_rules_v1_20260804"

# Uncertainty-sampling thresholds (Settles 2009) that decide what
# auto-enqueues into feedback.review_queue_lite - see migration 013 comment.
LOW_RATING_THRESHOLD = 2


def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _normalize_for_hash(question: str) -> str:
    """Whitespace/case normalization ONLY (not the paraphrase-masking
    normalization in scripts/split_router_benchmark.py - that's for
    template clustering, a different purpose). This hash is for literal
    dedup ("was this exact question asked before"), not paraphrase
    grouping."""
    return " ".join(question.strip().lower().split())


def extract_evidence(raw: dict | None, branch: str) -> dict[str, Any]:
    """Best-effort, JSON-safe evidence extraction from a route() result's
    "raw" field. Never raises - a malformed/unexpected raw shape degrades to
    an empty evidence dict rather than breaking the chat turn (same
    resilience pattern as _resolve_chat_context in app_streamlit.py).

    Shape matches the evidence JSONB comment in migration 013:
    {chunk_ids:[], prediction_id:, experiment_run_id:, sql_text:} - only
    keys that actually apply to the branch are populated.
    """
    evidence: dict[str, Any] = {"branch": branch}
    if not raw:
        return evidence
    try:
        if branch in ("rag", "absa") and raw.get("sources"):
            evidence["chunk_ids"] = [
                {
                    "ticker": s.get("ticker"), "doctype": s.get("doctype"),
                    "source_file": s.get("source_file"), "page": s.get("page"),
                    "similarity": s.get("similarity"),
                }
                for s in raw["sources"][:10]  # cap - this is evidence metadata, not a full dump
            ]
        if branch == "causal" and raw.get("ok"):
            evidence["experiment_run_id"] = raw.get("experiment_run_id")
            evidence["point_estimate"] = raw.get("point_estimate")
            evidence["ci_low"] = raw.get("ci_low")
            evidence["ci_high"] = raw.get("ci_high")
        if branch == "dag" and raw.get("ok"):
            evidence["sql_text"] = raw.get("sql_text") or raw.get("query")
        # predictive branch (not wired into router yet - Phase P2, blocked on
        # 4-year backfill per docs/CHECKLIST_PHASES.md 1b) will add
        # prediction_id here once it exists.
        json.dumps(evidence)  # validate JSON-safety before returning
        return evidence
    except Exception:
        return {"branch": branch, "evidence_extraction_error": True}


def log_answer_run(
    *, session_id: str, question_text: str, resolved_question: str | None,
    branch: str, fallback_from: str | None, answer_text: str,
    evidence: dict, latency_ms: int | None = None,
    persona: str | None = None, username: str | None = None,
) -> int:
    """INSERT one feedback.answer_runs row, returns its id.

    `branch` phải nằm trong tập CHECK của migration 014 (014 mở rộng 013 -
    013 THIẾU 'auditor_lookup', 'dag_trend' và các nhãn '*_unsupported'
    thật, khiến những nhánh đó ghi log thất bại IM LẶNG; xem comment đầu
    file 014 để hiểu vì sao im lặng lại nguy hiểm hơn là vỡ hẳn).

    `persona`/`username` (2026-08-07, migration 014): vai trò NGHIỆP VỤ và
    danh tính người hỏi - để trả lời được câu hỏi nghiên cứu "vai trò nào
    thấy nhánh nào hữu ích". Cả hai để None được (lượt không đăng nhập,
    script benchmark, hoặc gọi từ CLI) - NULL nghĩa là "không biết", không
    bịa giá trị mặc định.
    """
    with get_session() as session:
        row = session.execute(
            text(
                """
                INSERT INTO feedback.answer_runs
                    (session_id, question_text, question_hash, resolved_question,
                     branch, fallback_from, latency_ms, answer_text, answer_hash,
                     evidence, router_version, persona, username)
                VALUES
                    (:session_id, :question_text, :question_hash, :resolved_question,
                     :branch, :fallback_from, :latency_ms, :answer_text, :answer_hash,
                     :evidence, :router_version, :persona, :username)
                RETURNING id
                """
            ),
            {
                "session_id": session_id,
                "question_text": question_text,
                "question_hash": _sha256(_normalize_for_hash(question_text)),
                "resolved_question": resolved_question if resolved_question != question_text else None,
                "branch": branch,
                "fallback_from": fallback_from,
                "latency_ms": latency_ms,
                "answer_text": answer_text,
                "answer_hash": _sha256(answer_text) if answer_text else None,
                "evidence": json.dumps(evidence),
                "router_version": ROUTER_VERSION_LABEL,
                "persona": persona,
                "username": username,
            },
        )
        return row.scalar_one()


def log_feedback(
    *, answer_run_id: int, rating: int | None = None, is_skip: bool = False,
    reason_code: str | None = None, free_text: str | None = None,
) -> int:
    """INSERT one feedback.answer_feedback row. Enforces the same
    rating-or-skip rule as the DB CHECK constraint client-side too, so the
    caller gets a clear Python error instead of an IntegrityError."""
    if not is_skip and rating is None:
        raise ValueError("log_feedback: cần rating (nếu không phải is_skip=True)")
    with get_session() as session:
        row = session.execute(
            text(
                """
                INSERT INTO feedback.answer_feedback
                    (answer_run_id, rating, is_skip, reason_code, free_text)
                VALUES (:answer_run_id, :rating, :is_skip, :reason_code, :free_text)
                RETURNING id
                """
            ),
            {
                "answer_run_id": answer_run_id, "rating": rating, "is_skip": is_skip,
                "reason_code": reason_code, "free_text": free_text,
            },
        )
        feedback_id = row.scalar_one()

    # Uncertainty-sampling auto-enqueue (Settles 2009) - low rating is one of
    # the 3 triggers migration 013 documents. The other 2 (fallback,
    # low_similarity) are enqueued from app_streamlit.py at answer time,
    # since they're known before any rating exists.
    if not is_skip and rating is not None and rating <= LOW_RATING_THRESHOLD:
        enqueue_review(answer_run_id=answer_run_id, source="low_rating", priority=1)

    return feedback_id


def enqueue_review(*, answer_run_id: int | None, source: str, priority: int = 3) -> int:
    """INSERT one feedback.review_queue_lite row. `source` must be one of
    the CHECK-constrained values (low_rating/fallback/low_similarity/manual)."""
    with get_session() as session:
        row = session.execute(
            text(
                """
                INSERT INTO feedback.review_queue_lite (answer_run_id, source, priority)
                VALUES (:answer_run_id, :source, :priority)
                RETURNING id
                """
            ),
            {"answer_run_id": answer_run_id, "source": source, "priority": priority},
        )
        return row.scalar_one()


# RAG similarity below this is the 3rd uncertainty-sampling trigger
# (migration 013 comment: "low_similarity") - the router doesn't reject a
# RAG answer at this threshold (that's a separate design decision, see
# rag_pipeline.py's own no-context threshold), it just flags the answer for
# human review because the retrieval that grounds it was weak.
LOW_SIMILARITY_THRESHOLD = 0.45


def maybe_enqueue_low_similarity(*, answer_run_id: int, evidence: dict) -> bool:
    """If evidence["chunk_ids"] (from extract_evidence) has a max similarity
    below LOW_SIMILARITY_THRESHOLD, enqueue a review item. Returns whether it
    enqueued. No-op (returns False) if evidence has no similarity data -
    e.g. non-RAG branches, or RAG returned zero sources."""
    chunk_ids = evidence.get("chunk_ids") or []
    similarities = [c["similarity"] for c in chunk_ids if c.get("similarity") is not None]
    if not similarities:
        return False
    if max(similarities) < LOW_SIMILARITY_THRESHOLD:
        enqueue_review(answer_run_id=answer_run_id, source="low_similarity", priority=2)
        return True
    return False


def list_pending_review(limit: int = 20) -> list[dict]:
    """Pending review_queue_lite rows joined with their answer_runs context,
    ordered by priority (1=highest) then oldest first - the small "queue
    review" UI in app_streamlit.py reads this directly. Returns plain dicts
    (not ORM rows) so the Streamlit layer doesn't need a live session open."""
    with get_session() as session:
        rows = session.execute(
            text(
                """
                SELECT q.id AS review_id, q.source, q.priority, q.created_at,
                       r.id AS answer_run_id, r.question_text, r.answer_text,
                       r.branch, r.fallback_from
                FROM feedback.review_queue_lite q
                LEFT JOIN feedback.answer_runs r ON r.id = q.answer_run_id
                WHERE q.status = 'pending'
                ORDER BY q.priority ASC, q.created_at ASC
                LIMIT :limit
                """
            ),
            {"limit": limit},
        ).mappings().all()
        return [dict(row) for row in rows]


def mark_review_status(*, review_id: int, status: str, resolution_note: str | None = None) -> None:
    """Transition a review_queue_lite row - the ONE mutable table in this
    schema (migration 013 comment). `status` must be one of
    reviewed/resolved/dismissed (pending is the initial state, never set
    here)."""
    if status not in ("reviewed", "resolved", "dismissed"):
        raise ValueError(f"status không hợp lệ: {status}")
    with get_session() as session:
        session.execute(
            text(
                """
                UPDATE feedback.review_queue_lite
                SET status = :status, resolution_note = :resolution_note, reviewed_at = now()
                WHERE id = :review_id
                """
            ),
            {"status": status, "resolution_note": resolution_note, "review_id": review_id},
        )


def promote_review_to_regression(
    *, review_id: int, expected_answer: str, expected_sources: str | None = None,
    reviewer_note: str | None = None,
    promoted_by: str | None = None, promoted_persona: str | None = None,
) -> int:
    """Turn one reviewed failure into an executable-test candidate.

    This is intentionally a curation step, not a fine-tuning trigger. A
    unique review_id prevents a reviewer from accidentally promoting the
    same queue item twice.

    `promoted_by`/`promoted_persona` (2026-09-22) record WHO wrote this
    expected answer - see the provenance section appended to migration 015.
    This row becomes ground truth once accepted, so an unattributable row
    means a polluted fixture can only be handled by distrusting the whole
    set. NULL when called outside the gateway (CLI, tests) - "unknown",
    never a fabricated default.
    """
    with get_session() as session:
        review = session.execute(
            text("""
                SELECT q.answer_run_id, r.question_text
                FROM feedback.review_queue_lite q
                JOIN feedback.answer_runs r ON r.id = q.answer_run_id
                WHERE q.id = :review_id
            """),
            {"review_id": review_id},
        ).mappings().one_or_none()
        if not review:
            raise ValueError("Không tìm thấy review item để tạo regression candidate.")
        if review["answer_run_id"] is None:
            raise ValueError("Review item thủ công không có answer run để tái chạy.")

        existing = session.execute(
            text("SELECT id FROM feedback.regression_candidates WHERE review_id = :review_id"),
            {"review_id": review_id},
        ).scalar_one_or_none()
        if existing:
            raise ValueError("Review item này đã được đưa vào regression candidates.")

        candidate_id = session.execute(
            text("""
                INSERT INTO feedback.regression_candidates
                    (review_id, answer_run_id, question_text, expected_answer,
                     expected_sources, reviewer_note, promoted_by, promoted_persona)
                VALUES
                    (:review_id, :answer_run_id, :question_text, :expected_answer,
                     :expected_sources, :reviewer_note, :promoted_by, :promoted_persona)
                RETURNING id
            """),
            {
                "review_id": review_id,
                "answer_run_id": review["answer_run_id"],
                "question_text": review["question_text"],
                "expected_answer": expected_answer,
                "expected_sources": expected_sources,
                "reviewer_note": reviewer_note,
                "promoted_by": promoted_by,
                "promoted_persona": promoted_persona,
            },
        ).scalar_one()
        session.execute(
            text("""
                UPDATE feedback.review_queue_lite
                SET status = 'resolved', resolution_note = :resolution_note, reviewed_at = now()
                WHERE id = :review_id
            """),
            {
                "review_id": review_id,
                "resolution_note": reviewer_note or "Promoted to regression candidate.",
            },
        )
        return candidate_id


# 2026-09-22 - tách vai (separation of duties). Bật/tắt bằng biến môi
# trường vì đây là luật NGHIỆP VỤ, không phải bất biến kỹ thuật: lúc phát
# triển một mình (hoặc demo với đúng một tài khoản) thì luật này tự khoá
# chính mình, và khi đó tắt nó phải dễ hơn là phải sửa code/migration.
# Mặc định BẬT - mặc định an toàn, muốn nới thì phải cố ý nới.
REQUIRE_SEPARATE_REVIEWER = os.getenv("FINEX_REQUIRE_SEPARATE_REVIEWER", "1") != "0"


def decide_regression_candidate(
    *, candidate_id: int, status: str, reviewer_note: str | None = None,
    decided_by: str | None = None, decided_persona: str | None = None,
    allow_reversal: bool = False,
) -> None:
    """Accept or reject one curated candidate for later export.

    Accepted candidates are the only records the export command will write
    into a runnable regression fixture.  This keeps ratings and unreviewed
    suggestions out of the test set.

    2026-09-22: enforces separation of duties - the account that WROTE the
    expected answer cannot be the account that approves it. Before this, the
    3-tier HITL design looked like it had two checks but they were two
    STEPS, not two PEOPLE: every account in gateway's ACCOUNTS has
    role='user', and role='user' may call every /api/feedback/* route, so
    one person could promote and then approve their own ground truth. A
    second check performed by the same person verifies nothing.

    Enforced here (application layer) rather than as a DB CHECK on purpose -
    see migration 015's provenance section: rows written before that
    migration have promoted_by NULL and must stay valid, and this is a
    business rule that may need relaxing (REQUIRE_SEPARATE_REVIEWER).
    """
    if status not in {"accepted", "rejected"}:
        raise ValueError("Regression candidate chỉ có thể được accepted hoặc rejected.")
    with get_session() as session:
        row = session.execute(
            text("""SELECT status, promoted_by, decided_by
                    FROM feedback.regression_candidates WHERE id = :id"""),
            {"id": candidate_id},
        ).mappings().one_or_none()
        if row is None:
            raise ValueError("Không tìm thấy regression candidate.")

        from_status = row["status"]
        is_reversal = from_status != "candidate"

        if is_reversal:
            # 2026-09-22: trước đây đây là ngõ cụt - mọi dòng đã rời trạng
            # thái 'candidate' đều bị ném lỗi, nên một đáp án rác lỡ được
            # accept thì nằm vĩnh viễn trong fixture, không gỡ được qua
            # giao diện. Giờ lật lại được, nhưng có ĐIỀU KIỆN, vì lật một
            # quyết định của người khác là thao tác nặng hơn quyết định
            # bình thường.
            if not allow_reversal:
                raise ValueError(
                    f"Regression candidate đã được quyết định ({from_status}). "
                    f"Muốn lật lại phải dùng thao tác đảo ngược và ghi rõ lý do."
                )
            if not (reviewer_note or "").strip():
                raise ValueError(
                    "Đảo ngược một quyết định bắt buộc phải ghi lý do - đây là "
                    "phần giải trình được của thao tác này."
                )
            # Người đã duyệt không được tự lật quyết định của chính mình,
            # cùng lý do với luật tách vai ở dưới: một người tự sửa lại
            # chính mình thì không có ai kiểm tra thêm.
            if (REQUIRE_SEPARATE_REVIEWER and decided_by and row["decided_by"]
                    and decided_by == row["decided_by"]):
                raise PermissionError(
                    f"Tách vai: '{decided_by}' chính là người đã ra quyết định "
                    f"'{from_status}' cho candidate này, nên không thể tự lật lại. "
                    f"Cần một tài khoản review khác."
                )
        else:
            # Quyết định LẦN ĐẦU: người viết expected answer không được tự duyệt.
            # Chỉ chặn khi BIẾT cả hai danh tính. promoted_by NULL = dòng cũ
            # trước migration 015's provenance, hoặc gọi ngoài gateway - không
            # có cơ sở để kết luận tự duyệt, nên không chặn (thà lọt còn hơn
            # khoá nhầm một dòng hợp lệ mà người dùng không hiểu vì sao).
            if (REQUIRE_SEPARATE_REVIEWER and decided_by and row["promoted_by"]
                    and decided_by == row["promoted_by"]):
                raise PermissionError(
                    f"Tách vai: '{decided_by}' là người đã viết expected answer cho "
                    f"candidate này, nên không thể tự duyệt. Cần một tài khoản khác "
                    f"duyệt. (Đặt FINEX_REQUIRE_SEPARATE_REVIEWER=0 để tắt luật này "
                    f"khi chạy một mình.)"
                )

        session.execute(
            text("""
                UPDATE feedback.regression_candidates
                SET status = :status,
                    reviewer_note = COALESCE(:reviewer_note, reviewer_note),
                    decided_by = :decided_by,
                    decided_persona = :decided_persona,
                    decided_at = now()
                WHERE id = :id
            """),
            {"id": candidate_id, "status": status, "reviewer_note": reviewer_note,
             "decided_by": decided_by, "decided_persona": decided_persona},
        )
        # Nhật ký CHỈ-GHI-THÊM. Dòng candidate ở trên bị ghi đè (nó giữ
        # TRẠNG THÁI HIỆN TẠI), còn dòng log này thì không bao giờ - nó giữ
        # DIỄN BIẾN. Ghi trong CÙNG transaction với UPDATE: hoặc cả hai cùng
        # thành công, hoặc không có gì xảy ra; một quyết định đã áp dụng mà
        # không có vết log sẽ đúng là thứ bảng này sinh ra để ngăn.
        session.execute(
            text("""
                INSERT INTO feedback.regression_decisions
                    (candidate_id, decision, from_status, to_status,
                     actor, actor_persona, note)
                VALUES (:candidate_id, :decision, :from_status, :to_status,
                        :actor, :actor_persona, :note)
            """),
            {
                "candidate_id": candidate_id,
                "decision": "reversed" if is_reversal else status,
                "from_status": from_status,
                "to_status": status,
                "actor": decided_by,
                "actor_persona": decided_persona,
                "note": reviewer_note,
            },
        )


def regression_decision_trail(candidate_id: int | None = None, limit: int = 100) -> list[dict]:
    """Đọc nhật ký quyết định (chỉ đọc, không đổi gì).

    Dùng bởi admin service - xem api/services/admin/main.py. Trả dict thuần
    (không phải ORM row) để tầng gọi không cần giữ session mở, cùng quy ước
    với pending_reviews() ở trên.
    """
    where = "WHERE d.candidate_id = :candidate_id" if candidate_id else ""
    with get_session() as session:
        rows = session.execute(
            text(f"""
                SELECT d.id, d.candidate_id, d.decision, d.from_status, d.to_status,
                       d.actor, d.actor_persona, d.note, d.created_at,
                       c.question_text, c.promoted_by
                FROM feedback.regression_decisions d
                JOIN feedback.regression_candidates c ON c.id = d.candidate_id
                {where}
                ORDER BY d.created_at DESC, d.id DESC
                LIMIT :limit
            """),
            {"candidate_id": candidate_id, "limit": limit} if candidate_id else {"limit": limit},
        ).mappings().all()
    return [dict(r) for r in rows]
