"""Admin service (mặt VẬN HÀNH) - STUB có chủ đích, hoàn thiện SAU thesis.

Tách service riêng NGAY TỪ ĐẦU (thay vì nhét endpoint admin vào qa/feedback)
để ranh giới user/admin là ranh giới deploy thật: sau thesis, gắn auth
(OAuth2/JWT), MLOps (model registry, retrain pipeline, promotion gate) và
warehouse (vintage/quality reports) vào đây mà không đụng mặt người dùng.

Hiện chỉ expose các view CHỈ ĐỌC trên những bảng đã có thật trong DB
(model_registry_lite, retrain_gate_log) để chứng minh đường dây hoạt động.
"""
from sqlalchemy import text

from webapp.backend.shared.bootstrap import make_app  # noqa: F401

from src.core.db import get_session  # noqa: E402

app = make_app(
    "admin-service",
    "Vận hành (STUB - sau thesis): model registry, retrain gates, warehouse views.",
)


@app.get("/model-registry")
def model_registry():
    """Chỉ đọc: các model đã đăng ký + champion flags (migration 012)."""
    with get_session() as session:
        rows = session.execute(text("""
            SELECT id, task, model_name, model_version, calibration_method,
                   is_champion, frozen_at
            FROM model_registry_lite ORDER BY task, frozen_at DESC
        """)).mappings().all()
    return {"models": [dict(r) for r in rows]}


@app.get("/retrain-gates")
def retrain_gates():
    """Chỉ đọc: lịch sử đánh giá batch gate (migration 013)."""
    with get_session() as session:
        rows = session.execute(text("""
            SELECT id, window_start, window_end, branch, n_feedback, n_skip,
                   mean_rating, skip_rate, gate_rule, gate_triggered, note
            FROM feedback.retrain_gate_log ORDER BY window_end DESC LIMIT 50
        """)).mappings().all()
    return {"gates": [dict(r) for r in rows]}


@app.get("/roadmap")
def roadmap():
    """Những gì service này SẼ làm sau thesis - ghi rõ để phản biện thấy
    ranh giới scope là quyết định thiết kế, không phải bỏ sót."""
    return {"post_thesis": [
        "AuthN/AuthZ (OAuth2/JWT) cho toàn bộ endpoint admin",
        "MLOps: trigger retrain khi gate mở, promotion gate 1.1-1.4, model card",
        "Warehouse: bảng vintage/restatement, data quality dashboard",
        "Chuyển /admin/* của feedback-service về đây khi có auth",
    ]}
