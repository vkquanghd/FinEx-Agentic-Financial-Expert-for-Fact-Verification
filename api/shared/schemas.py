"""Pydantic schema dùng chung giữa các service và frontend.

Đây là hợp đồng (contract) giữa các service - đổi ở đây là đổi API công
khai, nên mỗi field đều có lý do tồn tại chứ không phải dump nguyên dict
nội bộ của router ra ngoài.
"""
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

Branch = Literal[
    "dag", "dag_trend", "rag", "whatif", "causal", "compare", "absa", "auditor_lookup",
    # 2026-09-14: "dag_multi" (tra cứu nhiều mã cùng lúc, src/router/rules.py)
    # và "out_of_scope" (dự đoán giá/ý kiến bên thứ ba/tư vấn đầu tư,
    # src/router/rules.py) - 2 nhánh mới thêm cùng ngày. THIẾU 1 trong 2 ở
    # đây làm Pydantic từ chối construct AskResponse (branch: Branch) và
    # /ask trả 500 cho MỌI câu hỏi rơi vào 2 nhánh này - phát hiện khi rà
    # lại api/services/qa/main.py's _log_run() docstring, vốn đã cảnh báo
    # đúng kịch bản này (thêm nhánh router mới mà quên cập nhật cả Literal
    # này LẪN CHECK constraint của migration 014 - xem file đó để biết vế
    # thứ 2 của cùng bug).
    "dag_multi", "out_of_scope",
    # 2026-09-15: "ratio_lookup" (src/router/rules.py, financial_ratios -
    # P/E, ROE, current ratio, debt/equity...) - same missing-Literal bug
    # class as dag_multi/out_of_scope above (see their comment for the
    # full failure mode: missing here -> Pydantic rejects AskResponse ->
    # /ask returns 500).
    "ratio_lookup",
    # 2026-09-15: "bank_rate_lookup" (src/router/rules.py, aggregate
    # deposit-rate ranking across macro_indicators' DEPOSIT_RATES category)
    # - same missing-Literal bug class as the entries above.
    "bank_rate_lookup",
    # 2026-09-21: "price_snapshot" (src/router/rules.py + src/causal/
    # market_factors.py's latest_price_snapshot() - "giá/cổ phiếu X hôm nay"
    # questions, company_price_snapshot EOD data) - same missing-Literal bug
    # class as the entries above. ALSO required in migration 014's CHECK
    # constraint (infra/postgres/migrations/014_answer_runs_branch_and_
    # persona.sql) - updated there too, see that file.
    "price_snapshot",
    "whatif_unsupported", "causal_unsupported", "compare_unsupported",
    "absa_unsupported",
]

# PHẢI khớp CHECK reason_code của feedback.answer_feedback (xem phần bổ sung
# cuối infra/postgres/migrations/013_feedback_hitl.sql). Hai danh sách này đã
# TỪNG lệch nhau: Literal có wrong_entity/wrong_period/missing_context mà DB
# CHECK không có, nên 3 lựa chọn đó qua được tầng API rồi mới chết ở tầng DB
# thành HTTP 502 - mất trắng đúng loại phản hồi có ích nhất (người dùng chịu
# khó nói rõ sai ở đâu). Sửa 2026-09-22 bằng cách nới CHECK phía DB; giữ hai
# danh sách này đồng bộ mỗi khi thêm mã mới.
ReasonCode = Literal[
    # dropdown hiện tại trên giao diện (api/frontend/app.js's _reasonsVi/_reasonsEn)
    "wrong_number", "wrong_entity", "wrong_period", "missing_context",
    "wrong_branch", "hallucination", "too_slow", "unclear", "other",
    # 2026-09-22: bỏ qua NGẦM - hỏi câu tiếp mà chưa chấm điểm. KHÔNG phải
    # lựa chọn trên dropdown, chỉ do frontend tự gửi (app.js's
    # _recordImplicitSkip); tách riêng khỏi lượt chủ động bấm "Bỏ qua".
    "implicit",
    # mã cũ còn tồn tại trong dữ liệu đã ghi (013) - Literal phải nhận, nếu
    # không thì một dòng cũ đọc lên sẽ không dựng lại được thành model.
    "good", "missing_evidence", "outdated",
]


# --- QA service -----------------------------------------------------------

class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=3000)
    session_id: str = Field(min_length=1, max_length=64)
    top_k: int = Field(default=5, ge=1, le=20)
    strategy: Literal["s1", "s2", "s3"] = "s3"
    history: list[str] = Field(default_factory=list, max_length=20)
    # 2026-09-22: the UI's language toggle, forwarded so the ANSWER can be
    # written in the user's language. It is only a TIE-BREAKER, never an
    # override - src.core.lang_detect.resolve_answer_language() prefers the
    # language of the question itself and consults this field only for
    # fragments with no language signal ("VNZ?"). Optional so every existing
    # caller (benchmark scripts, CLI, tests) keeps its current behaviour.
    ui_lang: Literal["vi", "en"] | None = None


class SourceRef(BaseModel):
    # ticker/doctype/source_file/page default to ""/0 (not required) since
    # 2026-09-07 - a web-fallback source (see src/retrieval/rag_pipeline.py's
    # web_fallback_answer(), infra/postgres/migrations/014_web_chunks.sql)
    # has NO doctype/source_file/page at all (it's a press URL, not a PDF
    # page) - the domain/source_url/published_date fields below are what
    # carry a web source's real identity instead. A disclosure_chunks
    # source (the normal case) still fills every field as before.
    ticker: str = ""
    doctype: str = ""
    source_file: str = ""
    page: int = 0
    year: int | None = None
    similarity: float | None = None
    # 2026-08-06: hybrid search gắn nhãn chunk này do nhánh nào tìm ra
    # (vector / lexical / both) - hiển thị được trên UI để người đọc thấy
    # vì sao một trích dẫn khớp-nguyên-văn lọt vào top-k.
    retrieval: str | None = None
    # Web-fallback-only fields (2026-09-07) - None for every normal
    # disclosure_chunks source, populated only when source_tier=="web_fallback"
    # (see AskResponse.signals below).
    domain: str | None = None
    source_url: str | None = None
    title: str | None = None
    published_date: str | None = None


class AskResponse(BaseModel):
    answer_run_id: int | None = None
    question: str
    resolved_question: str | None = None
    # 2026-09-24 (RQ3): `resolved_question` khác câu gốc vì HAI lý do rất
    # khác nhau - (a) khai triển viết tắt bằng từ điển (expand_display), và
    # (b) giải nghĩa câu hỏi nối tiếp từ lịch sử hội thoại
    # (LLMPromptOnlyResolver). Giao diện trước đây gộp cả hai vào một nhãn
    # "Abbreviation expanded", nên một lượt bộ nhớ hoạt động ĐÚNG lại bị
    # dán nhãn sai thành khai triển viết tắt - không chứng minh được gì cho
    # RQ3, vốn là câu hỏi nghiên cứu về CHÍNH bộ nhớ hội thoại.
    # Tách cờ này để UI gọi đúng tên hiện tượng và trưng ra câu đã viết lại.
    context_resolved: bool = False
    answer: str
    branch: Branch
    fallback_from: Branch | None = None
    intent_reason: str
    sources: list[SourceRef] = []
    latency_ms: int
    # Vì sao tách khỏi `sources`: đây là tín hiệu router dùng để quyết định,
    # không phải bằng chứng cho câu trả lời. UI hiển thị ở 2 chỗ khác nhau.
    # signals["source_tier"] == "web_fallback" (2026-09-07) is the one FE
    # MUST check to render the "not an official disclosure" warning banner
    # - see src/retrieval/rag_pipeline.py's web_fallback_answer().
    signals: dict[str, Any] = {}


class CompanyRef(BaseModel):
    ticker: str
    company_name: str
    exchange: str | None = None
    sector: str | None = None
    has_structured: bool = False
    has_documents: bool = False


# --- Feedback service (HITL) ---------------------------------------------

class FeedbackRequest(BaseModel):
    answer_run_id: int
    rating: int | None = Field(default=None, ge=1, le=5)
    is_skip: bool = False
    reason_code: ReasonCode | None = None
    free_text: str | None = Field(default=None, max_length=2000)


class FeedbackResponse(BaseModel):
    feedback_id: int
    enqueued_for_review: bool
    # Nói thẳng vì sao bị đưa vào hàng đợi - minh bạch với người dùng là
    # một phần của thiết kế HITL (Amershi et al. 2014), không phải log ẩn.
    enqueue_reason: str | None = None


class ReviewItem(BaseModel):
    review_id: int
    answer_run_id: int
    question_text: str
    answer_text: str
    branch: str
    source: str          # low_rating | fallback | low_similarity | manual
    source_label: str    # diễn giải tiếng Việt của `source`
    priority: int
    status: str
    created_at: datetime | None = None


class ReviewActionRequest(BaseModel):
    review_id: int
    status: Literal["reviewed", "resolved", "dismissed"]
    resolution_note: str | None = Field(default=None, max_length=2000)


class RegressionCandidateRequest(BaseModel):
    review_id: int
    expected_answer: str = Field(min_length=8, max_length=8000)
    expected_sources: str | None = Field(default=None, max_length=4000)
    reviewer_note: str | None = Field(default=None, max_length=2000)


class RegressionCandidate(BaseModel):
    id: int
    review_id: int
    answer_run_id: int
    question_text: str
    expected_answer: str
    expected_sources: str | None = None
    reviewer_note: str | None = None
    status: Literal["candidate", "accepted", "rejected"]
    created_at: datetime | None = None
    # Provenance (2026-09-22, migration 015's appended section). Một dòng
    # 'accepted' trở thành ground truth trong fixture hồi quy, nên phải biết
    # AI viết và AI duyệt: có thế mới lọc bỏ được đóng góp của một người
    # review nhập ẩu mà không phải nghi ngờ toàn bộ bộ test. NULL = dòng ghi
    # trước migration, hoặc gọi thẳng service không qua gateway.
    promoted_by: str | None = None
    promoted_persona: str | None = None
    decided_by: str | None = None
    decided_persona: str | None = None


class RegressionDecisionRequest(BaseModel):
    status: Literal["accepted", "rejected"]
    reviewer_note: str | None = Field(default=None, max_length=2000)
    # 2026-09-22: bật cờ này để LẬT LẠI một quyết định đã có (vd người duyệt
    # trước lỡ accept một đáp án không khớp câu hỏi). Phải cố ý gửi lên -
    # một cú bấm Accept bình thường không bao giờ vô tình ghi đè quyết định
    # của người khác. Khi bật, reviewer_note trở thành BẮT BUỘC và người đã
    # ra quyết định cũ không được tự lật (xem feedback_store).
    allow_reversal: bool = False


class RegressionDecisionLogRow(BaseModel):
    """Một dòng trong nhật ký chỉ-ghi-thêm feedback.regression_decisions.

    Chỉ dùng cho admin (read-only) - đây là dấu vết kiểm toán của các quyết
    định CON NGƯỜI sinh ra ground truth, nên nó phải đọc được tách biệt với
    trạng thái hiện tại của candidate.
    """
    id: int
    candidate_id: int
    decision: Literal["accepted", "rejected", "reversed"]
    from_status: str | None = None
    to_status: str
    actor: str | None = None
    actor_persona: str | None = None
    note: str | None = None
    created_at: datetime | None = None
    question_text: str
    promoted_by: str | None = None


class GateStatus(BaseModel):
    """Trạng thái cổng retrain - trái tim của HITL theo lô.

    Không bao giờ retrain theo từng nhãn lẻ (Christiano et al. 2017): phải
    đủ N feedback VÀ chất lượng trung bình dưới ngưỡng thì mới cân nhắc.
    Frontend vẽ đúng 2 điều kiện này thành 2 thanh tiến trình.
    """
    n_feedback: int
    n_skip: int
    mean_rating: float | None
    skip_rate: float | None
    min_feedback_required: int
    mean_rating_threshold: float
    condition_volume_met: bool
    condition_quality_met: bool
    gate_triggered: bool
    window_start: datetime | None = None
    window_end: datetime | None = None


class BranchStat(BaseModel):
    branch: str
    n_answers: int
    n_rated: int
    mean_rating: float | None
    skip_rate: float | None


# --- Admin service --------------------------------------------------------

class CoverageRow(BaseModel):
    ticker: str
    n_structured_periods: int
    year_min: int | None = None
    year_max: int | None = None
    n_chunks: int
    n_doctypes: int
    modality_count: int


class PriceSnapshotRow(BaseModel):
    """One row per (ticker, trade_date) in company_price_snapshot - EOD OHLCV
    only (scripts/market_snapshot.py via vnstock, once/day through the
    thesis_market_snapshot Airflow DAG). No intraday/tick data exists
    anywhere in this project - see /admin/prices' docstring in
    api/services/admin/main.py before treating this as live market data."""
    trade_date: str  # ISO date string, not datetime - trade_date column is DATE
    open_price: float | None = None
    high_price: float | None = None
    low_price: float | None = None
    close_price: float | None = None
    volume: int | None = None
    pct_change: float | None = None  # vs previous row in the SAME result set, null on the first row


class ModelRegistryRow(BaseModel):
    model_id: int
    task: str
    model_name: str
    model_version: str
    calibration_method: str | None = None
    is_champion: bool
    frozen_at: datetime | None = None
    validation_metrics: dict[str, Any] = {}


class ServiceHealth(BaseModel):
    service: str
    status: Literal["ok", "degraded", "down"]
    detail: str | None = None


class TableInventoryRow(BaseModel):
    """One row per tracked table/schema.table (2026-09-08, added so admin
    can answer "các bảng đã có những gì" directly instead of only the
    hand-picked counters in /admin/pipeline/summary)."""
    table: str
    row_count: int | None  # None = query failed (table missing/DB error) - see api/services/admin/main.py's TRACKED_TABLES
    error: str | None = None


class PdfDocumentRow(BaseModel):
    """One row per physical PDF/report file (pdf_documents table) - the
    file-level view /admin/files exposes so admin can see exactly which
    ticker has which file, including files sourced from the TiniX OCR
    dataset (see origin's docstring in api/services/admin/main.py for why
    that field is a best-effort GUESS, not a stored fact)."""
    id: int
    ticker: str | None
    doctype: str | None
    source_file: str
    fiscal_year: int | None
    n_pages_total: int | None
    extraction_version: str | None
    document_version: int
    is_active: bool
    pdf_missing: bool  # True if pdf_hash is the MISSING_SOURCE_PDF_HASH_SENTINEL placeholder (real PDF bytes gone)
    origin_guess: str  # best-effort: "TiniX (đoán)" / "PDF gốc dự án (đoán)" - see the docstring above
    extracted_at: datetime | None = None
    updated_at: datetime | None = None
