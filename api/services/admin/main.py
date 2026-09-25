"""Admin service (admin-plane) - port 8103.

Toàn bộ route nằm sau `AdminKeyDep` khai báo Ở CẤP ROUTER, không phải từng
hàm - thêm endpoint mới thì mặc định đã được bảo vệ, không thể quên.

Phạm vi hiện tại là CHỈ ĐỌC (read-only) và có chủ ý: luận văn chưa cần
admin ghi gì cả, mà một mặt phẳng admin ghi được thì phải có kiểm toán,
phân quyền, xác nhận - những thứ chưa dựng. Các endpoint ghi (kích hoạt
retrain, phong champion model, chạy lại ETL) để lại dạng khung ở cuối file
kèm ghi chú điều kiện cần trước khi mở.

Đây là chỗ MLOps + warehouse sau này mọc lên: model registry, coverage dữ
liệu, sức khỏe pipeline đều đã có endpoint đọc sẵn.
"""
import re
from typing import Annotated

from fastapi import APIRouter, FastAPI, Query
from sqlalchemy import text

from api.shared.config import ensure_repo_on_path
from api.shared.schemas import (
    CoverageRow,
    ModelRegistryRow,
    PdfDocumentRow,
    PriceSnapshotRow,
    RegressionDecisionLogRow,
    ServiceHealth,
    TableInventoryRow,
)
from api.shared.security import AdminKeyDep

ensure_repo_on_path()

from src.core.db import get_session  # noqa: E402
from scripts.register_pdf_documents import MISSING_SOURCE_PDF_HASH_SENTINEL  # noqa: E402

app = FastAPI(
    title="Financial QA - Admin Service",
    description="Admin-plane: quan sát dữ liệu, model registry, sức khỏe hệ thống. "
                "Mọi endpoint yêu cầu header X-Admin-Key.",
    version="1.0.0",
)

router = APIRouter(prefix="/admin", tags=["admin"])


@app.get("/health")
def health() -> ServiceHealth:
    """Không yêu cầu key: chỉ nói service có sống không, không lộ dữ liệu."""
    return ServiceHealth(service="admin", status="ok")


@router.get("/coverage")
def data_coverage(
    _: AdminKeyDep,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
) -> list[CoverageRow]:
    """Coverage theo mã: bao nhiêu kỳ số liệu, bao nhiêu chunk tài liệu.

    Cùng câu hỏi mà scripts/audit_data_coverage.py trả lời offline, nhưng
    trực tuyến - để sau này warehouse có nguồn đọc thay vì chạy script tay.
    """
    sql = text("""
        SELECT c.ticker,
               COALESCE(f.n_periods, 0)  AS n_structured_periods,
               f.y0 AS year_min, f.y1 AS year_max,
               COALESCE(d.n_chunks, 0)   AS n_chunks,
               COALESCE(d.n_doctypes, 0) AS n_doctypes
        FROM companies c
        LEFT JOIN (
            SELECT ticker, COUNT(DISTINCT (fiscal_year, period)) AS n_periods,
                   MIN(fiscal_year) AS y0, MAX(fiscal_year) AS y1
            FROM financial_line_items GROUP BY ticker
        ) f ON f.ticker = c.ticker
        LEFT JOIN (
            SELECT ticker, COUNT(*) AS n_chunks, COUNT(DISTINCT doctype) AS n_doctypes
            FROM disclosure_chunks GROUP BY ticker
        ) d ON d.ticker = c.ticker
        WHERE COALESCE(f.n_periods, 0) > 0 OR COALESCE(d.n_chunks, 0) > 0
        ORDER BY n_structured_periods DESC, n_chunks DESC
        LIMIT :limit
    """)
    with get_session() as session:
        rows = session.execute(sql, {"limit": limit}).mappings().all()
    return [
        CoverageRow(
            ticker=r["ticker"],
            n_structured_periods=int(r["n_structured_periods"]),
            year_min=r["year_min"], year_max=r["year_max"],
            n_chunks=int(r["n_chunks"]), n_doctypes=int(r["n_doctypes"]),
            modality_count=int(bool(r["n_structured_periods"])) + int(bool(r["n_chunks"])),
        )
        for r in rows
    ]


@router.get("/models")
def model_registry(_: AdminKeyDep) -> list[ModelRegistryRow]:
    """Model registry (migration 012). Nguồn sự thật về model nào đang là
    champion cho từng task, đóng băng lúc nào, chỉ số validation ra sao."""
    sql = text("""
        SELECT id AS model_id, task, model_name, model_version,
               calibration_method, is_champion, frozen_at, validation_metrics
        FROM model_registry_lite
        ORDER BY task, is_champion DESC, frozen_at DESC NULLS LAST
    """)
    with get_session() as session:
        rows = session.execute(sql).mappings().all()
    return [
        ModelRegistryRow(
            model_id=r["model_id"], task=r["task"], model_name=r["model_name"],
            model_version=r["model_version"],
            calibration_method=r["calibration_method"],
            is_champion=bool(r["is_champion"]), frozen_at=r["frozen_at"],
            validation_metrics=r["validation_metrics"] or {},
        )
        for r in rows
    ]


_YEAR_RE = re.compile(r"(?<!\d)(20\d{2})(?!\d)")

# Every table this project actually writes to, tracked here so
# /admin/tables can answer "các bảng đã có những gì" for the WHOLE schema,
# not just the hand-picked counters /admin/pipeline/summary already had.
# schema-qualified where needed (feedback.*) - see infra/postgres/migrations/
# 001_init_schema.sql for the single source of truth on what these tables
# are. Added 2026-09-08 alongside PdfDocumentRow/`/admin/files` (same
# request: admin needs real visibility into what data exists, per ticker,
# per file, not just aggregate counts).
TRACKED_TABLES = [
    "companies", "financial_line_items", "financial_ratios", "disclosures",
    "pdf_documents", "disclosure_chunks", "web_chunks", "macro_indicators",
    # 2026-09-22: company_price_snapshot từng bị liệt kê HAI lần (thêm nhầm
    # lúc làm tab giá 2026-09-21) - bảng kiểm kê hiện đúng một dòng trùng
    # lặp, vô hại nhưng gây hiểu nhầm là có 2 bảng khác nhau. Gộp lại 1.
    "company_price_snapshot", "causal_panel_snapshot", "predictive_features",
    "predictive_split_manifest", "causal_experiment_runs",
    "model_registry_lite", "ml_prediction_outputs", "ingestion_log",
    "feedback.answer_runs", "feedback.answer_feedback",
    "feedback.review_queue_lite", "feedback.retrain_gate_log",
    # 2026-09-22: hai bảng này trước đây admin HOÀN TOÀN không thấy, kể cả
    # số dòng - trong khi regression_candidates là nơi sinh ra ground truth
    # cho fixture hồi quy, tức là bảng có thẩm quyền cao nhất trong schema
    # feedback. Không giám sát được nó là một lỗ hổng thật, không phải thiếu
    # sót trang trí.
    "feedback.regression_candidates", "feedback.regression_decisions",
    "vietstock_ingestion_runs", "vietstock_ticker_sync",
    "vietstock_document_manifest", "vietstock_news_events",
    "vietstock_structured_runs", "vietstock_structured_tables",
]


def _guess_file_origin(source_file: str) -> str:
    """Best-effort GUESS of which corpus a file came from - NOT a stored/
    authoritative field (pdf_documents has no origin column; the two
    corpora were never tagged with an explicit origin at ingest time, only
    discovered afterwards). Reuses the exact filename-shape observation
    already documented in src/retrieval/vector_store.py's extract_year()
    docstring (not invented here): TiniX's OCR'd report folders encode TWO
    different 20xx years (filing date + fiscal year) and use lowercase
    'kiemtoan'/'baocaotaichinh' tokens, e.g.
    '..._vi_baocaotaichinhrieng_kiemtoan_2024' - while this project's own
    downloaded real-PDF corpus uses single-year, Title-Case Vietstock names
    (e.g. 'PNJ_Baocaotaichinh_2024_Kiemtoan_Hopnhat') or CBTT-style names
    with neither token. Same "guess_*, always labelled as a guess" spirit
    as guess_ticker()/guess_doctype() in scripts/pdf_ingest_router.py -
    displayed in the UI with "(đoán)" suffix, never presented as fact."""
    years = set(_YEAR_RE.findall(source_file))
    low = source_file.lower()
    if len(years) >= 2 and ("kiemtoan" in low or "baocaotaichinh" in low):
        return "TiniX (đoán)"
    return "PDF gốc dự án (đoán)"


@router.get("/feedback/regression-trail")
def regression_trail(
    _: AdminKeyDep,
    candidate_id: Annotated[int | None, Query(ge=1)] = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
) -> list[RegressionDecisionLogRow]:
    """Nhật ký MỌI quyết định accept/reject/đảo ngược trên regression candidates.

    2026-09-22. Vì sao admin cần route này: regression_candidates sinh ra
    ground truth (dòng 'accepted' được export thành fixture hồi quy rồi dùng
    để KẾT LUẬN hệ thống đúng hay sai), nên nó là bảng có thẩm quyền cao
    nhất trong schema feedback - mà trước đây admin không thấy gì về nó, kể
    cả số dòng. Không giám sát được nơi sinh ra ground truth là một lỗ hổng
    thật: một người review nhập ẩu, hoặc một quyết định bị lật đi lật lại,
    đều diễn ra hoàn toàn trong bóng tối.

    ĐỌC THUẦN TUÝ - đúng nguyên tắc read-only của service này. Admin xem
    được toàn bộ dấu vết nhưng KHÔNG sửa, không duyệt, không lật quyết định
    nào; thao tác đảo ngược nằm ở feedback service và phải do một tài khoản
    review khác thực hiện (xem feedback_store.decide_regression_candidate).
    Đếm số dòng decision='reversed' ở đây là cách đo chất lượng của chính
    quy trình review, tách khỏi chất lượng của hệ thống QA.
    """
    from src.core.feedback_store import regression_decision_trail

    rows = regression_decision_trail(candidate_id=candidate_id, limit=limit)
    return [RegressionDecisionLogRow(**r) for r in rows]


@router.get("/tables")
def table_inventory(_: AdminKeyDep) -> list[TableInventoryRow]:
    """Số dòng của MỌI bảng dự án có ghi vào - trả lời trực tiếp "các bảng
    đã có những gì" thay vì chỉ vài số đếm chọn sẵn ở /pipeline/summary.
    Một bảng lỗi/thiếu không làm hỏng cả danh sách - báo lỗi ngay tại dòng
    đó (cùng nguyên tắc fail-soft-per-item như pipeline_summary() bên dưới)."""
    out = []
    with get_session() as session:
        for table in TRACKED_TABLES:
            try:
                n = session.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
                out.append(TableInventoryRow(table=table, row_count=int(n or 0)))
            except Exception as e:
                out.append(TableInventoryRow(table=table, row_count=None, error=str(e)))
    return out


@router.get("/files")
def pdf_document_files(
    _: AdminKeyDep,
    ticker: Annotated[str | None, Query(description="Lọc theo mã, ví dụ VNM - để trống xem tất cả")] = None,
    limit: Annotated[int, Query(ge=1, le=2000)] = 500,
) -> list[PdfDocumentRow]:
    """Danh sách file (pdf_documents) - mã nào đã có file gì, kể cả file
    nguồn TiniX (đoán qua _guess_file_origin() - xem docstring hàm đó) hay
    file thiếu PDF gốc thật (pdf_missing, xem
    scripts/register_pdf_documents.py's MISSING_SOURCE_PDF_HASH_SENTINEL).
    Đây CHÍNH LÀ bảng nguồn cho câu "mã nào đã có file gì" - không suy ra từ
    disclosure_chunks (grain khác: chunk, không phải file)."""
    sql = text("""
        SELECT id, ticker, doctype, source_file, fiscal_year, n_pages_total,
               extraction_version, document_version, is_active, pdf_hash,
               extracted_at, updated_at
        FROM pdf_documents
        WHERE (:ticker IS NULL OR ticker = :ticker)
        ORDER BY ticker NULLS LAST, source_file
        LIMIT :limit
    """)
    with get_session() as session:
        rows = session.execute(sql, {"ticker": ticker, "limit": limit}).mappings().all()
    return [
        PdfDocumentRow(
            id=r["id"], ticker=r["ticker"], doctype=r["doctype"], source_file=r["source_file"],
            fiscal_year=r["fiscal_year"], n_pages_total=r["n_pages_total"],
            extraction_version=r["extraction_version"], document_version=r["document_version"],
            is_active=bool(r["is_active"]),
            pdf_missing=(r["pdf_hash"] == MISSING_SOURCE_PDF_HASH_SENTINEL),
            origin_guess=_guess_file_origin(r["source_file"]),
            extracted_at=r["extracted_at"], updated_at=r["updated_at"],
        )
        for r in rows
    ]


@router.get("/prices/tickers")
def price_tickers(_: AdminKeyDep) -> list[dict]:
    """Mọi mã ĐANG có ít nhất 1 dòng trong company_price_snapshot, kèm số
    dòng và ngày gần nhất - để tab "Biến động giá" chỉ cho chọn mã thật sự
    có dữ liệu, thay vì để trống rồi báo lỗi sau khi bấm."""
    sql = text("""
        SELECT ticker, COUNT(*) AS n_days, MIN(trade_date) AS date_min, MAX(trade_date) AS date_max
        FROM company_price_snapshot
        GROUP BY ticker
        ORDER BY ticker
    """)
    try:
        with get_session() as session:
            rows = session.execute(sql).mappings().all()
    except Exception:
        return []
    return [dict(r) for r in rows]


@router.get("/prices")
def price_series(
    _: AdminKeyDep,
    ticker: Annotated[str, Query(description="Mã cần xem, ví dụ VNM")],
    days: Annotated[int, Query(ge=1, le=1825, description="Số ngày gần nhất")] = 90,
) -> list[PriceSnapshotRow]:
    """Chuỗi giá EOD (đóng cửa/mở/cao/thấp/khối lượng) cho MỘT mã, dùng để vẽ
    biểu đồ biến động trong tab admin.

    QUAN TRỌNG - đây là dữ liệu cuối ngày (end-of-day), nạp 1 lần/ngày qua
    scripts/market_snapshot.py (vnstock) trong Airflow DAG
    thesis_market_snapshot - KHÔNG phải dữ liệu tick-by-tick/thời gian thực.
    "Biến động" ở đây nghĩa là dao động open/high/low/close VÀ %thay đổi so
    với ngày giao dịch liền trước trong chính tập kết quả trả về (LAG() theo
    trade_date đã lọc, không phải so với hôm nay theo lịch nếu thị trường
    nghỉ/chưa có dòng mới nhất) - KHÔNG phải biến động trong ngày (không có
    cột đó trong company_price_snapshot). Xem thêm PriceSnapshotRow's
    docstring và cuộc thảo luận 2026-09-21 trong docs/PROGRESS.md về việc
    câu hỏi "biến động hôm nay" có trả lời được hay không.
    """
    sql = text("""
        SELECT trade_date, open_price, high_price, low_price, close_price, volume,
               close_price - LAG(close_price) OVER (ORDER BY trade_date) AS abs_change,
               LAG(close_price) OVER (ORDER BY trade_date) AS prev_close
        FROM company_price_snapshot
        WHERE ticker = :ticker
        ORDER BY trade_date DESC
        LIMIT :days
    """)
    with get_session() as session:
        rows = session.execute(sql, {"ticker": ticker.upper(), "days": days}).mappings().all()
    out = []
    for r in reversed(rows):  # trả về theo thứ tự thời gian tăng dần cho biểu đồ
        pct = None
        if r["prev_close"] not in (None, 0) and r["close_price"] is not None:
            pct = float((r["close_price"] - r["prev_close"]) / r["prev_close"] * 100)
        out.append(PriceSnapshotRow(
            trade_date=str(r["trade_date"]),
            open_price=float(r["open_price"]) if r["open_price"] is not None else None,
            high_price=float(r["high_price"]) if r["high_price"] is not None else None,
            low_price=float(r["low_price"]) if r["low_price"] is not None else None,
            close_price=float(r["close_price"]) if r["close_price"] is not None else None,
            volume=int(r["volume"]) if r["volume"] is not None else None,
            pct_change=pct,
        ))
    return out


@router.get("/pipeline/summary")
def pipeline_summary(_: AdminKeyDep) -> dict:
    """Một ảnh chụp nhanh toàn hệ thống cho trang admin sau này."""
    queries = {
        "n_companies": "SELECT COUNT(*) FROM companies",
        "n_line_items": "SELECT COUNT(*) FROM financial_line_items",
        "n_pdf_documents": "SELECT COUNT(*) FROM pdf_documents WHERE is_active",
        "n_chunks": "SELECT COUNT(*) FROM disclosure_chunks",
        "n_chunks_embedded": "SELECT COUNT(*) FROM disclosure_chunks WHERE embedding IS NOT NULL",
        "n_web_chunks": "SELECT COUNT(*) FROM web_chunks",
        "n_answer_runs": "SELECT COUNT(*) FROM feedback.answer_runs",
        "n_feedback": "SELECT COUNT(*) FROM feedback.answer_feedback",
        "n_review_pending": "SELECT COUNT(*) FROM feedback.review_queue_lite WHERE status = 'pending'",
        "n_vietstock_documents": "SELECT COUNT(*) FROM vietstock_document_manifest",
        "n_vietstock_downloaded": "SELECT COUNT(*) FROM vietstock_document_manifest WHERE download_status = 'downloaded'",
        "n_vietstock_news_events": "SELECT COUNT(*) FROM vietstock_news_events",
        "n_vietstock_structured_tables": "SELECT COUNT(*) FROM vietstock_structured_tables",
    }
    out: dict = {}
    with get_session() as session:
        for key, sql in queries.items():
            try:
                out[key] = int(session.execute(text(sql)).scalar() or 0)
            except Exception:
                # Một bảng thiếu không được làm hỏng cả bảng điều khiển -
                # báo null cho đúng ô đó rồi đi tiếp.
                out[key] = None
    return out


@router.get("/vietstock/summary")
def vietstock_summary(_: AdminKeyDep) -> dict:
    """Read-only Premium-sync status for the host-side crawler.

    A missing migration returns null counters rather than taking down the
    dashboard. The endpoint never returns session cookies, request headers,
    source HTML, or downloaded file contents.
    """
    queries = {
        "runs": "SELECT COUNT(*) FROM vietstock_ingestion_runs",
        "last_run_at": "SELECT MAX(started_at) FROM vietstock_ingestion_runs",
        "documents": "SELECT COUNT(*) FROM vietstock_document_manifest",
        "downloaded": "SELECT COUNT(*) FROM vietstock_document_manifest WHERE download_status = 'downloaded'",
        "access_denied": "SELECT COUNT(*) FROM vietstock_document_manifest WHERE download_status = 'access_denied'",
        "failed_documents": "SELECT COUNT(*) FROM vietstock_document_manifest WHERE download_status = 'failed'",
        "news_events": "SELECT COUNT(*) FROM vietstock_news_events",
        "structured_runs": "SELECT COUNT(*) FROM vietstock_structured_runs",
        "structured_tables": "SELECT COUNT(*) FROM vietstock_structured_tables",
    }
    out: dict = {}
    with get_session() as session:
        for key, sql in queries.items():
            try:
                out[key] = session.execute(text(sql)).scalar()
                if key not in {"last_run_at"} and out[key] is not None:
                    out[key] = int(out[key])
            except Exception:
                out[key] = None
    return out


@router.get("/vietstock/runs")
def vietstock_runs(
    _: AdminKeyDep,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
) -> list[dict]:
    """Latest crawl runs with counters. This is observation only; starting
    an authenticated browser process remains a local host command."""
    sql = text("""
        SELECT id, scope, status, started_at, finished_at, counters, error_summary
        FROM vietstock_ingestion_runs
        ORDER BY started_at DESC
        LIMIT :limit
    """)
    try:
        with get_session() as session:
            rows = session.execute(sql, {"limit": limit}).mappings().all()
    except Exception:
        return []
    return [{
        "id": row["id"], "scope": row["scope"], "status": row["status"],
        "started_at": row["started_at"], "finished_at": row["finished_at"],
        "counters": row["counters"] or {}, "error_summary": row["error_summary"],
    } for row in rows]


@router.get("/vietstock/tickers")
def vietstock_tickers(
    _: AdminKeyDep,
    limit: Annotated[int, Query(ge=1, le=500)] = 200,
) -> list[dict]:
    """One latest row per ticker, exposing crawl quality without raw HTML."""
    sql = text("""
        SELECT DISTINCT ON (ticker)
               ticker, company_url, status, fetched_at, http_status,
               document_count, news_count, error_message
        FROM vietstock_ticker_sync
        ORDER BY ticker, fetched_at DESC
        LIMIT :limit
    """)
    try:
        with get_session() as session:
            rows = session.execute(sql, {"limit": limit}).mappings().all()
    except Exception:
        return []
    return [dict(row) for row in rows]


# --- Khung cho MLOps giai đoạn sau ----------------------------------------
#
# Chưa mở, và đây là quyết định có chủ ý. Trước khi bật bất kỳ endpoint GHI
# nào ở mặt phẳng admin, cần có đủ 3 thứ:
#   1. Bảng kiểm toán (ai làm gì lúc nào) - hiện chưa có.
#   2. Xác thực có danh tính (JWT/OAuth2) thay cho API key tĩnh dùng chung,
#      vì "ai bấm nút" phải truy được về một con người cụ thể.
#   3. Bước xác nhận hai lớp cho hành động không thể hoàn tác (phong
#      champion model, chạy lại ETL đè dữ liệu).
#
# @router.post("/models/{model_id}/promote")   -> phong champion
# @router.post("/retrain/trigger")             -> kích hoạt huấn luyện lại
# @router.post("/etl/run")                     -> chạy lại pipeline nạp dữ liệu

app.include_router(router)
