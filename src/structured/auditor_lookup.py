"""Tra cứu ĐẢO NGƯỢC: từ tên công ty kiểm toán -> danh sách doanh nghiệp
được kiểm toán, theo từng năm. (2026-08-07, phản hồi trực tiếp câu hỏi test
"KPMG đang kiểm toán các công ty nào qua từng năm?")

Vì sao đây là DAG-shaped, không phải RAG: `disclosures.auditor` là cột CÓ
CẤU TRÚC, nạp trực tiếp từ Vietstock Bao-cao-tai-chinh (row 10) bởi
src.ingestion.vietstock_etl.parse_bao_cao_tai_chinh() - xem
scripts/export_auditor_ground_truth.py, vốn đã dùng chính cột này làm
ground truth cho benchmark RAG (câu hỏi 1-CHIỀU "kiểm toán của X là ai").
Câu hỏi NGƯỢC (1 hãng kiểm toán -> N công ty x N năm) là một phép tổng hợp
(GROUP BY) trên toàn bộ cột đó - RAG (top-k similarity trên một câu hỏi)
về bản chất không làm được việc "quét và gộp qua mọi tài liệu", nên trước
khi có nhánh này câu hỏi loại đó luôn rơi vào RAG và trả lời "không tìm
thấy" dù dữ liệu thật ra có sẵn và chính xác tuyệt đối trong DB.

Không đi qua RAG/LLM nên không có rủi ro bịa (hallucination) - kết quả là
chính xác những gì `disclosures.auditor` ghi nhận, đúng tinh thần "RAG/DAG
phải làm thật tốt vì không được gọi API ngoài" mà dự án đặt ra.
"""
import re

# Tên hãng kiểm toán thường gặp trên thị trường VN - đủ để nhận diện câu
# hỏi, KHÔNG phải danh sách đầy đủ mọi hãng đã từng xuất hiện trong
# disclosures.auditor. Giữ ngắn n để tránh false-positive; mẫu \b để không
# khớp nhầm bên trong từ khác. Chuỗi thứ 2 trong mỗi tuple là chuỗi dùng để
# lọc SQL (ILIKE '%...%') - khớp với cách auditor thường được ghi
# ("Công ty TNHH KPMG Việt Nam", "Công ty TNHH PwC (Việt Nam)"...).
_AUDITOR_FIRM_PATTERNS: list[tuple[str, str]] = [
    (r"kpmg", "KPMG"),
    (r"pricewaterhousecoopers", "PwC"),
    (r"\bpwc\b", "PwC"),
    (r"ernst\s*&?\s*young", "Ernst & Young"),
    (r"\bey\b", "EY"),
    (r"deloitte", "Deloitte"),
    (r"\baasc\b", "AASC"),
    (r"a\s*&\s*c\b", "A&C"),
    (r"\bcrowe\b", "Crowe"),
    (r"grant\s*thornton", "Grant Thornton"),
    (r"\bmazars\b", "Mazars"),
    (r"\buhy\b", "UHY"),
    (r"\bafc\b", "AFC"),
    (r"\bs4b\b", "S4B"),
    (r"\brsm\b", "RSM"),
    (r"\bbdo\b", "BDO"),
    (r"\bimmanuel\b", "Immanuel"),
    (r"\bvaco\b", "VACO"),
    (r"\bmoore\b", "Moore"),
]


# Viết tắt chuẩn -> các cách viết ĐẦY ĐỦ có thể gặp trong văn bản
# (2026-08-07, dựng từ bằng chứng thật: 23 câu bị keyword_faithfulness chấm
# 0 trong lần chạy eval đầy đủ đầu tiên - xem
# docs/eval_run1_faithfulness_analysis.md).
#
# Vì sao cần: ground truth `disclosures.auditor` (Vietstock) lưu VIẾT TẮT
# ('EY', 'VAE', 'NVT', 'IAV'), còn RAG trả lời trích TÊN PHÁP LÝ ĐẦY ĐỦ từ
# chính tài liệu ('Công ty TNHH Ernst & Young Việt Nam'). Khớp chuỗi thô
# giữa hai quy ước đặt tên khác nhau -> chấm sai 11/69 câu TRẢ LỜI ĐÚNG
# thành 0 điểm, kéo faithfulness từ ~0.82 xuống 0.66.
#
# Nguyên tắc khi mở rộng bảng này: THÀ THIẾU CÒN HƠN THỪA. Một alias quá
# rộng sẽ khiến câu trả lời SAI hãng được chấm đúng - hỏng theo hướng nguy
# hiểm hơn nhiều (thổi phồng kết quả) so với việc bỏ sót một alias (chỉ làm
# điểm thấp hơn thực tế). Ví dụ cụ thể: KHÔNG khai "kiểm toán quốc tế" cho
# iCPA dù tên đăng ký của họ là "INTERNATIONAL AUDITING CO., LTD", vì cụm
# đó trùng với IAV ("Kiểm toán và Định giá Quốc tế") - để cả hai cùng không
# khớp còn hơn khớp nhầm chéo.
AUDITOR_ALIASES: dict[str, list[str]] = {
    "EY": [r"ernst\s*&?\s*young"],
    "PWC": [r"pricewaterhousecoopers", r"\bpwc\b"],
    "KPMG": [r"\bkpmg\b"],
    "DELOITTE": [r"\bdeloitte\b"],
    # Các hãng nội địa gặp trong corpus thật:
    "VAE": [r"ki[eể]m to[aá]n v[aà] [dđ][iị]nh gi[aá] vi[eệ]t nam",
            r"vietnam auditing and evaluation"],
    "NVT": [r"ki[eể]m to[aá]n nh[aâ]n t[aâ]m vi[eệ]t",
            r"nhan tam viet auditing"],
    "IAV": [r"ki[eể]m to[aá]n v[aà] [dđ][iị]nh gi[aá] qu[oố]c t[eế]",
            r"international auditing and valuation"],
    "AAC": [r"ki[eể]m to[aá]n v[aà] k[eế] to[aá]n aac", r"\baac\b"],
    "A&C": [r"a\s*&\s*c\b"],
    "AASC": [r"\baasc\b"],
    "BDO": [r"\bbdo\b"],
    "CROWE": [r"\bcrowe\b"],
}


def canonical_auditors(text: str) -> set[str]:
    """Tập viết tắt chuẩn của MỌI hãng kiểm toán nhận ra được trong `text`.

    Dùng để chuẩn hoá CẢ HAI phía (ground truth và câu trả lời) trước khi so
    khớp trong src/evaluation/metrics.py's keyword_faithfulness() - thay vì
    so chuỗi thô giữa hai quy ước đặt tên khác nhau.
    """
    if not text:
        return set()
    low = text.lower()
    out = set()
    for canon, patterns in AUDITOR_ALIASES.items():
        for pat in patterns:
            if re.search(pat, low):
                out.add(canon)
                break
    return out


def extract_auditor_firm(question: str) -> str | None:
    """Trả về chuỗi dùng để lọc SQL (ILIKE) nếu câu hỏi nhắc tới một hãng
    kiểm toán đã biết, ngược lại None."""
    low = question.lower()
    for pattern, canonical in _AUDITOR_FIRM_PATTERNS:
        if re.search(pattern, low):
            return canonical
    return None


def lookup_auditors_by_company(ticker: str, years: list[int]) -> list[dict]:
    """Annual filing metadata only; do not substitute interim reviewers."""
    from sqlalchemy import text
    from src.core.db import get_session

    with get_session() as session:
        rows = session.execute(text("""
            SELECT fiscal_year, period, consolidation, auditor, source_pdf_path
            FROM disclosures
            WHERE upper(ticker) = :ticker AND period = 'YEAR'
              AND auditor IS NOT NULL AND trim(auditor) <> ''
            ORDER BY fiscal_year DESC, consolidation
        """), {"ticker": ticker.upper()}).mappings().all()
    return [dict(row) for row in rows if not years or row['fiscal_year'] in years]


def lookup_companies_by_auditor_fts(firm_query: str, limit: int = 500) -> dict:
    """Bổ sung cho lookup_companies_by_auditor(): quét TOÀN VĂN mọi
    disclosure_chunks (bao gồm cả kho TiniX - scripts/tinix_ingest_adapter.py
    - phủ đủ báo cáo kiểm toán của 17 mã trọng tâm, rộng hơn nhiều so với
    disclosures.auditor vốn chỉ nạp từ Vietstock Bao-cao-tai-chinh và có
    thể THIẾU nhiều mã/năm - đúng như phiên test 2026-08-07 cho thấy: kết
    quả theo disclosures.auditor chỉ ra toàn công ty chứng khoán/bảo hiểm,
    không mã CORE_16 nào, dù các mã đó chắc chắn có văn bản BCTC kiểm toán
    đã nạp qua TiniX.

    QUAN TRỌNG - khác mức độ tin cậy với lookup_companies_by_auditor():
    khớp toàn văn "chunk văn bản BCTC có nhắc tới tên hãng" KHÔNG đồng
    nghĩa "hãng này là auditor chính thức của công ty đó" - tên hãng có
    thể xuất hiện trong thuyết minh vì lý do khác (công ty con dùng auditor
    khác, auditor tiền nhiệm được nhắc để so sánh, v.v). Vì vậy hàm này chỉ
    trả về DANH SÁCH ỨNG VIÊN cần đối chiếu thêm, KHÔNG được trình bày
    ngang hàng với kết quả có cấu trúc - xem _format_auditor_lookup() ở
    query_router.py, tách rõ 2 mức "đã xác nhận" vs "nhắc tới, chưa xác
    nhận" để không tái phạm đúng lỗi "trả lời chưa được tính toán/xác thực
    đủ kỹ" mà demo test ban đầu đã chỉ ra.

    Dùng GIN full-text index có sẵn (to_tsvector('simple', chunk_text) -
    infra/pgvector/migrations/002_chunk_text_fts.sql), không phải ANN vector
    search - đây là tra cứu từ khóa chính xác (tên hãng kiểm toán), không
    cần similarity ngữ nghĩa.
    """
    from sqlalchemy import text

    from src.core.db import get_session
    from src.retrieval.vector_store import extract_year

    sql = text(
        """
        SELECT DISTINCT ticker, source_file
        FROM disclosure_chunks
        WHERE doctype = 'financial_statement'
          AND to_tsvector('simple', chunk_text) @@ plainto_tsquery('simple', :firm)
        LIMIT :limit
        """
    )
    with get_session() as session:
        rows = session.execute(sql, {"firm": firm_query, "limit": limit}).mappings().all()

    by_year: dict[int, set] = {}
    no_year: set = set()
    for r in rows:
        year = extract_year(r["source_file"])
        if year is None:
            no_year.add(r["ticker"])
        else:
            by_year.setdefault(year, set()).add(r["ticker"])

    return {
        "ok": True,
        "firm_query": firm_query,
        "n_chunks_matched": len(rows),
        "by_year": {y: sorted(ts) for y, ts in sorted(by_year.items(), reverse=True)},
        "no_year_tickers": sorted(no_year),
    }


def lookup_companies_by_auditor(firm_query: str, period: str = "YEAR") -> dict:
    """Quét `disclosures.auditor` (đã nạp từ Vietstock, không qua RAG) và
    trả về mọi (ticker, fiscal_year) có tên hãng khớp `firm_query`.

    `period="YEAR"` mặc định vì đây là câu hỏi mang tính tổng hợp/năm
    (giống lựa chọn year-1 mặc định ở whatif/rag khác trong project) -
    kỳ quý sẽ làm bảng kết quả phình to với cùng một thông tin kiểm toán
    lặp lại 4 lần/năm, không thêm giá trị cho câu hỏi "công ty nào".
    """
    from sqlalchemy import text

    from src.core.db import get_session

    sql = text(
        """
        SELECT ticker, fiscal_year, auditor
        FROM disclosures
        WHERE period = :period AND auditor ILIKE :pattern
        ORDER BY fiscal_year DESC, ticker
        """
    )
    with get_session() as session:
        rows = session.execute(
            sql, {"period": period, "pattern": f"%{firm_query}%"}
        ).mappings().all()

    by_year: dict[int, list[str]] = {}
    for r in rows:
        by_year.setdefault(int(r["fiscal_year"]), []).append(r["ticker"])

    return {
        "ok": True,
        "firm_query": firm_query,
        "period": period,
        "n_rows": len(rows),
        "by_year": dict(sorted(by_year.items(), reverse=True)),
        "raw_auditor_names": sorted({r["auditor"] for r in rows}),
    }
