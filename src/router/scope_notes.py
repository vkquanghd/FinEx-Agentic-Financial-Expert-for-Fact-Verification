"""Câu trả lời "khéo" khi ra ngoài phạm vi dữ liệu (2026-08-06).

Yêu cầu trực tiếp từ phiên test Chat 2026-08-06: "các câu hỏi ngoài phạm vi
không có trong dữ liệu phải trả lời thật khéo - ta hiện tại chỉ tập trung
vào chính data trọng tâm thôi."

Nguyên tắc: KHÔNG sửa system prompt của LLM để nó "nói khéo" (rủi ro nó
nói khéo bằng cách bịa). Giữ nguyên câu từ chối của model, rồi NỐI THÊM
một đoạn hướng dẫn phạm vi sinh deterministically ở đây - luôn đúng, không
tốn thêm lượt gọi LLM, và kiểm soát được từng chữ.

Mọi truy vấn DB đều cache theo process + fail-soft về text tĩnh.

BUG THẬT tìm thấy 2026-09-08 (test trực diện song ngữ trên máy thật): với
câu hỏi TIẾNG ANH, `polish_not_found()` chỉ kiểm tra marker tiếng Việt
(`NOT_FOUND_MARKERS`) nên KHÔNG bao giờ tự kích hoạt khi model trả lời đúng
bằng tiếng Anh ("No information found...") - nghe có vẻ ổn, NHƯNG khi model
(Qwen3 8B chạy local) lệch ngôn ngữ và trả lời "Không tìm thấy thông tin..."
dù được hỏi bằng tiếng Anh (vì mọi đoạn trích/context đưa vào đều là tiếng
Việt - xem generation.py's SYSTEM_PROMPT_EN rule 0 mới thêm cùng ngày để
giảm rủi ro này), `polish_not_found()` VẪN khớp marker tiếng Việt và nối
thêm nguyên một đoạn `scope_hint()` tiếng Việt vào - kết quả là câu hỏi
tiếng Anh nhận về một câu trả lời lẫn/toàn tiếng Việt. Sửa bằng cách nhận
`lang` ("vi"/"en", suy ra từ CÂU HỎI qua src.core.lang_detect.detect_language
- không suy từ câu trả lời, vì câu trả lời chính là thứ có thể bị lệch) và
chọn đúng bộ marker + đoạn hướng dẫn theo ngôn ngữ đó - xem
query_router.py's 3 điểm gọi polish_not_found().
"""
from functools import lru_cache
from typing import Optional

from sqlalchemy import text

from src.core.db import get_session

# Các mở đầu "không tìm thấy" mà generation.py/rag_pipeline.py có thể trả -
# dùng để nhận diện một câu trả lời rỗng-thông-tin cần được nói khéo lại.
NOT_FOUND_MARKERS = (
    "Không tìm thấy thông tin",
    "Không tìm thấy đoạn trích",
)

# English counterpart (2026-09-08) - matches generation.SYSTEM_PROMPT_EN's
# rule 2 phrasing ("No information found in the provided documents") and
# rag_pipeline.NO_CONTEXT_MSG_EN's opening ("No relevant excerpt was found").
NOT_FOUND_MARKERS_EN = (
    "No information found",
    "No relevant excerpt",
)

# Bộ mã trọng tâm demo (CORE_16 + ad-hoc) - đồng bộ tay với
# scripts/seed_companies.py; chỉ dùng để GIỚI THIỆU phạm vi trong câu trả
# lời, không dùng làm điều kiện chặn.
FOCUS_TICKERS = ["VNM", "SAB", "MSN", "KDC", "VIC", "VHM", "NVL", "KDH",
                 "HPG", "GAS", "DPM", "DCM", "MWG", "PNJ", "DGW", "FRT", "VNZ"]

_SUPPORTED_KINDS = (
    "Hiện demo trả lời tốt nhất các dạng câu: "
    "(1) tra cứu số liệu BCTC một mã theo kỳ (vd: \"Doanh thu của VNZ quý 1 năm 2026?\"); "
    "(2) tìm thông tin trong tài liệu công bố (vd: \"VIC nêu rủi ro gì trong báo cáo thường niên?\"); "
    "(3) kịch bản what-if (vd: \"Nếu COGS của HPG tăng 5%...?\"); "
    "(4) so sánh hiệu suất giá 2 mã (vd: \"So sánh HPG và GAS\")."
)

# English counterpart (2026-09-08) - same 4 kinds, same example tickers (no
# need to translate the tickers themselves).
_SUPPORTED_KINDS_EN = (
    "The demo currently answers these question types best: "
    "(1) a single ticker's financial-statement figure for one period (e.g. "
    "\"What was VNZ's revenue in Q1 2026?\"); "
    "(2) information from a disclosure document (e.g. \"What risks does VIC's "
    "annual report mention?\"); "
    "(3) a what-if scenario (e.g. \"If HPG's COGS increased by 5%...?\"); "
    "(4) comparing 2 tickers' price performance (e.g. \"Compare HPG and GAS\")."
)


@lru_cache(maxsize=256)
def _ticker_data_status(ticker: str) -> str:
    """-> "has_data" | "known_no_data" | "unknown". Fail-soft về "unknown"
    khi DB không truy vấn được (không để phần nói khéo làm vỡ câu trả lời)."""
    try:
        with get_session() as session:
            known = session.execute(
                text("SELECT 1 FROM companies WHERE ticker = :t"), {"t": ticker}
            ).first()
            if not known:
                return "unknown"
            has = session.execute(text(
                "SELECT (EXISTS (SELECT 1 FROM financial_line_items WHERE ticker = :t)) OR "
                "(EXISTS (SELECT 1 FROM disclosure_chunks WHERE ticker = :t))"
            ), {"t": ticker}).scalar()
            return "has_data" if has else "known_no_data"
    except Exception:
        return "unknown"


def scope_hint(ticker: Optional[str], lang: str = "vi") -> str:
    """Đoạn hướng dẫn phạm vi để nối sau một câu trả lời "không tìm thấy".
    `lang="en"` (thêm 2026-09-08) trả bản tiếng Anh - xem module docstring
    cho bug thật đã sửa (câu hỏi tiếng Anh nhận nhầm đoạn hướng dẫn tiếng
    Việt)."""
    if lang == "en":
        return _scope_hint_en(ticker)

    focus = ", ".join(FOCUS_TICKERS[:8]) + ",... (16 mã phi ngân hàng) và VNZ"
    if ticker:
        status = _ticker_data_status(ticker)
        if status == "known_no_data":
            head = (f"Mã {ticker} có trong danh mục nhưng CHƯA được nạp dữ liệu "
                    f"chi tiết. Demo hiện tập trung vào bộ mã trọng tâm: {focus}.")
        elif status == "unknown":
            head = (f"Mã/công ty \"{ticker}\" chưa nằm trong bộ dữ liệu trọng tâm "
                    f"của demo (gồm: {focus}). Lưu ý nhóm ngân hàng (VCB, BID, ACB, "
                    f"VPB...) được chủ đích loại khỏi phạm vi vì dùng mẫu BCTC khác.")
        else:
            return (f"Hệ thống có dữ liệu của {ticker}, nhưng các đoạn trích được chọn "
                    "chưa đủ để xác nhận câu trả lời. Điều này không chứng minh thông tin "
                    "không có trong tài liệu; cần kiểm tra độ phủ dữ liệu và bước truy xuất.")
    else:
        head = (f"Câu hỏi này nằm ngoài phạm vi dữ liệu demo. Demo tập trung vào "
                f"bộ mã trọng tâm ({focus}), giai đoạn 2024-2026, và chưa hỗ trợ "
                f"câu hỏi tổng hợp toàn thị trường (vd: \"công ty nào cùng công ty "
                f"kiểm toán với X\").")
    return head + "\n" + _SUPPORTED_KINDS


def _scope_hint_en(ticker: Optional[str]) -> str:
    """English counterpart of scope_hint()'s body (2026-09-08) - same 3
    branches (known_no_data / unknown / has_data-but-question-mismatch),
    same underlying _ticker_data_status() DB lookup, just English wording."""
    focus = ", ".join(FOCUS_TICKERS[:8]) + ",... (16 non-bank tickers) and VNZ"
    if ticker:
        status = _ticker_data_status(ticker)
        if status == "known_no_data":
            head = (f"{ticker} is in the company list but detailed data has NOT been "
                    f"loaded yet. The demo currently focuses on this ticker set: {focus}.")
        elif status == "unknown":
            head = (f"\"{ticker}\" is not in the demo's focus ticker set ({focus}). "
                    f"Note: banks (VCB, BID, ACB, VPB...) are deliberately excluded "
                    f"from scope since they use a different financial-statement template.")
        else:
            return (f"The system has data for {ticker}, but the selected excerpts did not "
                    "establish an answer. This does not prove the information is absent "
                    "from the documents; data coverage and retrieval need checking.")
    else:
        head = ("This question is out of scope for the demo's data. The demo focuses "
                f"on this ticker set ({focus}), the 2024-2026 period, and does not yet "
                "support market-wide aggregate questions (e.g. \"which companies share "
                "the same auditor as X\").")
    return head + "\n" + _SUPPORTED_KINDS_EN


def polish_not_found(answer: str, ticker: Optional[str], lang: str = "vi") -> str:
    """Nếu `answer` là dạng "không tìm thấy" thì nối thêm hướng dẫn phạm vi;
    ngược lại trả nguyên văn. `lang` ("vi"/"en", mặc định "vi" để KHÔNG đổi
    hành vi cũ ở bất kỳ call site nào chưa được cập nhật để truyền `lang`) -
    QUYẾT ĐỊNH bởi ngôn ngữ CÂU HỎI (src.core.lang_detect.detect_language),
    không phải ngôn ngữ thực tế của `answer` - xem module docstring: chính
    `answer` là thứ có thể đã lệch ngôn ngữ, nên không thể dùng nó để tự
    quyết định ngôn ngữ của phần nối thêm."""
    # Luôn kiểm tra CẢ 2 bộ marker (VI + EN), không chỉ bộ theo `lang` - nếu
    # model lệch ngôn ngữ (câu hỏi EN nhưng answer ra tiếng Việt "Không tìm
    # thấy..."), vẫn phải nhận diện được đó là câu "không tìm thấy" để nối
    # đúng phần hướng dẫn theo NGÔN NGỮ CÂU HỎI (lang), không phải ngôn ngữ
    # thực tế của answer.
    all_markers = NOT_FOUND_MARKERS + NOT_FOUND_MARKERS_EN
    if any(answer.strip().startswith(m) or m in answer[:120] for m in all_markers):
        return answer.rstrip() + "\n\n" + scope_hint(ticker, lang=lang)
    return answer
