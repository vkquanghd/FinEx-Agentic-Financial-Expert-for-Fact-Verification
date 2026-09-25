"""Từ viết tắt tài chính tiếng Việt -> cụm đầy đủ (2026-08-06).

Yêu cầu từ phiên test Chat 2026-08-06: "cần rà soát lại các tên viết tắt,
các từ viết tắt thông dụng của Việt Nam". Đã cân nhắc thư viện NLP tiếng
Việt (underthesea, pyvi) - chúng làm tách từ/POS/NER tốt nhưng KHÔNG có
sẵn từ điển khai triển viết tắt tài chính (LNST, VCSH...), nên phần lõi
vẫn phải là từ điển tự quản lý; vì vậy giữ dependency-free và deterministic
(một dict + regex word-boundary), khỏi kéo thêm thư viện chỉ để làm việc
mà thư viện không làm.

Nguyên tắc chọn từ:
  - Chỉ nhận viết tắt >= 3 ký tự, KHỚP THEO TỪ NGUYÊN VẸN (\\b...\\b),
    không khớp substring - tránh "vlnst" ăn nhầm "lnst".
  - Không nhận viết tắt 2 ký tự ("DT", "TS", "LN"...) - va chạm quá nhiều
    (DT = doanh thu / diện tích / điện thoại...).
  - Mỗi viết tắt map về đúng MỘT cụm chuẩn đã có trong INDICATOR_ALIASES
    của schema.py - expand() chỉ dịch chữ, việc resolve về MS code vẫn đi
    qua một đường duy nhất (resolve_indicator), không tạo đường tắt mới.

Đồng bộ: mọi key ở đây nên có mặt trong NON_TICKER_ABBREVIATIONS của
query_builder.py (dạng UPPER) để không bị regex ticker ăn nhầm khi DB
không kiểm tra được (fail-open path của _extract_ticker).
"""
import re

# viết tắt (lower, không dấu) -> cụm đầy đủ (đúng chính tả alias đã đăng ký
# trong schema.py để resolve_indicator khớp exact sau khi _normalize)
FINANCIAL_ABBREVIATIONS: dict[str, str] = {
    "lnst": "loi nhuan sau thue",
    "lntt": "loi nhuan truoc thue",
    "lng": "loi nhuan gop",          # lưu ý: trong ngữ cảnh chỉ tiêu, không phải khí LNG
    "dtt": "doanh thu thuan",
    "gvhb": "gia von hang ban",
    "tts": "tong tai san",
    "tsnh": "tai san ngan han",
    "tsdh": "tai san dai han",
    "vcsh": "von chu so huu",
    "npt": "no phai tra",
    "htk": "hang ton kho",
    "tdtt": "tien va tuong duong tien",
    "dtkd": "dong tien kinh doanh",
}

_PATTERN = re.compile(
    r"\b(" + "|".join(sorted(FINANCIAL_ABBREVIATIONS, key=len, reverse=True)) + r")\b"
)


def expand(normalized_text: str) -> str:
    """Khai triển viết tắt trong text ĐÃ normalize (lower, không dấu - đầu
    ra của schema._normalize). Trả về text sau khai triển; không thay đổi
    gì nếu không có viết tắt nào."""
    return _PATTERN.sub(lambda m: FINANCIAL_ABBREVIATIONS[m.group(1)], normalized_text)


# --- Khai triển trên text THÔ (có dấu, giữ nguyên hoa/thường) -------------
#
# 2026-08-07: phát hiện qua Chat rằng LLM context-resolver có thể "đoán bừa"
# viết tắt TRƯỚC khi câu hỏi tới router ("TTS của VNZ nâm 2025" bị viết lại
# thành "Tỷ giá trung tâm của VNZ năm 2025" - bịa hoàn toàn). Fix: khai
# triển viết tắt bằng từ điển NGAY trên câu người dùng gõ, trước khi đưa
# qua resolver - deterministic đi trước, LLM chỉ còn xử lý phần ngữ cảnh
# hội thoại thật sự. Chỉ khớp khi token viết HOA TOÀN BỘ (đúng cách người
# ta gõ viết tắt); "tts" thường/lẫn lộn thì để nguyên cho an toàn.
FINANCIAL_ABBREVIATIONS_DISPLAY: dict[str, str] = {
    "LNST": "lợi nhuận sau thuế",
    "LNTT": "lợi nhuận trước thuế",
    "LNG": "lợi nhuận gộp",
    "DTT": "doanh thu thuần",
    "GVHB": "giá vốn hàng bán",
    "TTS": "tổng tài sản",
    "TSNH": "tài sản ngắn hạn",
    "TSDH": "tài sản dài hạn",
    "VCSH": "vốn chủ sở hữu",
    "NPT": "nợ phải trả",
    "HTK": "hàng tồn kho",
    "TDTT": "tiền và tương đương tiền",
    "DTKD": "dòng tiền kinh doanh",
}

_DISPLAY_PATTERN = re.compile(
    r"\b(" + "|".join(sorted(FINANCIAL_ABBREVIATIONS_DISPLAY, key=len, reverse=True)) + r")\b"
)


def expand_display(raw_text: str) -> str:
    """Khai triển viết tắt tài chính trong câu hỏi THÔ của người dùng (giữ
    dấu tiếng Việt). Chỉ thay token viết HOA nguyên vẹn. Dùng ở app layer
    TRƯỚC khi câu đi qua LLM context-resolver / router."""
    return _DISPLAY_PATTERN.sub(
        lambda m: FINANCIAL_ABBREVIATIONS_DISPLAY[m.group(1)], raw_text
    )
