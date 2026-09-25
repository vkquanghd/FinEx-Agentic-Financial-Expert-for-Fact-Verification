"""Đăng nhập tách biệt user/admin (2026-08-07, theo yêu cầu người dùng).

Thiết kế có chủ ý ĐƠN GIẢN và YẾU về mật khẩu - đây là demo luận văn chạy
nội bộ trên máy cá nhân, một người dùng, không phải hệ thống nhiều người
dùng thật. Ghi rõ ở đây để không ai nhầm đây là mức bảo mật production:

  - Đúng 2 tài khoản, cứng trong code: "user" (vai trò user, dùng Chat) và
    "admin" (vai trò admin, dùng dashboard kiểm tra database). Mật khẩu
    của MỖI tài khoản = CHÍNH TÊN TÀI KHOẢN ĐÓ, đúng yêu cầu người dùng.
  - Không có bảng người dùng trong DB - đổi tài khoản/mật khẩu là sửa
    ACCOUNTS bên dưới, không phải một tính năng "quản lý user" đầy đủ.
  - Không hash mật khẩu (không cần - "mật khẩu" ở đây chỉ là tên tài
    khoản gõ lại lần 2, hash nó không thêm bảo mật thật nào).

Phiên đăng nhập: cookie HttpOnly chứa một token tự ký bằng HMAC-SHA256
(thuần thư viện chuẩn `hmac`/`hashlib`, không kéo thêm PyJWT/itsdangerous
chỉ để làm việc mà 20 dòng code đã làm được). Gateway là NƠI DUY NHẤT xác
minh cookie này - qa/feedback/admin service không biết gì về phiên đăng
nhập, chúng chỉ thấy request đã được gateway cho qua.
"""
import base64
import hashlib
import hmac
import json
import time
from typing import Literal, Optional

Role = Literal["user", "admin"]

# username -> (role kiểm soát truy cập, persona nghiệp vụ).
#
# HAI khái niệm KHÁC NHAU, cố ý không gộp (2026-08-07):
#   - role    = ai được gọi API nào (user -> /api/qa|feedback, admin ->
#               /api/admin). Đây là kiểm soát truy cập.
#   - persona = vai trò NGHIỆP VỤ, dùng để phân tích nghiên cứu "vai trò
#               nào thấy nhánh nào hữu ích" (xem src/core/personas.py).
# Mọi persona nghiệp vụ đều có role='user' - nếu gộp hai thứ này làm một
# thì cột phân tích sẽ toàn giá trị 'user', vô dụng cho nghiên cứu.
#
# Mật khẩu hợp lệ DUY NHẤT cho mỗi tài khoản chính là tên tài khoản đó
# (kiểm tra trong authenticate() bên dưới) - đúng quy ước người dùng chốt
# 2026-08-07, áp dụng cho cả các tài khoản persona thêm mới.
ACCOUNTS: dict[str, tuple[Role, Optional[str]]] = {
    # 4 persona nghiệp vụ chốt 2026-08-07 (xem src/core/personas.py)
    "analyst":   ("user", "analyst"),
    "auditor":   ("user", "auditor"),
    "executive": ("user", "executive"),
    "investor":  ("user", "investor"),
    # Giữ nguyên 2 tài khoản gốc để không phá thói quen đăng nhập đang có:
    # "user" không gắn persona nào (None -> ghi NULL vào answer_runs.persona,
    # trung thực hơn là gán bừa một persona).
    "user":      ("user", None),
    "admin":     ("admin", None),
}

COOKIE_NAME = "fqa_session"


def authenticate(username: str, password: str) -> Optional[tuple[Role, Optional[str]]]:
    """Trả về (role, persona) nếu (username, password) hợp lệ, None nếu sai.

    So sánh bằng hmac.compare_digest dù input ở đây không phải bí mật thật
    sự (password == username là công khai) - giữ thói quen so sánh thời
    gian hằng định nhất quán với require_admin() trong shared/security.py,
    phòng trường hợp sau này ACCOUNTS đổi sang mật khẩu thật.
    """
    entry = ACCOUNTS.get(username)
    if entry is None:
        return None
    if not hmac.compare_digest(password, username):
        return None
    return entry


def _b64e(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _b64d(s: str) -> bytes:
    pad = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(s + pad)


def create_session_token(username: str, role: Role, secret: str, ttl_seconds: int,
                         persona: Optional[str] = None) -> str:
    """payload.signature, cả hai base64url. payload chứa username/role/
    persona/hạn dùng (exp) - đủ để verify_session_token tự đứng độc lập,
    không cần tra DB/bộ nhớ nào khác (stateless, sống sót qua việc gateway
    tự reload).

    `persona` nằm TRONG token (không tra lại ACCOUNTS mỗi request) để
    gateway không phải đọc bảng nào khi forward - cùng lý do role đã nằm
    sẵn trong đó. Key viết tắt "p" cho nhất quán với "u"/"r"/"exp".
    """
    payload = json.dumps({"u": username, "r": role, "p": persona,
                          "exp": int(time.time()) + ttl_seconds},
                         separators=(",", ":")).encode()
    sig = hmac.new(secret.encode(), payload, hashlib.sha256).digest()
    return f"{_b64e(payload)}.{_b64e(sig)}"


def verify_session_token(token: str, secret: str) -> Optional[dict]:
    """Trả {"u", "r", "exp"} nếu token hợp lệ và chưa hết hạn, None nếu
    không. Không raise - lỗi định dạng bất kỳ (cookie bị sửa tay, cũ từ
    lần chạy với secret khác, v.v.) đều coi là "chưa đăng nhập", không
    phải lỗi 500."""
    try:
        payload_b64, sig_b64 = token.split(".", 1)
        payload = _b64d(payload_b64)
        expected_sig = hmac.new(secret.encode(), payload, hashlib.sha256).digest()
        actual_sig = _b64d(sig_b64)
        if not hmac.compare_digest(expected_sig, actual_sig):
            return None
        data = json.loads(payload)
        if data.get("exp", 0) < time.time():
            return None
        return data
    except Exception:
        return None
