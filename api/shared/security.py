"""Ranh giới bảo mật giữa user-plane và admin-plane.

Nguyên tắc: phân tách theo TIẾN TRÌNH, không chỉ theo đường dẫn URL.
Admin service là một process riêng, bind vào 127.0.0.1, và mọi route của
nó nằm sau `AdminKeyDep`. Ngay cả khi gateway bị cấu hình sai và lộ
/admin/* ra ngoài, request vẫn phải có API key mới đi tiếp - không có
đường nào chạm tới hàm xử lý mà bỏ qua lớp này.

Vì sao chưa dùng OAuth2/JWT: demo luận văn chạy nội bộ, một người dùng,
chưa có user store. API key tĩnh là mức bảo mật tương xứng với mối đe dọa
hiện tại và không giả vờ an toàn hơn thực tế. Khi lên MLOps thật (nhiều
người, phân quyền), thay thân hàm `require_admin` bằng verify JWT là đủ -
mọi route đã sẵn sàng vì chúng phụ thuộc vào dependency chứ không phải
vào cách xác thực cụ thể.

2026-08-07: gateway giờ có đăng nhập phiên riêng (cookie, 2 tài khoản
user/admin - xem api/services/gateway/auth.py) và chỉ cho phiên role=admin
đi qua tới /api/admin/*, TỰ gắn X-Admin-Key thật lúc forward - trình
duyệt không bao giờ thấy key. Nhưng lớp này (`require_admin` bên dưới)
KHÔNG bị bỏ đi hay thay thế: admin service vẫn tự đòi key thật độc lập
với gateway, đúng nguyên tắc "phân tách theo tiến trình" ở đầu file - nếu
gateway có bị cấu hình sai và quên gọi _require_session(), hoặc ai đó gọi
thẳng 127.0.0.1:8103 bỏ qua gateway, request vẫn bị chặn ở đây.
"""
import hmac
from typing import Annotated

from fastapi import Depends, Header, HTTPException, status

from api.shared.config import get_settings


def require_admin(x_admin_key: Annotated[str | None, Header()] = None) -> str:
    settings = get_settings()
    if not settings.admin_api_key:
        # Fail-closed: thiếu cấu hình thì KHÓA, không mở.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Admin plane chưa được cấu hình (thiếu ADMIN_API_KEY).",
        )
    if not x_admin_key or not hmac.compare_digest(x_admin_key, settings.admin_api_key):
        # compare_digest: so sánh thời gian hằng định, tránh timing attack.
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Cần header X-Admin-Key hợp lệ.",
            headers={"WWW-Authenticate": "AdminKey"},
        )
    return x_admin_key


AdminKeyDep = Annotated[str, Depends(require_admin)]
