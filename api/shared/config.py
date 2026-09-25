"""Cấu hình dùng chung cho toàn bộ service trong api/.

Mọi service đọc cùng file .env (api/.env) nhưng CHỈ dùng phần của mình -
đây là ranh giới đầu tiên giữa user-plane và admin-plane: qa/feedback
service không bao giờ đọc ADMIN_API_KEY, và admin service không bao giờ
cần biết port của nhau ngoài qua gateway.
"""
import os
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

API_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = API_DIR.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=API_DIR / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- cổng nội bộ của từng service (chỉ gateway biết hết) ---
    qa_service_url: str = "http://127.0.0.1:8101"
    feedback_service_url: str = "http://127.0.0.1:8102"
    admin_service_url: str = "http://127.0.0.1:8103"
    gateway_port: int = 8443

    # --- xác thực ---
    # Admin-plane: BẮT BUỘC có API key. Không có key mặc định - service
    # admin từ chối khởi động nếu thiếu, để không bao giờ tồn tại một bản
    # deploy có admin mở toang vì quên set biến môi trường. Key này KHÔNG
    # bao giờ lộ ra trình duyệt - gateway giữ nó và tự gắn vào request khi
    # đã xác thực phiên đăng nhập có role=admin (xem
    # api/services/gateway/auth.py).
    admin_api_key: str = ""

    # 2026-08-07: đăng nhập riêng biệt cho 2 vai trò (yêu cầu người dùng) -
    # phiên đăng nhập là một cookie ký HMAC (thuần stdlib, không thêm thư
    # viện), khoá ký là SESSION_SECRET này. BẮT BUỘC đặt - fail-closed
    # giống ADMIN_API_KEY: thiếu thì /api/auth/login trả 503 thay vì chạy
    # với khoá rỗng đoán được. Sinh bằng:
    #   python3 -c "import secrets; print(secrets.token_urlsafe(32))"
    # Đặt cố định trong .env (KHÔNG để rỗng tự sinh ngẫu nhiên lúc khởi
    # động) vì gateway chạy --reload - một secret ngẫu nhiên mỗi lần nạp
    # lại code sẽ đăng xuất người dùng liên tục trong lúc phát triển.
    session_secret: str = ""
    session_ttl_seconds: int = 12 * 3600

    # --- SSL (gateway) ---
    ssl_certfile: str = str(API_DIR / "certs" / "dev-cert.pem")
    ssl_keyfile: str = str(API_DIR / "certs" / "dev-key.pem")

    # --- Gateway có tự phục vụ frontend tĩnh không? (2026-08-20) ---
    # Mặc định True để cách chạy CŨ (bash api/dev.sh, không Docker) không
    # đổi hành vi - gateway vẫn vừa là API vừa serve SPA như trước.
    #
    # Trong Docker thì đặt SERVE_FRONTEND=0: nginx là container riêng, nó
    # chấm dứt TLS và phục vụ api/frontend/, còn gateway chỉ còn là REST
    # API thuần (đúng ranh giới frontend/backend người dùng chốt
    # 2026-08-20). Hai chế độ dùng CHUNG một file code - không fork logic,
    # không có "bản Docker" và "bản không-Docker" lệch nhau.
    serve_frontend: bool = True

    # --- CORS: chỉ cho gateway; service nội bộ không mở CORS ---
    allowed_origins: list[str] = ["https://localhost:8443", "https://127.0.0.1:8443"]

    # --- HITL gate (khớp infra/postgres/migrations/013_feedback_hitl.sql) ---
    retrain_gate_min_feedback: int = 50
    retrain_gate_mean_rating: float = 3.5

    request_timeout_s: float = 180.0


@lru_cache
def get_settings() -> Settings:
    return Settings()


def ensure_repo_on_path() -> None:
    """Cho phép các service import src/* của luận văn.

    Chủ ý KHÔNG copy code luận văn vào api/: router, RAG pipeline, ETL vẫn
    là một nguồn sự thật duy nhất ở src/. api/ chỉ là lớp phục vụ (serving
    layer) bọc quanh nó - đúng ranh giới Layer 2/3 trong DSA v5.
    """
    import sys

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
