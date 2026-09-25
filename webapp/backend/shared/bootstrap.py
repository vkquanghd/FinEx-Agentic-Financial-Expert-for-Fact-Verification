"""Shared bootstrap cho cả 3 service (qa / feedback / admin).

Kiến trúc microservices tối giản cho demo thesis (2026-08-07):
  - Mỗi service là MỘT app FastAPI độc lập, chạy port riêng, deploy/scale
    riêng được - KHÔNG import lẫn nhau; điểm chung duy nhất là module này
    (config + app factory) và tầng dữ liệu (src.core.db).
  - Ranh giới user/admin rõ từ đầu: service `qa` + `feedback` là mặt
    người dùng; `admin` là mặt vận hành (MLOps/warehouse SAU thesis) -
    tách sẵn để sau này gắn auth/quyền riêng mà không đụng mặt người dùng.
  - SSL: dev chạy HTTP; chạy HTTPS bằng self-signed cert (xem
    certs/gen_dev_certs.sh + README) - uvicorn nhận --ssl-keyfile/--ssl-certfile.

Chạy từ REPO ROOT (Thesis_Project/) để `src.*` import được:
    uvicorn webapp.backend.services.qa.main:app --port 8101
"""
import os
import sys
from pathlib import Path

# repo root = .../Thesis_Project (parents[3] tính từ file này)
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fastapi import FastAPI  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

# Frontend dev origins (Vite mặc định 5173). Thêm origin qua env khi cần.
CORS_ORIGINS = os.getenv(
    "WEBAPP_CORS_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173,https://localhost:5173",
).split(",")

QA_PORT = int(os.getenv("QA_SERVICE_PORT", "8101"))
FEEDBACK_PORT = int(os.getenv("FEEDBACK_SERVICE_PORT", "8102"))
ADMIN_PORT = int(os.getenv("ADMIN_SERVICE_PORT", "8103"))


def make_app(title: str, description: str = "") -> FastAPI:
    app = FastAPI(title=title, description=description, version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health():
        return {"status": "ok", "service": title}

    return app
