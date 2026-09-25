"""Gateway (biên public, có SSL) - port 8443.

Vai trò: là cửa vào DUY NHẤT, không chứa logic nghiệp vụ. Năm việc -
kết thúc TLS, đăng nhập/phiên đăng nhập, phân tuyến theo tiền tố VÀ theo
vai trò, chuyển tiếp header xác thực admin (server-side, không lộ ra
trình duyệt), và phục vụ frontend đã build.

Vì sao cần một lớp riêng thay vì mở thẳng 3 service ra ngoài:
  - Chỉ MỘT cổng chạm được từ ngoài. qa/feedback/admin bind vào 127.0.0.1,
    nên kể cả khi firewall cấu hình lỏng, chúng vẫn không lộ.
  - Chứng chỉ TLS quản lý ở một chỗ, không nhân ba.
  - Phiên đăng nhập (cookie) cũng quản lý ở một chỗ - qa/feedback/admin
    không biết gì về "ai đang đăng nhập", chúng chỉ thấy request đã được
    gateway cho qua.
  - Sau này thêm rate limit, log truy cập, WAF thì làm ở đây, không phải
    sửa cả 3 service.

Tiền tố định tuyến (ranh giới user-plane / admin-plane nhìn thấy được ngay
trên URL) + vai trò BẮT BUỘC để gọi được (2026-08-07, đăng nhập riêng biệt
user/admin - xem api/services/gateway/auth.py):
    /api/qa/*        -> qa service        (cần phiên đăng nhập role=user)
    /api/feedback/*  -> feedback service  (cần phiên đăng nhập role=user)
    /api/admin/*     -> admin service     (cần phiên đăng nhập role=admin;
                                            gateway TỰ gắn X-Admin-Key thật,
                                            trình duyệt không bao giờ thấy)
    /api/auth/*      -> xử lý ngay tại gateway, không proxy đi đâu
    /*               -> frontend tĩnh (public - trang đăng nhập cũng nằm
                                        trong SPA này, phải tải được TRƯỚC
                                        khi đăng nhập)
"""
from collections.abc import AsyncIterable
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Request, Response as FastAPIResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

from api.services.gateway.auth import (
    COOKIE_NAME,
    Role,
    authenticate,
    create_session_token,
    verify_session_token,
)
from api.shared.config import get_settings
from api.shared.schemas import ServiceHealth

settings = get_settings()
FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"

UPSTREAMS = {
    "qa": settings.qa_service_url,
    "feedback": settings.feedback_service_url,
    "admin": settings.admin_service_url,
}

# Vai trò bắt buộc để gọi mỗi service qua gateway - đúng ranh giới "riêng
# biệt" người dùng yêu cầu: role user KHÔNG gọi được /api/admin/*, role
# admin KHÔNG gọi được /api/qa|feedback/*. Tách biệt tuyệt đối, không phải
# admin "cũng là user nâng cao".
ROLE_FOR_SERVICE: dict[str, Role] = {"qa": "user", "feedback": "user", "admin": "admin"}

# Header không được chuyển tiếp nguyên xi: host phải theo upstream, còn
# các header về độ dài/mã hoá do httpx tự tính lại. X-Admin-Key bị lọc
# khỏi request TỪ TRÌNH DUYỆT (client không bao giờ được tự xưng admin
# bằng header tự chế) - gateway tự gắn key thật sau khi xác thực phiên.
_HOP_BY_HOP = {"host", "content-length", "transfer-encoding", "connection",
               "keep-alive", "upgrade", "x-admin-key", "cookie",
               # 2026-08-07: persona/username do GATEWAY gắn từ phiên đã xác
               # thực, KHÔNG bao giờ nhận từ trình duyệt - nếu không lọc,
               # client tự chế header là tự xưng persona bất kỳ, làm hỏng
               # toàn bộ dữ liệu phân tích theo vai trò. Cùng nguyên tắc với
               # x-admin-key ở trên.
               "x-user-persona", "x-user-name"}


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=64)


class SessionInfo(BaseModel):
    username: str
    role: Role
    # Vai trò NGHIỆP VỤ (analyst/auditor/executive/investor) - KHÁC `role`
    # ở trên vốn là kiểm soát truy cập. None với tài khoản user/admin gốc.
    # Xem src/core/personas.py và api/services/gateway/auth.py's ACCOUNTS.
    persona: str | None = None


def _get_session(request: Request) -> SessionInfo | None:
    token = request.cookies.get(COOKIE_NAME)
    if not token or not settings.session_secret:
        return None
    data = verify_session_token(token, settings.session_secret)
    if not data:
        return None
    # .get("p") chứ không data["p"]: cookie phát hành TRƯỚC 2026-08-07 không
    # có khóa "p" - vẫn phải đăng nhập được, không đá người dùng ra ngoài
    # chỉ vì thêm một trường mới.
    return SessionInfo(username=data["u"], role=data["r"], persona=data.get("p"))


def _require_session(request: Request, service: str) -> SessionInfo:
    """401 nếu chưa đăng nhập, 403 nếu đăng nhập sai vai trò cho service
    này. Gọi ĐẦU TIÊN trong mọi proxy - không request nào chạm tới
    qa/feedback/admin mà chưa qua đây."""
    session = _get_session(request)
    if not session:
        raise HTTPException(status_code=401, detail="Chưa đăng nhập.")
    required = ROLE_FOR_SERVICE.get(service)
    if required and session.role != required:
        raise HTTPException(
            status_code=403,
            detail=f"Tài khoản vai trò '{session.role}' không có quyền dùng '{service}'.",
        )
    return session


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Một client dùng lại kết nối cho mọi request - tạo client mới mỗi
    # request là lỗi hiệu năng kinh điển khi viết proxy.
    async with httpx.AsyncClient(timeout=settings.request_timeout_s) as client:
        app.state.client = client
        yield


app = FastAPI(
    title="Financial QA - Gateway",
    description="Biên public có TLS. Phân tuyến user-plane / admin-plane.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    # True vì phiên đăng nhập giờ là cookie (2026-08-07) - trình duyệt chỉ
    # gửi kèm cookie trên request cross-origin nếu server khai báo rõ cho
    # phép. Trong vận hành bình thường frontend và API cùng origin
    # (https://localhost:8443 phục vụ cả hai) nên điều này hiếm khi thật
    # sự được dùng tới, nhưng khai đúng để không âm thầm gãy nếu sau này
    # frontend được host ở domain khác.
    allow_credentials=True,
    # Đủ 4 phương thức REST (2026-08-20) - khớp với proxy_write bên dưới.
    # Để thiếu PUT/DELETE ở đây thì trình duyệt chặn ngay từ preflight,
    # trước cả khi request chạm tới gateway.
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Content-Type"],
)


def _forward_headers(request: Request, *, admin: bool = False,
                     session: "SessionInfo | None" = None) -> dict[str, str]:
    headers = {k: v for k, v in request.headers.items() if k.lower() not in _HOP_BY_HOP}
    if admin:
        # Gắn key thật NGAY TẠI ĐÂY, sau khi _require_session() đã xác
        # nhận đây là phiên role=admin hợp lệ - trình duyệt chưa từng và
        # sẽ không bao giờ nhìn thấy settings.admin_api_key.
        headers["X-Admin-Key"] = settings.admin_api_key
    # persona/username lấy TỪ PHIÊN ĐÃ XÁC THỰC (cookie ký HMAC), không
    # phải từ header client gửi lên - header cùng tên đã bị lọc ở
    # _HOP_BY_HOP. Service qa dùng chúng để ghi vào answer_runs.
    if session:
        headers["X-User-Name"] = session.username
        if session.persona:
            headers["X-User-Persona"] = session.persona
    return headers


async def _proxy_sse(client: httpx.AsyncClient, url: str, request: Request,
                     session: "SessionInfo | None" = None) -> StreamingResponse:
    """Chuyển tiếp Server-Sent Events mà KHÔNG gom hết vào bộ nhớ.

    Nếu proxy đọc trọn response rồi mới trả, toàn bộ ý nghĩa của SSE mất
    sạch - người dùng vẫn ngồi nhìn spinner câm rồi nhận tất cả một lúc.
    Vì vậy phải stream từng byte, và tắt buffer của proxy tầng trên
    (X-Accel-Buffering) phòng khi sau này đặt Nginx phía trước.
    """
    async def event_stream() -> AsyncIterable[bytes]:
        async with client.stream(
            "GET", url, params=dict(request.query_params),
            headers=_forward_headers(request, session=session),
        ) as upstream:
            async for chunk in upstream.aiter_raw():
                yield chunk

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# --- Đăng nhập ---------------------------------------------------------

@app.post("/api/auth/login")
def login(payload: LoginRequest, response: FastAPIResponse) -> SessionInfo:
    if not settings.session_secret:
        raise HTTPException(
            status_code=503,
            detail="Đăng nhập chưa được cấu hình (thiếu SESSION_SECRET trong api/.env).",
        )
    result = authenticate(payload.username, payload.password)
    if not result:
        raise HTTPException(status_code=401, detail="Sai tài khoản hoặc mật khẩu.")
    role, persona = result

    token = create_session_token(payload.username, role, settings.session_secret,
                                 settings.session_ttl_seconds, persona=persona)
    response.set_cookie(
        key=COOKIE_NAME, value=token, max_age=settings.session_ttl_seconds,
        path="/", httponly=True, secure=True, samesite="lax",
    )
    return SessionInfo(username=payload.username, role=role, persona=persona)


@app.post("/api/auth/logout")
def logout(response: FastAPIResponse) -> dict[str, bool]:
    response.delete_cookie(COOKIE_NAME, path="/")
    return {"ok": True}


@app.get("/api/auth/me")
def me(request: Request) -> SessionInfo:
    session = _get_session(request)
    if not session:
        raise HTTPException(status_code=401, detail="Chưa đăng nhập.")
    return session


@app.get("/__devreload")
def dev_reload_marker() -> dict:
    """Mốc thời gian sửa đổi mới nhất trong frontend/ - frontend/livereload.js
    poll endpoint này mỗi giây và tự location.reload() khi thấy mốc mới
    hơn lần trước, để sửa index.html/app.js/styles.css KHÔNG cần bấm F5
    tay. Chỉ đọc mtime (stat, không đọc nội dung file) nên rẻ, an toàn để
    bật thường trực trong lúc phát triển - không phải build step, không
    WebSocket, chỉ một con số."""
    if not FRONTEND_DIR.is_dir():
        # Chạy trong Docker với SERVE_FRONTEND=0: thư mục frontend không
        # được mount vào container gateway (nginx giữ nó). Trả 0 thay vì
        # để .glob() ném FileNotFoundError - livereload.js hiểu "mốc không
        # đổi" và im lặng, đúng hành vi mong muốn vì lúc đó nginx phục vụ
        # file tĩnh và người dùng chỉ cần F5.
        return {"mtime": 0}
    mtimes = [f.stat().st_mtime for f in FRONTEND_DIR.glob("*") if f.is_file()]
    return {"mtime": max(mtimes) if mtimes else 0}


@app.get("/health")
async def health() -> list[ServiceHealth]:
    """Sức khỏe tổng hợp - một lời gọi thấy cả 3 service."""
    client: httpx.AsyncClient = app.state.client
    out = [ServiceHealth(service="gateway", status="ok")]
    for name, base in UPSTREAMS.items():
        try:
            resp = await client.get(f"{base}/health", timeout=5.0)
            data = resp.json()
            out.append(ServiceHealth(service=name, status=data.get("status", "ok"),
                                     detail=data.get("detail")))
        except Exception as exc:
            out.append(ServiceHealth(service=name, status="down",
                                     detail=f"{type(exc).__name__}"))
    return out


@app.get("/api/{service}/{path:path}")
async def proxy_get(service: str, path: str, request: Request):
    base = UPSTREAMS.get(service)
    if not base:
        raise HTTPException(status_code=404, detail=f"Không có service '{service}'.")
    session = _require_session(request, service)
    is_admin = service == "admin"

    client: httpx.AsyncClient = app.state.client
    url = f"{base}/{path}"

    # SSE nhận diện theo đường dẫn, không theo header Accept: EventSource
    # của trình duyệt có gửi Accept đúng, nhưng curl/script thì thường
    # không - dựa vào đường dẫn thì hành vi giống nhau ở mọi client.
    if path.endswith("/stream") or path == "ask/stream":
        return await _proxy_sse(client, url, request, session=session)

    resp = await client.get(url, params=dict(request.query_params),
                            headers=_forward_headers(request, admin=is_admin, session=session))
    return Response(content=resp.content, status_code=resp.status_code,
                    media_type=resp.headers.get("content-type"))


# POST/PUT/DELETE dùng CHUNG một handler (2026-08-20). Trước đó chỉ có
# proxy_get và proxy_post, nghĩa là một route PUT/DELETE thêm vào
# qa/feedback/admin sau này sẽ bị gateway trả 405 dù service đích có hỗ
# trợ - lỗi rất khó đoán vì gọi thẳng service thì chạy, qua gateway thì
# không. Giờ gateway phủ đủ 4 phương thức REST.
#
# DELETE vẫn chuyển tiếp body: HTTP cho phép DELETE có body và một số
# route xoá-theo-lô dùng tới. Nếu không có body thì `await request.body()`
# trả b"" - vô hại.
@app.api_route("/api/{service}/{path:path}", methods=["POST", "PUT", "DELETE"])
async def proxy_write(service: str, path: str, request: Request):
    base = UPSTREAMS.get(service)
    if not base:
        raise HTTPException(status_code=404, detail=f"Không có service '{service}'.")
    session = _require_session(request, service)
    is_admin = service == "admin"

    client: httpx.AsyncClient = app.state.client
    resp = await client.request(
        request.method, f"{base}/{path}",
        params=dict(request.query_params),
        content=await request.body(),
        headers=_forward_headers(request, admin=is_admin, session=session),
    )
    return Response(content=resp.content, status_code=resp.status_code,
                    media_type=resp.headers.get("content-type"))


# Frontend phục vụ CUỐI CÙNG: route API khớp trước, phần còn lại rơi về
# SPA (kể cả đường dẫn client-side routing).
#
# 2026-08-20: chuyển thành CÓ ĐIỀU KIỆN. Khi chạy bằng Docker,
# SERVE_FRONTEND=0 và container nginx riêng đảm nhiệm việc này - gateway
# lúc đó là REST API thuần, không đụng gì tới HTML/CSS/JS. Khi chạy kiểu
# cũ (bash api/dev.sh) biến này mặc định True nên hành vi không đổi.
if settings.serve_frontend:
    app.frontend("/", directory=str(FRONTEND_DIR))
