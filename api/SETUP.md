# Tầng phục vụ (`api/`) — hướng dẫn cài đặt tay

Bốn dịch vụ FastAPI độc lập + một frontend không cần build. Chạy tay từng
bước, mỗi bước có cách kiểm chứng trước khi sang bước sau.

```
                          ┌─────────────────────────┐
   trình duyệt ──TLS───►  │  gateway  :8443         │  ← cổng DUY NHẤT ra ngoài
                          │  TLS · phân tuyến · SPA │
                          └────┬──────┬──────┬──────┘
                   /api/qa/*   │      │      │   /api/admin/*
                               ▼      ▼      ▼
                          ┌────────┐┌────────┐┌────────┐
                          │qa :8101││fb :8102││ad :8103│  ← chỉ bind 127.0.0.1
                          └────────┘└────────┘└────────┘
                            user-plane          admin-plane
                                 │                  │
                                 └──── src/ ────────┘   (code luận văn, dùng chung)
```

**Vì sao chia như vậy**

| Ranh giới | Lý do |
|---|---|
| `qa` tách `feedback` | Hai vòng đời khác nhau: qa phục vụ một câu rồi quên; feedback tích lũy trạng thái qua thời gian và là thứ MLOps sau này sẽ đọc. Tách từ đầu để lúc dựng warehouse không phải mổ qa. |
| `admin` là tiến trình riêng | Phân tách theo tiến trình, không chỉ theo URL. Kể cả khi gateway cấu hình sai, admin vẫn nằm sau `X-Admin-Key` và bind `127.0.0.1`. |
| `gateway` không có logic nghiệp vụ | Một chỗ duy nhất quản TLS, CORS, rate limit sau này. |
| `api/` không copy code từ `src/` | `src/` vẫn là nguồn sự thật duy nhất cho router/RAG/ETL. `api/` chỉ là lớp vận chuyển. |

---

## Bước 1 — Kiểm tra thứ đã có sẵn

```bash
cd ~/Desktop/Project/Thesis_Project
python3 --version          # cần >= 3.11
psql -V                    # Postgres client
openssl version
```

Postgres + pgvector phải đang chạy và `.env` **ở gốc repo** đã trỏ đúng DB
(file đó là nguồn cấu hình DB duy nhất — `api/.env` không lặp lại).

---

## Bước 2 — Môi trường ảo riêng cho `api/`

Tách hẳn khỏi venv ở gốc repo (vốn nặng: torch, streamlit, jupyter).

```bash
cd ~/Desktop/Project/Thesis_Project/api
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

Kiểm chứng:

```bash
python -c "import fastapi; from fastapi.sse import EventSourceResponse; print(fastapi.__version__)"
```

Nếu báo lỗi `No module named 'fastapi.sse'` thì bản FastAPI quá cũ:
`pip install --upgrade 'fastapi>=0.121.0'`.

> **`requirements.txt` đã gồm cả `torch` + `sentence-transformers`** để
> nhánh RAG chạy được ngay (chúng chỉ load lazy - lúc câu hỏi thực sự rơi
> vào nhánh RAG - nên không làm chậm lúc khởi động service, nhưng
> `pip install` vẫn tải chúng về máy, khá nặng). Lần đầu nhánh RAG chạy sẽ
> tự tải thêm ~2.2GB trọng số model BAAI/bge-m3 từ HuggingFace - cần mạng
> thật. Nếu muốn bỏ qua để cài nhanh hơn: xoá 2 dòng đó khỏi
> `api/requirements.txt` trước khi `pip install` - nhánh DAG/what-if/
> causal/compare vẫn chạy đủ, riêng RAG sẽ báo lỗi thiếu module khi được gọi.

---

## Bước 3 — Cấu hình

```bash
cp .env.example .env
python3 -c "import secrets; print(secrets.token_urlsafe(32))"   # -> dán vào ADMIN_API_KEY
python3 -c "import secrets; print(secrets.token_urlsafe(32))"   # -> dán vào SESSION_SECRET (khác giá trị)
```

Dán 2 chuỗi vừa in vào `ADMIN_API_KEY=` và `SESSION_SECRET=` trong
`api/.env` (hai khoá khác nhau, đừng dùng chung một chuỗi).

Cả hai đều **bắt buộc**, để trống thì:
- `ADMIN_API_KEY` trống → admin service trả 503 cho mọi route.
- `SESSION_SECRET` trống → `/api/auth/login` trả 503 (không ai đăng nhập được).

Fail-closed có chủ ý cho cả hai — không bao giờ tồn tại một bản chạy có
admin mở toang hoặc phiên đăng nhập ký bằng khoá rỗng vì quên đặt biến.

> `SESSION_SECRET` nên đặt **cố định** trong `.env`, không để trống rồi
> service tự sinh ngẫu nhiên lúc khởi động — gateway chạy `--reload`, một
> khoá đổi mỗi lần nạp lại code sẽ đăng xuất bạn liên tục lúc đang code.

---

## Bước 4 — Chứng chỉ TLS cho môi trường dev

**Cách hết cảnh báo trình duyệt hoàn toàn (khuyến nghị):**

```bash
brew install mkcert nss
bash certs/generate_dev_certs.sh
```

`mkcert` tạo một CA cục bộ và cài vào Keychain của macOS — Chrome/Safari
**tin ngay**, không còn "Your connection is not private" nữa.

**Không cài mkcert cũng chạy được**, script tự lùi về `openssl` tự ký:

```bash
bash certs/generate_dev_certs.sh
```

Sinh `dev-cert.pem` + `dev-key.pem` (365 ngày, có `subjectAltName` — thiếu
SAN thì Chrome/Safari từ chối kể cả khi bấm "vẫn tiếp tục"). Trình duyệt
**sẽ** cảnh báo "không tin cậy" — đúng, vì không có CA nào ký. Ở Bước 7 có
cách bỏ qua cảnh báo này bằng tay.

Lên máy chủ thật thì thay bằng chứng chỉ Let's Encrypt và trỏ
`SSL_CERTFILE` / `SSL_KEYFILE` sang đó; **không phải sửa dòng code nào**.

---

## Bước 5 — Chạy 4 dịch vụ

**Cách A — một lệnh, một terminal (khuyến nghị dùng hàng ngày):**

```bash
source ~/Desktop/Project/Thesis_Project/api/.venv/bin/activate
bash ~/Desktop/Project/Thesis_Project/api/dev.sh
```

Gộp log cả 4 service vào một chỗ, mỗi dòng có tiền tố màu để phân biệt
(`qa`/`feedback`/`admin`/`gateway`). **Ctrl+C một lần** tắt cả 4 — script
tự dọn cả tiến trình con của `--reload`, không để sót tiến trình chạy nền.

**Cách B — 4 terminal riêng (dùng khi cần soi kỹ log một service):**

Mỗi cửa sổ chạy **đúng 2 dòng** này (thay `qa` bằng tên dịch vụ):

```bash
source ~/Desktop/Project/Thesis_Project/api/.venv/bin/activate
bash ~/Desktop/Project/Thesis_Project/api/run.sh qa
```

| Terminal | Lệnh | Cổng |
|---|---|---|
| 1 | `bash api/run.sh qa` | 8101 |
| 2 | `bash api/run.sh feedback` | 8102 |
| 3 | `bash api/run.sh admin` | 8103 |
| 4 | `bash api/run.sh gateway` | 8443 (TLS) |

`run.sh`/`dev.sh` đều tự `cd` về gốc repo và giới hạn phạm vi `--reload`
— gọi từ thư mục nào cũng chạy đúng.

> **Vì sao không dùng Docker ở bước này:** `torch`/`sentence-transformers`
> nặng vài trăm MB-GB — build lại image mỗi lần sửa code sẽ chậm hơn nhiều
> so với `--reload` chạy thẳng trên venv, và Postgres/pgvector đang chạy
> sẵn trên máy host nên container còn cần cấu hình network riêng để gọi
> ra ngoài. Docker Compose hợp lý hơn ở giai đoạn triển khai thật (sau khi
> bảo vệ xong thesis), không phải lúc còn sửa code hàng ngày — có thể quay
> lại việc này khi tới bước "host máy chủ".

> **Hai lỗi mà script này tồn tại để chặn**
>
> - `ModuleNotFoundError: No module named 'api'` — uvicorn thêm *thư mục
>   hiện tại* vào `sys.path`; đứng trong `api/` thì không có package tên
>   `api` để import. Phải chạy từ gốc repo.
> - **Reload lặp vô tận** ngay sau khi `pip install` — `--reload` mặc định
>   theo dõi cả `api/.venv/`, nên mọi thay đổi trong `site-packages` đều
>   kích hoạt khởi động lại. Script chỉ theo dõi `api/services`,
>   `api/shared`, `src`.
>
> Nếu vẫn muốn gõ tay, đây là lệnh đầy đủ:
> ```bash
> cd ~/Desktop/Project/Thesis_Project     # bắt buộc: gốc repo
> uvicorn api.services.qa.main:app --host 127.0.0.1 --port 8101 \
>   --reload --reload-dir api/services --reload-dir api/shared --reload-dir src
> ```

Ba service đầu bind `127.0.0.1` — không chạm được từ máy khác. Chỉ gateway
mở ra ngoài.

---

## Bước 6 — Kiểm chứng

```bash
# từng service (-k: bỏ qua cảnh báo chứng chỉ tự ký)
curl     http://127.0.0.1:8101/health
curl     http://127.0.0.1:8102/health
curl     http://127.0.0.1:8103/health

# tổng hợp qua gateway — một lời gọi thấy cả 4
curl -sk https://localhost:8443/health | python3 -m json.tool
```

Mong đợi: bốn dòng `"status": "ok"`. Nếu `feedback` báo `degraded` thì
schema `feedback` chưa có — chạy
`psql <db> -f infra/postgres/migrations/013_feedback_hitl.sql`.

**Đăng nhập trước khi gọi bất kỳ `/api/qa`, `/api/feedback`, `/api/admin`
nào** (2026-08-07) — gateway giờ đòi phiên đăng nhập hợp lệ, dùng cookie
jar để `curl` giữ cookie qua nhiều lệnh:

```bash
# đăng nhập role=user, lưu cookie vào user.jar
curl -sk -c /tmp/user.jar -X POST https://localhost:8443/api/auth/login \
  -H 'Content-Type: application/json' -d '{"username":"user","password":"user"}'

# hỏi một câu, DÙNG LẠI cookie vừa lưu (-b)
curl -sk -b /tmp/user.jar -X POST https://localhost:8443/api/qa/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"TTS của VNZ năm 2025?","session_id":"test"}' \
  | python3 -m json.tool
```

Thử ranh giới vai trò (đúng thứ tự tách biệt user/admin):

```bash
# chưa đăng nhập -> 401
curl -sk https://localhost:8443/api/admin/admin/coverage

# đăng nhập role=user rồi gọi /api/admin -> 403 (đúng, "riêng biệt")
curl -sk -b /tmp/user.jar https://localhost:8443/api/admin/admin/coverage

# đăng nhập role=admin, lưu cookie riêng -> gọi /api/admin -> 200
curl -sk -c /tmp/admin.jar -X POST https://localhost:8443/api/auth/login \
  -H 'Content-Type: application/json' -d '{"username":"admin","password":"admin"}'
curl -sk -b /tmp/admin.jar https://localhost:8443/api/admin/admin/coverage \
  | python3 -m json.tool

# cookie admin lại KHÔNG gọi được /api/qa -> 403 (tách biệt cả hai chiều)
curl -sk -b /tmp/admin.jar https://localhost:8443/api/qa/companies
```

`ADMIN_API_KEY` trong `.env` giờ là bí mật NỘI BỘ giữa gateway và admin
service — không bao giờ cần gõ tay vào `curl`/trình duyệt nữa, gateway tự
gắn nó sau khi xác nhận cookie admin hợp lệ.

---

## Bước 7 — Mở giao diện

```
https://localhost:8443
```

Sẽ thấy màn hình **đăng nhập** trước (2026-08-07 — user/admin tách biệt).
Tài khoản demo, in luôn trên màn hình đăng nhập cho tiện lúc bảo vệ thesis:

| Tài khoản | Mật khẩu | Vào giao diện nào |
|---|---|---|
| `user` | `user` | Chat (hỏi đáp + đánh giá HITL) |
| `admin` | `admin` | Dashboard admin (kiểm tra database, model registry) |

Mật khẩu = tên tài khoản — **có chủ ý yếu**, đây là demo luận văn chạy nội
bộ một người dùng, không phải hệ thống nhiều người dùng thật. Đổi tài
khoản/mật khẩu: sửa `ACCOUNTS` trong `api/services/gateway/auth.py`.

Nếu dùng `mkcert` ở Bước 4 thì vào thẳng, không cảnh báo gì.

Nếu dùng chứng chỉ tự ký (openssl), trình duyệt sẽ chặn lại
(`net::ERR_CERT_AUTHORITY_INVALID` / "Your connection is not private") —
đây là hành vi đúng, không phải lỗi. Bỏ qua:

- **Chrome**: nếu thấy nút *Advanced/Nâng cao* → bấm rồi chọn
  *Proceed to localhost (unsafe)*. Nếu KHÔNG thấy nút đó (một số bản Chrome
  ẩn nó): click vào bất kỳ chỗ trống nào trên trang cảnh báo, rồi gõ thẳng
  `thisisunsafe` (gõ mù, không có ô nhập hiện ra) — Chrome sẽ vào thẳng.
- **Safari**: *Show Details → visit this website*.
- **Firefox**: *Advanced → Accept the Risk and Continue*.

Tài liệu API tự sinh:

| | |
|---|---|
| QA | http://127.0.0.1:8101/docs |
| Feedback | http://127.0.0.1:8102/docs |
| Admin | http://127.0.0.1:8103/docs |

---

## Xử lý sự cố

| Hiện tượng | Nguyên nhân thường gặp |
|---|---|
| `ModuleNotFoundError: api` | Đang chạy uvicorn từ `api/`. Dùng `bash api/run.sh <tên>` — nó tự về gốc repo. |
| `ModuleNotFoundError: src` | Như trên — `ensure_repo_on_path()` tính đường dẫn từ vị trí file, nhưng import `api.*` cần cwd ở gốc repo. |
| Reload lặp vô tận, log tràn `WatchFiles detected changes in '.venv/...'` | `--reload` đang theo dõi `api/.venv/`. Dùng `run.sh`, hoặc thêm `--reload-dir` như ở Bước 5. |
| SSE không chảy, đợi rồi hiện một lúc | Có proxy đệm ở giữa. Gateway đã gửi `X-Accel-Buffering: no`; nếu đặt Nginx phía trước thì thêm `proxy_buffering off`. |
| Frontend trắng trang | Mở DevTools → Console. Thường do CDN bị chặn mạng (Alpine/Chart/Lucide tải từ jsDelivr + unpkg). |
| `/api/admin/*` trả 503 | `ADMIN_API_KEY` còn trống trong `api/.env`. |
| Nhánh RAG lỗi import | Chưa cài `sentence-transformers` trong venv `api/` — xem ghi chú ở Bước 2. |
| `/api/auth/login` trả 503 | `SESSION_SECRET` còn trống trong `api/.env` — xem Bước 3. |
| Đăng nhập xong bị đăng xuất ngay, hoặc đăng xuất liên tục lúc đang code | `SESSION_SECRET` để trống nên mỗi lần gateway `--reload` lại đổi khoá ngẫu nhiên. Đặt cố định trong `.env`. |
| Đăng nhập `user` xong gọi `/api/admin/*` bị 403 | Đúng thiết kế — 2 vai trò tách biệt tuyệt đối. Đăng xuất, đăng nhập lại bằng `admin`/`admin`. |

---

## Còn thiếu gì trước khi lên máy chủ thật

Ghi ra đây để không nhầm mức độ sẵn sàng của bản hiện tại:

1. **Xác thực có danh tính thật** — hiện chỉ 2 tài khoản cứng trong code
   (`user`/`user`, `admin`/`admin`), không có bảng người dùng, không hash
   mật khẩu (không cần - "mật khẩu" chỉ là gõ lại tên tài khoản). Lên
   nhiều người dùng thật thì cần bảng user + mật khẩu hash (bcrypt/argon2)
   + có thể OAuth2 - route đã sẵn sàng vì phụ thuộc vào cookie phiên chứ
   không phụ thuộc cách xác thực cụ thể.
2. **Bảng kiểm toán** — chưa ghi lại ai làm gì lúc nào ở mặt phẳng admin.
3. **Rate limit** — gateway chưa giới hạn tần suất.
4. **Chứng chỉ thật** — thay chứng chỉ tự ký bằng Let's Encrypt.
5. **Quản lý tiến trình** — hiện chạy tay 4 terminal. Lên máy chủ nên dùng
   systemd hoặc Docker Compose.

Các endpoint **ghi** ở mặt phẳng admin (phong champion model, kích hoạt
retrain, chạy lại ETL) đã để sẵn khung trong `api/services/admin/main.py`
nhưng **chưa mở**, đúng vì thiếu ba mục đầu ở trên.
