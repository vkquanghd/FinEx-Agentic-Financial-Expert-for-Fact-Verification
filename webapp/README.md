> # ⚠️ NGỪNG DÙNG — 2026-08-20
>
> Thư mục này là **bản nháp đầu tiên** của tầng phục vụ (3 service + React/Vite).
> Nó đã được thay bằng `api/` — bản đó có thêm **gateway** (TLS, đăng nhập,
> chặn theo vai trò user/admin), và từ 2026-08-20 chạy trọn trong Docker cùng
> nginx + Ollama.
>
> **Đọc `docs/DOCKER.md` và `api/SETUP.md` thay cho file này.**
>
> Giữ lại chứ không xoá vì repo hiện **chưa đặt dưới git**, nên xoá là mất
> vĩnh viễn — và bản React ở đây có thể còn dùng lại được nếu sau này muốn
> thay frontend Alpine.js. Chỉ tốn 112KB. Khi nào repo có git, xoá lúc đó là
> an toàn.
>
> Không có gì trong `src/`, `api/` hay `docker-compose.yml` trỏ tới thư mục
> này — nó không ảnh hưởng hệ thống đang chạy.

# Webapp — FastAPI microservices + React frontend (2026-08-07)

Giao diện web mới thay Streamlit cho phần demo người dùng. Folder này **độc lập**
với code thesis (`src/` không đổi — backend chỉ *import* nó).

## Kiến trúc

```
┌─────────────────────────────  MẶT NGƯỜI DÙNG (làm ngay)  ────────────────┐
│  frontend (React+Vite, :5173)                                            │
│     │ REST/JSON (CORS)                                                   │
│     ├──> qa-service        :8101  POST /ask   (Router S3 + log HITL L1)  │
│     └──> feedback-service  :8102  POST /rate  (L1)                       │
│                                   GET/POST /admin/review-queue (L2)      │
│                                   GET /admin/hitl-stats        (L3 gate) │
├─────────────────────────────  MẶT VẬN HÀNH (sau thesis)  ────────────────┤
│  admin-service :8103 — STUB chỉ-đọc: /model-registry /retrain-gates      │
│  (sau thesis: auth OAuth2/JWT, MLOps retrain pipeline, warehouse views;  │
│   /admin/* của feedback-service sẽ dời về đây khi có auth)               │
└──────────────────────────────────────────────────────────────────────────┘
        Tầng dữ liệu chung: Postgres (companies, financial_line_items,
        feedback.*) + pgvector (disclosure_chunks) + Ollama (local LLM)
```

Vì sao chia vậy: ranh giới user/admin là ranh giới **deploy** thật (mỗi service
1 process/port, scale & auth riêng được) — chuẩn bị sẵn cho MLOps + warehouse
sau thesis mà không đụng mặt người dùng.

## Setup từng bước

### 0. Yêu cầu sẵn có
- Postgres + pgvector đang chạy (như khi chạy Streamlit), Ollama đã pull model.
- Node.js >= 18 (`node -v`), npm.

### 1. Backend
```bash
cd <repo root Thesis_Project>
source .venv/bin/activate                       # venv sẵn có của repo
pip install -r webapp/backend/requirements.txt  # fastapi, uvicorn, pydantic
```

Chạy dev (HTTP, 3 service cùng lúc):
```bash
bash webapp/backend/run_dev.sh
```
Hoặc từng service riêng:
```bash
uvicorn webapp.backend.services.qa.main:app       --port 8101
uvicorn webapp.backend.services.feedback.main:app --port 8102
uvicorn webapp.backend.services.admin.main:app    --port 8103
```
Kiểm tra: mở http://localhost:8101/docs (Swagger tự sinh), /health từng port.

### 2. SSL (dev)
```bash
bash webapp/backend/certs/gen_dev_certs.sh   # tạo self-signed cert localhost
bash webapp/backend/run_dev.sh --ssl         # 3 service chạy HTTPS
```
Trình duyệt sẽ cảnh báo cert tự ký — chấp nhận cho localhost. Prod sau này:
cert thật (Let's Encrypt) đặt ở reverse proxy (nginx/caddy), không đổi code.
Nếu backend chạy `--ssl`, đổi URL trong frontend qua env:
`VITE_QA_URL=https://localhost:8101 VITE_FEEDBACK_URL=https://localhost:8102 npm run dev`
và bật khối `server.https` trong `frontend/vite.config.js`.

### 3. Frontend
```bash
cd webapp/frontend
npm install
npm run dev        # -> http://localhost:5173
```

### 4. Thử luồng đầy đủ
1. Mở http://localhost:5173 — TopBar có đồng hồ thời gian thực + toggle dark/light
   + đèn trạng thái QA/HITL (xanh = service sống).
2. Hỏi "TTS của VNZ năm 2025?" — thấy pipeline "đang chạy" theo bước, rồi câu
   trả lời kèm badge nhánh (màu theo DAG/RAG/what-if/causal/compare), latency,
   nút xem "vì sao nhánh này" + nguồn.
3. Panel HITL bên phải (tách khỏi chat, đúng bản chất 3 lớp):
   - **L1**: rate 1-5 sao + lý do (hoặc Bỏ qua — skip cũng là data). Chưa
     rate/skip thì ô nhập câu hỏi bị khóa (mỗi câu trả lời phải có tín hiệu).
   - **L2**: hàng đợi review (uncertainty sampling) — rating ≤2 / fallback /
     similarity thấp tự rơi vào đây; adjudicate ngay trên panel.
   - **L3**: tiến độ batch gate (N≥50 & mean<3.5) + thống kê theo nhánh.

## Ghi chú scope
- Admin UI **chưa làm** (đúng thỏa thuận: sau thesis) — admin-service chỉ có
  endpoint chỉ-đọc để chứng minh đường dây; Swagger ở :8103/docs.
- Streamlit (`app_streamlit.py`) giữ nguyên — vẫn là công cụ nội bộ cho các
  tab thí nghiệm (RQ3/RQ4/RQ7, what-if, causal); webapp là mặt demo người dùng.
- Microservices ở mức tối giản có chủ đích (3 process, không message broker/
  service mesh) — đủ để ranh giới đúng, không over-engineer trong thời gian thesis.
