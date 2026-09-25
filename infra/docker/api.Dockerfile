# Image dùng chung cho CẢ 4 service FastAPI (gateway/qa/feedback/admin).
#
# Vì sao MỘT image cho cả 4: chúng cùng import `src/*` nên cùng bộ phụ
# thuộc (xem api/requirements.txt) - build 4 image gần giống hệt nhau chỉ
# tốn thêm thời gian và dung lượng. Khác nhau duy nhất là lệnh `command`
# trong docker-compose.yml (module uvicorn + cổng nào).
#
# ĐIỂM QUAN TRỌNG NHẤT - vì sao sửa code KHÔNG phải build lại image:
# image này CỐ Ý không `COPY` thư mục src/ hay api/ vào trong. Code được
# gắn vào lúc chạy bằng bind mount (xem `volumes:` trong
# docker-compose.yml) và uvicorn chạy với `--reload`. Nghĩa là:
#     sửa file .py trên máy  ->  container tự nạp lại trong ~1 giây
#     thêm thư viện mới      ->  lúc đó mới cần `docker compose build`
# Đây chính là lý do api/dev.sh trước đây từ chối dùng Docker ("build lại
# image mỗi lần sửa code sẽ chậm hơn --reload chạy thẳng trên venv") -
# ràng buộc đó biến mất khi code không nằm trong image.
#
# Đánh đổi đã biết: torch + sentence-transformers (cần cho nhánh RAG) làm
# image nặng ~4-6GB. Chỉ build MỘT lần; các lần sau Docker dùng lại layer
# cache miễn api/requirements.txt không đổi. Muốn image nhẹ (~600MB) để
# demo 4 nhánh DAG/whatif/causal/compare mà bỏ RAG, build với:
#     docker compose build --build-arg WITH_RAG=0
FROM python:3.12-slim

ARG WITH_RAG=1

# poppler-utils: pdfinfo/pdftotext - src/retrieval/text_splitter.py và
# scripts/pdf_ingest_router.py gọi qua subprocess, KHÔNG phải gói pip.
# curl: dùng cho HEALTHCHECK bên dưới.
# build-essential + libpq-dev: psycopg2-binary có wheel sẵn cho hầu hết
# nền tảng, nhưng trên linux/arm64 (Docker Desktop trên Mac Apple Silicon)
# đôi khi phải biên dịch - thiếu 2 gói này thì pip lỗi khó hiểu.
RUN apt-get update && apt-get install -y --no-install-recommends \
        poppler-utils curl build-essential libpq-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Chỉ COPY file requirements (KHÔNG copy code) - để layer pip bên dưới
# chỉ mất hiệu lực cache khi danh sách thư viện đổi, không phải mỗi lần
# sửa một dòng Python.
COPY api/requirements.txt /tmp/api-requirements.txt

# Tách 2 nhóm: phần lõi luôn cài, phần RAG (torch ~2.5GB +
# sentence-transformers) chỉ cài khi WITH_RAG=1. grep -v loại 2 dòng cuối
# của api/requirements.txt; nếu ai đó sắp xếp lại file đó thì lệnh này vẫn
# đúng vì lọc theo TÊN GÓI chứ không theo số dòng.
RUN grep -viE '^(torch|sentence-transformers)' /tmp/api-requirements.txt > /tmp/core.txt \
    && pip install --no-cache-dir -r /tmp/core.txt \
    && if [ "$WITH_RAG" = "1" ]; then \
         pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
         && pip install --no-cache-dir 'sentence-transformers>=3.0'; \
       fi

# PYTHONPATH=/app: các service chạy `from src...` và `from api...`.
# ensure_repo_on_path() trong api/shared/config.py cũng tự thêm, nhưng đặt
# sẵn ở đây để `python -c "import src"` trong container cũng chạy được khi
# debug bằng tay.
# PYTHONUNBUFFERED: không có nó, log uvicorn bị đệm và `docker compose
# logs -f` trông như service bị treo.
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/opt/hf-cache

# Trọng số BAAI/bge-m3 (~2.2GB) tải về lần đầu chạy nhánh RAG. Đặt ngoài
# /app (nơi bind-mount code) và trỏ vào một named volume trong compose -
# CÙNG lý do đã ghi cho airflow: /app map tới thư mục project nằm dưới
# ~/Desktop mà iCloud đồng bộ, ghi 2.2GB vào đó là tự chuốc chậm.
RUN mkdir -p /opt/hf-cache

# Không có CMD: mỗi service tự đặt `command:` riêng trong
# docker-compose.yml (cùng image, khác module + cổng).
