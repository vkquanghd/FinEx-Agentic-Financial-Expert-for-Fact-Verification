#!/usr/bin/env bash
# Chạy một dịch vụ. Tự về gốc repo trước, nên gọi từ đâu cũng được.
#
#   bash api/run.sh qa | feedback | admin | gateway
#
# Vì sao cần script thay vì gõ uvicorn tay (2 lỗi thật gặp 2026-08-07):
#   1. `uvicorn api.services.qa.main:app` BẮT BUỘC chạy từ gốc repo -
#      uvicorn thêm thư mục hiện tại vào sys.path, đứng trong api/ thì
#      không có package tên `api` để import.
#   2. `--reload` mặc định theo dõi cả thư mục hiện tại, tức là cả
#      api/.venv/ - mỗi lần pip đụng vào site-packages là reload storm.
#      Vì vậy chỉ định rõ --reload-dir cho đúng phần code của mình.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SERVICE="${1:-}"
RELOAD_DIRS=(--reload-dir api/services --reload-dir api/shared --reload-dir src)

case "$SERVICE" in
  qa)       MODULE=api.services.qa.main:app;       PORT=8101 ;;
  feedback) MODULE=api.services.feedback.main:app; PORT=8102 ;;
  admin)    MODULE=api.services.admin.main:app;    PORT=8103 ;;
  gateway)
    CERT="api/certs/dev-cert.pem"
    KEY="api/certs/dev-key.pem"
    if [[ ! -f "$CERT" || ! -f "$KEY" ]]; then
      echo "Chưa có chứng chỉ. Chạy trước: bash api/certs/generate_dev_certs.sh" >&2
      exit 1
    fi
    echo "→ gateway  https://localhost:8443   (Ctrl+C để dừng)"
    exec uvicorn api.services.gateway.main:app \
      --host 0.0.0.0 --port 8443 \
      --ssl-certfile "$CERT" --ssl-keyfile "$KEY" \
      --reload "${RELOAD_DIRS[@]}"
    ;;
  *)
    echo "Dùng: bash api/run.sh {qa|feedback|admin|gateway}" >&2
    exit 1 ;;
esac

echo "→ $SERVICE  http://127.0.0.1:$PORT/docs   (Ctrl+C để dừng)"
exec uvicorn "$MODULE" --host 127.0.0.1 --port "$PORT" --reload "${RELOAD_DIRS[@]}"
