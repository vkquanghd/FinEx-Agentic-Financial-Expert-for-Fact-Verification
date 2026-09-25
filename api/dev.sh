#!/usr/bin/env bash
# Chạy CẢ 4 SERVICE trong MỘT terminal, một lệnh duy nhất:
#
#   source api/.venv/bin/activate
#   bash api/dev.sh
#
# Log của 4 service gộp chung, mỗi dòng có tiền tố màu để phân biệt. Ctrl+C
# MỘT LẦN tắt hết cả 4 - không cần đóng từng terminal.
#
# Vẫn giữ cách chạy 4 terminal riêng (xem SETUP.md Bước 5) vì tách log
# từng service dễ đọc hơn khi debug sâu một service cụ thể - dùng script
# này cho ngày thường, dùng 4 terminal khi cần soi kỹ một service.
#
# Không dùng Docker (có chủ ý, xem ghi chú cuối file): torch/sentence-
# transformers nặng vài trăm MB->GB, build lại image mỗi lần sửa code sẽ
# chậm hơn nhiều so với --reload chạy thẳng trên venv. Docker Compose hợp
# lý hơn ở giai đoạn deploy thật (sau khi bảo vệ xong thesis), không phải
# lúc còn đang sửa code hàng ngày.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
set -m  # monitor mode: mỗi service chạy nền được cấp process group riêng,
        # nhờ vậy Ctrl+C dọn được luôn tiến trình con của --reload (uvicorn
        # spawns một reloader process con) chứ không sót lại tiến trình mồ côi.

# --- kiểm tra trước khi chạy, báo lỗi rõ ràng thay vì để uvicorn tự sập ---
if ! python -c "import fastapi" 2>/dev/null; then
  echo "Chưa activate venv của api/ (hoặc chưa cài requirements)." >&2
  echo "  source api/.venv/bin/activate" >&2
  exit 1
fi

if [[ ! -f api/.env ]]; then
  echo "Chưa có api/.env - copy từ mẫu:" >&2
  echo "  cp api/.env.example api/.env" >&2
  exit 1
fi

if [[ ! -f api/certs/dev-cert.pem || ! -f api/certs/dev-key.pem ]]; then
  echo "Chưa có chứng chỉ TLS - đang tạo tự động..."
  bash api/certs/generate_dev_certs.sh
fi

RELOAD_DIRS=(--reload-dir api/services --reload-dir api/shared --reload-dir src)
PIDS=()
NAMES=()

# In một dòng log với tiền tố có màu + căn cột, đọc từ stdin theo dòng -
# giữ luồng log chảy real-time (không đợi cả service kết thúc mới in).
prefix() {
  local name="$1" color="$2"
  while IFS= read -r line; do
    printf "\033[%sm[%-8s]\033[0m %s\n" "$color" "$name" "$line"
  done
}

start() {
  local name="$1" color="$2"; shift 2
  # `> >(...)`: process substitution, KHÔNG bọc lệnh chính trong subshell -
  # $! ngay sau đó là PID THẬT của uvicorn, không phải PID của pipe. Quan
  # trọng để kill đúng tiến trình lúc dọn dẹp.
  "$@" > >(prefix "$name" "$color") 2>&1 &
  PIDS+=("$!")
  NAMES+=("$name")
}

cleanup() {
  echo
  echo "Đang tắt tất cả service..."
  for i in "${!PIDS[@]}"; do
    # PID âm = kill cả process group (bắt được cả tiến trình reloader con
    # của --reload, không chỉ tiến trình cha).
    kill -TERM "-${PIDS[$i]}" 2>/dev/null || kill -TERM "${PIDS[$i]}" 2>/dev/null || true
  done
  sleep 1
  for i in "${!PIDS[@]}"; do
    kill -KILL "-${PIDS[$i]}" 2>/dev/null || kill -KILL "${PIDS[$i]}" 2>/dev/null || true
  done
  echo "Đã tắt."
}
trap cleanup EXIT INT TERM

echo "Khởi động 4 service..."
echo

start "qa"       36 uvicorn api.services.qa.main:app       --host 127.0.0.1 --port 8101 --reload "${RELOAD_DIRS[@]}"
start "feedback" 35 uvicorn api.services.feedback.main:app --host 127.0.0.1 --port 8102 --reload "${RELOAD_DIRS[@]}"
start "admin"    33 uvicorn api.services.admin.main:app    --host 127.0.0.1 --port 8103 --reload "${RELOAD_DIRS[@]}"

# Gateway chờ 3 service kia bắt đầu nạp trước một nhịp - không bắt buộc
# (gateway tự proxy lỗi 502 nếu upstream chưa sẵn sàng), chỉ để log gọn hơn
# lúc khởi động thay vì xen kẽ hỗn loạn ngay giây đầu.
sleep 1
start "gateway"  32 uvicorn api.services.gateway.main:app --host 0.0.0.0 --port 8443 \
  --ssl-certfile api/certs/dev-cert.pem --ssl-keyfile api/certs/dev-key.pem \
  --reload "${RELOAD_DIRS[@]}"

sleep 2
echo
echo "───────────────────────────────────────────────────────────"
echo "  Giao diện:     https://localhost:8443"
echo "  QA docs:       http://127.0.0.1:8101/docs"
echo "  Feedback docs: http://127.0.0.1:8102/docs"
echo "  Admin docs:    http://127.0.0.1:8103/docs"
echo "  Ctrl+C MỘT LẦN để tắt cả 4 service."
echo "───────────────────────────────────────────────────────────"
echo

wait
