#!/bin/bash
# Double-click file này để chạy đồng bộ Vietstock Premium NGAY LẬP TỨC,
# không cần đợi tới 18:20. Chạy đúng 2 lệnh mà launchd job hằng ngày chạy,
# theo đúng thứ tự:
#   1) sync thật (Playwright, dùng phiên đăng nhập Premium đã lưu sẵn)
#   2) in báo cáo tóm tắt tiếng Việt ngay sau khi xong
#      (scripts/vietstock_daily_sync_report.py)
#
# An toàn chạy nhiều lần trong ngày - đây chỉ là bản chạy THỦ CÔNG của
# đúng lệnh trong launchd job (xem scripts/install_vietstock_daily_sync.py),
# không đụng tới lịch 18:20 tự động đang chạy song song.
#
# Lưu ý tần suất: dữ liệu GIÁ cổ phiếu (company_price_snapshot) chỉ có giá
# trị mới 1 lần/ngày sau khi HOSE/HNX đóng cửa - chạy nút này nhiều lần
# trong cùng một ngày sẽ tải lại đúng số cũ cho phần giá, không có gì mới.
# Phần tài liệu/tin tức (document/news) có thể có cập nhật rải rác hơn nên
# chạy lại vẫn có ích cho phần đó.

set -e
cd "$(dirname "$0")"

PYTHON="$(dirname "$0")/.venv/bin/python"
if [ ! -x "$PYTHON" ]; then
  echo "Không tìm thấy .venv/bin/python trong $(pwd) - kiểm tra lại đường dẫn project."
  read -p "Nhấn Enter để đóng cửa sổ này..."
  exit 1
fi

echo "=== Đang đồng bộ Vietstock Premium (có thể mất vài phút cho ~200 mã)... ==="
"$PYTHON" -m scripts.vietstock_premium_sync sync --all-known --documents

echo ""
echo "=== Xong đồng bộ. Đang in báo cáo tóm tắt... ==="
"$PYTHON" -m scripts.vietstock_daily_sync_report

echo ""
read -p "Nhấn Enter để đóng cửa sổ này..."
