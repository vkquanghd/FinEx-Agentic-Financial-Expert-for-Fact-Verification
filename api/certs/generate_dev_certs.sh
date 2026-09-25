#!/usr/bin/env bash
# Sinh chứng chỉ TLS cho môi trường phát triển.
#
# Ưu tiên `mkcert` nếu có: nó tạo một CA cục bộ và cài thẳng vào Keychain
# của macOS (hoặc kho chứng chỉ hệ điều hành tương ứng), nên Chrome/Safari
# TIN chứng chỉ ngay - không còn cảnh báo "Your connection is not private"
# nữa. Không có mkcert thì fallback về openssl tự ký như trước (vẫn chạy
# đúng, chỉ là trình duyệt sẽ cảnh báo và cần bấm qua bằng tay).
#
# Lên máy chủ thật thì thay bằng chứng chỉ Let's Encrypt (certbot) và trỏ
# SSL_CERTFILE/SSL_KEYFILE sang đó; code không cần đổi dòng nào.
#
# Cách chạy:  bash api/certs/generate_dev_certs.sh
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CERT="$DIR/dev-cert.pem"
KEY="$DIR/dev-key.pem"

if [[ -f "$CERT" && -f "$KEY" ]]; then
  echo "Đã có chứng chỉ tại $DIR - xoá 2 file dev-*.pem nếu muốn tạo lại."
  exit 0
fi

if command -v mkcert >/dev/null 2>&1; then
  echo "Dùng mkcert (chứng chỉ sẽ được trình duyệt TIN NGAY, không cảnh báo)."
  mkcert -install
  mkcert -cert-file "$CERT" -key-file "$KEY" localhost 127.0.0.1 ::1
  chmod 600 "$KEY"
  echo "Đã tạo (mkcert):"
  echo "  $CERT"
  echo "  $KEY  (chmod 600)"
  exit 0
fi

echo "Không thấy mkcert - dùng openssl tự ký (trình duyệt SẼ cảnh báo)."
echo "Cài mkcert để hết cảnh báo:  brew install mkcert nss && mkcert -install"
echo "  rồi chạy lại script này."
echo

# subjectAltName là bắt buộc: trình duyệt hiện đại bỏ qua Common Name.
# Thiếu SAN thì Chrome/Safari từ chối kể cả khi đã bấm "vẫn tiếp tục".
openssl req -x509 -newkey rsa:2048 -nodes -days 365 \
  -keyout "$KEY" -out "$CERT" \
  -subj "/C=VN/ST=HCM/O=RMIT Thesis/CN=localhost" \
  -addext "subjectAltName=DNS:localhost,IP:127.0.0.1"

chmod 600 "$KEY"
echo "Đã tạo (tự ký - trình duyệt sẽ cảnh báo, đây là bình thường):"
echo "  $CERT"
echo "  $KEY  (chmod 600)"
echo
echo "Cách bỏ qua cảnh báo trên Chrome: click vào trang cảnh báo rồi gõ"
echo "  thisisunsafe   (gõ mù, không cần ô nhập hiện ra)"
