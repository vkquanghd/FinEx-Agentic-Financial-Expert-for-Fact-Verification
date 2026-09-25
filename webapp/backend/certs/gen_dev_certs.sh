#!/usr/bin/env bash
# Self-signed cert cho dev HTTPS (yêu cầu bảo mật SSL của webapp).
# Chạy 1 lần: bash webapp/backend/certs/gen_dev_certs.sh
# Prod sau này: thay bằng cert thật (Let's Encrypt) ở tầng reverse proxy.
set -euo pipefail
cd "$(dirname "$0")"
openssl req -x509 -newkey rsa:2048 -nodes -days 365 \
  -keyout dev-key.pem -out dev-cert.pem \
  -subj "/C=VN/ST=HCM/O=ThesisDemo/CN=localhost" \
  -addext "subjectAltName=DNS:localhost,IP:127.0.0.1"
echo "OK: dev-key.pem + dev-cert.pem (self-signed, localhost, 365 ngày)"
