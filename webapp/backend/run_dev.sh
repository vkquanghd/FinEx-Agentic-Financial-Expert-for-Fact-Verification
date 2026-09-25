#!/usr/bin/env bash
# Chạy cả 3 service ở chế độ dev (HTTP). Chạy TỪ REPO ROOT:
#   bash webapp/backend/run_dev.sh          # HTTP
#   bash webapp/backend/run_dev.sh --ssl    # HTTPS self-signed (chạy gen_dev_certs.sh trước)
set -euo pipefail
cd "$(dirname "$0")/../.."   # -> repo root

SSL_ARGS=""
if [[ "${1:-}" == "--ssl" ]]; then
  C=webapp/backend/certs
  SSL_ARGS="--ssl-keyfile $C/dev-key.pem --ssl-certfile $C/dev-cert.pem"
fi

trap 'kill 0' EXIT
uvicorn webapp.backend.services.qa.main:app       --port "${QA_SERVICE_PORT:-8101}"       $SSL_ARGS &
uvicorn webapp.backend.services.feedback.main:app --port "${FEEDBACK_SERVICE_PORT:-8102}" $SSL_ARGS &
uvicorn webapp.backend.services.admin.main:app    --port "${ADMIN_SERVICE_PORT:-8103}"    $SSL_ARGS &
wait
