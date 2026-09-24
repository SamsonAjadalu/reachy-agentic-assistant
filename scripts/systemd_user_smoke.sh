#!/usr/bin/env bash
# Install, start, probe, restart, and uninstall user-level units in mock mode.
# Does not use sudo and does not enable lingering. Does not modify the repo .env.
#
#   ./scripts/systemd_user_smoke.sh
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

SMOKE_ROOT="${TMPDIR:-/tmp}/pa-systemd-smoke-$$"
DATA_DIR="${SMOKE_ROOT}/data"
ENV_FILE="${SMOKE_ROOT}/.env"
SECRET_KEY_FILE="${SMOKE_ROOT}/pa_secret_key"
PORT="${PA_SMOKE_PORT:-18080}"
TOKEN="smoke-token-0123456789abcdef0123456789abcdef"
PY="${REPO_ROOT}/.venv/bin/python"
[[ -x "${PY}" ]] || PY="$(command -v python3)"

cleanup() {
  systemctl --user stop reachy-personal-assistant-api.service 2>/dev/null || true
  "${REPO_ROOT}/deployment/main_pc/uninstall-user.sh" >/dev/null 2>&1 || true
  rm -rf "${SMOKE_ROOT}"
}
trap cleanup EXIT

mkdir -p "${DATA_DIR}"
chmod 700 "${DATA_DIR}"

"${PY}" "${REPO_ROOT}/scripts/generate_api_token.py" --secret-key --key-file "${SECRET_KEY_FILE}"

cat > "${ENV_FILE}" <<EOF
APP_ENV=development
MOCK_MODE=true
APP_HOST=127.0.0.1
APP_PORT=${PORT}
APP_DATA_DIR=${DATA_DIR}
PA_API_TOKEN=${TOKEN}
PA_API_ALLOWED_NETWORKS=127.0.0.0/8,::1/128
SCHEDULER_ENABLED=true
WORKER_ENABLED=true
TELEGRAM_ENABLED=false
GOOGLE_ENABLED=false
NOTION_ENABLED=false
APP_LOG_FORMAT=console
APP_LOG_LEVEL=WARNING
EOF

export PA_ENV_FILE="${ENV_FILE}"
export APP_DATA_DIR="${DATA_DIR}"
export APP_ENV=development
export MOCK_MODE=true
export DATABASE_URL="sqlite+aiosqlite:///${DATA_DIR}/assistant.db"
export PA_API_TOKEN="${TOKEN}"
export PA_SECRET_KEY_FILE="${SECRET_KEY_FILE}"
export SCHEDULER_ENABLED=false
export WORKER_ENABLED=false
export TELEGRAM_ENABLED=false

"${PY}" "${REPO_ROOT}/scripts/init_database.py"

echo "Installing user units (non-interactive)..."
ENV_FILE="${ENV_FILE}" APP_DATA_DIR="${DATA_DIR}" SECRET_KEY_FILE="${SECRET_KEY_FILE}" \
  "${REPO_ROOT}/deployment/main_pc/install-user.sh" --yes --no-linger --start

UNIT="${HOME}/.config/systemd/user/reachy-personal-assistant-api.service"
if command -v systemd-analyze >/dev/null 2>&1; then
  echo "Verifying unit syntax..."
  # Some distros warn on user credentials; treat hard errors only.
  if ! systemd-analyze verify "${UNIT}" 2>"${SMOKE_ROOT}/analyze.err"; then
    if grep -qiE 'error|failed' "${SMOKE_ROOT}/analyze.err"; then
      cat "${SMOKE_ROOT}/analyze.err" >&2
      exit 1
    fi
  fi
fi

echo "Waiting for health on port ${PORT}..."
ok=0
for _ in $(seq 1 40); do
  if curl -sfS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    ok=1
    break
  fi
  sleep 0.5
done
if [[ "${ok}" -ne 1 ]]; then
  journalctl --user -u reachy-personal-assistant-api.service -n 40 --no-pager || true
  exit 1
fi

curl -sfS "http://127.0.0.1:${PORT}/health" | "${PY}" -m json.tool
curl -sfS "http://127.0.0.1:${PORT}/ready" | "${PY}" -m json.tool
curl -sfS -H "Authorization: Bearer ${TOKEN}" \
  "http://127.0.0.1:${PORT}/api/v1/ping" | "${PY}" -m json.tool

echo "Restarting API..."
systemctl --user restart reachy-personal-assistant-api.service
ok=0
for _ in $(seq 1 40); do
  if curl -sfS "http://127.0.0.1:${PORT}/ready" >/dev/null 2>&1; then
    ok=1
    break
  fi
  sleep 0.5
done
[[ "${ok}" -eq 1 ]]
curl -sfS "http://127.0.0.1:${PORT}/ready" | "${PY}" -m json.tool

echo "Stopping and uninstalling..."
systemctl --user stop reachy-personal-assistant-api.service || true
"${REPO_ROOT}/deployment/main_pc/uninstall-user.sh"

echo "systemd user smoke passed."
