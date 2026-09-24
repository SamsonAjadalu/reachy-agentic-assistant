#!/usr/bin/env bash
# Start the API in mock mode (if not already running), seed demo data, and walk
# through major HTTP workflows.
#
#   ./scripts/run_mock_demo.sh
#
# Requires curl and a Python environment with project dependencies installed.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if command -v uv >/dev/null 2>&1 && [[ -f pyproject.toml ]]; then
  PYTHON="uv run python"
  UVICORN="uv run uvicorn"
else
  PYTHON="${PYTHON:-python3}"
  UVICORN="${UVICORN:-python3 -m uvicorn}"
fi

HOST="${APP_HOST:-127.0.0.1}"
PORT="${APP_PORT:-8080}"
BASE="http://${HOST}:${PORT}"
DEMO_TOKEN="${PA_API_TOKEN:-demo-mock-token-0123456789abcdef0123456789ab}"
DATA_DIR="${APP_DATA_DIR:-${HOME}/.local/share/reachy-personal-assistant-demo}"
PIDFILE="${DATA_DIR}/mock-api.pid"
LOGFILE="${DATA_DIR}/mock-api.log"

mkdir -p "${DATA_DIR}"

export APP_ENV=development
export MOCK_MODE=true
export APP_DATA_DIR="${DATA_DIR}"
export PA_API_TOKEN="${DEMO_TOKEN}"
export SCHEDULER_ENABLED=false
export WORKER_ENABLED=false
export TELEGRAM_ENABLED=false
export APP_LOG_FORMAT=console

# Fernet key outside APP_DATA_DIR (mirrors production layout).
SECRET_DIR="${DATA_DIR%/*}/reachy-demo-secrets"
SECRET_FILE="${SECRET_DIR}/pa_secret_key"
if [[ ! -f "${SECRET_FILE}" ]]; then
  mkdir -p "${SECRET_DIR}"
  chmod 700 "${SECRET_DIR}"
  ${PYTHON} scripts/generate_api_token.py --secret-key --key-file "${SECRET_FILE}" >/dev/null
fi
export PA_SECRET_KEY_FILE="${SECRET_FILE}"
unset PA_SECRET_KEY

started_api=false

cleanup() {
  if [[ "${started_api}" == true && -f "${PIDFILE}" ]]; then
    kill "$(cat "${PIDFILE}")" 2>/dev/null || true
    rm -f "${PIDFILE}"
  fi
}
trap cleanup EXIT

if curl -sf "${BASE}/health" >/dev/null 2>&1; then
  echo "API already running at ${BASE}"
else
  echo "Starting mock API on ${BASE} ..."
  ${PYTHON} scripts/init_database.py >/dev/null
  nohup ${UVICORN} app.main:app --host "${HOST}" --port "${PORT}" >"${LOGFILE}" 2>&1 &
  echo $! >"${PIDFILE}"
  started_api=true
  for _ in $(seq 1 30); do
    if curl -sf "${BASE}/health" >/dev/null 2>&1; then
      break
    fi
    sleep 0.5
  done
  if ! curl -sf "${BASE}/health" >/dev/null 2>&1; then
    echo "API failed to start. See ${LOGFILE}" >&2
    exit 1
  fi
fi

echo
echo "=== Seeding demo data ==="
${PYTHON} scripts/seed_demo_data.py --reset-first

auth=(-H "Authorization: Bearer ${DEMO_TOKEN}")

section() {
  echo
  echo "=== $1 ==="
}

section "Health and readiness (unauthenticated)"
curl -s "${BASE}/health" | ${PYTHON} -m json.tool
curl -s "${BASE}/ready" | ${PYTHON} -m json.tool

section "Authenticated ping"
curl -s "${auth[@]}" "${BASE}/api/v1/ping" | ${PYTHON} -m json.tool

section "Integration inventory"
curl -s "${auth[@]}" "${BASE}/api/v1/integrations" | ${PYTHON} -m json.tool

section "Tasks"
curl -s "${auth[@]}" "${BASE}/api/v1/tasks" | ${PYTHON} -m json.tool

section "Reminders"
curl -s "${auth[@]}" "${BASE}/api/v1/reminders" | ${PYTHON} -m json.tool

section "Weather (mock provider)"
curl -s "${auth[@]}" "${BASE}/api/v1/weather/current" | ${PYTHON} -m json.tool

section "Briefing preview (assemble only, no send)"
curl -s "${auth[@]}" "${BASE}/api/v1/briefings/today" | ${PYTHON} -m json.tool

section "Wardrobe recommendation"
curl -s "${auth[@]}" "${BASE}/api/v1/wardrobe/recommend" | ${PYTHON} -m json.tool

section "Service status"
curl -s "${auth[@]}" "${BASE}/api/v1/status" | ${PYTHON} -m json.tool

section "Calendar propose (dry-run)"
curl -s "${auth[@]}" -X POST "${BASE}/api/v1/calendar/events/propose" \
  -H "Content-Type: application/json" \
  -d '{"title":"Demo coffee","duration_minutes":30}' | ${PYTHON} -m json.tool

section "Gmail attachments + archive (mock)"
curl -s "${auth[@]}" "${BASE}/api/v1/gmail/messages/msg-004/attachments" | ${PYTHON} -m json.tool
curl -s "${auth[@]}" -X POST "${BASE}/api/v1/gmail/messages/msg-002/actions/archive" | ${PYTHON} -m json.tool

echo
echo "Mock demo complete."
echo "  API base:    ${BASE}"
echo "  Data dir:    ${DATA_DIR}"
echo "  Bearer token: ${DEMO_TOKEN}"
if [[ "${started_api}" == true ]]; then
  echo "  Log file:    ${LOGFILE}"
  echo
  echo "The API will stop when this script exits."
fi
