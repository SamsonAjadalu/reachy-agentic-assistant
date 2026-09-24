#!/usr/bin/env bash
# Start the personal-assistant API.
#
#   ./scripts/run.sh
#
# Uses .venv through `uv run` (no manual activation). The app loads `.env`
# itself. Defaults below apply only when those variables are unset.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if [[ ! -x .venv/bin/python ]]; then
  echo "error: .venv missing. Run ./scripts/install.sh first." >&2
  exit 1
fi

DEFAULT_DATA_DIR="${HOME}/.local/share/reachy-personal-assistant"
export APP_ENV="${APP_ENV:-development}"
export MOCK_MODE="${MOCK_MODE:-true}"
export APP_HOST="${APP_HOST:-127.0.0.1}"
export APP_PORT="${APP_PORT:-8080}"
export APP_DATA_DIR="${APP_DATA_DIR:-$DEFAULT_DATA_DIR}"
APP_DATA_DIR="${APP_DATA_DIR/#\~/$HOME}"
export APP_DATA_DIR
export APP_LOG_FORMAT="${APP_LOG_FORMAT:-console}"

# Placeholder tokens are fine in mock mode; never print their values.
if [[ -z "${PA_API_TOKEN:-}" ]]; then
  export PA_API_TOKEN="demo-mock-token-0123456789abcdef0123456789ab"
fi

mkdir -p "${APP_DATA_DIR}"

# Ensure schema exists (idempotent). Quiet on success.
uv run python scripts/init_database.py >/dev/null

echo "Starting API on http://${APP_HOST}:${APP_PORT} (MOCK_MODE=${MOCK_MODE})"
echo "OpenAPI docs: http://${APP_HOST}:${APP_PORT}/docs"
exec uv run uvicorn app.main:app --host "${APP_HOST}" --port "${APP_PORT}"
