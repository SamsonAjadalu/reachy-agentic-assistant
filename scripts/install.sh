#!/usr/bin/env bash
# One-time setup for Reachy Agentic Assistant.
#
#   ./scripts/install.sh
#
# Checks Python 3.12 and uv, creates .venv, installs dependencies, creates
# .env from .env.example when missing, and applies database migrations.
# Preserves an existing .env and keeps secrets out of output.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

DEFAULT_DATA_DIR="${HOME}/.local/share/reachy-personal-assistant"

die() { echo "error: $*" >&2; exit 1; }

echo "==> Checking uv"
if ! command -v uv >/dev/null 2>&1; then
  cat >&2 <<'EOF'
error: uv is not installed.

Install uv, then re-run ./scripts/install.sh:
  curl -LsSf https://astral.sh/uv/install.sh | sh

Documentation: https://docs.astral.sh/uv/getting-started/installation/
EOF
  exit 1
fi
echo "    found $(uv --version)"

echo "==> Checking Python 3.12"
if command -v python3.12 >/dev/null 2>&1; then
  PY312="$(command -v python3.12)"
elif uv python find 3.12 >/dev/null 2>&1; then
  PY312="$(uv python find 3.12)"
elif command -v python3 >/dev/null 2>&1 && python3 -c 'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)'; then
  PY312="$(command -v python3)"
else
  die "Python 3.12 is required. Install it (uv can fetch it with: uv python install 3.12), then re-run ./scripts/install.sh"
fi
"${PY312}" -c 'import sys; print(f"    found Python {sys.version.split()[0]}")'

echo "==> Creating virtual environment (.venv)"
if [[ -x .venv/bin/python ]]; then
  echo "    .venv already present — leaving it in place"
else
  uv venv --python 3.12 .venv
fi

echo "==> Installing project and development dependencies"
# Editable install keeps the committed lockfile unchanged; `uv sync` may rewrite it.
uv pip install -e '.[dev]'

if [[ ! -f .env ]]; then
  echo "==> Creating .env from .env.example (not overwriting later)"
  [[ -f .env.example ]] || die ".env.example is missing"
  cp .env.example .env
  # Safe local defaults for a first checkout. Existing .env files stay unchanged.
  if grep -q '^APP_DATA_DIR=' .env; then
    sed -i "s|^APP_DATA_DIR=.*|APP_DATA_DIR=${DEFAULT_DATA_DIR}|" .env
  else
    echo "APP_DATA_DIR=${DEFAULT_DATA_DIR}" >> .env
  fi
  if grep -q '^MOCK_MODE=' .env; then
    sed -i 's|^MOCK_MODE=.*|MOCK_MODE=true|' .env
  else
    echo 'MOCK_MODE=true' >> .env
  fi
  if grep -q '^APP_ENV=' .env; then
    sed -i 's|^APP_ENV=.*|APP_ENV=development|' .env
  fi
  echo "    wrote .env with APP_DATA_DIR=${DEFAULT_DATA_DIR} and MOCK_MODE=true"
  echo "    edit .env before enabling live integrations; keep it local"
else
  echo "==> .env already exists — leaving it unchanged"
fi

# Prefer APP_DATA_DIR from the environment, then .env, then the default.
if [[ -z "${APP_DATA_DIR:-}" && -f .env ]]; then
  # shellcheck disable=SC1091
  APP_DATA_DIR="$(grep -E '^APP_DATA_DIR=' .env | tail -1 | cut -d= -f2- || true)"
fi
APP_DATA_DIR="${APP_DATA_DIR:-$DEFAULT_DATA_DIR}"
# Expand a leading ~
APP_DATA_DIR="${APP_DATA_DIR/#\~/$HOME}"
export APP_DATA_DIR
mkdir -p "${APP_DATA_DIR}"

echo "==> Applying database migrations (APP_DATA_DIR=${APP_DATA_DIR})"
uv run python scripts/init_database.py

cat <<EOF

Setup complete.

Next:
  ./scripts/run.sh       # start the API (uses .venv via uv; no manual activate)
  ./scripts/verify.sh    # offline/mock quality gate

Optional: edit .env to enable Telegram, Google, Notion, Reachy, or vision.
EOF
