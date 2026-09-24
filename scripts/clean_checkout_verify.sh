#!/usr/bin/env bash
# Verify the project from an isolated copy of tracked files only.
#
#   ./scripts/clean_checkout_verify.sh
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="${TMPDIR:-/tmp}/pa-clean-checkout-$$"
ARCHIVE="${WORK}/tree.tar"
COPY="${WORK}/src"
DATA_DIR="${WORK}/data"

cleanup() { rm -rf "${WORK}"; }
trap cleanup EXIT

mkdir -p "${WORK}" "${DATA_DIR}"
chmod 700 "${DATA_DIR}"

echo "Exporting tracked tree from ${REPO_ROOT}..."
git -C "${REPO_ROOT}" archive --format=tar HEAD > "${ARCHIVE}"
mkdir -p "${COPY}"
tar -xf "${ARCHIVE}" -C "${COPY}"

cd "${COPY}"
export PYTHONPATH=""
export PYTHONDONTWRITEBYTECODE=1
export APP_ENV=test
export PA_ENV_FILE="${COPY}/tests/.env.absent"
export APP_DATA_DIR="${DATA_DIR}"

if command -v uv >/dev/null 2>&1; then
  uv venv --python 3.12 .venv
  # Prefer the lockfile when present.
  if [[ -f uv.lock ]]; then
    uv sync --extra dev
  else
    uv pip install -e '.[dev]'
  fi
else
  python3 -m venv .venv
  .venv/bin/pip install -U pip
  .venv/bin/pip install -e '.[dev]'
fi

echo "Running verify.sh in isolated copy..."
./scripts/verify.sh

echo "clean checkout verify passed."
