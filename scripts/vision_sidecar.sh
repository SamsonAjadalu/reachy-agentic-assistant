#!/usr/bin/env bash
#
# Run the perception sidecar in its own process and (optionally) venv.
#
#   ./scripts/vision_sidecar.sh              # serve on 127.0.0.1:8090
#   ./scripts/vision_sidecar.sh bakeoff      # verify upstream; GPU bench only if ollama idle
#   ./scripts/vision_sidecar.sh bakeoff --gpu
#
# The API virtualenv must stay free of torch. Prefer .venv-vision for serving
# real providers; unit tests use the API venv and MOCK_MODE.
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

VISION_VENV="${VISION_VENV:-${REPO_ROOT}/.venv-vision}"
API_VENV="${REPO_ROOT}/.venv"

if [[ -x "${VISION_VENV}/bin/python" ]]; then
  PY="${VISION_VENV}/bin/python"
elif [[ -x "${API_VENV}/bin/python" ]]; then
  PY="${API_VENV}/bin/python"
else
  echo "No virtualenv found. Create a sidecar venv with:"
  echo "  uv venv --python 3.12 .venv-vision && uv pip install -e '.[vision]' --python .venv-vision"
  echo "or, for mock-only:"
  echo "  uv venv --python 3.12 .venv && uv pip install -e '.[dev]'"
  exit 1
fi

export PYTHONPATH="${REPO_ROOT}"
export PYTHONDONTWRITEBYTECODE=1
# HF_HOME is forced by SidecarSettings.apply_hf_env to ${APP_DATA_DIR}/vision/hf.

cmd="${1:-serve}"
if [[ "${cmd}" == "bakeoff" ]]; then
  shift
  exec "$PY" -m vision_sidecar.bakeoff "$@"
fi
if [[ "${cmd}" == "serve" ]]; then
  shift || true
  exec "$PY" -m vision_sidecar "$@"
fi
exec "$PY" -m vision_sidecar "$@"
