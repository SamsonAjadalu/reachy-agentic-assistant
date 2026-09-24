#!/usr/bin/env bash
# Install user-level systemd units for the Reachy Personal Assistant.
#
#   ./deployment/workstation/install-user.sh
#   ./deployment/workstation/install-user.sh --yes --no-start
#   PA_INSTALL_NONINTERACTIVE=1 ./deployment/workstation/install-user.sh --start
#
# Does not run sudo. Prompts before enabling systemd user lingering unless
# non-interactive mode is set.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SYSTEMD_SRC="${REPO_ROOT}/deployment/workstation/systemd"
SYSTEMD_USER="${HOME}/.config/systemd/user"
ENV_FILE="${ENV_FILE:-${REPO_ROOT}/.env}"
APP_DATA_DIR="${APP_DATA_DIR:-${HOME}/.local/share/reachy-personal-assistant}"
SECRET_DIR="${HOME}/.config/reachy-personal-assistant"
SECRET_KEY_FILE="${SECRET_KEY_FILE:-${SECRET_DIR}/pa_secret_key}"

NONINTERACTIVE=0
DO_START=""
DO_LINGER=""
for arg in "$@"; do
  case "$arg" in
    --yes|-y) NONINTERACTIVE=1 ;;
    --start) DO_START=1 ;;
    --no-start) DO_START=0 ;;
    --linger) DO_LINGER=1 ;;
    --no-linger) DO_LINGER=0 ;;
    *)
      echo "Unknown option: $arg" >&2
      exit 2
      ;;
  esac
done
if [[ "${PA_INSTALL_NONINTERACTIVE:-0}" == "1" ]]; then
  NONINTERACTIVE=1
fi

# Uses the configured workflow.
if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
  PYTHON="${REPO_ROOT}/.venv/bin/python"
elif [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
  PYTHON="${VIRTUAL_ENV}/bin/python"
elif command -v uv >/dev/null 2>&1 && [[ -f "${REPO_ROOT}/pyproject.toml" ]]; then
  UV_BIN="$(command -v uv)"
  PYTHON="${UV_BIN} run --directory ${REPO_ROOT} python"
else
  PYTHON="$(command -v python3 || command -v python)"
fi

echo "Reachy Personal Assistant — user install"
echo "  Repository:     ${REPO_ROOT}"
echo "  APP_DATA_DIR:   ${APP_DATA_DIR}"
echo "  Environment:    ${ENV_FILE}"
echo "  Secret key:     ${SECRET_KEY_FILE}"
echo "  Python:         ${PYTHON}"
echo

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "WARNING: ${ENV_FILE} not found."
  echo "  cp .env.example .env && python scripts/generate_api_token.py --write"
  echo
fi

if [[ ! -f "${SECRET_KEY_FILE}" ]]; then
  echo "Encryption key not found at ${SECRET_KEY_FILE}."
  if [[ "${NONINTERACTIVE}" -eq 1 ]]; then
    mkdir -p "${SECRET_DIR}"
    chmod 700 "${SECRET_DIR}"
    ${PYTHON} "${REPO_ROOT}/scripts/generate_api_token.py" --secret-key --key-file "${SECRET_KEY_FILE}"
    echo "Created ${SECRET_KEY_FILE} (mode 600) in non-interactive mode."
  else
    read -r -p "Generate one now? [y/N] " reply
    if [[ "${reply,,}" == "y" ]]; then
      mkdir -p "${SECRET_DIR}"
      chmod 700 "${SECRET_DIR}"
      ${PYTHON} "${REPO_ROOT}/scripts/generate_api_token.py" --secret-key --key-file "${SECRET_KEY_FILE}"
      echo "Created ${SECRET_KEY_FILE} (mode 600)."
      echo "Ensure PA_SECRET_KEY is unset in .env and PA_SECRET_KEY_FILE points at this path"
      echo "  (the unit sets PA_SECRET_KEY_FILE via LoadCredential automatically)."
      echo
    else
      echo "Create a key before starting the service:"
      echo "  python scripts/generate_api_token.py --secret-key --key-file ${SECRET_KEY_FILE}"
      echo
    fi
  fi
fi

echo "Creating runtime data directory..."
mkdir -p "${APP_DATA_DIR}"/{secrets,cache,task_outputs,documents,backups,wardrobe/items,wardrobe/thumbnails,wardrobe/outfits,script_runs}
chmod 700 "${APP_DATA_DIR}/secrets"

echo "Installing user units to ${SYSTEMD_USER}..."
mkdir -p "${SYSTEMD_USER}"

substitute() {
  local python_bin="${1:-${PYTHON}}"
  sed \
    -e "s|@REPO_ROOT@|${REPO_ROOT}|g" \
    -e "s|@APP_DATA_DIR@|${APP_DATA_DIR}|g" \
    -e "s|@ENV_FILE@|${ENV_FILE}|g" \
    -e "s|@PYTHON@|${python_bin}|g" \
    -e "s|@SECRET_KEY_FILE@|${SECRET_KEY_FILE}|g"
}

for unit in "${SYSTEMD_SRC}"/*.service "${SYSTEMD_SRC}"/*.timer; do
  [[ -f "${unit}" ]] || continue
  name="$(basename "${unit}")"
  python_bin="${PYTHON}"
  if [[ "${name}" == "reachy-vision-sidecar.service" && -x "${REPO_ROOT}/.venv-vision/bin/python" ]]; then
    python_bin="${REPO_ROOT}/.venv-vision/bin/python"
  fi
  substitute "${python_bin}" < "${unit}" > "${SYSTEMD_USER}/${name}"
  echo "  ${name}"
done

systemctl --user daemon-reload

echo
echo "Enabling API service and maintenance timers..."
systemctl --user enable reachy-personal-assistant-api.service
systemctl --user enable reachy-personal-assistant-backup.timer
systemctl --user enable reachy-personal-assistant-healthcheck.timer

echo
echo "User services stop when you log out unless lingering is enabled."
echo "Without lingering, the assistant will not run headlessly after logout."
echo
if [[ -n "${DO_LINGER}" ]]; then
  linger_choice="${DO_LINGER}"
elif [[ "${NONINTERACTIVE}" -eq 1 ]]; then
  linger_choice=0
else
  read -r -p "Enable lingering for ${USER}? (loginctl enable-linger ${USER}) [y/N] " linger
  [[ "${linger,,}" == "y" ]] && linger_choice=1 || linger_choice=0
fi
if [[ "${linger_choice}" -eq 1 ]]; then
  loginctl enable-linger "${USER}"
  echo "Lingering enabled."
else
  echo "Skipped. Enable later with:  loginctl enable-linger ${USER}"
fi

echo
echo "Next steps:"
echo "  1. Edit ${ENV_FILE} — set APP_DATA_DIR=${APP_DATA_DIR} if not already."
echo "  2. Run:  python scripts/check_configuration.py"
echo "  3. Run:  python scripts/init_database.py"
echo "  4. Start:  systemctl --user start reachy-personal-assistant-api.service"
echo "  5. Logs:   journalctl --user -u reachy-personal-assistant-api.service -f"
echo
echo "Telegram polls inside the scheduler-owning API process — no separate telegram unit."
echo "Optional worker split: disable WORKER_ENABLED on the API unit, then enable"
echo "  reachy-personal-assistant-worker.service (uses port 8081 by default)."
echo "Perception sidecar is copied but not enabled (needs .venv-vision + VISION_SIDECAR_TOKEN):"
echo "  systemctl --user enable --now reachy-vision-sidecar.service"
echo
if [[ -n "${DO_START}" ]]; then
  start_choice="${DO_START}"
elif [[ "${NONINTERACTIVE}" -eq 1 ]]; then
  start_choice=0
else
  read -r -p "Start the API now? [y/N] " start_now
  [[ "${start_now,,}" == "y" ]] && start_choice=1 || start_choice=0
fi
if [[ "${start_choice}" -eq 1 ]]; then
  systemctl --user start reachy-personal-assistant-api.service
  systemctl --user start reachy-personal-assistant-backup.timer
  systemctl --user start reachy-personal-assistant-healthcheck.timer
  echo "Started. Check:  curl -s http://127.0.0.1:8080/health"
fi

echo "Done."
