#!/usr/bin/env bash
# Remove user-level systemd units installed by install-user.sh.
#
#   ./deployment/workstation/uninstall-user.sh
#
# Does not delete APP_DATA_DIR or secrets. Does not disable lingering.

set -euo pipefail

SYSTEMD_USER="${HOME}/.config/systemd/user"
UNITS=(
  reachy-personal-assistant-api.service
  reachy-personal-assistant-worker.service
  reachy-personal-assistant-backup.service
  reachy-personal-assistant-backup.timer
  reachy-personal-assistant-healthcheck.service
  reachy-personal-assistant-healthcheck.timer
  reachy-personal-assistant-document-index.service
  reachy-personal-assistant-document-index.timer
  reachy-vision-sidecar.service
)

echo "Stopping and disabling Reachy Personal Assistant user units..."

for unit in "${UNITS[@]}"; do
  systemctl --user disable --now "${unit}" 2>/dev/null || true
  rm -f "${SYSTEMD_USER}/${unit}"
done

systemctl --user daemon-reload
echo "User units removed from ${SYSTEMD_USER}."
echo
echo "Runtime data was NOT deleted. APP_DATA_DIR is unchanged."
echo "To remove lingering:  loginctl disable-linger ${USER}"
