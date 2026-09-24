#!/usr/bin/env bash
# Print exact sudo commands for a system-level install. Does NOT run sudo.
#
#   ./deployment/workstation/install-system.sh
#
# Review and run each block manually. Adjust paths, users, and ports to match
# your host.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
INSTALL_USER="${INSTALL_USER:-reachy}"
INSTALL_GROUP="${INSTALL_GROUP:-${INSTALL_USER}}"
APP_DATA_DIR="${APP_DATA_DIR:-/var/lib/reachy-personal-assistant}"
ENV_FILE="${ENV_FILE:-/etc/reachy-personal-assistant/env}"
SECRET_KEY_FILE="${SECRET_KEY_FILE:-/etc/reachy-personal-assistant/pa_secret_key}"
SYSTEMD_SRC="${REPO_ROOT}/deployment/workstation/systemd"
PYTHON="${PYTHON:-/opt/reachy-personal-assistant/.venv/bin/python}"

cat <<EOF
Reachy Personal Assistant — system install commands
===================================================

Review each block before running. Nothing here is executed automatically.

1. Create service account and directories
-----------------------------------------
sudo useradd --system --home-dir ${APP_DATA_DIR} --shell /usr/sbin/nologin ${INSTALL_USER} || true
sudo mkdir -p ${APP_DATA_DIR}/{secrets,cache,task_outputs,documents,backups,wardrobe/items,wardrobe/thumbnails,wardrobe/outfits,script_runs}
sudo mkdir -p /etc/reachy-personal-assistant
sudo chown -R ${INSTALL_USER}:${INSTALL_GROUP} ${APP_DATA_DIR}
sudo chmod 700 ${APP_DATA_DIR}/secrets

2. Deploy application code (example — use your preferred layout)
----------------------------------------------------------------
sudo mkdir -p /opt/reachy-personal-assistant
sudo rsync -a --delete \\
  --exclude .git --exclude .venv --exclude __pycache__ --exclude .env \\
  ${REPO_ROOT}/ /opt/reachy-personal-assistant/
sudo chown -R ${INSTALL_USER}:${INSTALL_GROUP} /opt/reachy-personal-assistant

3. Python environment
---------------------
sudo -u ${INSTALL_USER} bash -c 'cd /opt/reachy-personal-assistant && uv sync'   # or pip install -e .

4. Configuration and secrets (outside APP_DATA_DIR)
---------------------------------------------------
sudo cp ${REPO_ROOT}/.env.example ${ENV_FILE}
sudo chmod 640 ${ENV_FILE}
sudo chown root:${INSTALL_GROUP} ${ENV_FILE}
# Edit ${ENV_FILE}: set APP_ENV=production, APP_DATA_DIR=${APP_DATA_DIR}, tokens, integrations.

sudo ${PYTHON} /opt/reachy-personal-assistant/scripts/generate_api_token.py \\
  --secret-key --key-file ${SECRET_KEY_FILE}
sudo chmod 600 ${SECRET_KEY_FILE}
sudo chown root:${INSTALL_GROUP} ${SECRET_KEY_FILE}
# Leave PA_SECRET_KEY unset in ${ENV_FILE}; the unit uses LoadCredential.

5. Initialise database
------------------------
sudo -u ${INSTALL_USER} env PA_ENV_FILE=${ENV_FILE} APP_DATA_DIR=${APP_DATA_DIR} \\
  ${PYTHON} /opt/reachy-personal-assistant/scripts/init_database.py

6. Install systemd units
------------------------
EOF

for unit in "${SYSTEMD_SRC}"/*.service "${SYSTEMD_SRC}"/*.timer; do
  [[ -f "${unit}" ]] || continue
  name="$(basename "${unit}")"
  cat <<EOF
sudo sed \\
  -e 's|@REPO_ROOT@|/opt/reachy-personal-assistant|g' \\
  -e 's|@APP_DATA_DIR@|${APP_DATA_DIR}|g' \\
  -e 's|@ENV_FILE@|${ENV_FILE}|g' \\
  -e 's|@PYTHON@|${PYTHON}|g' \\
  -e 's|@SECRET_KEY_FILE@|${SECRET_KEY_FILE}|g' \\
  ${SYSTEMD_SRC}/${name} | sudo tee /etc/systemd/system/${name} > /dev/null
EOF
done

cat <<EOF

# Patch User= and Group= into service units (templates omit these for user installs):
for svc in reachy-personal-assistant-api reachy-personal-assistant-worker \\
           reachy-personal-assistant-backup reachy-personal-assistant-healthcheck \\
           reachy-personal-assistant-document-index; do
  sudo sed -i '/^\\[Service\\]/a User=${INSTALL_USER}\\nGroup=${INSTALL_GROUP}' \\
    /etc/systemd/system/\${svc}.service
done

sudo systemctl daemon-reload

7. Enable and start
-------------------
sudo systemctl enable --now reachy-personal-assistant-api.service
sudo systemctl enable --now reachy-personal-assistant-backup.timer
sudo systemctl enable --now reachy-personal-assistant-healthcheck.timer
# Optional worker split (disable WORKER_ENABLED in ${ENV_FILE} first):
# sudo systemctl enable --now reachy-personal-assistant-worker.service
# Optional document re-index timer:
# sudo systemctl enable --now reachy-personal-assistant-document-index.timer

8. Verify
---------
curl -s http://127.0.0.1:8080/health
curl -s http://127.0.0.1:8080/ready
sudo journalctl -u reachy-personal-assistant-api.service -f

Notes
-----
- System units use the system service manager directly.
- Telegram long-polling runs inside the scheduler-owning API process; there is
  no separate telegram unit.
- Only one process may hold ${APP_DATA_DIR}/scheduler.lock.
- Upgrade: rsync new code, run init_database.py, systemctl restart the API.
- Rollback: restore database from ${APP_DATA_DIR}/backups, redeploy previous code.

EOF
