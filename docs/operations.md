# Operations

Day-to-day running of Reachy Agentic Assistant on the workstation.

## Layout

| Path | Purpose |
|------|---------|
| Application files | Code, config templates, and systemd unit templates |
| `APP_DATA_DIR` (default `~/.local/share/reachy-personal-assistant`) | SQLite databases, encrypted tokens, wardrobe images, document index, backups |
| Fernet key file | **Outside** `APP_DATA_DIR` — e.g. `~/.config/reachy-personal-assistant/pa_secret_key` or systemd `LoadCredential` |

Runtime data is stored under `APP_DATA_DIR`. Store the decryption key separately from APP_DATA_DIR backups.

## Install (user systemd)

```bash
./scripts/install.sh
# Edit .env for live credentials when ready, then:
uv run python scripts/generate_api_token.py --write
uv run python scripts/check_configuration.py
./deployment/workstation/install-user.sh
systemctl --user start reachy-personal-assistant-api.service
```

For a local mock run without systemd, use `./scripts/run.sh` after `./scripts/install.sh`.

The install-user script copies unit templates from `deployment/workstation/systemd/`, creates `APP_DATA_DIR`, wires `LoadCredential` for the Fernet key, and enables the API plus backup and health-check timers.

### User lingering

User services stop when you log out unless **lingering** is enabled:

```bash
loginctl enable-linger $USER
loginctl show-user $USER -p Linger
```

`install-user.sh` prompts for this. The assistant runs headlessly after lingering is enabled.

### System install

For a dedicated service account and paths under `/var/lib` and `/etc`, run:

```bash
./deployment/workstation/install-system.sh
```

That script **prints** exact `sudo` commands for manual execution. System units use the system service manager directly.

## Process model

One **scheduler-owning** API process is the normal layout:

- `SCHEDULER_ENABLED=true`, `WORKER_ENABLED=true` (defaults)
- Holds `${APP_DATA_DIR}/scheduler.lock`
- Runs APScheduler jobs (reminders, briefings, document index queue, maintenance)
- Runs the background worker in-process
- Starts the **Telegram long-poll listener** when `TELEGRAM_ENABLED=true` and not in mock mode

There is **no separate Telegram unit**. Two pollers would fight over `getUpdates` offsets.

Optional split:

| Unit | `SCHEDULER_ENABLED` | `WORKER_ENABLED` | Notes |
|------|---------------------|------------------|-------|
| `reachy-personal-assistant-api` | true | false | Scheduler + Telegram |
| `reachy-personal-assistant-worker` | false | true | Worker only (port 8081 in template) |

Use one scheduler-owning instance.

## Encryption key

Prefer systemd credentials (see unit templates):

```
Environment=PA_SECRET_KEY_FILE=${CREDENTIALS_DIRECTORY}/pa_secret_key
LoadCredential=pa_secret_key:/path/to/pa_secret_key
```

Leave `PA_SECRET_KEY` unset in production. Alternatives: `PA_SECRET_KEY_FILE` pointing at a chmod-600 file outside `APP_DATA_DIR`, or `PA_SECRET_KEY_KEYRING=true`. See `security/keyring.py`.

## Upgrade

1. Pull or rsync new code into the checkout (or `/opt/reachy-personal-assistant`).
2. Install dependencies (`uv sync` or equivalent).
3. `python scripts/init_database.py` — applies Alembic migrations.
4. `python scripts/check_configuration.py`
5. `systemctl --user restart reachy-personal-assistant-api.service` (or `sudo systemctl restart …`).

## Rollback

1. Stop the API (and worker unit if enabled).
2. Redeploy the previous code revision.
3. Restore databases: `python scripts/restore_database.py --latest --force`
4. Start the service.

OAuth tokens in `${APP_DATA_DIR}/secrets/google_tokens.enc` use a separate backup policy from SQLite data.

## Logs

```bash
journalctl --user -u reachy-personal-assistant-api.service -f
journalctl --user -u reachy-personal-assistant-backup.service
journalctl --user -u reachy-personal-assistant-healthcheck.service
```

System install: drop `--user`.

## Health checks

Unauthenticated probes (also used by `scripts/healthcheck.py`):

- `GET /health` — process alive
- `GET /ready` — database reachable; scheduler/worker fields are informational

Timer: `reachy-personal-assistant-healthcheck.timer` (every 5 minutes by default).

Manual:

```bash
python scripts/healthcheck.py
curl -s http://127.0.0.1:8080/health
curl -s http://127.0.0.1:8080/ready
```

## Backups

Timer: `reachy-personal-assistant-backup.timer` (daily 03:15 by default).

Manual:

```bash
python scripts/backup_database.py
python scripts/restore_database.py --latest --force   # stop API first
```

Archives land in `${APP_DATA_DIR}/backups/` unless `BACKUP_ROOT` is set. Retention: `BACKUP_RETENTION_DAYS` (default 30).

## Document indexing

Incremental re-indexing is scheduled inside the API process (`DOCUMENT_INDEX_INTERVAL_SECONDS`, default 3600). Optional timer `reachy-personal-assistant-document-index.timer` POSTs to `/api/v1/documents/reindex` for a manual nudge.

## Uninstall

```bash
./deployment/workstation/uninstall-user.sh
```

Removes user units only; `APP_DATA_DIR` and secrets are kept.

## Demo / mock mode

```bash
chmod +x scripts/run_mock_demo.sh
./scripts/run_mock_demo.sh
```

Uses `MOCK_MODE=true` and a throwaway data directory under `~/.local/share/reachy-personal-assistant-demo`.

## Related docs

- [API overview](api.md)
- [Telegram setup](telegram_setup.md)
- [Google setup](google_setup.md)
- [Notion setup](notion_setup.md)
- [Wardrobe](wardrobe.md)
