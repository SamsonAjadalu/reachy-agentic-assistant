# Reachy Agentic Assistant

Reachy Agentic Assistant is a workstation service that extends Reachy Mini with
Gmail, Calendar, Telegram, Notion, scheduling, memory, and visual-perception
tools.

The official Reachy Mini conversation app runs on the robot. Reachy Agentic
Assistant provides the external agentic and perception services that the robot
connects to.

## Workstation

This repository runs on a workstation (or other always-on host) and provides:

- Authenticated HTTP API for reminders, tasks, mail, calendar, Notion, documents,
  wardrobe, and approvals
- Persistent memory, scheduling, and background jobs
- Optional visual memory and CUDA perception sidecar
- Mock mode for offline development without hardware or cloud credentials

## Reachy Mini

On the robot, the official conversation app handles voice, wake word, motion,
and camera. Install the overlays in [`pi_integration/`](pi_integration/) so the
app can call this service over the network.

## Installation

Requirements: Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/SamsonAjadalu/reachy-agentic-assistant
cd reachy-agentic-assistant
./scripts/install.sh
./scripts/run.sh
```

[`scripts/install.sh`](scripts/install.sh) checks Python and uv, creates
`.venv`, installs dependencies, creates `.env` from
[`.env.example`](.env.example) only if missing, and applies database migrations.
Runtime data stays outside the repo under `APP_DATA_DIR` (default
`~/.local/share/reachy-personal-assistant`).

API schema: `http://127.0.0.1:8080/docs`

### Reachy Mini connection

1. Install the official Reachy Mini conversation app on the robot.
2. Install the Pi overlays into that checkout:

   ```bash
   ./pi_integration/install_to_reachy.sh /path/to/conversation_app
   ```

3. On the robot, set `PA_API_URL` and `PA_API_TOKEN` to this service.
4. Run this service with [`./scripts/run.sh`](scripts/run.sh).
5. Run the conversation app on the robot.

## Configuration

Edit `.env` (gitignored). [`.env.example`](.env.example) contains placeholders
only. Keep secrets in local environment variables.

| Variable | Default | Purpose |
|----------|---------|---------|
| `MOCK_MODE` | `true` after install | Mock providers (no live credentials) |
| `TELEGRAM_ENABLED` | `false` | Approvals and alerts |
| `GOOGLE_ENABLED` | `false` | Gmail and Calendar |
| `NOTION_ENABLED` | `false` | Notion pages |
| `REACHY_TEXT_TURN_ENABLED` | `false` | Telegram → robot chat |
| `REACHY_CAMERA_ENABLED` | `false` | Robot camera API |
| `VISUAL_ENABLED` | `false` | Visual memory |
| `VISUAL_SIDECAR_ENABLED` | `false` | CUDA perception sidecar |

```env
APP_DATA_DIR=<your-data-directory>
REACHY_CAMERA_URL=http://reachy-mini.local:7861
VISION_SIDECAR_HOST=127.0.0.1
```

## Optional integrations

All cloud and robot integrations are **off by default**. With `MOCK_MODE=true`,
you can exercise the API without Reachy hardware, Telegram, Google, Notion, or a
GPU. Enable each integration in `.env` and follow the matching setup guide under
[Documentation](#documentation).

## Verification

```bash
./scripts/verify.sh
```

Offline gate: format, lint, type checks, migrations, and the mock test suite.
See [`scripts/verify.sh`](scripts/verify.sh).

## Documentation

- [API overview](docs/api.md)
- [Operations](docs/operations.md)
- [Telegram setup](docs/telegram_setup.md)
- [Google setup](docs/google_setup.md)
- [Notion setup](docs/notion_setup.md)
- [Reachy Mini handoff](docs/reachy_pi_handoff.md)
- [Wardrobe](docs/wardrobe.md)
- [Vision sidecar (optional GPU host)](deployment/vision_3090/README.md)
- [Reachy tools](reachy_tools/README.md)
- [Pi integration installer](pi_integration/install_to_reachy.sh)
