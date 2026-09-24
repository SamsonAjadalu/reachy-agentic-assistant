# Vision sidecar on a CUDA GPU host

**Role:** run **only** the perception sidecar (Grounding DINO, SAM 2, DA3, DINOv2) on a
machine with a CUDA GPU. Use the dedicated sidecar environment for the CUDA host. Keep workstation data, robot
tokens, Telegram credentials, or local LLM config onto this host.

## Layout

| Machine | Role |
|---|---|
| Workstation (Linux) | Reachy Agentic Assistant API, SQLite, robot client |
| Optional GPU host | `vision_sidecar` HTTP on `:8090` |

When the GPU is local to the workstation, set `VISION_SIDECAR_HOST=127.0.0.1` and skip the
remote firewall steps.

## One-time setup on the GPU host

1. Install Python 3.12.
2. Sync the sidecar source and its dedicated configuration: `vision_sidecar/`, modules it imports
   (`app/` helpers, `shared/`, `security/`), and `pyproject.toml`.
3. Create a dedicated venv and install deps:

```powershell
cd <path-to-this-repo>
py -3.12 -m venv .venv-vision
.\.venv-vision\Scripts\python.exe -m pip install -U pip wheel
.\.venv-vision\Scripts\python.exe -m pip install -r deployment\vision_3090\requirements.txt
.\.venv-vision\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

4. Copy `env.example` → `%LOCALAPPDATA%\reachy-vision-sidecar\.env` and set
   `VISION_SIDECAR_TOKEN`.
5. Allow inbound TCP 8090 only from the workstation network (firewall / host firewall).
6. Start:

```powershell
# Optional overrides:
#   $env:REACHY_PA_REPO = "C:\path\to\reachy-agentic-assistant"
#   $env:REACHY_VISION_DATA = "$env:LOCALAPPDATA\reachy-vision-sidecar"
powershell -ExecutionPolicy Bypass -File deployment\vision_3090\start-sidecar.ps1
```

## Workstation

```env
VISUAL_ENABLED=true
VISUAL_SIDECAR_ENABLED=true
MOCK_MODE=false
VISION_SIDECAR_HOST=<vision-sidecar-host-or-127.0.0.1>
VISION_SIDECAR_PORT=8090
VISION_SIDECAR_TOKEN=<same secret>
REACHY_CAMERA_ENABLED=true
```

Keep `./scripts/verify.sh` on mock/localhost; live flags are for attended runs only.
