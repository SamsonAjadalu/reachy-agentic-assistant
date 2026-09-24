# Reachy Pi integration

Notes for connecting Reachy Agentic Assistant tools to the official Reachy Mini
conversation application on the robot.

## Conversation app layout

- Conversation app checkout: `<REACHY_CONVERSATION_APP_ROOT>`
- Profile directory:
  `~/.config/reachy_mini_conversation_app/user_personalities/<your_profile>`

## Tool mechanism

A tool subclasses `reachy_mini_conversation_app.tools.core_tools.Tool` and
declares `name`, `description`, and an object-shaped JSON `parameters_schema`.
Its entrypoint is `async def __call__(deps: ToolDependencies, **kwargs) -> dict`.

The selected profile’s `tools.txt` is an allowlist of module names. For each
entry, the app looks for a same-named `.py` file in the profile directory, then
falls back to built-in or configured external tools.

The bridge in this repository is `reachy_tools/reachy_app_tools.py`. Typical
wiring:

1. Symlink or copy `reachy_app_tools.py` into the profile directory.
2. Add `reachy_app_tools` to the profile `tools.txt`.
3. Add a `.pth` file in the conversation virtualenv that points at this repository
   checkout (so `reachy_client` imports resolve). Install the Pi-specific dependencies
   Reachy Agentic Assistant service on the robot.

## Configuration

| Variable | Required | Default | Meaning |
|---|---:|---:|---|
| `PA_API_URL` | yes | none | Absolute HTTP(S) Reachy Agentic Assistant API base URL |
| `PA_API_TOKEN` | yes | none | Bearer token shared with Reachy Agentic Assistant |
| `PA_API_CONNECT_TIMEOUT` | no | `3` | TCP connection timeout, seconds |
| `PA_API_TIMEOUT` | no | `10` | Request read/write timeout, seconds |

Keep `.env` and tokens in the robot's local configuration.

## Behaviour notes

- Tools must stay async; blocking work stalls the conversation event loop.
- Fast API calls use bounded HTTP timeouts. Long work (indexing, scripts)
  stays on the workstation service and returns tickets / watch IDs.
- Supported creates use idempotency keys. Irreversible writes use one approved request.
- Externally visible writes require Telegram approval via Reachy Agentic Assistant.

## Camera / visual memory

When enabling vision on the workstation, point `REACHY_CAMERA_URL` at the robot
camera HTTP origin (for example `http://reachy-mini.local:7861`) and share a
bearer token. Attended scans require an explicit robot-side enable flag; they
are off by default.

See the root [README.md](../README.md) for install, mock mode, and integration flags.
