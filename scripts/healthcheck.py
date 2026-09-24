#!/usr/bin/env python3
"""Probe /health and /ready. Exit non-zero when either fails.

    python scripts/healthcheck.py
    python scripts/healthcheck.py --host 127.0.0.1 --port 8080
    python scripts/healthcheck.py --json

Used by reachy-personal-assistant-healthcheck.service and external monitors.
/ready must report ready=true (database reachable). Scheduler and worker state
is informational only.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app.config import Settings  # noqa: E402


def _fetch(url: str, timeout: float) -> tuple[int, dict[str, Any] | None, str | None]:
    request = urllib.request.Request(url, method="GET")
    try:
        # Local health probes only; host/port come from Settings, not user input.
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            payload = json.loads(body) if body else None
            return response.status, payload, None
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            payload = json.loads(body) if body else None
        except json.JSONDecodeError:
            payload = None
        return exc.code, payload, body or str(exc)
    except Exception as exc:
        return 0, None, str(exc)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--host", help="Override APP_HOST from settings.")
    parser.add_argument("--port", type=int, help="Override APP_PORT from settings.")
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    settings = Settings()
    host = args.host or settings.app_host
    port = args.port or settings.app_port
    base = f"http://{host}:{port}"

    report: dict[str, Any] = {"base_url": base, "checks": {}}
    failed = False

    for name, path in (("health", "/health"), ("ready", "/ready")):
        status, payload, error = _fetch(f"{base}{path}", args.timeout)
        ok = status == 200
        if name == "ready" and payload is not None:
            ok = ok and bool(payload.get("ready"))
        if name == "health" and payload is not None:
            ok = ok and payload.get("status") == "ok"

        report["checks"][name] = {
            "ok": ok,
            "status_code": status,
            "body": payload,
            "error": error,
        }
        if not ok:
            failed = True

    if args.json:
        report["ok"] = not failed
        print(json.dumps(report, indent=2))
    else:
        for name, detail in report["checks"].items():
            state = "ok" if detail["ok"] else "FAIL"
            print(f"[{state}] {name}  HTTP {detail['status_code']}")
            if detail["error"]:
                print(f"       {detail['error']}")
            elif name == "ready" and detail["body"]:
                checks = detail["body"].get("checks", {})
                print(f"       database={checks.get('database', {}).get('ok')}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
