#!/usr/bin/env python3
"""Report which real integrations are configured without exposing secrets.

    python scripts/live_readiness_check.py
    python scripts/live_readiness_check.py --json

Exit 0 always when the process can read configuration. Exit 1 only if the
base app would refuse to start (same bar as check_configuration for hard fails).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app.config import Settings, looks_like_placeholder  # noqa: E402
from scripts.check_configuration import check  # noqa: E402


def _secret_state(value: str, *, min_len: int = 1) -> str:
    if not value or looks_like_placeholder(value):
        return "missing"
    if len(value) < min_len:
        return f"too_short({len(value)})"
    return f"set({len(value)} chars)"


def integration_matrix(settings: Settings) -> dict[str, Any]:
    matrix: dict[str, Any] = {
        "mock_mode": settings.mock_mode,
        "app_env": settings.app_env.value,
        "telegram": {
            "enabled": settings.telegram_enabled,
            "token": _secret_state(settings.telegram_bot_token.get_secret_value(), min_len=20),
            "allowed_chats": bool(settings.telegram_allowed_chat_ids),
            "ready_for_live": bool(
                settings.telegram_enabled
                and not looks_like_placeholder(settings.telegram_bot_token.get_secret_value())
                and settings.telegram_allowed_chat_ids
                and not settings.mock_mode
            ),
        },
        "google": {
            "enabled": settings.google_enabled,
            "client_id": _secret_state(settings.google_client_id),
            "client_secret": _secret_state(
                settings.google_client_secret.get_secret_value(), min_len=10
            ),
            "secret_key_source": settings.secret_key_source(),
            "token_store_configured": True,
            "ready_for_live": bool(
                settings.google_enabled
                and not settings.mock_mode
                and not looks_like_placeholder(settings.google_client_id)
                and not looks_like_placeholder(settings.google_client_secret.get_secret_value())
                and settings.secret_key_source() != "unset"
            ),
        },
        "notion": {
            "enabled": settings.notion_enabled,
            "token": _secret_state(settings.notion_token.get_secret_value(), min_len=10),
            "allowed_pages": bool(settings.notion_allowed_page_ids),
            "ready_for_live": bool(
                settings.notion_enabled
                and not settings.mock_mode
                and not looks_like_placeholder(settings.notion_token.get_secret_value())
                and settings.notion_allowed_page_ids
            ),
        },
        "weather": {
            "provider": str(settings.weather_provider.value),
            "ready_for_live": (not settings.mock_mode)
            and (
                settings.weather_provider.value == "open_meteo"
                or (
                    settings.weather_provider.value == "openweathermap"
                    and not looks_like_placeholder(settings.weather_api_key.get_secret_value())
                )
            ),
        },
    }
    return matrix


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    settings = Settings()
    base = check(settings)
    matrix = integration_matrix(settings)
    payload = {
        "configuration": base.entries,
        "integrations": matrix,
        "would_refuse_start": base.failed,
    }

    if args.json:
        print(json.dumps(payload, indent=2, default=str))
    else:
        print(base.render())
        print()
        print("Integration readiness (no secrets shown):")
        for name, info in matrix.items():
            if name in {"mock_mode", "app_env"}:
                print(f"  {name}: {info}")
                continue
            ready = info.get("ready_for_live")
            print(f"  {name}: ready_for_live={ready}")
            for key, value in info.items():
                if key == "ready_for_live":
                    continue
                print(f"    - {key}: {value}")
        print()
        print("Live provider calls are not performed by this script.")
        print("Use scripts/run_live_smoke_tests.py after credentials are set.")

    return 1 if base.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
