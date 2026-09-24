#!/usr/bin/env python3
"""Validate the configuration and report what is ready, missing or mocked.

    python scripts/check_configuration.py
    python scripts/check_configuration.py --json

Output is always redacted: it reports whether a secret is set and how long it
Uses the configured workflow.
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
from security.keyring import resolve_secret_key  # noqa: E402
from shared.errors import ConfigurationError  # noqa: E402

OK = "  ok  "
WARN = " warn "
FAIL = " fail "


class Report:
    def __init__(self) -> None:
        self.entries: list[dict[str, Any]] = []
        self.failed = False

    def add(self, level: str, area: str, message: str, hint: str | None = None) -> None:
        self.entries.append(
            {"level": level.strip(), "area": area, "message": message, "hint": hint}
        )
        if level is FAIL:
            self.failed = True

    def render(self) -> str:
        lines = []
        for entry in self.entries:
            level = {"ok": OK, "warn": WARN, "fail": FAIL}[entry["level"]]
            lines.append(f"[{level}] {entry['area']:<22} {entry['message']}")
            if entry["hint"]:
                lines.append(f"{'':>9}{entry['hint']}")
        return "\n".join(lines)


def check(settings: Settings) -> Report:
    report = Report()

    report.add(OK, "environment", f"APP_ENV={settings.app_env.value}")
    report.add(OK, "timezone", settings.app_timezone)

    data_dir = settings.app_data_dir
    if data_dir.exists():
        writable = _is_writable(data_dir)
        report.add(
            OK if writable else FAIL,
            "data directory",
            f"{data_dir} ({'writable' if writable else 'NOT writable'})",
            None if writable else "Create it and grant the service user write access.",
        )
    else:
        report.add(
            WARN,
            "data directory",
            f"{data_dir} does not exist yet",
            "It is created on first start, or run: python scripts/init_database.py",
        )

    token = settings.pa_api_token.get_secret_value()
    if not token:
        report.add(
            FAIL, "api token", "PA_API_TOKEN is unset", "Run: python scripts/generate_api_token.py"
        )
    elif looks_like_placeholder(token) or len(token) < 32:
        level = FAIL if settings.is_production else WARN
        report.add(
            level,
            "api token",
            "PA_API_TOKEN is a placeholder or shorter than 32 characters",
            "Run: python scripts/generate_api_token.py --write",
        )
    else:
        report.add(OK, "api token", f"set ({len(token)} chars)")

    report.add(OK, "allowed networks", ", ".join(settings.pa_api_allowed_networks) or "any")
    if settings.app_host == "0.0.0.0" and settings.is_production:  # noqa: S104
        report.add(
            WARN,
            "bind address",
            "APP_HOST=0.0.0.0 exposes the API on every interface",
            "Prefer a Tailscale address or a reverse proxy. See SECURITY.md.",
        )

    _check_secret_key(settings, report)
    _check_providers(settings, report)
    _check_paths(settings, report)
    return report


def _check_secret_key(settings: Settings, report: Report) -> None:
    source = settings.secret_key_source()
    if source == "unset":
        needs_key = settings.google_enabled and not settings.mock_mode
        report.add(
            FAIL if needs_key else WARN,
            "encryption key",
            "No encryption key is configured, so OAuth tokens cannot be stored",
            "Run: python scripts/generate_api_token.py --secret-key",
        )
        return
    try:
        resolve_secret_key(settings)
    except ConfigurationError as exc:
        report.add(FAIL, "encryption key", str(exc).split("\n")[0])
        return
    report.add(OK, "encryption key", f"valid, source={source}")
    if source == "environment" and settings.is_production:
        report.add(
            WARN,
            "encryption key",
            "The key sits in the environment in production",
            "Prefer systemd LoadCredential or PA_SECRET_KEY_FILE outside APP_DATA_DIR.",
        )


def _check_providers(settings: Settings, report: Report) -> None:
    if settings.mock_mode:
        report.add(
            WARN if settings.is_production else OK,
            "mock mode",
            "MOCK_MODE=true - every external provider is simulated",
        )

    providers = [
        (
            "telegram",
            settings.telegram_enabled,
            bool(settings.telegram_bot_token.get_secret_value()),
            "TELEGRAM_BOT_TOKEN from @BotFather. See docs/telegram_setup.md.",
        ),
        (
            "google",
            settings.google_enabled,
            bool(settings.google_client_id and settings.google_client_secret.get_secret_value()),
            "GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET. See docs/google_setup.md.",
        ),
        (
            "notion",
            settings.notion_enabled,
            bool(settings.notion_token.get_secret_value()),
            "NOTION_TOKEN. See docs/notion_setup.md.",
        ),
    ]
    for name, enabled, configured, hint in providers:
        if not enabled:
            report.add(OK, name, "disabled - the mock provider will be used")
        elif configured:
            report.add(OK, name, "enabled and configured")
        else:
            report.add(FAIL, name, "enabled but its credentials are missing", hint)

    if settings.telegram_enabled and not settings.telegram_allowed_chat_ids:
        report.add(
            FAIL,
            "telegram",
            "TELEGRAM_ALLOWED_CHAT_IDS is empty, so no chat may approve anything",
            "Message the bot, then read the chat id from docs/telegram_setup.md.",
        )

    provider = settings.weather_provider.value
    if provider == "openweathermap" and not settings.weather_api_key.get_secret_value():
        report.add(FAIL, "weather", "openweathermap needs WEATHER_API_KEY")
    else:
        report.add(OK, "weather", f"provider={provider} (open_meteo needs no key)")


def _check_paths(settings: Settings, report: Report) -> None:
    if settings.workstation_script_registry.exists():
        report.add(OK, "script registry", str(settings.workstation_script_registry))
    else:
        report.add(
            WARN,
            "script registry",
            f"{settings.workstation_script_registry} not found - no local scripts can run",
            "Copy config/registered_scripts.example.yaml to that path.",
        )

    if settings.document_index_roots:
        missing = [
            root for root in settings.document_index_roots if not Path(root).expanduser().is_dir()
        ]
        if missing:
            report.add(WARN, "document roots", f"missing: {', '.join(missing)}")
        else:
            report.add(OK, "document roots", f"{len(settings.document_index_roots)} configured")
    else:
        report.add(
            WARN, "document roots", "DOCUMENT_INDEX_ROOTS is empty; document search is inert"
        )


def _is_writable(path: Path) -> bool:
    import os

    return os.access(path, os.W_OK)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable output.")
    args = parser.parse_args()

    try:
        settings = Settings()
    except Exception as exc:
        message = str(exc)
        if args.json:
            print(json.dumps({"ok": False, "error": message}, indent=2))
        else:
            print(f"[{FAIL}] configuration could not be loaded:\n{message}")
        return 1

    report = check(settings)

    if args.json:
        print(
            json.dumps(
                {
                    "ok": not report.failed,
                    "checks": report.entries,
                    "settings": settings.redacted_diagnostics(),
                },
                indent=2,
                default=str,
            )
        )
    else:
        print(report.render())
        print()
        print(
            "FAILED - the service would refuse to start."
            if report.failed
            else "Configuration is usable."
        )
    return 1 if report.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
