#!/usr/bin/env python3
"""Non-destructive live smoke checks against configured providers.

    python scripts/run_live_smoke_tests.py
    python scripts/run_live_smoke_tests.py --allow-external-write

By default only read-only / status probes run. Actions that could create
mail, calendar events, or Notion blocks require
``--allow-external-write`` and still raise approval tickets for manual review.

A provider passes only when the remote call succeeds in this run.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app.config import Settings  # noqa: E402


class SmokeReport:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str]] = []

    def ok(self, name: str, detail: str) -> None:
        self.rows.append((name, "PASS", detail))

    def skip(self, name: str, detail: str) -> None:
        self.rows.append((name, "SKIP", detail))

    def fail(self, name: str, detail: str) -> None:
        self.rows.append((name, "FAIL", detail))

    @property
    def failed(self) -> bool:
        return any(status == "FAIL" for _, status, _ in self.rows)

    def render(self) -> str:
        lines = []
        for name, status, detail in self.rows:
            lines.append(f"[{status}] {name}: {detail}")
        return "\n".join(lines)


async def _http_get(url: str, token: str | None = None) -> httpx.Response:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    async with httpx.AsyncClient(timeout=15.0) as client:
        return await client.get(url, headers=headers)


async def check_local_api(settings: Settings, report: SmokeReport) -> None:
    base = f"http://{settings.app_host}:{settings.app_port}"
    try:
        health = await _http_get(f"{base}/health")
        if health.status_code == 200:
            report.ok("api.health", f"{base}/health")
        else:
            report.fail("api.health", f"status={health.status_code}")
            return
        ready = await _http_get(f"{base}/ready")
        if ready.status_code == 200:
            report.ok("api.ready", "ready")
        else:
            report.fail("api.ready", f"status={ready.status_code}")
        token = settings.pa_api_token.get_secret_value()
        ping = await _http_get(f"{base}/api/v1/ping", token=token)
        if ping.status_code == 200:
            report.ok("api.ping", "authenticated")
        else:
            report.fail("api.ping", f"status={ping.status_code}")
    except httpx.HTTPError as exc:
        report.fail("api.health", f"unreachable: {exc}")


async def check_telegram(settings: Settings, report: SmokeReport) -> None:
    if not settings.telegram_enabled or settings.mock_mode:
        report.skip("telegram.getMe", "disabled or mock mode")
        return
    token = settings.telegram_bot_token.get_secret_value()
    if not token:
        report.skip("telegram.getMe", "token missing")
        return
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(f"https://api.telegram.org/bot{token}/getMe")
        if response.status_code == 200 and response.json().get("ok"):
            username = response.json().get("result", {}).get("username", "?")
            report.ok("telegram.getMe", f"@{username}")
        else:
            report.fail("telegram.getMe", f"status={response.status_code}")
    except httpx.HTTPError as exc:
        report.fail("telegram.getMe", str(exc))


async def check_google(settings: Settings, report: SmokeReport) -> None:
    if not settings.google_enabled or settings.mock_mode:
        report.skip("google.status", "disabled or mock mode")
        return
    base = f"http://{settings.app_host}:{settings.app_port}"
    token = settings.pa_api_token.get_secret_value()
    try:
        response = await _http_get(f"{base}/api/v1/integrations/google", token=token)
        if response.status_code != 200:
            report.fail("google.status", f"status={response.status_code}")
            return
        body = response.json()
        authorised = bool(body.get("authorised") or body.get("authorized"))
        if authorised:
            report.ok("google.status", "authorised grant present")
        else:
            report.fail("google.status", "not authorised — run setup_google_oauth.py")
    except httpx.HTTPError as exc:
        report.fail("google.status", str(exc))


async def check_notion(settings: Settings, report: SmokeReport) -> None:
    if not settings.notion_enabled or settings.mock_mode:
        report.skip("notion.search", "disabled or mock mode")
        return
    if not settings.notion_allowed_page_ids:
        report.skip("notion.search", "no allowlisted pages")
        return
    base = f"http://{settings.app_host}:{settings.app_port}"
    token = settings.pa_api_token.get_secret_value()
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.get(
                f"{base}/api/v1/notion/search",
                params={"q": "a", "limit": 1},
                headers={"Authorization": f"Bearer {token}"},
            )
        if response.status_code == 200:
            report.ok("notion.search", "read succeeded")
        else:
            report.fail("notion.search", f"status={response.status_code}")
    except httpx.HTTPError as exc:
        report.fail("notion.search", str(exc))


async def check_external_write_gate(settings: Settings, report: SmokeReport, allow: bool) -> None:
    if not allow:
        report.skip(
            "external_write",
            "pass --allow-external-write to create approval tickets only",
        )
        return
    report.skip(
        "external_write",
        "flag set but no automatic approve — create tickets manually via API",
    )


async def main_async(allow_external_write: bool) -> int:
    settings = Settings()
    report = SmokeReport()
    if settings.mock_mode:
        report.skip("live_mode", "MOCK_MODE=true — refusing to claim live provider results")
    await check_local_api(settings, report)
    await check_telegram(settings, report)
    await check_google(settings, report)
    await check_notion(settings, report)
    await check_external_write_gate(settings, report, allow_external_write)
    print(report.render())
    return 1 if report.failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-external-write",
        action="store_true",
        help="Acknowledge that approval-gated write tests may be invoked manually.",
    )
    args = parser.parse_args()
    return asyncio.run(main_async(args.allow_external_write))


if __name__ == "__main__":
    raise SystemExit(main())
