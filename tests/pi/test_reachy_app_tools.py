"""Pi-only contract tests against the installed Reachy conversation app."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import logging
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import pytest

_reachy_app_python = Path(os.environ.get("REACHY_CONVERSATION_APP_PYTHON", ""))
if _reachy_app_python.is_file() or importlib.util.find_spec("reachy_mini") is not None:
    try:
        import reachy_mini  # noqa: F401
    except ImportError:
        pytest.skip("Reachy Mini SDK is not installed in this interpreter", allow_module_level=True)
else:
    pytest.skip("Reachy conversation app is not installed on this machine", allow_module_level=True)

from reachy_mini_conversation_app.config import DEFAULT_PROFILES_DIRECTORY, config
from reachy_mini_conversation_app.tools import core_tools
from reachy_mini_conversation_app.tools.background_tool_manager import (
    BackgroundToolManager,
    ToolCallRoutine,
    ToolNotification,
)
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TOOLS_ROOT = REPOSITORY_ROOT / "reachy_tools"
TOKEN = "fake-pi-token-that-must-never-appear-in-logs"
NOW = "2026-08-07T20:00:00Z"
LOCAL_NOW = "2026-08-07T16:00:00-04:00"


def _reminder(reminder_id: str = "rem-1") -> dict[str, Any]:
    return {
        "id": reminder_id,
        "text": "Call supervisor",
        "status": "active",
        "channel": "telegram",
        "timezone": "America/Toronto",
        "trigger_at": "2026-08-07T21:00:00Z",
        "trigger_at_local": "2026-08-07 17:00 EDT",
        "original_trigger_at": "2026-08-07T21:00:00Z",
        "created_at": NOW,
        "snooze_count": 0,
    }


def _task() -> dict[str, Any]:
    return {
        "id": "task-1",
        "description": "Finish report",
        "status": "open",
        "priority": "normal",
        "created_at": NOW,
    }


class MockPersonalAssistantServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), MockPersonalAssistantHandler)
        self.requests: list[dict[str, Any]] = []
        self.forced_status: int | None = None
        self.forced_payload: dict[str, Any] | None = None
        self.fail_first_status: int | None = None
        self.malformed = False
        self.delay_seconds = 0.0


class MockPersonalAssistantHandler(BaseHTTPRequestHandler):
    server: MockPersonalAssistantServer

    def log_message(self, format: str, *args: object) -> None:
        del format, args

    def do_GET(self) -> None:
        self._handle()

    def do_POST(self) -> None:
        self._handle()

    def do_PATCH(self) -> None:
        self._handle()

    def _handle(self) -> None:
        if self.server.delay_seconds:
            time.sleep(self.server.delay_seconds)
        length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(length) if length else b""
        body = json.loads(raw_body) if raw_body else None
        parsed = urlsplit(self.path)
        self.server.requests.append(
            {
                "method": self.command,
                "path": parsed.path,
                "headers": dict(self.headers),
                "body": body,
            }
        )

        if self.server.malformed:
            self._send_bytes(200, b"not-json")
            return
        if self.server.fail_first_status is not None:
            status = self.server.fail_first_status
            self.server.fail_first_status = None
            self._send_json(
                status,
                {"ok": False, "error": {"code": "transient", "message": "try again"}},
            )
            return
        if self.server.forced_status is not None:
            payload = self.server.forced_payload or {
                "ok": False,
                "error": {"code": "forced", "message": "forced failure"},
            }
            self._send_json(self.server.forced_status, payload)
            return

        status, payload = self._fixture(self.command, parsed.path, body)
        self._send_json(status, payload)

    def _fixture(
        self, method: str, path: str, body: dict[str, Any] | None
    ) -> tuple[int, dict[str, Any]]:
        if method == "GET" and path == "/api/v1/ping":
            return 200, {
                "pong": True,
                "message": "Personal assistant reachable.",
                "server_time_utc": NOW,
                "server_time_local": LOCAL_NOW,
                "timezone": "America/Toronto",
            }
        if method == "GET" and path == "/api/v1/status":
            return 200, {
                "version": "0.2.0",
                "environment": "test",
                "mock_mode": True,
                "timezone": "America/Toronto",
                "server_time_utc": NOW,
                "database": {"ok": True},
            }
        if path == "/api/v1/reminders" and method == "GET":
            return 200, {"items": [_reminder()], "total": 1, "limit": 5, "offset": 0}
        if path == "/api/v1/reminders" and method == "POST":
            assert body and body["text"] == "Call supervisor"
            return 201, _reminder()
        if path.endswith("/snooze") and method == "POST":
            result = _reminder()
            result["snooze_count"] = 1
            return 200, result
        if path == "/api/v1/tasks" and method == "GET":
            return 200, {"items": [_task()], "total": 1, "limit": 5, "offset": 0}
        if path == "/api/v1/tasks" and method == "POST":
            return 201, _task()
        if path == "/api/v1/tasks/task-1" and method == "PATCH":
            result = _task()
            result["description"] = str((body or {}).get("description", result["description"]))
            return 200, result
        if path == "/api/v1/gmail/search" and method == "GET":
            return 200, {
                "items": [
                    {
                        "id": "msg-1",
                        "thread_id": "thread-1",
                        "sender": "friend@example.com",
                        "sender_name": "Friend",
                        "subject": "Report",
                        "snippet": "The report is ready.",
                        "received_at": NOW,
                    }
                ],
                "total": 1,
            }
        if path == "/api/v1/gmail/messages/msg-1" and method == "GET":
            return 200, {
                "message": {
                    "id": "msg-1",
                    "thread_id": "thread-1",
                    "sender": "friend@example.com",
                    "subject": "Report",
                    "snippet": "The report is ready.",
                    "received_at": NOW,
                    "body_text": "The report is ready for review.",
                },
                "content_is_untrusted": True,
            }
        if path == "/api/v1/gmail/drafts" and method == "POST":
            assert body and body["subject"] == "Draft subject"
            return 201, {
                "draft": {
                    "id": "draft-1",
                    "to": body["to"],
                    "subject": body["subject"],
                    "body": body["body"],
                    "content_hash": "hash-1",
                    "sent": False,
                }
            }
        if path == "/api/v1/gmail/messages/msg-1/reply-draft" and method == "POST":
            assert body and body["body"] == "Thanks for the update."
            return 201, {
                "draft": {
                    "id": "draft-reply-1",
                    "message_id": "msg-1",
                    "thread_id": "thread-1",
                    "to": ["friend@example.com"],
                    "subject": "Re: Report",
                    "body": body["body"],
                    "content_hash": "reply-hash-1",
                    "sent": False,
                }
            }
        if path == "/api/v1/gmail/actions/send-draft" and method == "POST":
            assert body and body["draft_id"] == "draft-1"
            return 202, {
                "approval_id": "approval-1",
                "action_id": "action-1",
                "status": "awaiting_approval",
                "summary": "Send draft",
                "preview": "Draft preview",
                "expires_at": "2026-08-07T21:00:00Z",
            }
        if path == "/api/v1/actions/action-1/status" and method == "GET":
            return 200, {
                "approval_id": "approval-1",
                "action_id": "action-1",
                "action_type": "google.gmail.send_draft",
                "status": "executed",
                "created_at": "2026-08-07T20:00:00Z",
                "executed_at": "2026-08-07T20:01:00Z",
                "result_summary": "sent",
            }
        if path == "/api/v1/calendar/events" and method == "GET":
            return 200, {
                "items": [
                    {
                        "id": "event-1",
                        "title": "Research meeting",
                        "starts_at": "2026-08-08T14:00:00Z",
                        "ends_at": "2026-08-08T15:00:00Z",
                    }
                ],
                "total": 1,
            }
        if path == "/api/v1/calendar/free-busy" and method == "GET":
            return 200, {
                "busy": [],
                "free_slots": [
                    {
                        "starts_at": "2026-08-08T14:00:00Z",
                        "ends_at": "2026-08-08T15:00:00Z",
                    }
                ],
                "queried_from": "2026-08-08T14:00:00Z",
                "queried_to": "2026-08-08T15:00:00Z",
            }
        if path == "/api/v1/calendar/events/propose" and method == "POST":
            return 200, {
                "title": "Coffee",
                "duration_minutes": 30,
                "suggestions": [
                    {
                        "starts_at": "2026-08-08T14:00:00Z",
                        "ends_at": "2026-08-08T14:30:00Z",
                        "conflicts": [],
                    }
                ],
                "preview_text": "Coffee at 10",
                "draft_create_payload": {
                    "title": "Coffee",
                    "starts_at": "2026-08-08T14:00:00Z",
                    "ends_at": "2026-08-08T14:30:00Z",
                },
            }
        if path == "/api/v1/calendar/events" and method == "POST":
            return 202, {
                "approval_id": "approval-calendar-create",
                "action_id": "action-calendar-create",
                "status": "awaiting_approval",
                "summary": "Create event",
                "preview": "Research meeting",
                "expires_at": "2026-08-07T21:00:00Z",
            }
        if path == "/api/v1/calendar/events/event-1" and method == "PATCH":
            return 202, {
                "approval_id": "approval-calendar-update",
                "action_id": "action-calendar-update",
                "status": "awaiting_approval",
                "summary": "Update event",
                "preview": "Updated research meeting",
                "expires_at": "2026-08-07T21:00:00Z",
            }
        if path == "/api/v1/contacts/search" and method == "GET":
            return 200, {
                "items": [
                    {
                        "id": "contact-1",
                        "display_name": "Sam Friend",
                        "emails": ["friend@example.com"],
                    }
                ],
                "total": 1,
            }
        if path == "/api/v1/notion/search" and method == "GET":
            return 200, {
                "items": [{"id": "page-1", "title": "Research notes"}],
                "total": 1,
            }
        if path == "/api/v1/notion/pages/page-1/content" and method == "GET":
            return 200, {
                "page": {"id": "page-1", "title": "Research notes"},
                "text": "Tactile ablation notes.",
                "truncated": False,
                "content_is_untrusted": True,
            }
        if path == "/api/v1/documents/search" and method == "GET":
            return 200, {
                "items": [
                    {
                        "path": "/fixtures/report.pdf",
                        "name": "report.pdf",
                        "kind": "pdf",
                        "snippet": "robotics report",
                        "score": 1.0,
                        "size_bytes": 100,
                        "modified_at": NOW,
                    }
                ],
                "total": 1,
                "query": "robotics",
            }
        if path == "/api/v1/documents/content" and method == "GET":
            return 200, {
                "path": "/fixtures/report.pdf",
                "name": "report.pdf",
                "kind": "pdf",
                "text": "The bounded document text.",
                "truncated": False,
            }
        if path == "/api/v1/weather/current" and method == "GET":
            return 200, {
                "report": {
                    "location": "Toronto",
                    "current": {
                        "location": "Toronto",
                        "latitude": 43.65,
                        "longitude": -79.38,
                        "observed_at": NOW,
                        "temperature_c": 22.0,
                        "feels_like_c": 23.0,
                        "condition": "clear",
                        "description": "Clear sky",
                    },
                    "retrieved_at": NOW,
                },
                "advice": {
                    "warmth_band": "warm",
                    "needs_umbrella": False,
                    "needs_winter_layers": False,
                    "needs_sun_protection": True,
                    "windy": False,
                    "reasons": [],
                    "summary": "Wear light layers.",
                },
            }
        if path == "/api/v1/wardrobe/recommend" and method == "GET":
            return 200, {
                "recommendations": [
                    {
                        "outfit_id": "outfit-1",
                        "outfit_name": "Blue casual",
                        "score": 0.9,
                        "explanation": "It suits the weather.",
                        "components": [],
                    }
                ],
                "considered": 1,
                "weather_summary": "Warm and clear.",
                "temperature_c": 22.0,
            }
        if path == "/api/v1/workstation/status" and method == "GET":
            return 200, {
                "status": {
                    "hostname": "main-pc",
                    "checked_at": NOW,
                    "uptime_seconds": 1000,
                    "cpu_percent": 10.0,
                    "cpu_count": 8,
                    "memory_total_gb": 32.0,
                    "memory_used_gb": 8.0,
                    "memory_percent": 25.0,
                    "swap_percent": 0.0,
                },
                "spoken_summary": "The workstation is healthy.",
            }
        if path == "/api/v1/reachy/pending" and method == "GET":
            return 200, {
                "items": [
                    {
                        "id": "note-1",
                        "kind": "task.complete",
                        "severity": "info",
                        "spoken_text": "Training finished.",
                        "created_at": NOW,
                    }
                ],
                "total": 1,
                "spoken_text": "Training finished.",
            }
        return 404, {"ok": False, "error": {"code": "not_found", "message": path}}

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        self._send_bytes(status, json.dumps(payload).encode())

    def _send_bytes(self, status: int, payload: bytes) -> None:
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        except BrokenPipeError:
            return


@pytest.fixture
def mock_api() -> MockPersonalAssistantServer:
    server = MockPersonalAssistantServer()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.fixture
def registered_tools(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ToolDependencies:
    profile = tmp_path / "pi_profile"
    profile.mkdir()
    (profile / "tools.txt").write_text("reachy_app_tools\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(REPOSITORY_ROOT))
    monkeypatch.setattr(config, "REACHY_MINI_CUSTOM_PROFILE", "pi_profile")
    monkeypatch.setattr(config, "PROFILES_DIRECTORY", tmp_path)
    monkeypatch.setattr(config, "TOOLS_DIRECTORY", TOOLS_ROOT)
    monkeypatch.setattr(config, "AUTOLOAD_EXTERNAL_TOOLS", False)
    core_tools.initialize_tools(force=True)
    yield ToolDependencies(reachy_mini=object(), movement_manager=object())
    config.REACHY_MINI_CUSTOM_PROFILE = None
    config.PROFILES_DIRECTORY = DEFAULT_PROFILES_DIRECTORY
    config.TOOLS_DIRECTORY = None
    config.AUTOLOAD_EXTERNAL_TOOLS = False
    core_tools.initialize_tools(force=True)


def _configure_api(monkeypatch: pytest.MonkeyPatch, server: MockPersonalAssistantServer) -> None:
    host, port = server.server_address
    monkeypatch.setenv("PA_API_URL", f"http://{host}:{port}")
    monkeypatch.setenv("PA_API_TOKEN", TOKEN)
    monkeypatch.setenv("PA_API_TIMEOUT", "1")
    monkeypatch.setenv("PA_API_CONNECT_TIMEOUT", "0.25")


async def _dispatch(name: str, arguments: dict[str, Any], deps: ToolDependencies) -> dict[str, Any]:
    return await core_tools.dispatch_tool_call(name, json.dumps(arguments), deps)


def test_real_registry_loads_unique_schemas(registered_tools: ToolDependencies) -> None:
    del registered_tools
    names = [spec["name"] for spec in core_tools.get_tool_specs()]
    assert len(names) == len(set(names))
    assert "personal_assistant_ping" in names
    assert "create_gmail_draft" in names
    assert "reply_to_email" in names
    assert "send_existing_draft" in names
    assert all(spec["parameters"]["type"] == "object" for spec in core_tools.get_tool_specs())


@pytest.mark.asyncio
async def test_representative_workflows_use_real_tools_and_http(
    registered_tools: ToolDependencies,
    mock_api: MockPersonalAssistantServer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_api(monkeypatch, mock_api)
    calls = [
        ("personal_assistant_ping", {}),
        ("list_reminders", {}),
        ("create_reminder", {"text": "Call supervisor", "delay_minutes": 60}),
        ("snooze_reminder", {"reminder_id": "rem-1", "minutes": 10}),
        ("list_tasks", {}),
        ("create_task", {"description": "Finish report"}),
        ("update_task", {"task_id": "task-1", "description": "Finish final report"}),
        ("search_gmail", {"query": "report"}),
        (
            "create_gmail_draft",
            {
                "to": ["friend@example.com"],
                "subject": "Draft subject",
                "body": "Draft body",
            },
        ),
        ("reply_to_email", {"message_id": "msg-1", "body": "Thanks for the update."}),
        ("send_existing_draft", {"draft_id": "draft-1", "content_hash": "hash-1"}),
        ("get_action_status", {"action_id": "action-1"}),
        ("get_calendar_events", {"days": 7}),
        (
            "check_calendar_free_busy",
            {
                "starts_at": "2026-08-08T14:00:00Z",
                "ends_at": "2026-08-08T15:00:00Z",
            },
        ),
        ("propose_calendar_event", {"title": "Coffee", "duration_minutes": 30}),
        (
            "request_create_calendar_event",
            {
                "title": "Research meeting",
                "starts_at": "2026-08-08T14:00:00Z",
                "ends_at": "2026-08-08T15:00:00Z",
            },
        ),
        (
            "request_update_calendar_event",
            {"event_id": "event-1", "title": "Updated research meeting"},
        ),
        ("search_contacts", {"query": "Sam"}),
        ("search_notion", {"query": "Research"}),
        ("read_notion_page", {"page_id": "page-1"}),
        ("search_documents", {"query": "robotics"}),
        ("read_document", {"path": "/fixtures/report.pdf"}),
        ("get_personal_weather", {"location": "Toronto"}),
        ("recommend_outfit", {"occasion": "work"}),
        ("get_workstation_status", {}),
        ("list_pending_notifications", {}),
    ]
    results = [await _dispatch(name, arguments, registered_tools) for name, arguments in calls]
    assert all(result["ok"] is True for result in results)

    paths = [request["path"] for request in mock_api.requests]
    assert "/api/v1/gmail/drafts" in paths
    assert "/api/v1/gmail/actions/send-draft" in paths
    assert "/api/v1/gmail/send" not in paths
    assert all(
        request["headers"].get("Authorization") == f"Bearer {TOKEN}"
        for request in mock_api.requests
    )

    idempotent_operations = {
        ("POST", "/api/v1/reminders"),
        ("POST", "/api/v1/tasks"),
        ("POST", "/api/v1/gmail/drafts"),
        ("POST", "/api/v1/gmail/actions/send-draft"),
        ("POST", "/api/v1/calendar/events"),
        ("PATCH", "/api/v1/calendar/events/event-1"),
    }
    writes = [
        request
        for request in mock_api.requests
        if (request["method"], request["path"]) in idempotent_operations
    ]
    assert len(writes) == len(idempotent_operations)
    assert all(request["headers"].get("Idempotency-Key") for request in writes)

    by_name = {name: result for (name, _), result in zip(calls, results, strict=True)}
    draft_result = by_name["create_gmail_draft"]
    approval_result = by_name["send_existing_draft"]
    assert draft_result["sent"] is False
    reply_result = by_name["reply_to_email"]
    assert reply_result["sent"] is False
    assert reply_result["message_id"] == "msg-1"
    assert reply_result["thread_id"] == "thread-1"
    assert approval_result["status"] == "awaiting_approval"
    assert approval_result["approval_id"] == "approval-1"
    assert approval_result["action_id"] == "action-1"
    assert by_name["get_action_status"]["spoken"] == "The action was sent and completed."
    assert by_name["request_create_calendar_event"]["status"] == "awaiting_approval"
    assert by_name["request_update_calendar_event"]["status"] == "awaiting_approval"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "expected_code"),
    [
        (401, "authentication_failed"),
        (403, "forbidden"),
        (404, "not_found"),
        (409, "conflict"),
        (422, "invalid_request"),
        (429, "rate_limited"),
        (500, "server_unavailable"),
        (503, "server_unavailable"),
    ],
)
async def test_http_failures_are_sanitized(
    status: int,
    expected_code: str,
    registered_tools: ToolDependencies,
    mock_api: MockPersonalAssistantServer,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _configure_api(monkeypatch, mock_api)
    mock_api.forced_status = status
    mock_api.forced_payload = {
        "ok": False,
        "error": {"code": "conflict" if status == 409 else "failure", "message": TOKEN},
    }
    with caplog.at_level(logging.WARNING):
        result = await _dispatch("personal_assistant_ping", {}, registered_tools)
    assert result["ok"] is False
    assert result["code"] == expected_code
    assert TOKEN not in result["spoken"]
    assert TOKEN not in caplog.text


@pytest.mark.asyncio
async def test_unreachable_server_is_bounded(
    registered_tools: ToolDependencies,
    mock_api: MockPersonalAssistantServer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host, port = mock_api.server_address
    mock_api.shutdown()
    mock_api.server_close()
    monkeypatch.setenv("PA_API_URL", f"http://{host}:{port}")
    monkeypatch.setenv("PA_API_TOKEN", TOKEN)
    monkeypatch.setenv("PA_API_TIMEOUT", "0.1")
    monkeypatch.setenv("PA_API_CONNECT_TIMEOUT", "0.1")
    started = time.monotonic()
    result = await _dispatch("personal_assistant_ping", {}, registered_tools)
    assert result["code"] == "server_unavailable"
    assert time.monotonic() - started < 1.0


@pytest.mark.asyncio
async def test_read_timeout_and_malformed_response(
    registered_tools: ToolDependencies,
    mock_api: MockPersonalAssistantServer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_api(monkeypatch, mock_api)
    monkeypatch.setenv("PA_API_TIMEOUT", "0.05")
    mock_api.delay_seconds = 0.2
    started = time.monotonic()
    timeout_result = await _dispatch("personal_assistant_ping", {}, registered_tools)
    assert timeout_result["code"] == "server_unavailable"
    assert time.monotonic() - started < 0.15

    mock_api.delay_seconds = 0
    mock_api.malformed = True
    malformed_result = await _dispatch("personal_assistant_ping", {}, registered_tools)
    assert malformed_result["code"] == "invalid_response"

    mock_api.malformed = False
    mock_api.forced_status = 200
    mock_api.forced_payload = {"pong": "not-a-boolean"}
    wrong_schema = await _dispatch("personal_assistant_ping", {}, registered_tools)
    assert wrong_schema["code"] == "invalid_response"


@pytest.mark.asyncio
async def test_configuration_errors_do_not_break_registration(
    registered_tools: ToolDependencies, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("PA_API_URL", raising=False)
    monkeypatch.delenv("PA_API_TOKEN", raising=False)
    missing = await _dispatch("personal_assistant_ping", {}, registered_tools)
    assert missing["code"] == "not_configured"

    monkeypatch.setenv("PA_API_URL", "not-a-url")
    monkeypatch.setenv("PA_API_TOKEN", TOKEN)
    malformed = await _dispatch("personal_assistant_ping", {}, registered_tools)
    assert malformed["code"] == "not_configured"
    assert TOKEN not in json.dumps(malformed)

    for invalid_url in ("http://pa.example/api", "http://user:secret@pa.example"):
        monkeypatch.setenv("PA_API_URL", invalid_url)
        invalid = await _dispatch("personal_assistant_ping", {}, registered_tools)
        assert invalid["code"] == "not_configured"

    monkeypatch.setenv("PA_API_URL", "http://pa.example")
    for invalid_timeout in ("nan", "inf", "0", "-1", "61", "not-a-number"):
        monkeypatch.setenv("PA_API_TIMEOUT", invalid_timeout)
        invalid = await _dispatch("personal_assistant_ping", {}, registered_tools)
        assert invalid["code"] == "not_configured"


@pytest.mark.asyncio
async def test_resource_ids_cannot_change_http_routes(
    registered_tools: ToolDependencies,
    mock_api: MockPersonalAssistantServer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_api(monkeypatch, mock_api)
    attacks = [
        ("read_email", {"message_id": "../../reminders/rem-1/complete?"}),
        ("read_email", {"message_id": "msg-1?download=true"}),
        ("update_task", {"task_id": "../task-1", "status": "completed"}),
        ("read_notion_page", {"page_id": "%2e%2e%2fsecrets"}),
    ]
    results = [await _dispatch(name, arguments, registered_tools) for name, arguments in attacks]
    assert all(result["code"] == "invalid_arguments" for result in results)
    assert mock_api.requests == []


@pytest.mark.asyncio
async def test_idempotent_retry_reuses_key_and_patch_is_not_retried(
    registered_tools: ToolDependencies,
    mock_api: MockPersonalAssistantServer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_api(monkeypatch, mock_api)
    mock_api.fail_first_status = 503
    created = await _dispatch(
        "create_reminder",
        {"text": "Call supervisor", "delay_minutes": 60},
        registered_tools,
    )
    assert created["ok"] is True
    assert len(mock_api.requests) == 2
    retry_keys = [request["headers"].get("Idempotency-Key") for request in mock_api.requests]
    assert retry_keys[0] and retry_keys[0] == retry_keys[1]

    mock_api.requests.clear()
    mock_api.forced_status = 503
    update = await _dispatch(
        "request_update_calendar_event",
        {"event_id": "event-1", "title": "Updated research meeting"},
        registered_tools,
    )
    assert update["code"] == "server_unavailable"
    assert len(mock_api.requests) == 1


@pytest.mark.asyncio
async def test_async_http_yields_to_the_realtime_event_loop(
    registered_tools: ToolDependencies,
    mock_api: MockPersonalAssistantServer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_api(monkeypatch, mock_api)
    mock_api.delay_seconds = 0.15
    tool_task = asyncio.create_task(_dispatch("personal_assistant_ping", {}, registered_tools))
    tick_started = time.monotonic()
    await asyncio.sleep(0.02)
    tick_elapsed = time.monotonic() - tick_started
    result = await tool_task
    assert tick_elapsed < 0.08
    assert result["ok"] is True


@pytest.mark.asyncio
async def test_background_manager_delivers_real_tool_result(
    registered_tools: ToolDependencies,
    mock_api: MockPersonalAssistantServer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_api(monkeypatch, mock_api)
    manager = BackgroundToolManager()
    completed = asyncio.Event()
    notifications: list[ToolNotification] = []

    async def receive(notification: ToolNotification) -> None:
        notifications.append(notification)
        completed.set()

    manager.start_up([receive])
    await manager.start_tool(
        "call-1",
        ToolCallRoutine(
            tool_name="personal_assistant_ping",
            args_json_str="{}",
            deps=registered_tools,
        ),
        is_idle_tool_call=False,
    )
    await asyncio.wait_for(completed.wait(), timeout=1)
    await manager.shutdown()
    assert notifications[0].result is not None
    assert notifications[0].result["ok"] is True
