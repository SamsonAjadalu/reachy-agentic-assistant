"""Reachy conversation-app tools for the Reachy Agentic Assistant API."""

from __future__ import annotations

import abc
import asyncio
import builtins
import json
import logging
import math
import os
import re
import uuid
from collections.abc import Mapping
from datetime import datetime
from typing import Any
from urllib.parse import urlsplit

from pydantic import ValidationError as ResponseValidationError
from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies

from reachy_client.client import AsyncPersonalAssistantClient
from reachy_client.exceptions import (
    ApiError,
    AuthError,
    NotFoundError,
    RateLimitError,
    ServiceUnavailable,
    ValidationError,
)
from reachy_client.exceptions import TimeoutError as ClientTimeoutError
from reachy_client.models import (
    CreateDraftRequest,
    CreateEventRequest,
    CreateReplyDraftRequest,
    Delay,
    ProposeEventRequest,
    ReminderCreate,
    ReminderSnooze,
    ScheduleInput,
    SendDraftRequest,
    TaskCreate,
    TaskUpdate,
    UpdateEventRequest,
)

logger = logging.getLogger(__name__)

_DEFAULT_CONNECT_TIMEOUT = 3.0
_DEFAULT_REQUEST_TIMEOUT = 10.0
_MAX_TIMEOUT = 60.0
_RESOURCE_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:@+-]*\Z")


class BridgeConfigurationError(ValueError):
    """The Pi bridge environment is absent or invalid."""


class ToolArgumentError(ValueError):
    """A model-supplied tool argument is absent or invalid."""


def _positive_timeout(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise BridgeConfigurationError(f"{name} must be a number") from exc
    if not math.isfinite(value) or value <= 0 or value > _MAX_TIMEOUT:
        raise BridgeConfigurationError(
            f"{name} must be greater than zero and at most {_MAX_TIMEOUT:g}"
        )
    return value


def _new_client() -> AsyncPersonalAssistantClient:
    base_url = os.getenv("PA_API_URL", "").strip()
    token = os.getenv("PA_API_TOKEN", "").strip()
    if not base_url or not token:
        raise BridgeConfigurationError("PA_API_URL and PA_API_TOKEN must both be configured")

    parsed = urlsplit(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise BridgeConfigurationError("PA_API_URL must be an absolute HTTP or HTTPS URL")
    if (
        parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise BridgeConfigurationError(
            "PA_API_URL must be a server origin without credentials, a path, a query, or a fragment"
        )

    request_timeout = _positive_timeout("PA_API_TIMEOUT", _DEFAULT_REQUEST_TIMEOUT)
    connect_timeout = _positive_timeout("PA_API_CONNECT_TIMEOUT", _DEFAULT_CONNECT_TIMEOUT)
    return AsyncPersonalAssistantClient(
        base_url,
        token,
        connect_timeout=connect_timeout,
        read_timeout=request_timeout,
        write_read_timeout=request_timeout,
    )


def _failure(exc: Exception) -> dict[str, Any]:
    logger.warning("Personal-assistant tool failed safely (%s)", type(exc).__name__)
    if isinstance(exc, BridgeConfigurationError):
        return {
            "ok": False,
            "code": "not_configured",
            "spoken": "The personal assistant connection is not configured on this robot.",
        }
    if isinstance(exc, AuthError):
        return {
            "ok": False,
            "code": "authentication_failed",
            "spoken": "The personal assistant server rejected the configured credentials.",
        }
    if isinstance(exc, NotFoundError):
        return {
            "ok": False,
            "code": "not_found",
            "spoken": "The requested personal assistant item was not found.",
        }
    if isinstance(exc, ValidationError):
        return {
            "ok": False,
            "code": "invalid_request",
            "spoken": "The personal assistant server could not accept that request.",
        }
    if isinstance(exc, RateLimitError):
        return {
            "ok": False,
            "code": "rate_limited",
            "spoken": "The personal assistant server is busy. Please try again shortly.",
        }
    if isinstance(exc, ClientTimeoutError | builtins.TimeoutError | ServiceUnavailable):
        return {
            "ok": False,
            "code": "server_unavailable",
            "spoken": "The personal assistant server is currently unavailable.",
        }
    if isinstance(exc, ResponseValidationError | json.JSONDecodeError):
        return {
            "ok": False,
            "code": "invalid_response",
            "spoken": "The personal assistant server returned an invalid response.",
        }
    if isinstance(exc, ApiError):
        if exc.status_code == 403:
            code = "forbidden"
            spoken = "The personal assistant server did not permit that request."
        elif exc.status_code == 409:
            code = "conflict"
            spoken = "That request conflicts with the current personal assistant state."
        else:
            code = "api_error"
            spoken = "The personal assistant server could not complete that request."
        return {"ok": False, "code": code, "spoken": spoken}
    return {
        "ok": False,
        "code": "client_error",
        "spoken": "The personal assistant request could not be completed.",
    }


def _required_text(arguments: Mapping[str, Any], name: str, *, max_length: int = 4000) -> str:
    value = arguments.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ToolArgumentError(f"{name} must be a non-empty string")
    result = value.strip()
    if len(result) > max_length:
        raise ToolArgumentError(f"{name} is too long")
    return result


def _resource_id(arguments: Mapping[str, Any], name: str, *, max_length: int = 200) -> str:
    """Return a path-safe opaque identifier supplied by the realtime model."""
    value = _required_text(arguments, name, max_length=max_length)
    if value in {".", ".."} or _RESOURCE_ID_PATTERN.fullmatch(value) is None:
        raise ToolArgumentError(f"{name} contains unsupported characters")
    return value


def _optional_text(
    arguments: Mapping[str, Any], name: str, *, max_length: int = 4000
) -> str | None:
    value = arguments.get(name)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ToolArgumentError(f"{name} must be a string")
    result = value.strip()
    if not result or len(result) > max_length:
        raise ToolArgumentError(
            f"{name} must be non-empty and no longer than {max_length} characters"
        )
    return result


def _bounded_int(arguments: Mapping[str, Any], name: str, default: int, low: int, high: int) -> int:
    value = arguments.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ToolArgumentError(f"{name} must be an integer from {low} to {high}")
    return value


def _datetime(arguments: Mapping[str, Any], name: str, *, required: bool = True) -> datetime | None:
    value = arguments.get(name)
    if value is None and not required:
        return None
    if not isinstance(value, str):
        raise ToolArgumentError(f"{name} must be an ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ToolArgumentError(f"{name} must be an ISO-8601 string") from exc
    if parsed.tzinfo is None:
        raise ToolArgumentError(f"{name} must include a timezone offset")
    return parsed


def _string_list(
    arguments: Mapping[str, Any], name: str, *, required: bool = False, max_items: int = 20
) -> list[str]:
    value = arguments.get(name)
    if value is None and not required:
        return []
    if (
        not isinstance(value, list)
        or (required and not value)
        or len(value) > max_items
        or not all(isinstance(item, str) and item.strip() for item in value)
    ):
        raise ToolArgumentError(f"{name} must be a non-empty list of at most {max_items} strings")
    return [item.strip() for item in value]


class PersonalAssistantTool(Tool, abc.ABC):
    """Base class that owns one bounded async client per tool invocation."""

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> dict[str, Any]:
        del deps
        try:
            total_timeout = _positive_timeout("PA_API_TIMEOUT", _DEFAULT_REQUEST_TIMEOUT)
            async with asyncio.timeout(total_timeout):
                async with _new_client() as client:
                    return await self.run(client, kwargs)
        except ToolArgumentError as exc:
            logger.info("Rejected invalid arguments for %s: %s", self.name, exc)
            return {
                "ok": False,
                "code": "invalid_arguments",
                "spoken": "The tool arguments were invalid. Please clarify the request.",
            }
        except Exception as exc:
            return _failure(exc)

    @abc.abstractmethod
    async def run(
        self, client: AsyncPersonalAssistantClient, arguments: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Validate arguments, call the API, and create a concise model-visible result."""


class PersonalAssistantPing(PersonalAssistantTool):
    name = "personal_assistant_ping"
    description = "Check whether Reachy Agentic Assistant is reachable."
    parameters_schema = {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    }

    async def run(
        self, client: AsyncPersonalAssistantClient, arguments: Mapping[str, Any]
    ) -> dict[str, Any]:
        del arguments
        result = await client.ping()
        return {"ok": True, "spoken": result.message, "timezone": result.timezone}


class PersonalAssistantStatus(PersonalAssistantTool):
    name = "personal_assistant_status"
    description = "Get the health summary of Reachy Agentic Assistant."
    parameters_schema = PersonalAssistantPing.parameters_schema

    async def run(
        self, client: AsyncPersonalAssistantClient, arguments: Mapping[str, Any]
    ) -> dict[str, Any]:
        del arguments
        result = await client.status()
        return {
            "ok": True,
            "spoken": f"The personal assistant is running version {result.version}.",
            "mock_mode": result.mock_mode,
            "database_ok": bool(result.database.get("ok")),
        }


class CreateReminder(PersonalAssistantTool):
    name = "create_reminder"
    description = (
        "Create a reminder on the workstation using either a relative delay "
        "or an exact ISO-8601 time."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "text": {"type": "string", "minLength": 1, "maxLength": 2000},
            "delay_minutes": {"type": "integer", "minimum": 1, "maximum": 525600},
            "at": {"type": "string", "description": "ISO-8601 time with timezone offset"},
        },
        "required": ["text"],
        "additionalProperties": False,
    }

    async def run(
        self, client: AsyncPersonalAssistantClient, arguments: Mapping[str, Any]
    ) -> dict[str, Any]:
        text = _required_text(arguments, "text", max_length=2000)
        has_delay = arguments.get("delay_minutes") is not None
        has_at = arguments.get("at") is not None
        if has_delay == has_at:
            raise ToolArgumentError("provide exactly one of delay_minutes or at")
        schedule = (
            ScheduleInput(
                delay=Delay(minutes=_bounded_int(arguments, "delay_minutes", 0, 1, 525600))
            )
            if has_delay
            else ScheduleInput(at=_datetime(arguments, "at"))
        )
        result = await client.create_reminder(
            ReminderCreate(text=text, schedule=schedule), idempotency_key=uuid.uuid4().hex
        )
        return {
            "ok": True,
            "spoken": f"Reminder set for {result.trigger_at_local}.",
            "reminder_id": result.id,
            "trigger_at_local": result.trigger_at_local,
        }


class ListReminders(PersonalAssistantTool):
    name = "list_reminders"
    description = "List upcoming reminders stored by Reachy Agentic Assistant."
    parameters_schema = {
        "type": "object",
        "properties": {"limit": {"type": "integer", "minimum": 1, "maximum": 20}},
        "required": [],
        "additionalProperties": False,
    }

    async def run(
        self, client: AsyncPersonalAssistantClient, arguments: Mapping[str, Any]
    ) -> dict[str, Any]:
        limit = _bounded_int(arguments, "limit", 5, 1, 20)
        result = await client.list_reminders(upcoming_only=True, limit=limit)
        if not result.items:
            return {"ok": True, "spoken": "You have no upcoming reminders.", "count": 0}
        summary = "; ".join(
            f"{item.text} at {item.trigger_at_local}" for item in result.items[:limit]
        )
        return {
            "ok": True,
            "spoken": f"You have {result.total} reminder(s). {summary}.",
            "count": result.total,
            "items": [
                {"id": item.id, "text": item.text, "when": item.trigger_at_local}
                for item in result.items
            ],
        }


class SnoozeReminder(PersonalAssistantTool):
    name = "snooze_reminder"
    description = "Snooze an existing reminder by a bounded number of minutes."
    parameters_schema = {
        "type": "object",
        "properties": {
            "reminder_id": {"type": "string", "minLength": 1},
            "minutes": {"type": "integer", "minimum": 1, "maximum": 10080},
        },
        "required": ["reminder_id"],
        "additionalProperties": False,
    }

    async def run(
        self, client: AsyncPersonalAssistantClient, arguments: Mapping[str, Any]
    ) -> dict[str, Any]:
        reminder_id = _resource_id(arguments, "reminder_id", max_length=100)
        minutes = _bounded_int(arguments, "minutes", 10, 1, 10080)
        result = await client.snooze_reminder(
            reminder_id, ReminderSnooze(delay=Delay(minutes=minutes))
        )
        return {
            "ok": True,
            "spoken": f"Snoozed for {minutes} minutes, until {result.trigger_at_local}.",
            "reminder_id": result.id,
        }


class CreateTask(PersonalAssistantTool):
    name = "create_task"
    description = "Create a task in Reachy Agentic Assistant."
    parameters_schema = {
        "type": "object",
        "properties": {"description": {"type": "string", "minLength": 1, "maxLength": 2000}},
        "required": ["description"],
        "additionalProperties": False,
    }

    async def run(
        self, client: AsyncPersonalAssistantClient, arguments: Mapping[str, Any]
    ) -> dict[str, Any]:
        description = _required_text(arguments, "description", max_length=2000)
        result = await client.create_task(
            TaskCreate(description=description), idempotency_key=uuid.uuid4().hex
        )
        return {"ok": True, "spoken": f"Added task: {result.description}.", "task_id": result.id}


class ListTasks(PersonalAssistantTool):
    name = "list_tasks"
    description = "List tasks from Reachy Agentic Assistant."
    parameters_schema = {
        "type": "object",
        "properties": {
            "status": {
                "type": "string",
                "enum": ["open", "in_progress", "completed", "cancelled"],
            },
            "limit": {"type": "integer", "minimum": 1, "maximum": 20},
        },
        "required": [],
        "additionalProperties": False,
    }

    async def run(
        self, client: AsyncPersonalAssistantClient, arguments: Mapping[str, Any]
    ) -> dict[str, Any]:
        status = _optional_text(arguments, "status", max_length=20) or "open"
        if status not in {"open", "in_progress", "completed", "cancelled"}:
            raise ToolArgumentError("unsupported task status")
        limit = _bounded_int(arguments, "limit", 5, 1, 20)
        result = await client.list_tasks(status=status, limit=limit)
        if not result.items:
            return {"ok": True, "spoken": f"You have no {status} tasks.", "count": 0}
        return {
            "ok": True,
            "spoken": (
                f"{result.total} {status} task(s): "
                f"{'; '.join(item.description for item in result.items[:limit])}."
            ),
            "count": result.total,
            "items": [
                {"id": item.id, "description": item.description, "status": item.status}
                for item in result.items
            ],
        }


class UpdateTask(PersonalAssistantTool):
    name = "update_task"
    description = "Update the description, priority, status, or due time of an existing task."
    parameters_schema = {
        "type": "object",
        "properties": {
            "task_id": {"type": "string", "minLength": 1},
            "description": {"type": "string", "minLength": 1, "maxLength": 2000},
            "priority": {"type": "string", "enum": ["low", "normal", "high", "urgent"]},
            "status": {
                "type": "string",
                "enum": ["open", "in_progress", "completed", "cancelled"],
            },
            "due_at": {"type": "string", "description": "ISO-8601 time with timezone offset"},
            "clear_due_at": {"type": "boolean"},
        },
        "required": ["task_id"],
        "additionalProperties": False,
    }

    async def run(
        self, client: AsyncPersonalAssistantClient, arguments: Mapping[str, Any]
    ) -> dict[str, Any]:
        task_id = _resource_id(arguments, "task_id", max_length=100)
        description = _optional_text(arguments, "description", max_length=2000)
        priority = _optional_text(arguments, "priority", max_length=20)
        status = _optional_text(arguments, "status", max_length=20)
        due_at = _datetime(arguments, "due_at", required=False)
        clear_due_at = arguments.get("clear_due_at", False)
        if not isinstance(clear_due_at, bool):
            raise ToolArgumentError("clear_due_at must be a boolean")
        if due_at is not None and clear_due_at:
            raise ToolArgumentError("due_at and clear_due_at cannot both be supplied")
        if (
            all(value is None for value in (description, priority, status, due_at))
            and not clear_due_at
        ):
            raise ToolArgumentError("no task update was supplied")
        result = await client.update_task(
            task_id,
            TaskUpdate(
                description=description,
                priority=priority,
                status=status,
                due_at=due_at,
                clear_due_at=clear_due_at,
            ),
        )
        return {"ok": True, "spoken": f"Updated task: {result.description}.", "task_id": result.id}


class SearchGmail(PersonalAssistantTool):
    name = "search_gmail"
    description = "Search Gmail through Reachy Agentic Assistant."
    parameters_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "minLength": 1, "maxLength": 200},
            "limit": {"type": "integer", "minimum": 1, "maximum": 10},
        },
        "required": ["query"],
        "additionalProperties": False,
    }

    async def run(
        self, client: AsyncPersonalAssistantClient, arguments: Mapping[str, Any]
    ) -> dict[str, Any]:
        result = await client.search_gmail(
            _required_text(arguments, "query", max_length=200),
            limit=_bounded_int(arguments, "limit", 5, 1, 10),
        )
        if not result.items:
            return {"ok": True, "spoken": "No messages matched that search.", "count": 0}
        messages = [
            {"id": item.id, "subject": item.subject, "from": item.sender} for item in result.items
        ]
        summary = "; ".join(
            f"{item.sender_name or item.sender}: {item.subject}" for item in result.items
        )
        return {
            "ok": True,
            "spoken": f"Found {result.total} message(s). {summary}.",
            "count": result.total,
            "messages": messages,
        }


class ReadEmail(PersonalAssistantTool):
    name = "read_email"
    description = "Read a bounded excerpt from one Gmail message returned by Gmail search."
    parameters_schema = {
        "type": "object",
        "properties": {"message_id": {"type": "string", "minLength": 1}},
        "required": ["message_id"],
        "additionalProperties": False,
    }

    async def run(
        self, client: AsyncPersonalAssistantClient, arguments: Mapping[str, Any]
    ) -> dict[str, Any]:
        result = await client.read_email(
            _resource_id(arguments, "message_id", max_length=200), max_chars=1000
        )
        message = result.message
        return {
            "ok": True,
            "spoken": (
                f"From {message.sender_name or message.sender}. "
                f"Subject: {message.subject}. {message.body_text[:500]}"
            ),
            "message_id": message.id,
            "truncated": message.truncated,
            "content_is_untrusted": True,
        }


class CreateGmailDraft(PersonalAssistantTool):
    name = "create_gmail_draft"
    description = (
        "Create a brand-new Gmail email draft only. Use reply_to_email for replying to "
        "an existing message. Email sending follows the approval flow."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "to": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 10},
            "subject": {"type": "string", "minLength": 1, "maxLength": 500},
            "body": {"type": "string", "minLength": 1, "maxLength": 20000},
            "cc": {"type": "array", "items": {"type": "string"}, "maxItems": 10},
        },
        "required": ["to", "subject", "body"],
        "additionalProperties": False,
    }

    async def run(
        self, client: AsyncPersonalAssistantClient, arguments: Mapping[str, Any]
    ) -> dict[str, Any]:
        payload = CreateDraftRequest(
            to=_string_list(arguments, "to", required=True, max_items=10),
            cc=_string_list(arguments, "cc", max_items=10),
            subject=_required_text(arguments, "subject", max_length=500),
            body=_required_text(arguments, "body", max_length=20000),
        )
        result = await client.create_gmail_draft(payload, idempotency_key=uuid.uuid4().hex)
        draft = result.draft
        return {
            "ok": True,
            "spoken": "The Gmail draft was saved. Nothing has been sent.",
            "draft_id": draft.id,
            "content_hash": draft.content_hash,
            "sent": draft.sent,
        }


class ReplyToEmail(PersonalAssistantTool):
    name = "reply_to_email"
    description = (
        "Create a reply draft for an existing Gmail message in its original thread. "
        "The server derives the recipient and thread from message_id; "
        "email sending follows the approval flow."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "message_id": {"type": "string", "minLength": 1, "maxLength": 200},
            "body": {"type": "string", "minLength": 1, "maxLength": 20000},
        },
        "required": ["message_id", "body"],
        "additionalProperties": False,
    }

    async def run(
        self, client: AsyncPersonalAssistantClient, arguments: Mapping[str, Any]
    ) -> dict[str, Any]:
        result = await client.create_gmail_reply_draft(
            _resource_id(arguments, "message_id", max_length=200),
            CreateReplyDraftRequest(body=_required_text(arguments, "body", max_length=20000)),
            idempotency_key=uuid.uuid4().hex,
        )
        draft = result.draft
        return {
            "ok": True,
            "spoken": "The reply draft was saved in the same email thread. Nothing has been sent.",
            "draft_id": draft.id,
            "content_hash": draft.content_hash,
            "message_id": draft.message_id,
            "thread_id": draft.thread_id,
            "sent": draft.sent,
        }


class SendExistingDraft(PersonalAssistantTool):
    name = "send_existing_draft"
    description = (
        "Request owner approval to send an existing Gmail draft revision through the approval flow."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "draft_id": {"type": "string", "minLength": 1},
            "content_hash": {"type": "string", "minLength": 1, "maxLength": 128},
        },
        "required": ["draft_id"],
        "additionalProperties": False,
    }

    async def run(
        self, client: AsyncPersonalAssistantClient, arguments: Mapping[str, Any]
    ) -> dict[str, Any]:
        ticket = await client.send_existing_draft(
            SendDraftRequest(
                draft_id=_resource_id(arguments, "draft_id", max_length=200),
                content_hash=_optional_text(arguments, "content_hash", max_length=128),
            ),
            idempotency_key=uuid.uuid4().hex,
        )
        return {
            "ok": True,
            "spoken": (
                "Send approval is pending. Nothing will be sent until the exact draft is approved."
            ),
            "approval_id": ticket.approval_id,
            "action_id": ticket.action_id,
            "status": ticket.status,
        }


class GetActionStatus(PersonalAssistantTool):
    name = "get_action_status"
    description = (
        "Check whether a previously requested action is awaiting approval, executing, "
        "completed, rejected, expired, or failed."
    )
    parameters_schema = {
        "type": "object",
        "properties": {"action_id": {"type": "string", "minLength": 1, "maxLength": 200}},
        "required": ["action_id"],
        "additionalProperties": False,
    }

    async def run(
        self, client: AsyncPersonalAssistantClient, arguments: Mapping[str, Any]
    ) -> dict[str, Any]:
        action_id = _resource_id(arguments, "action_id", max_length=200)
        status = await client.get_action_status(action_id)
        spoken_by_status = {
            "pending": "The action is awaiting approval.",
            "awaiting_approval": "The action is awaiting approval.",
            "executing": "The action is executing.",
            "executed": "The action was sent and completed.",
            "sent": "The action was sent and completed.",
            "completed": "The action was sent and completed.",
            "approved": "The action was approved and is being processed.",
            "rejected": "The action was rejected.",
            "expired": "The action expired before approval.",
            "failed": "The action failed.",
        }
        normalized = status.status.strip().lower()
        return {
            "ok": True,
            "spoken": spoken_by_status.get(
                normalized, f"The action status is {normalized.replace('_', ' ')}."
            ),
            "status": status.status,
            "action_id": status.action_id,
            "approval_id": status.approval_id,
            "action_type": status.action_type,
            "executed_at": status.executed_at.isoformat() if status.executed_at else None,
        }


class GetCalendarEvents(PersonalAssistantTool):
    name = "get_calendar_events"
    description = "List upcoming calendar events from Reachy Agentic Assistant."
    parameters_schema = {
        "type": "object",
        "properties": {"days": {"type": "integer", "minimum": 1, "maximum": 90}},
        "required": [],
        "additionalProperties": False,
    }

    async def run(
        self, client: AsyncPersonalAssistantClient, arguments: Mapping[str, Any]
    ) -> dict[str, Any]:
        days = _bounded_int(arguments, "days", 7, 1, 90)
        result = await client.get_calendar_events(days=days, limit=10)
        if not result.items:
            return {
                "ok": True,
                "spoken": f"Nothing is scheduled in the next {days} days.",
                "count": 0,
            }
        summary = "; ".join(
            f"{item.title} at {item.starts_at.isoformat()}" for item in result.items[:5]
        )
        return {
            "ok": True,
            "spoken": f"{result.total} upcoming event(s). {summary}.",
            "count": result.total,
        }


class CheckCalendarFreeBusy(PersonalAssistantTool):
    name = "check_calendar_free_busy"
    description = "Check calendar availability over an exact ISO-8601 interval."
    parameters_schema = {
        "type": "object",
        "properties": {
            "starts_at": {"type": "string"},
            "ends_at": {"type": "string"},
        },
        "required": ["starts_at", "ends_at"],
        "additionalProperties": False,
    }

    async def run(
        self, client: AsyncPersonalAssistantClient, arguments: Mapping[str, Any]
    ) -> dict[str, Any]:
        starts_at = _datetime(arguments, "starts_at")
        ends_at = _datetime(arguments, "ends_at")
        assert starts_at is not None and ends_at is not None
        if ends_at <= starts_at:
            raise ToolArgumentError("ends_at must be after starts_at")
        if ends_at <= starts_at:
            raise ToolArgumentError("ends_at must be after starts_at")
        result = await client.check_free_busy(starts_at, ends_at)
        return {
            "ok": True,
            "spoken": f"There are {len(result.free_slots)} free slot(s) in that interval.",
            "free_slots": [slot.model_dump(mode="json") for slot in result.free_slots[:5]],
        }


class ProposeCalendarEvent(PersonalAssistantTool):
    name = "propose_calendar_event"
    description = "Find free times for a calendar event without creating or changing an event."
    parameters_schema = {
        "type": "object",
        "properties": {
            "title": {"type": "string", "minLength": 1},
            "duration_minutes": {"type": "integer", "minimum": 5, "maximum": 1440},
        },
        "required": ["title"],
        "additionalProperties": False,
    }

    async def run(
        self, client: AsyncPersonalAssistantClient, arguments: Mapping[str, Any]
    ) -> dict[str, Any]:
        title = _required_text(arguments, "title", max_length=500)
        result = await client.propose_calendar_event(
            ProposeEventRequest(
                title=title,
                duration_minutes=_bounded_int(arguments, "duration_minutes", 60, 5, 1440),
            )
        )
        first = result.suggestions[0]
        return {
            "ok": True,
            "spoken": (
                f"The best proposed time for {title} starts at {first.starts_at.isoformat()}."
            ),
            "suggestions": [slot.model_dump(mode="json") for slot in result.suggestions],
            "draft_create_payload": result.draft_create_payload.model_dump(mode="json"),
        }


class RequestCreateCalendarEvent(PersonalAssistantTool):
    name = "request_create_calendar_event"
    description = (
        "Request owner approval to create a calendar event. Does not create it immediately."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "title": {"type": "string", "minLength": 1},
            "starts_at": {"type": "string"},
            "ends_at": {"type": "string"},
        },
        "required": ["title", "starts_at", "ends_at"],
        "additionalProperties": False,
    }

    async def run(
        self, client: AsyncPersonalAssistantClient, arguments: Mapping[str, Any]
    ) -> dict[str, Any]:
        title = _required_text(arguments, "title", max_length=500)
        starts_at = _datetime(arguments, "starts_at")
        ends_at = _datetime(arguments, "ends_at")
        assert starts_at is not None and ends_at is not None
        ticket = await client.request_create_calendar_event(
            CreateEventRequest(title=title, starts_at=starts_at, ends_at=ends_at),
            idempotency_key=uuid.uuid4().hex,
        )
        return {
            "ok": True,
            "spoken": "Calendar creation is awaiting owner approval.",
            "approval_id": ticket.approval_id,
            "status": ticket.status,
        }


class RequestUpdateCalendarEvent(PersonalAssistantTool):
    name = "request_update_calendar_event"
    description = "Request owner approval to update an existing calendar event."
    parameters_schema = {
        "type": "object",
        "properties": {
            "event_id": {"type": "string", "minLength": 1},
            "title": {"type": "string", "minLength": 1},
            "starts_at": {"type": "string"},
            "ends_at": {"type": "string"},
        },
        "required": ["event_id"],
        "additionalProperties": False,
    }

    async def run(
        self, client: AsyncPersonalAssistantClient, arguments: Mapping[str, Any]
    ) -> dict[str, Any]:
        event_id = _resource_id(arguments, "event_id", max_length=200)
        title = _optional_text(arguments, "title", max_length=500)
        starts_at = _datetime(arguments, "starts_at", required=False)
        ends_at = _datetime(arguments, "ends_at", required=False)
        if title is None and starts_at is None and ends_at is None:
            raise ToolArgumentError("no calendar update was supplied")
        if (starts_at is None) != (ends_at is None):
            raise ToolArgumentError("starts_at and ends_at must be supplied together")
        if starts_at is not None and ends_at is not None and ends_at <= starts_at:
            raise ToolArgumentError("ends_at must be after starts_at")
        ticket = await client.request_update_calendar_event(
            event_id,
            UpdateEventRequest(title=title, starts_at=starts_at, ends_at=ends_at),
            idempotency_key=uuid.uuid4().hex,
        )
        return {
            "ok": True,
            "spoken": "Calendar update is awaiting owner approval.",
            "approval_id": ticket.approval_id,
            "status": ticket.status,
        }


class SearchContacts(PersonalAssistantTool):
    name = "search_contacts"
    description = "Look up contacts through Reachy Agentic Assistant."
    parameters_schema = {
        "type": "object",
        "properties": {"query": {"type": "string", "minLength": 1, "maxLength": 100}},
        "required": ["query"],
        "additionalProperties": False,
    }

    async def run(
        self, client: AsyncPersonalAssistantClient, arguments: Mapping[str, Any]
    ) -> dict[str, Any]:
        result = await client.search_contacts(
            _required_text(arguments, "query", max_length=100), limit=5
        )
        if not result.items:
            return {"ok": True, "spoken": "No contacts matched.", "count": 0}
        contacts = [
            {"id": item.id, "name": item.display_name, "emails": item.emails, "phones": item.phones}
            for item in result.items
        ]
        return {
            "ok": True,
            "spoken": f"Found {result.total} matching contact(s).",
            "count": result.total,
            "contacts": contacts,
        }


class SearchNotion(PersonalAssistantTool):
    name = "search_notion"
    description = "Search allowlisted Notion pages through Reachy Agentic Assistant."
    parameters_schema = SearchContacts.parameters_schema

    async def run(
        self, client: AsyncPersonalAssistantClient, arguments: Mapping[str, Any]
    ) -> dict[str, Any]:
        result = await client.search_notion(
            _required_text(arguments, "query", max_length=100), limit=5
        )
        return {
            "ok": True,
            "spoken": "No Notion pages matched."
            if not result.items
            else f"Found: {'; '.join(item.title for item in result.items)}.",
            "count": result.total,
            "pages": [{"id": item.id, "title": item.title} for item in result.items],
        }


class ReadNotionPage(PersonalAssistantTool):
    name = "read_notion_page"
    description = "Read a bounded text excerpt from an allowlisted Notion page."
    parameters_schema = {
        "type": "object",
        "properties": {"page_id": {"type": "string", "minLength": 1}},
        "required": ["page_id"],
        "additionalProperties": False,
    }

    async def run(
        self, client: AsyncPersonalAssistantClient, arguments: Mapping[str, Any]
    ) -> dict[str, Any]:
        result = await client.read_notion_page(
            _resource_id(arguments, "page_id", max_length=200), max_chars=2000
        )
        return {
            "ok": True,
            "spoken": f"{result.page.title}: {result.text[:700]}",
            "page_id": result.page.id,
            "truncated": result.truncated,
            "content_is_untrusted": True,
        }


class SearchDocuments(PersonalAssistantTool):
    name = "search_documents"
    description = "Search the workstation's allowlisted document index."
    parameters_schema = {
        "type": "object",
        "properties": {"query": {"type": "string", "minLength": 1, "maxLength": 200}},
        "required": ["query"],
        "additionalProperties": False,
    }

    async def run(
        self, client: AsyncPersonalAssistantClient, arguments: Mapping[str, Any]
    ) -> dict[str, Any]:
        result = await client.search_documents(
            _required_text(arguments, "query", max_length=200), limit=5
        )
        hits = [
            {"path": item.path, "name": item.name, "snippet": item.snippet} for item in result.items
        ]
        return {
            "ok": True,
            "spoken": "No documents matched."
            if not hits
            else f"Found {result.total} matching document(s).",
            "count": result.total,
            "documents": hits,
        }


class ReadDocument(PersonalAssistantTool):
    name = "read_document"
    description = "Read a bounded excerpt from a document returned by document search."
    parameters_schema = {
        "type": "object",
        "properties": {"path": {"type": "string", "minLength": 1, "maxLength": 4096}},
        "required": ["path"],
        "additionalProperties": False,
    }

    async def run(
        self, client: AsyncPersonalAssistantClient, arguments: Mapping[str, Any]
    ) -> dict[str, Any]:
        result = await client.read_document(
            _required_text(arguments, "path", max_length=4096), max_chars=2000
        )
        return {
            "ok": True,
            "spoken": f"{result.name}: {result.text[:700]}",
            "path": result.path,
            "truncated": result.truncated,
            "content_is_untrusted": True,
        }


class GetPersonalWeather(PersonalAssistantTool):
    name = "get_personal_weather"
    description = "Get weather and clothing advice from Reachy Agentic Assistant."
    parameters_schema = {
        "type": "object",
        "properties": {"location": {"type": "string", "minLength": 1, "maxLength": 100}},
        "required": [],
        "additionalProperties": False,
    }

    async def run(
        self, client: AsyncPersonalAssistantClient, arguments: Mapping[str, Any]
    ) -> dict[str, Any]:
        result = await client.get_weather(
            location=_optional_text(arguments, "location", max_length=100)
        )
        report = result.report
        return {
            "ok": True,
            "spoken": (
                f"In {report.location}, it feels like {report.current.feels_like_c:.0f} degrees. "
                f"{result.advice.summary}"
            ),
            "temperature_c": report.current.temperature_c,
            "advice": result.advice.summary,
        }


class SearchWardrobe(PersonalAssistantTool):
    name = "search_wardrobe"
    description = "Search the wardrobe inventory on the workstation."
    parameters_schema = {
        "type": "object",
        "properties": {
            "category": {"type": "string", "minLength": 1, "maxLength": 40},
            "color": {"type": "string", "minLength": 1, "maxLength": 40},
        },
        "required": [],
        "additionalProperties": False,
    }

    async def run(
        self, client: AsyncPersonalAssistantClient, arguments: Mapping[str, Any]
    ) -> dict[str, Any]:
        result = await client.search_wardrobe(
            category=_optional_text(arguments, "category", max_length=40),
            color=_optional_text(arguments, "color", max_length=40),
            available_only=True,
            limit=10,
        )
        return {
            "ok": True,
            "spoken": "No matching garments."
            if not result.items
            else f"Found {result.total} matching garment(s).",
            "count": result.total,
            "items": [
                {
                    "id": item.id,
                    "name": item.name,
                    "category": item.category,
                    "color": item.primary_color,
                }
                for item in result.items
            ],
        }


class RecommendOutfit(PersonalAssistantTool):
    name = "recommend_outfit"
    description = "Recommend an outfit using wardrobe data and weather on the workstation."
    parameters_schema = {
        "type": "object",
        "properties": {"occasion": {"type": "string", "minLength": 1, "maxLength": 60}},
        "required": [],
        "additionalProperties": False,
    }

    async def run(
        self, client: AsyncPersonalAssistantClient, arguments: Mapping[str, Any]
    ) -> dict[str, Any]:
        result = await client.recommend_outfit(
            occasion=_optional_text(arguments, "occasion", max_length=60), limit=2
        )
        if not result.recommendations:
            return {
                "ok": True,
                "spoken": f"No suitable outfit was found. {result.weather_summary}",
                "count": 0,
            }
        top = result.recommendations[0]
        return {
            "ok": True,
            "spoken": f"I suggest {top.outfit_name}. {top.explanation} {result.weather_summary}",
            "outfit_id": top.outfit_id,
            "score": top.score,
        }


class GetWorkstationStatus(PersonalAssistantTool):
    name = "get_workstation_status"
    description = "Read bounded workstation health metrics from the workstation."
    parameters_schema = PersonalAssistantPing.parameters_schema

    async def run(
        self, client: AsyncPersonalAssistantClient, arguments: Mapping[str, Any]
    ) -> dict[str, Any]:
        del arguments
        result = await client.get_workstation_status()
        return {"ok": True, "spoken": result.spoken_summary, "hostname": result.status.hostname}


class ListPendingNotifications(PersonalAssistantTool):
    name = "list_pending_notifications"
    description = "List pending proactive notifications queued for Reachy by the workstation."
    parameters_schema = PersonalAssistantPing.parameters_schema

    async def run(
        self, client: AsyncPersonalAssistantClient, arguments: Mapping[str, Any]
    ) -> dict[str, Any]:
        del arguments
        result = await client.list_pending_notifications(limit=5)
        if not result.items:
            return {"ok": True, "spoken": "Nothing is queued for Reachy.", "count": 0}
        return {
            "ok": True,
            "spoken": result.spoken_text or result.items[0].spoken_text,
            "count": result.total,
            "notifications": [
                {"id": item.id, "kind": item.kind, "spoken_text": item.spoken_text}
                for item in result.items
            ],
        }
