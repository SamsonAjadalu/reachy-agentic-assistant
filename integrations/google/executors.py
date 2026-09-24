"""Approval executors for the Google write actions.

Registration is an explicit call rather than an import side effect, so startup
controls when it happens and a pending action written before a restart still
finds its handler afterwards. Each executor re-reads its payload from the
Uses the configured workflow.
the payload hash check meaningful.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings
from approvals.state_machine import has_executor, register_executor
from integrations.registry import get_calendar, get_gmail

# Historical action type kept registered so any in-flight approvals created
# before the draft/send split can still resolve without crashing startup.
SEND_EMAIL = "google.gmail.send"
SEND_DRAFT = "google.gmail.send_draft"
CREATE_EVENT = "google.calendar.create_event"
UPDATE_EVENT = "google.calendar.update_event"
DELETE_EVENT = "google.calendar.delete_event"


async def send_draft(
    session: AsyncSession, payload: dict[str, Any], settings: Settings
) -> dict[str, Any]:
    result = await get_gmail(settings).send_draft(
        str(payload["draft_id"]),
        expected_content_hash=str(payload["content_hash"]),
    )
    return {
        "message_id": result.get("message_id"),
        "draft_id": result.get("draft_id") or payload["draft_id"],
        "recipients": result.get("recipients"),
        "external_id": result.get("external_id") or result.get("message_id"),
    }


async def send_email_legacy(
    session: AsyncSession, payload: dict[str, Any], settings: Settings
) -> dict[str, Any]:
    """Bridge for approvals created before draft/send were split.

    New compose-and-send requests are rejected at the HTTP layer. If an old
    ticket somehow still holds a raw compose payload, refuse rather than
    inventing a send path that bypasses draft revision checks.
    """
    if "draft_id" in payload and "content_hash" in payload:
        return await send_draft(session, payload, settings)
    raise ValueError(
        "Legacy google.gmail.send approvals without a draft id are no longer executable. "
        "Create a draft and request send-draft approval instead."
    )


async def create_event(
    session: AsyncSession, payload: dict[str, Any], settings: Settings
) -> dict[str, Any]:
    event = await get_calendar(settings).create_event(payload)
    return {"event_id": event.id, "title": event.title, "starts_at": event.starts_at.isoformat()}


async def update_event(
    session: AsyncSession, payload: dict[str, Any], settings: Settings
) -> dict[str, Any]:
    event = await get_calendar(settings).update_event(
        str(payload["event_id"]),
        payload,
        calendar_id=str(payload.get("calendar_id") or "primary"),
    )
    return {
        "event_id": event.id,
        "title": event.title,
        "starts_at": event.starts_at.isoformat(),
        "external_id": event.id,
    }


async def delete_event(
    session: AsyncSession, payload: dict[str, Any], settings: Settings
) -> dict[str, Any]:
    await get_calendar(settings).delete_event(
        str(payload["event_id"]), calendar_id=str(payload.get("calendar_id") or "primary")
    )
    return {"event_id": payload["event_id"], "deleted": True}


def register_google_executors() -> None:
    """Idempotent so startup, the CLI and tests can all call it."""
    handlers = {
        SEND_DRAFT: send_draft,
        SEND_EMAIL: send_email_legacy,
        CREATE_EVENT: create_event,
        UPDATE_EVENT: update_event,
        DELETE_EVENT: delete_event,
    }
    for action_type, handler in handlers.items():
        if not has_executor(action_type):
            register_executor(action_type)(handler)


def draft_send_preview(payload: dict[str, Any]) -> str:
    """What the owner reads before approving. Every consequential field appears."""
    lines = [
        f"Draft: {payload['draft_id']}",
        f"Revision: {payload['content_hash'][:12]}",
        f"To: {', '.join(payload['to'])}",
    ]
    if payload.get("cc"):
        lines.append(f"Cc: {', '.join(payload['cc'])}")
    lines.append(f"Subject: {payload['subject']}")
    lines.append("")
    body = str(payload["body"])
    lines.append(body if len(body) <= 1500 else body[:1500] + "\n[body truncated in preview]")
    return "\n".join(lines)


def email_preview(payload: dict[str, Any]) -> str:
    """Compatibility wrapper; new code should call draft_send_preview."""
    if "draft_id" in payload:
        return draft_send_preview(payload)
    lines = [f"To: {', '.join(payload['to'])}"]
    if payload.get("cc"):
        lines.append(f"Cc: {', '.join(payload['cc'])}")
    lines.append(f"Subject: {payload['subject']}")
    lines.append("")
    body = str(payload["body"])
    lines.append(body if len(body) <= 1500 else body[:1500] + "\n[body truncated in preview]")
    return "\n".join(lines)


def event_preview(payload: dict[str, Any]) -> str:
    lines = [
        f"Title: {payload['title']}",
        f"Starts: {payload['starts_at']}",
        f"Ends: {payload['ends_at']}",
    ]
    if payload.get("location"):
        lines.append(f"Location: {payload['location']}")
    if payload.get("attendees"):
        lines.append(f"Attendees: {', '.join(payload['attendees'])}")
    if payload.get("description"):
        lines.append(f"\n{payload['description']}")
    return "\n".join(lines)


def event_update_preview(before: dict[str, Any], changes: dict[str, Any]) -> str:
    lines = [
        f"Update event: {before.get('title')} ({changes.get('event_id')})",
        f"Currently: {before.get('starts_at')} → {before.get('ends_at')}",
        "",
        "Proposed changes:",
    ]
    for field in ("title", "starts_at", "ends_at", "location", "description", "attendees"):
        if field in changes and changes[field] is not None:
            lines.append(f"  {field}: {before.get(field)} → {changes[field]}")
    return "\n".join(lines)
