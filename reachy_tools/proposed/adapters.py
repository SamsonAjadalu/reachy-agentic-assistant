"""Proposed thin adapters for a Reachy Pi voice agent.

These are plain Python functions — not verified against any particular Reachy
tool decorator or conversation-app schema. Each adapter validates lightly,
calls the personal assistant client, and returns a short dict for one voice turn.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from reachy_client.client import PersonalAssistantClient
from reachy_client.exceptions import ApiError, ApprovalRequired, TimeoutError
from reachy_client.models import (
    CreateDraftRequest,
    CreateEventRequest,
    CreateReplyDraftRequest,
    Delay,
    ProposeEventRequest,
    ReminderCreate,
    ScheduleInput,
    SendDraftRequest,
    TaskCreate,
    UpdateEventRequest,
)


def _error_result(exc: Exception) -> dict[str, Any]:
    if isinstance(exc, TimeoutError):
        return {
            "ok": False,
            "error": "timeout",
            "spoken": "The assistant did not answer in time. Try again in a moment.",
            "correlation_id": exc.correlation_id,
        }
    if isinstance(exc, ApprovalRequired):
        return {
            "ok": False,
            "error": "approval_required",
            "spoken": exc.message,
            "approval_id": exc.approval_id,
            "correlation_id": exc.correlation_id,
        }
    if isinstance(exc, ApiError):
        return {
            "ok": False,
            "error": exc.code or "api_error",
            "spoken": exc.message,
            "correlation_id": exc.correlation_id,
        }
    return {"ok": False, "error": "unexpected", "spoken": str(exc)}


def assistant_ping(client: PersonalAssistantClient) -> dict[str, Any]:
    try:
        result = client.ping()
        return {
            "ok": True,
            "spoken": result.message,
            "server_time_local": result.server_time_local,
            "timezone": result.timezone,
        }
    except Exception as exc:
        return _error_result(exc)


def create_reminder(
    client: PersonalAssistantClient,
    *,
    text: str,
    delay_minutes: int | None = None,
    at: datetime | None = None,
    idempotency_key: str | None = None,
) -> dict[str, Any]:
    if not text.strip():
        return {"ok": False, "spoken": "Reminder text cannot be empty."}
    if (delay_minutes is None) == (at is None):
        return {"ok": False, "spoken": "Provide either delay_minutes or a specific time."}
    try:
        schedule = (
            ScheduleInput(delay=Delay(minutes=delay_minutes or 0))
            if delay_minutes is not None
            else ScheduleInput(at=at)
        )
        reminder = client.create_reminder(
            ReminderCreate(text=text.strip(), schedule=schedule),
            idempotency_key=idempotency_key,
        )
        return {
            "ok": True,
            "spoken": f"Reminder set for {reminder.trigger_at_local}.",
            "reminder_id": reminder.id,
            "trigger_at_local": reminder.trigger_at_local,
        }
    except Exception as exc:
        return _error_result(exc)


def list_reminders(
    client: PersonalAssistantClient,
    *,
    upcoming_only: bool = True,
    limit: int = 5,
) -> dict[str, Any]:
    try:
        result = client.list_reminders(upcoming_only=upcoming_only, limit=limit)
        if not result.items:
            return {"ok": True, "spoken": "You have no upcoming reminders.", "count": 0}
        lines = [f"{item.text} at {item.trigger_at_local}" for item in result.items[:limit]]
        items = [{"id": i.id, "text": i.text, "when": i.trigger_at_local} for i in result.items]
        return {
            "ok": True,
            "spoken": f"You have {result.total} reminder(s). Next: {'; '.join(lines)}.",
            "count": result.total,
            "items": items,
        }
    except Exception as exc:
        return _error_result(exc)


def create_task(
    client: PersonalAssistantClient,
    *,
    description: str,
    idempotency_key: str | None = None,
) -> dict[str, Any]:
    if not description.strip():
        return {"ok": False, "spoken": "Task description cannot be empty."}
    try:
        task = client.create_task(
            TaskCreate(description=description.strip()),
            idempotency_key=idempotency_key,
        )
        return {
            "ok": True,
            "spoken": f"Added task: {task.description}.",
            "task_id": task.id,
        }
    except Exception as exc:
        return _error_result(exc)


def list_tasks(
    client: PersonalAssistantClient,
    *,
    status: str = "open",
    limit: int = 5,
) -> dict[str, Any]:
    try:
        result = client.list_tasks(status=status, limit=limit)
        if not result.items:
            return {"ok": True, "spoken": f"No {status} tasks.", "count": 0}
        lines = [item.description for item in result.items[:limit]]
        return {
            "ok": True,
            "spoken": f"{result.total} {status} task(s): {'; '.join(lines)}.",
            "count": result.total,
            "items": [{"id": i.id, "description": i.description} for i in result.items],
        }
    except Exception as exc:
        return _error_result(exc)


def search_gmail(client: PersonalAssistantClient, query: str, *, limit: int = 5) -> dict[str, Any]:
    if not query.strip():
        return {"ok": False, "spoken": "Tell me what to search for in your mail."}
    try:
        result = client.search_gmail(query.strip(), limit=limit)
        if not result.items:
            return {"ok": True, "spoken": "No messages matched that search.", "count": 0}
        lines = [f"{m.sender_name or m.sender}: {m.subject}" for m in result.items]
        messages = [{"id": m.id, "subject": m.subject, "from": m.sender} for m in result.items]
        return {
            "ok": True,
            "spoken": f"Found {result.total} message(s). {'; '.join(lines)}.",
            "count": result.total,
            "messages": messages,
        }
    except Exception as exc:
        return _error_result(exc)


def read_email(client: PersonalAssistantClient, message_id: str) -> dict[str, Any]:
    if not message_id.strip():
        return {"ok": False, "spoken": "Which message should I read?"}
    try:
        result = client.read_email(message_id.strip())
        body = result.message.body_text
        if result.message.truncated:
            body += " (truncated)"
        return {
            "ok": True,
            "spoken": f"From {result.message.sender_name or result.message.sender}. "
            f"Subject: {result.message.subject}. {body[:500]}",
            "message_id": result.message.id,
            "truncated": result.message.truncated,
        }
    except Exception as exc:
        return _error_result(exc)


def create_gmail_draft(
    client: PersonalAssistantClient,
    *,
    to: list[str],
    subject: str,
    body: str,
    idempotency_key: str | None = None,
) -> dict[str, Any]:
    if not to or not subject.strip() or not body.strip():
        return {"ok": False, "spoken": "I need recipients, a subject, and a message body."}
    try:
        result = client.create_gmail_draft(
            CreateDraftRequest(to=to, subject=subject.strip(), body=body.strip()),
            idempotency_key=idempotency_key,
        )
        draft = result.draft
        return {
            "ok": True,
            "spoken": (
                f"I saved a draft to {', '.join(draft.to)}. "
                "Say when you want me to ask for send approval."
            ),
            "draft_id": draft.id,
            "content_hash": draft.content_hash,
        }
    except Exception as exc:
        return _error_result(exc)


def create_reply_draft(
    client: PersonalAssistantClient,
    *,
    message_id: str,
    body: str,
    idempotency_key: str | None = None,
) -> dict[str, Any]:
    if not message_id.strip() or not body.strip():
        return {"ok": False, "spoken": "I need the message id and a reply body."}
    try:
        result = client.create_gmail_reply_draft(
            message_id.strip(),
            CreateReplyDraftRequest(body=body.strip()),
            idempotency_key=idempotency_key,
        )
        draft = result.draft
        return {
            "ok": True,
            "spoken": (
                f"I saved a reply draft to {', '.join(draft.to)} in the same thread. "
                "Say when you want me to ask for send approval."
            ),
            "draft_id": draft.id,
            "content_hash": draft.content_hash,
            "thread_id": draft.thread_id,
        }
    except Exception as exc:
        return _error_result(exc)


def send_existing_draft(
    client: PersonalAssistantClient,
    *,
    draft_id: str,
    content_hash: str | None = None,
    idempotency_key: str | None = None,
) -> dict[str, Any]:
    if not draft_id.strip():
        return {"ok": False, "spoken": "I need the draft id to request send approval."}
    try:
        ticket = client.send_existing_draft(
            SendDraftRequest(draft_id=draft_id.strip(), content_hash=content_hash),
            idempotency_key=idempotency_key,
        )
        return {
            "ok": True,
            "spoken": (
                "I asked for approval to send that draft. "
                "Nothing goes out until you approve the exact revision."
            ),
            "approval_id": ticket.approval_id,
            "action_id": ticket.action_id,
            "summary": ticket.summary,
        }
    except Exception as exc:
        return _error_result(exc)


def get_action_status(client: PersonalAssistantClient, *, action_id: str) -> dict[str, Any]:
    if not action_id.strip():
        return {"ok": False, "spoken": "I need the action id to check its status."}
    try:
        status = client.get_action_status(action_id.strip())
        spoken = status.result_summary or f"The action status is {status.status.replace('_', ' ')}."
        return {
            "ok": True,
            "spoken": spoken,
            "status": status.status,
            "action_type": status.action_type,
            "approval_id": status.approval_id,
            "action_id": status.action_id,
            "executed_at": status.executed_at.isoformat() if status.executed_at else None,
        }
    except Exception as exc:
        return _error_result(exc)


def list_gmail_attachments(client: PersonalAssistantClient, *, message_id: str) -> dict[str, Any]:
    try:
        result = client.list_gmail_attachments(message_id)
        if not result.items:
            return {"ok": True, "spoken": "That message has no attachments.", "count": 0}
        names = ", ".join(item.filename for item in result.items[:5])
        return {
            "ok": True,
            "spoken": f"{result.total} attachment(s): {names}.",
            "count": result.total,
            "content_is_untrusted": True,
        }
    except Exception as exc:
        return _error_result(exc)


def archive_gmail_message(client: PersonalAssistantClient, *, message_id: str) -> dict[str, Any]:
    try:
        result = client.archive_gmail_message(message_id)
        return {
            "ok": True,
            "spoken": "Archived.",
            "labels": result.labels,
        }
    except Exception as exc:
        return _error_result(exc)


def get_calendar_events(client: PersonalAssistantClient, *, days: int = 7) -> dict[str, Any]:
    try:
        result = client.get_calendar_events(days=days, limit=10)
        if not result.items:
            return {
                "ok": True,
                "spoken": f"Nothing on the calendar in the next {days} days.",
                "count": 0,
            }
        lines = [f"{e.title} at {e.starts_at.isoformat()}" for e in result.items[:5]]
        return {
            "ok": True,
            "spoken": f"{result.total} event(s) coming up. {'; '.join(lines)}.",
            "count": result.total,
        }
    except Exception as exc:
        return _error_result(exc)


def check_free_busy(client: PersonalAssistantClient, *, hours_ahead: int = 8) -> dict[str, Any]:
    try:
        start = datetime.now().astimezone()
        end = start + timedelta(hours=hours_ahead)
        result = client.check_free_busy(start, end)
        if not result.free_slots:
            return {"ok": True, "spoken": f"You appear busy for the next {hours_ahead} hours."}
        first = result.free_slots[0]
        return {
            "ok": True,
            "spoken": (
                f"Next free slot starts at {first.starts_at.isoformat()} "
                f"and ends at {first.ends_at.isoformat()}."
            ),
            "free_slots": len(result.free_slots),
        }
    except Exception as exc:
        return _error_result(exc)


def propose_calendar_event(
    client: PersonalAssistantClient,
    *,
    title: str,
    duration_minutes: int = 60,
) -> dict[str, Any]:
    if not title.strip():
        return {"ok": False, "spoken": "What should I call the event?"}
    try:
        result = client.propose_calendar_event(
            ProposeEventRequest(title=title.strip(), duration_minutes=duration_minutes)
        )
        first = result.suggestions[0]
        return {
            "ok": True,
            "spoken": (
                f"I found {len(result.suggestions)} free slot(s) for {title}. "
                f"Best option starts at {first.starts_at.isoformat()}."
            ),
            "suggestions": len(result.suggestions),
            "draft_create_payload": result.draft_create_payload.model_dump(mode="json"),
        }
    except Exception as exc:
        return _error_result(exc)


def request_create_calendar_event(
    client: PersonalAssistantClient,
    *,
    title: str,
    starts_at: datetime,
    ends_at: datetime,
    idempotency_key: str | None = None,
) -> dict[str, Any]:
    try:
        ticket = client.request_create_calendar_event(
            CreateEventRequest(title=title, starts_at=starts_at, ends_at=ends_at),
            idempotency_key=idempotency_key,
        )
        return {
            "ok": True,
            "spoken": f"I asked for approval to add {title}.",
            "approval_id": ticket.approval_id,
        }
    except Exception as exc:
        return _error_result(exc)


def request_update_calendar_event(
    client: PersonalAssistantClient,
    *,
    event_id: str,
    title: str | None = None,
    starts_at: datetime | None = None,
    ends_at: datetime | None = None,
    idempotency_key: str | None = None,
) -> dict[str, Any]:
    try:
        ticket = client.request_update_calendar_event(
            event_id,
            UpdateEventRequest(title=title, starts_at=starts_at, ends_at=ends_at),
            idempotency_key=idempotency_key,
        )
        return {
            "ok": True,
            "spoken": "I asked for approval to update that event.",
            "approval_id": ticket.approval_id,
        }
    except Exception as exc:
        return _error_result(exc)


def search_contacts(client: PersonalAssistantClient, query: str) -> dict[str, Any]:
    if not query.strip():
        return {"ok": False, "spoken": "Who are you looking for?"}
    try:
        result = client.search_contacts(query.strip(), limit=5)
        if not result.items:
            return {"ok": True, "spoken": "No contacts matched.", "count": 0}
        lines = [
            f"{c.display_name} ({c.emails[0] if c.emails else 'no email'})" for c in result.items
        ]
        return {
            "ok": True,
            "spoken": f"Found: {'; '.join(lines)}.",
            "count": result.total,
        }
    except Exception as exc:
        return _error_result(exc)


def search_notion(client: PersonalAssistantClient, query: str) -> dict[str, Any]:
    if not query.strip():
        return {"ok": False, "spoken": "What should I search for in Notion?"}
    try:
        result = client.search_notion(query.strip(), limit=5)
        if not result.items:
            return {"ok": True, "spoken": "No Notion pages matched.", "count": 0}
        lines = [p.title for p in result.items]
        return {"ok": True, "spoken": f"Pages: {'; '.join(lines)}.", "count": result.total}
    except Exception as exc:
        return _error_result(exc)


def search_documents(client: PersonalAssistantClient, query: str) -> dict[str, Any]:
    if not query.strip():
        return {"ok": False, "spoken": "What document should I look for?"}
    try:
        result = client.search_documents(query.strip(), limit=5)
        if not result.items:
            return {"ok": True, "spoken": "No documents matched.", "count": 0}
        lines = [f"{hit.name}: {hit.snippet[:80]}" for hit in result.items]
        spoken = f"Found {result.total}. {'; '.join(lines)}."
        return {"ok": True, "spoken": spoken, "count": result.total}
    except Exception as exc:
        return _error_result(exc)


def get_weather(client: PersonalAssistantClient, *, location: str | None = None) -> dict[str, Any]:
    try:
        result = client.get_weather(location=location)
        report = result.report
        return {
            "ok": True,
            "spoken": (
                f"In {report.location}, it feels like {report.current.feels_like_c:.0f} degrees. "
                f"{result.advice.summary}"
            ),
            "location": report.location,
            "temperature_c": report.current.temperature_c,
            "advice": result.advice.summary,
        }
    except Exception as exc:
        return _error_result(exc)


def search_wardrobe(
    client: PersonalAssistantClient,
    *,
    category: str | None = None,
    color: str | None = None,
) -> dict[str, Any]:
    try:
        result = client.search_wardrobe(
            category=category, color=color, available_only=True, limit=10
        )
        if not result.items:
            return {"ok": True, "spoken": "No matching garments in the wardrobe.", "count": 0}
        lines = [f"{i.name} ({i.primary_color} {i.category})" for i in result.items[:5]]
        spoken = f"{result.total} item(s): {'; '.join(lines)}."
        return {"ok": True, "spoken": spoken, "count": result.total}
    except Exception as exc:
        return _error_result(exc)


def recommend_outfit(
    client: PersonalAssistantClient,
    *,
    occasion: str | None = None,
) -> dict[str, Any]:
    try:
        result = client.recommend_outfit(occasion=occasion, limit=2)
        if not result.recommendations:
            return {
                "ok": True,
                "spoken": f"No suitable outfits. {result.weather_summary}",
                "count": 0,
            }
        top = result.recommendations[0]
        return {
            "ok": True,
            "spoken": (
                f"I suggest {top.outfit_name}. {top.explanation} Forecast: {result.weather_summary}"
            ),
            "outfit_id": top.outfit_id,
            "score": top.score,
        }
    except Exception as exc:
        return _error_result(exc)


def get_workstation_status(client: PersonalAssistantClient) -> dict[str, Any]:
    try:
        result = client.get_workstation_status()
        return {"ok": True, "spoken": result.spoken_summary, "hostname": result.status.hostname}
    except Exception as exc:
        return _error_result(exc)


def list_pending_notifications(client: PersonalAssistantClient) -> dict[str, Any]:
    try:
        result = client.list_pending_notifications()
        if not result.items:
            return {"ok": True, "spoken": "Nothing queued for Reachy.", "count": 0}
        return {
            "ok": True,
            "spoken": result.spoken_text or result.items[0].spoken_text,
            "count": result.total,
            "ids": [item.id for item in result.items],
        }
    except Exception as exc:
        return _error_result(exc)


def get_camera_status(client: PersonalAssistantClient) -> dict[str, Any]:
    try:
        result = client.get_camera_status()
        return {
            "ok": True,
            "spoken": result.phrase or f"Camera health is {result.camera_health}.",
            "camera_health": result.camera_health,
            "mocked": result.mocked,
            "may_move_head": False,
        }
    except Exception as exc:
        return _error_result(exc)


def look_now(
    client: PersonalAssistantClient,
    *,
    query: str | None = None,
    zone: str | None = None,
    preset_id: str | None = None,
    allow_scan: bool = False,
    idempotency_key: str | None = None,
) -> dict[str, Any]:
    try:
        result = client.look_now(
            query=query,
            zone=zone,
            preset_id=preset_id,
            allow_scan=allow_scan,
            idempotency_key=idempotency_key,
        )
        payload: dict[str, Any] = {
            "ok": True,
            "spoken": result.phrase,
            "presence": result.presence,
            "may_move_head": result.may_move_head,
            "world_frame_status": result.world_frame_status,
        }
        if result.deferred:
            payload["task_id"] = result.task_id
            payload["poll_url"] = result.poll_url
            payload["spoken"] = (
                "I queued another look. I will follow up; this is not on the spoken path."
            )
        return payload
    except Exception as exc:
        return _error_result(exc)


def find_last_seen(client: PersonalAssistantClient, query: str) -> dict[str, Any]:
    if not query.strip():
        return {"ok": False, "spoken": "What should I look up in visual memory?"}
    try:
        result = client.find_last_seen(query.strip())
        return {
            "ok": True,
            "spoken": result.phrase,
            "presence": result.presence,
            "count": len(result.items),
            "may_move_head": False,
        }
    except Exception as exc:
        return _error_result(exc)


def search_visual_memory(client: PersonalAssistantClient, query: str) -> dict[str, Any]:
    if not query.strip():
        return {"ok": False, "spoken": "What should I search for in visual memory?"}
    try:
        result = client.search_visual_memory(query.strip())
        return {
            "ok": True,
            "spoken": result.phrase,
            "presence": result.presence,
            "may_move_head": False,
        }
    except Exception as exc:
        return _error_result(exc)


def describe_previous_scene(client: PersonalAssistantClient, snapshot_id: str) -> dict[str, Any]:
    if not snapshot_id.strip():
        return {"ok": False, "spoken": "Which previous look should I describe?"}
    try:
        result = client.describe_previous_scene(snapshot_id.strip())
        return {
            "ok": True,
            "spoken": result.phrase,
            "labels": result.labels,
            "may_move_head": False,
        }
    except Exception as exc:
        return _error_result(exc)


def compare_visual_scenes(
    client: PersonalAssistantClient, *, left_snapshot_id: str, right_snapshot_id: str
) -> dict[str, Any]:
    try:
        result = client.compare_visual_scenes(left_snapshot_id, right_snapshot_id)
        return {
            "ok": True,
            "spoken": result.phrase,
            "appeared": result.appeared,
            "disappeared": result.disappeared,
            "may_move_head": False,
        }
    except Exception as exc:
        return _error_result(exc)


def create_visual_watch(
    client: PersonalAssistantClient,
    *,
    label: str,
    event: str = "change",
    zone: str | None = None,
    idempotency_key: str | None = None,
) -> dict[str, Any]:
    if not label.strip():
        return {"ok": False, "spoken": "What should I watch for?"}
    try:
        result = client.create_visual_watch(
            label=label.strip(), event=event, zone=zone, idempotency_key=idempotency_key
        )
        return {
            "ok": True,
            "spoken": result.phrase,
            "watch_id": result.id,
            "may_move_head": False,
        }
    except Exception as exc:
        return _error_result(exc)


def list_visual_watches(client: PersonalAssistantClient) -> dict[str, Any]:
    try:
        result = client.list_visual_watches()
        return {
            "ok": True,
            "spoken": result.phrase or f"{result.total} visual watch(es).",
            "count": result.total,
            "ids": [item.id for item in result.items],
            "may_move_head": False,
        }
    except Exception as exc:
        return _error_result(exc)


def cancel_visual_watch(client: PersonalAssistantClient, watch_id: str) -> dict[str, Any]:
    if not watch_id.strip():
        return {"ok": False, "spoken": "Which watch should I cancel?"}
    try:
        result = client.cancel_visual_watch(watch_id.strip())
        return {"ok": True, "spoken": result.phrase, "watch_id": result.id, "may_move_head": False}
    except Exception as exc:
        return _error_result(exc)
