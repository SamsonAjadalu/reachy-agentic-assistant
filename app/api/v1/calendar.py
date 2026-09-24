"""Calendar endpoints.

Reads and dry-run proposals are synchronous. Creating, updating or deleting an
event is a write other people can see, so those go through approval.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Annotated

from fastapi import APIRouter, Depends, Query, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, get_settings
from app.dependencies.auth import Principal, require_token
from app.dependencies.common import get_session, idempotency_key
from app.schemas.google import (
    ApprovalTicket,
    CreateEventRequest,
    EventListResponse,
    EventProposalResponse,
    FreeBusyResponse,
    ProposeEventRequest,
    UpdateEventRequest,
)
from approvals.state_machine import request_approval
from integrations.google.calendar import suggest_event_slots
from integrations.google.executors import (
    CREATE_EVENT,
    DELETE_EVENT,
    UPDATE_EVENT,
    event_preview,
    event_update_preview,
)
from integrations.google.models import FreeBusySlot, ProposedSlot
from integrations.registry import get_calendar, require_enabled
from shared.enums import RiskLevel
from shared.errors import ValidationError
from shared.timeutils import end_of_local_day, ensure_utc, start_of_local_day, utcnow

router = APIRouter(prefix="/api/v1/calendar", tags=["calendar"])

SessionDep = Annotated[AsyncSession, Depends(get_session)]
SettingsDep = Annotated[Settings, Depends(get_settings)]
PrincipalDep = Annotated[Principal, Depends(require_token)]
IdempotencyDep = Annotated[str | None, Depends(idempotency_key)]


@router.get("/events", response_model=EventListResponse, summary="List events in a range")
async def list_events(
    _: PrincipalDep,
    settings: SettingsDep,
    starts_after: Annotated[datetime | None, Query()] = None,
    ends_before: Annotated[datetime | None, Query()] = None,
    days: Annotated[int, Query(ge=1, le=90, description="Used when no range is given.")] = 7,
    limit: Annotated[int, Query(ge=1, le=100)] = 25,
) -> EventListResponse:
    start = ensure_utc(starts_after) if starts_after else utcnow()
    end = ensure_utc(ends_before) if ends_before else start + timedelta(days=days)
    items = await get_calendar(settings).list_events(
        starts_after=start, ends_before=end, limit=limit
    )
    return EventListResponse(items=items, total=len(items))


@router.get("/today", response_model=EventListResponse, summary="Today's events")
async def today(
    _: PrincipalDep,
    settings: SettingsDep,
    limit: Annotated[int, Query(ge=1, le=100)] = 25,
) -> EventListResponse:
    """Today in the owner's timezone, not in UTC: 'today' is a local idea."""
    now = utcnow()
    items = await get_calendar(settings).list_events(
        starts_after=start_of_local_day(now, settings.app_timezone),
        ends_before=end_of_local_day(now, settings.app_timezone),
        limit=limit,
    )
    return EventListResponse(items=items, total=len(items))


@router.get("/next", response_model=EventListResponse, summary="The next few events")
async def next_events(
    _: PrincipalDep,
    settings: SettingsDep,
    limit: Annotated[int, Query(ge=1, le=10)] = 3,
) -> EventListResponse:
    now = utcnow()
    items = await get_calendar(settings).list_events(
        starts_after=now, ends_before=now + timedelta(days=30), limit=limit
    )
    return EventListResponse(items=items, total=len(items))


@router.get("/free-busy", response_model=FreeBusyResponse, summary="Busy and free blocks")
async def free_busy(
    _: PrincipalDep,
    settings: SettingsDep,
    starts_at: Annotated[datetime, Query()],
    ends_at: Annotated[datetime, Query()],
) -> FreeBusyResponse:
    start, end = ensure_utc(starts_at), ensure_utc(ends_at)
    busy = await get_calendar(settings).free_busy(starts_at=start, ends_at=end)
    return FreeBusyResponse(
        busy=busy,
        free_slots=_invert(busy, start, end),
        queried_from=start,
        queried_to=end,
    )


@router.get("/events/{event_id}", summary="Get one event")
async def get_event(event_id: str, _: PrincipalDep, settings: SettingsDep) -> dict[str, object]:
    event = await get_calendar(settings).get_event(event_id)
    return {"event": event.model_dump(mode="json")}


@router.post(
    "/events/propose",
    response_model=EventProposalResponse,
    summary="Propose event times (dry-run)",
    description=(
        "Uses free/busy to suggest slots. Does not create a calendar event and does "
        "not raise an approval."
    ),
)
async def propose_event(
    payload: ProposeEventRequest,
    _: PrincipalDep,
    settings: SettingsDep,
) -> EventProposalResponse:
    require_enabled("calendar", settings)
    duration = timedelta(minutes=payload.duration_minutes)
    search_from = ensure_utc(payload.search_from) if payload.search_from else utcnow()
    search_to = (
        ensure_utc(payload.search_to) if payload.search_to else search_from + timedelta(days=7)
    )
    preferred = ensure_utc(payload.starts_at) if payload.starts_at else None

    calendar = get_calendar(settings)
    busy = await calendar.free_busy(
        starts_at=search_from, ends_at=search_to, calendar_id=payload.calendar_id
    )
    suggestions_raw = suggest_event_slots(
        busy=busy,
        search_from=search_from,
        search_to=search_to,
        duration=duration,
        preferred_start=preferred,
        max_suggestions=payload.max_suggestions,
    )
    if not suggestions_raw:
        raise ValidationError("No free slots found in the requested window.")

    suggestions = [
        ProposedSlot(starts_at=start, ends_at=end, conflicts=conflicts)
        for start, end, conflicts in suggestions_raw
    ]
    first = suggestions[0]
    draft = CreateEventRequest(
        title=payload.title,
        starts_at=first.starts_at,
        ends_at=first.ends_at,
        description=payload.description,
        location=payload.location,
        attendees=payload.attendees,
        timezone=payload.timezone or settings.app_timezone,
        calendar_id=payload.calendar_id,
    )
    preview_payload = {
        "title": draft.title,
        "starts_at": draft.starts_at.isoformat(),
        "ends_at": draft.ends_at.isoformat(),
        "description": draft.description,
        "location": draft.location,
        "attendees": draft.attendees,
    }
    return EventProposalResponse(
        title=payload.title,
        duration_minutes=payload.duration_minutes,
        suggestions=suggestions,
        preview_text=event_preview(preview_payload),
        draft_create_payload=draft,
    )


@router.post(
    "/events",
    response_model=ApprovalTicket,
    status_code=202,
    summary="Request to create an event",
    description="Creates a pending action. The event appears only after the owner approves.",
)
async def create_event(
    payload: CreateEventRequest,
    request: Request,
    _: PrincipalDep,
    session: SessionDep,
    settings: SettingsDep,
    key: IdempotencyDep,
) -> ApprovalTicket:
    require_enabled("calendar", settings)

    action_payload = {
        "title": payload.title,
        "starts_at": ensure_utc(payload.starts_at).isoformat(),
        "ends_at": ensure_utc(payload.ends_at).isoformat(),
        "description": payload.description,
        "location": payload.location,
        "attendees": payload.attendees,
        "timezone": payload.timezone or settings.app_timezone,
        "calendar_id": payload.calendar_id,
    }

    action, approval = await request_approval(
        session,
        action_type=CREATE_EVENT,
        summary=f"Add '{payload.title}' to the calendar",
        payload=action_payload,
        preview_text=event_preview(action_payload),
        risk_level=RiskLevel.EXTERNAL_WRITE,
        settings=settings,
        idempotency_key=key,
        correlation_id=getattr(request.state, "correlation_id", None),
    )
    return ApprovalTicket(
        approval_id=approval.id,
        action_id=action.id,
        summary=action.summary,
        preview=action.preview_text,
        expires_at=approval.expires_at,
    )


@router.patch(
    "/events/{event_id}",
    response_model=ApprovalTicket,
    status_code=202,
    summary="Request to update an event",
    description="Creates a pending action. The calendar is changed only after approval.",
)
async def update_event(
    event_id: str,
    payload: UpdateEventRequest,
    request: Request,
    _: PrincipalDep,
    session: SessionDep,
    settings: SettingsDep,
    key: IdempotencyDep,
) -> ApprovalTicket:
    require_enabled("calendar", settings)
    event = await get_calendar(settings).get_event(event_id, calendar_id=payload.calendar_id)
    action_payload: dict[str, object] = {
        "event_id": event_id,
        "calendar_id": payload.calendar_id,
        "timezone": payload.timezone or settings.app_timezone,
    }
    before = {
        "title": event.title,
        "starts_at": event.starts_at.isoformat(),
        "ends_at": event.ends_at.isoformat(),
        "location": event.location,
        "description": event.description,
        "attendees": event.attendees,
    }
    for field in ("title", "description", "location", "attendees"):
        value = getattr(payload, field)
        if value is not None:
            action_payload[field] = value
    if payload.starts_at is not None and payload.ends_at is not None:
        action_payload["starts_at"] = ensure_utc(payload.starts_at).isoformat()
        action_payload["ends_at"] = ensure_utc(payload.ends_at).isoformat()

    action, approval = await request_approval(
        session,
        action_type=UPDATE_EVENT,
        summary=f"Update '{event.title}' on the calendar",
        payload=action_payload,
        preview_text=event_update_preview(before, action_payload),
        risk_level=RiskLevel.EXTERNAL_WRITE,
        settings=settings,
        idempotency_key=key,
        correlation_id=getattr(request.state, "correlation_id", None),
    )
    return ApprovalTicket(
        approval_id=approval.id,
        action_id=action.id,
        summary=action.summary,
        preview=action.preview_text,
        expires_at=approval.expires_at,
    )


@router.delete(
    "/events/{event_id}",
    response_model=ApprovalTicket,
    status_code=202,
    summary="Request to delete an event",
)
async def delete_event(
    event_id: str,
    request: Request,
    _: PrincipalDep,
    session: SessionDep,
    settings: SettingsDep,
    key: IdempotencyDep,
    calendar_id: Annotated[str, Query()] = "primary",
) -> ApprovalTicket:
    require_enabled("calendar", settings)

    # Naming the event in the prompt: "delete evt-8fa2" is not something anyone
    # can meaningfully approve.
    event = await get_calendar(settings).get_event(event_id, calendar_id=calendar_id)
    action_payload = {"event_id": event_id, "calendar_id": calendar_id}

    action, approval = await request_approval(
        session,
        action_type=DELETE_EVENT,
        summary=f"Delete '{event.title}' from the calendar",
        payload=action_payload,
        preview_text=(
            f"Delete this event:\n\n{event.title}\n"
            f"{event.starts_at.isoformat()} to {event.ends_at.isoformat()}"
        ),
        risk_level=RiskLevel.EXTERNAL_WRITE,
        settings=settings,
        idempotency_key=key,
        correlation_id=getattr(request.state, "correlation_id", None),
    )
    return ApprovalTicket(
        approval_id=approval.id,
        action_id=action.id,
        summary=action.summary,
        preview=action.preview_text,
        expires_at=approval.expires_at,
    )


def _invert(busy: list[FreeBusySlot], start: datetime, end: datetime) -> list[FreeBusySlot]:
    """Turn busy blocks into the gaps between them."""
    free: list[FreeBusySlot] = []
    cursor = start
    for slot in sorted(busy, key=lambda entry: entry.starts_at):
        if slot.starts_at > cursor:
            free.append(FreeBusySlot(starts_at=cursor, ends_at=slot.starts_at))
        cursor = max(cursor, slot.ends_at)
    if cursor < end:
        free.append(FreeBusySlot(starts_at=cursor, ends_at=end))
    return free
