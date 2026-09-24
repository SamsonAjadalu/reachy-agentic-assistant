"""Reminder endpoints.

All of these are fast local operations, so every one answers synchronously.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, Query, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, get_settings
from app.dependencies.auth import Principal, require_token
from app.dependencies.common import Pagination, get_session, idempotency_key, pagination
from app.schemas.common import DeleteResponse
from app.schemas.reminders import (
    ReminderCreate,
    ReminderList,
    ReminderOut,
    ReminderSnooze,
    ReminderUpdate,
)
from app.services import idempotency as idem
from app.services import reminders as service
from shared.enums import ReminderStatus
from shared.timeutils import duration_from_parts

router = APIRouter(prefix="/api/v1/reminders", tags=["reminders"])

SessionDep = Annotated[AsyncSession, Depends(get_session)]
SettingsDep = Annotated[Settings, Depends(get_settings)]
PrincipalDep = Annotated[Principal, Depends(require_token)]
PaginationDep = Annotated[Pagination, Depends(pagination)]
IdempotencyDep = Annotated[str | None, Depends(idempotency_key)]

IDEMPOTENCY_SCOPE = "reminders.create"


@router.post(
    "",
    response_model=ReminderOut,
    status_code=201,
    summary="Create a reminder",
    description=(
        "Accepts either an absolute time (`schedule.at`, ISO-8601 with an offset) or a "
        "structured relative delay (`schedule.delay`). Free-text times are not parsed here: "
        "the conversational model resolves them before calling. Supply `Idempotency-Key` to "
        "make a retried voice turn safe."
    ),
)
async def create_reminder(
    payload: ReminderCreate,
    request: Request,
    _: PrincipalDep,
    session: SessionDep,
    settings: SettingsDep,
    key: IdempotencyDep,
) -> ReminderOut:
    replay = await idem.lookup(session, IDEMPOTENCY_SCOPE, key, payload)
    if replay is not None:
        return ReminderOut.model_validate(replay)

    reminder = await service.create_reminder(
        session,
        payload,
        settings=settings,
        idempotency_key=key,
        correlation_id=getattr(request.state, "correlation_id", None),
    )
    result = await service.to_out(session, reminder)
    await idem.remember(
        session,
        IDEMPOTENCY_SCOPE,
        key,
        payload,
        result.model_dump(mode="json"),
        resource_id=reminder.id,
    )
    return result


@router.get("", response_model=ReminderList, summary="List reminders")
async def list_reminders(
    _: PrincipalDep,
    session: SessionDep,
    page: PaginationDep,
    status: Annotated[ReminderStatus | None, Query()] = None,
    upcoming_only: Annotated[
        bool, Query(description="Only future, still-active reminders.")
    ] = False,
    before: Annotated[
        datetime | None, Query(description="Only reminders due at or before this instant.")
    ] = None,
) -> ReminderList:
    rows, total = await service.list_reminders(
        session,
        status=status,
        upcoming_only=upcoming_only,
        before=before,
        limit=page.limit,
        offset=page.offset,
    )
    return ReminderList(
        items=[await service.to_out(session, row) for row in rows],
        total=total,
        limit=page.limit,
        offset=page.offset,
    )


@router.get("/{reminder_id}", response_model=ReminderOut, summary="Get one reminder")
async def get_reminder(reminder_id: str, _: PrincipalDep, session: SessionDep) -> ReminderOut:
    return await service.to_out(session, await service.get_reminder(session, reminder_id))


@router.patch("/{reminder_id}", response_model=ReminderOut, summary="Edit a reminder")
async def update_reminder(
    reminder_id: str, payload: ReminderUpdate, _: PrincipalDep, session: SessionDep
) -> ReminderOut:
    reminder = await service.update_reminder(session, reminder_id, payload)
    return await service.to_out(session, reminder)


@router.post(
    "/{reminder_id}/snooze",
    response_model=ReminderOut,
    summary="Snooze a reminder",
    description="Pushes the reminder out by a structured delay. Defaults to ten minutes.",
)
async def snooze_reminder(
    reminder_id: str, payload: ReminderSnooze, _: PrincipalDep, session: SessionDep
) -> ReminderOut:
    delay = duration_from_parts(
        days=payload.delay.days,
        hours=payload.delay.hours,
        minutes=payload.delay.minutes,
        seconds=payload.delay.seconds,
    )
    reminder = await service.snooze_reminder(session, reminder_id, delay)
    return await service.to_out(session, reminder)


@router.post("/{reminder_id}/complete", response_model=ReminderOut, summary="Mark done")
async def complete_reminder(reminder_id: str, _: PrincipalDep, session: SessionDep) -> ReminderOut:
    reminder = await service.complete_reminder(session, reminder_id)
    return await service.to_out(session, reminder)


@router.delete("/{reminder_id}", response_model=DeleteResponse, summary="Cancel a reminder")
async def cancel_reminder(reminder_id: str, _: PrincipalDep, session: SessionDep) -> DeleteResponse:
    reminder = await service.cancel_reminder(session, reminder_id)
    return DeleteResponse(id=reminder.id)
