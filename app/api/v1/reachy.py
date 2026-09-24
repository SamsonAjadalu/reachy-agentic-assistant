"""The robot-facing surface.

The workstation cannot make Reachy speak. What it can do is hold a queue of things
worth saying and let the Pi collect them at the start of a conversation, which
is the only moment the robot has the owner's attention anyway.

Collection is deliberately two-step: fetching does not acknowledge. If the Pi
crashes between reading and speaking, the message is still waiting rather than
silently consumed.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query
from pydantic import Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, get_settings
from app.dependencies.auth import Principal, require_token
from app.dependencies.common import get_session
from app.schemas.common import ApiModel
from database.models import PendingReachyNotification
from shared.enums import AlertSeverity
from shared.errors import NotFoundError, ValidationError
from shared.timeutils import utcnow

router = APIRouter(prefix="/api/v1/reachy", tags=["reachy"])

SessionDep = Annotated[AsyncSession, Depends(get_session)]
SettingsDep = Annotated[Settings, Depends(get_settings)]
PrincipalDep = Annotated[Principal, Depends(require_token)]

MAX_BATCH = 10


class PendingOut(ApiModel):
    id: str
    kind: str
    severity: str
    spoken_text: str
    detail: str | None = None
    created_at: Any
    resource_type: str | None = None
    resource_id: str | None = None


class PendingList(ApiModel):
    items: list[PendingOut] = Field(default_factory=list)
    total: int
    spoken_text: str = ""
    """Everything joined into one utterance, so the Pi can speak it in one turn."""


class AcknowledgeRequest(ApiModel):
    ids: list[str] = Field(default_factory=list, max_length=MAX_BATCH)


class QueueRequest(ApiModel):
    kind: Annotated[str, Field(min_length=1, max_length=60)]
    spoken_text: Annotated[str, Field(min_length=1, max_length=1000)]
    detail: Annotated[str | None, Field(max_length=4000)] = None
    severity: AlertSeverity = AlertSeverity.INFO


async def _collect(session: AsyncSession, limit: int) -> PendingList:
    now = utcnow()
    rows = list(
        await session.scalars(
            select(PendingReachyNotification)
            .where(PendingReachyNotification.acknowledged.is_(False))
            .order_by(PendingReachyNotification.created_at)
            .limit(limit)
        )
    )
    live = [row for row in rows if row.expires_at is None or row.expires_at > now]

    items = [
        PendingOut(
            id=row.id,
            kind=row.kind,
            severity=row.severity,
            spoken_text=row.spoken_text,
            detail=row.detail,
            created_at=row.created_at,
            resource_type=row.resource_type,
            resource_id=row.resource_id,
        )
        for row in live
    ]
    return PendingList(
        items=items,
        total=len(items),
        spoken_text=" ".join(item.spoken_text.rstrip(".") + "." for item in items),
    )


async def _acknowledge_ids(session: AsyncSession, ids: list[str]) -> int:
    if not ids:
        raise ValidationError("Provide at least one notification id.")
    now = utcnow()
    acknowledged = 0
    for identifier in ids:
        row = await session.get(PendingReachyNotification, identifier)
        if row is None:
            raise NotFoundError(f"No pending notification {identifier}.")
        if not row.acknowledged:
            row.acknowledged = True
            row.acknowledged_at = now
            acknowledged += 1
    await session.commit()
    return acknowledged


@router.get(
    "/pending",
    response_model=PendingList,
    summary="What Reachy should mention",
    description=(
        "Reading does not acknowledge. Call the acknowledge endpoint once the robot has "
        "actually said it, so a crash mid-turn does not lose the message."
    ),
)
@router.get(
    "/pending-notifications",
    response_model=PendingList,
    summary="What Reachy should mention",
    include_in_schema=False,
)
async def pending(
    _: PrincipalDep,
    session: SessionDep,
    limit: Annotated[int, Query(ge=1, le=MAX_BATCH)] = 5,
) -> PendingList:
    return await _collect(session, limit)


@router.post("/pending/acknowledge", summary="Mark messages as delivered")
async def acknowledge_batch(
    payload: AcknowledgeRequest, _: PrincipalDep, session: SessionDep
) -> dict[str, Any]:
    return {"acknowledged": await _acknowledge_ids(session, list(payload.ids))}


@router.post(
    "/pending-notifications/{notification_id}/acknowledge",
    summary="Mark one message as delivered",
)
async def acknowledge_one(
    notification_id: str, _: PrincipalDep, session: SessionDep
) -> dict[str, Any]:
    return {"acknowledged": await _acknowledge_ids(session, [notification_id])}


@router.post("/pending", response_model=PendingOut, status_code=201, summary="Queue a message")
async def queue(payload: QueueRequest, _: PrincipalDep, session: SessionDep) -> PendingOut:
    row = PendingReachyNotification(
        kind=payload.kind,
        severity=payload.severity.value,
        spoken_text=payload.spoken_text,
        detail=payload.detail,
    )
    session.add(row)
    await session.commit()
    await session.refresh(row)
    return PendingOut(
        id=row.id,
        kind=row.kind,
        severity=row.severity,
        spoken_text=row.spoken_text,
        detail=row.detail,
        created_at=row.created_at,
    )


@router.get(
    "/greeting",
    summary="What to say when a conversation starts",
    description=(
        "Combines anything queued with a short status line. Intended as the first thing "
        "the Pi asks for when it detects the owner."
    ),
)
async def greeting(_: PrincipalDep, session: SessionDep, settings: SettingsDep) -> dict[str, Any]:
    from proactive.briefing import assemble

    waiting = await _collect(session, 3)
    briefing = await assemble(
        session, settings, sections=["calendar", "tasks"], formatter_name="spoken"
    )

    parts = [briefing.text]
    if waiting.items:
        parts.append(waiting.spoken_text)

    return {
        "spoken_text": " ".join(parts),
        "pending_count": waiting.total,
        "pending_ids": [item.id for item in waiting.items],
    }
