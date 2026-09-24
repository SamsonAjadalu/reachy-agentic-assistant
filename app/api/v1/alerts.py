"""Alert endpoints.

Evaluation is exposed so the owner can ask "is anything wrong?" directly rather
than waiting for the scheduled sweep, and so the cooldown ledger is inspectable
when an expected alert did not arrive.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, get_settings
from app.dependencies.auth import Principal, require_token
from app.dependencies.common import get_session
from app.schemas.common import ApiModel, DeleteResponse
from database.models import AlertDeduplication, Notification
from notifications.dispatcher import clear_dedup
from proactive import alerts as service
from shared.errors import ValidationError

router = APIRouter(prefix="/api/v1/alerts", tags=["alerts"])

SessionDep = Annotated[AsyncSession, Depends(get_session)]
SettingsDep = Annotated[Settings, Depends(get_settings)]
PrincipalDep = Annotated[Principal, Depends(require_token)]


class NotificationOut(ApiModel):
    id: str
    kind: str
    severity: str
    title: str
    body: str
    channel: str
    status: str
    created_at: Any
    sent_at: Any = None


class CooldownOut(ApiModel):
    dedup_key: str
    alert_kind: str
    first_seen_at: Any
    last_notified_at: Any
    cooldown_seconds: int
    notify_count: int


@router.get("/rules", summary="Which conditions are checked")
async def rules(_: PrincipalDep) -> dict[str, Any]:
    return {"rules": service.rule_names()}


@router.post(
    "/evaluate",
    summary="Check every condition now",
    description="Set 'dry_run' to see what would fire without sending anything.",
)
@router.post(
    "/test",
    summary="Check every condition now",
    description="Alias of /evaluate.",
    include_in_schema=False,
)
async def evaluate(
    _: PrincipalDep,
    session: SessionDep,
    settings: SettingsDep,
    dry_run: bool = False,
    only: Annotated[str | None, Query(max_length=200)] = None,
) -> dict[str, Any]:
    wanted = [part.strip() for part in only.split(",") if part.strip()] if only else None
    if wanted:
        unknown = sorted(set(wanted) - set(service.rule_names()))
        if unknown:
            raise ValidationError(f"Unknown alert rule(s): {', '.join(unknown)}.")

    result = await service.evaluate(session, settings, rules=wanted, dry_run=dry_run)
    await session.commit()
    return result


@router.get("", summary="Recent notifications", include_in_schema=False)
@router.get("/history", summary="Recent notifications")
async def history(
    _: PrincipalDep,
    session: SessionDep,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    severity: Annotated[str | None, Query(max_length=20)] = None,
) -> dict[str, Any]:
    query = select(Notification).order_by(Notification.created_at.desc()).limit(limit)
    if severity:
        query = query.where(Notification.severity == severity)

    rows = list(await session.scalars(query))
    return {
        "notifications": [
            NotificationOut(
                id=row.id,
                kind=row.kind,
                severity=row.severity,
                title=row.title,
                body=row.body,
                channel=row.channel,
                status=row.status,
                created_at=row.created_at,
                sent_at=row.sent_at,
            )
            for row in rows
        ],
        "total": len(rows),
    }


@router.get(
    "/cooldowns",
    summary="Conditions currently in their cooldown window",
    description="Explains why a condition that is still true is not notifying again.",
)
async def cooldowns(_: PrincipalDep, session: SessionDep) -> dict[str, Any]:
    rows = list(
        await session.scalars(
            select(AlertDeduplication).order_by(AlertDeduplication.last_notified_at.desc())
        )
    )
    return {
        "cooldowns": [
            CooldownOut(
                dedup_key=row.dedup_key,
                alert_kind=row.alert_kind,
                first_seen_at=row.first_seen_at,
                last_notified_at=row.last_notified_at,
                cooldown_seconds=row.cooldown_seconds,
                notify_count=row.notify_count,
            )
            for row in rows
        ],
        "total": len(rows),
    }


@router.delete(
    "/cooldowns/{dedup_key:path}",
    response_model=DeleteResponse,
    summary="Forget a cooldown so the condition notifies again",
)
async def reset_cooldown(dedup_key: str, _: PrincipalDep, session: SessionDep) -> DeleteResponse:
    await clear_dedup(session, dedup_key)
    await session.commit()
    return DeleteResponse(id=dedup_key, deleted=True)
