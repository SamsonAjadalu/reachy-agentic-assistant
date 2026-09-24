"""Visual watches: standing queries, scheduler evaluation, Telegram photos."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings
from database.models.vision import EvidenceAsset, VisualEvent, VisualWatch
from notifications.dispatcher import dispatch
from shared.enums import AlertSeverity, NotificationChannel
from shared.errors import NotFoundError, ValidationError
from shared.timeutils import utcnow
from vision.backend import build_backend
from vision.camera import build_camera
from vision.change import structural_dedup_key
from vision.enums import ObservationTrigger, SnapshotKind, VisualEventType, VisualWatchStatus
from vision.phrases import event_verb, render
from vision.pipeline import create_request, create_snapshot, observe_frame
from vision.schemas import dump_json, load_json

EVENT_ALIASES = {
    "appear": "appeared",
    "appeared": "appeared",
    "disappear": "disappeared",
    "disappeared": "disappeared",
    "move": "moved",
    "moved": "moved",
    "change": "changed",
    "changed": "changed",
}


async def create_watch(
    session: AsyncSession,
    *,
    query: str | dict[str, Any],
    event_types: list[str] | None = None,
    cooldown_seconds: int = 3600,
    zone_id: str | None = None,
    quiet_hours: dict[str, Any] | None = None,
    idempotency_key: str | None = None,
) -> VisualWatch:
    if isinstance(query, dict):
        text = str(query.get("label") or query.get("query") or "").strip()
        if event_types is None and query.get("event"):
            event_types = [str(query["event"])]
        payload_query = query
    else:
        text = query.strip()
        payload_query = {"query": text}
    if not text:
        raise ValidationError("Watch query cannot be empty.")
    if event_types:
        event_types = [EVENT_ALIASES.get(item.lower(), item.lower()) for item in event_types]
    if idempotency_key:
        existing = (
            await session.execute(
                select(VisualWatch).where(VisualWatch.idempotency_key == idempotency_key)
            )
        ).scalar_one_or_none()
        if existing is not None:
            return existing
    types = event_types or [
        VisualEventType.APPEARED.value,
        VisualEventType.DISAPPEARED.value,
        VisualEventType.MOVED.value,
        VisualEventType.CHANGED.value,
    ]
    payload = {**payload_query, "query": text, "event_types": types}
    now = utcnow()
    row = VisualWatch(
        query_json=dump_json(payload),
        trigger="schedule",
        status=VisualWatchStatus.ACTIVE.value,
        zone_id=zone_id,
        next_check_at=now,
        cooldown_seconds=max(30, cooldown_seconds),
        quiet_hours_json=dump_json(quiet_hours) if quiet_hours else None,
        idempotency_key=idempotency_key,
    )
    session.add(row)
    await session.flush()
    return row


async def list_watches(
    session: AsyncSession,
    *,
    status: str | None = None,
    limit: int = 50,
    offset: int = 0,
    include_inactive: bool = False,
) -> tuple[list[VisualWatch], int]:
    from sqlalchemy import func

    stmt = select(VisualWatch)
    count_stmt = select(func.count()).select_from(VisualWatch)
    if status:
        stmt = stmt.where(VisualWatch.status == status)
        count_stmt = count_stmt.where(VisualWatch.status == status)
    elif not include_inactive:
        stmt = stmt.where(VisualWatch.status == VisualWatchStatus.ACTIVE.value)
        count_stmt = count_stmt.where(VisualWatch.status == VisualWatchStatus.ACTIVE.value)
    total = int((await session.execute(count_stmt)).scalar_one())
    rows = list(
        (
            await session.execute(
                stmt.order_by(VisualWatch.created_at.desc()).offset(offset).limit(limit)
            )
        ).scalars()
    )
    return rows, total


async def cancel_watch(session: AsyncSession, watch_id: str) -> VisualWatch:
    row = await session.get(VisualWatch, watch_id)
    if row is None:
        raise NotFoundError("Visual watch not found.")
    row.status = VisualWatchStatus.CANCELLED.value
    row.next_check_at = None
    await session.flush()
    return row


def watch_as_dict(row: VisualWatch) -> dict[str, Any]:
    payload = load_json(row.query_json, {})
    return {
        "id": row.id,
        "query": payload.get("query"),
        "event_types": payload.get("event_types"),
        "status": row.status,
        "next_check_at": None if row.next_check_at is None else row.next_check_at.isoformat(),
        "cooldown_seconds": row.cooldown_seconds,
        "last_fired_at": None if row.last_fired_at is None else row.last_fired_at.isoformat(),
        "zone_id": row.zone_id,
    }


async def evaluate_due_watches(session: AsyncSession, settings: Settings) -> dict[str, int]:
    if not settings.visual_enabled:
        return {"polled": 0, "fired": 0, "skipped_quiet": 0}
    now = utcnow()
    rows = list(
        (
            await session.execute(
                select(VisualWatch).where(
                    VisualWatch.status == VisualWatchStatus.ACTIVE.value,
                    VisualWatch.next_check_at.is_not(None),
                    VisualWatch.next_check_at <= now,
                )
            )
        ).scalars()
    )
    fired = 0
    skipped_quiet = 0
    camera = build_camera(settings)
    backend = await build_backend(settings)
    try:
        for watch in rows:
            if _in_quiet_hours(watch, now, settings.app_timezone):
                watch.next_check_at = now + timedelta(seconds=watch.cooldown_seconds)
                skipped_quiet += 1
                continue
            payload = load_json(watch.query_json, {})
            query = str(payload.get("query") or "object")
            wanted = set(payload.get("event_types") or [])
            request = await create_request(
                session,
                trigger=ObservationTrigger.WATCH,
                preset_id=None,
                query=query,
                correlation_id=None,
            )
            snapshot = await create_snapshot(
                session, kind=SnapshotKind.WATCH, request_id=request.id, zone_id=watch.zone_id
            )
            frame = await camera.get_latest_frame()
            result = await observe_frame(
                session,
                settings,
                frame,
                backend=backend,
                query=query,
                trigger=ObservationTrigger.WATCH,
                snapshot_id=snapshot.id,
                request_id=request.id,
                zone_id=watch.zone_id,
            )
            matching = [event for event in result.events if event["event_type"] in wanted]
            watch.next_check_at = now + timedelta(seconds=watch.cooldown_seconds)
            if not matching:
                continue
            if await _notify_watch(session, settings, watch, matching, result.thumbnail_id, query):
                fired += 1
                watch.last_fired_at = now
    finally:
        await camera.aclose()
        await backend.aclose()
    return {"polled": len(rows), "due": len(rows), "fired": fired, "skipped_quiet": skipped_quiet}


async def _notify_watch(
    session: AsyncSession,
    settings: Settings,
    watch: VisualWatch,
    events: list[dict[str, Any]],
    thumbnail_id: str | None,
    query: str,
) -> bool:
    event = events[0]
    key = structural_dedup_key(
        watch_id=watch.id,
        event_type=event["event_type"],
        label=event.get("label") or query,
        observation_id=event.get("observation_id"),
    )
    # Bind the watch onto a VisualEvent row; uniqueness is structural.
    existing = (
        await session.execute(select(VisualEvent).where(VisualEvent.dedup_key == key))
    ).scalar_one_or_none()
    if existing is None:
        row = VisualEvent(
            occurred_after=utcnow(),
            occurred_at=utcnow(),
            event_type=event["event_type"],
            watch_id=watch.id,
            zone_id=watch.zone_id,
            after_evidence_id=thumbnail_id,
            payload_json=dump_json(event),
            dedup_key=key,
        )
        session.add(row)
        await session.flush()
    attachment = None
    if thumbnail_id:
        asset = await session.get(EvidenceAsset, thumbnail_id)
        if asset is not None:
            attachment = str(settings.visual_evidence_path / asset.relative_path)
    phrase = render(
        "watch_fired",
        label=event.get("label") or query,
        event=event_verb(event["event_type"]),
        zone="view",
    )
    await dispatch(
        session,
        kind="visual_watch",
        title="Visual watch",
        body=phrase,
        severity=AlertSeverity.INFO,
        channel=NotificationChannel.TELEGRAM,
        dedup_key=key,
        resource_type="visual_watch",
        resource_id=watch.id,
        attachment_path=attachment,
        settings=settings,
    )
    return True


def in_quiet_hours(watch: Any, now: datetime | None = None, timezone: str | None = None) -> bool:
    when = now or utcnow()
    zone = timezone or "UTC"
    return _in_quiet_hours(watch, when, zone)


def _in_quiet_hours(watch: Any, now: datetime, default_tz: str) -> bool:
    payload = load_json(watch.quiet_hours_json, None)
    if not payload:
        return False
    start = str(payload.get("start") or "")
    end = str(payload.get("end") or "")
    if not start or not end:
        return False
    timezone = ZoneInfo(str(payload.get("timezone") or default_tz))
    local = now.astimezone(timezone)
    start_h, start_m = _parse_hhmm(start)
    end_h, end_m = _parse_hhmm(end)
    minutes = local.hour * 60 + local.minute
    start_min = start_h * 60 + start_m
    end_min = end_h * 60 + end_m
    if start_min == end_min:
        return False
    if start_min < end_min:
        return start_min <= minutes < end_min
    return minutes >= start_min or minutes < end_min


def _parse_hhmm(value: str) -> tuple[int, int]:
    parts = value.strip().split(":")
    return int(parts[0]), int(parts[1]) if len(parts) > 1 else 0
