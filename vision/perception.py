"""Active perception: look_now, last-seen, search, compare, scans (preset_id only)."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings
from database.models.vision import (
    ObjectDetection,
    ObjectEntity,
    ObservationRequest,
    VisualObservation,
    VisualZone,
)
from integrations.reachy.camera.models import ScanAccepted, ScanRejected
from shared.errors import ValidationError
from shared.timeutils import utcnow
from vision.backend import PerceptionBackend, build_backend
from vision.camera import assert_preset_only, build_camera
from vision.enums import (
    EntityStatus,
    LookOutcome,
    ObservationRequestStatus,
    ObservationTrigger,
    Presence,
    ScanOutcomeKind,
    SnapshotKind,
    SnapshotState,
)
from vision.labels import normalise_label
from vision.need_look import should_scan
from vision.phrases import qualitative_when, render
from vision.pipeline import create_request, create_snapshot, observe_frame
from vision.results import SceneResult
from workers.queue import enqueue

SCAN_TASK = "vision.scan"

DEFAULT_PRESET = "CLOSE_LOOK"


async def get_camera_status(settings: Settings) -> dict[str, Any]:
    camera = build_camera(settings)
    try:
        status = await camera.get_status()
        return {
            "adapter_version": status.adapter_version,
            "mocked": settings.mock_mode or not settings.reachy_camera_enabled,
            "camera_health": None if status.camera_health is None else status.camera_health.value,
            "image_width": status.image_width,
            "image_height": status.image_height,
            "stream_active": status.stream_active,
            "time_since_last_frame_ms": status.time_since_last_frame_ms,
            "world_frame_status": "unknown",
            "may_move_head": False,
        }
    finally:
        await camera.aclose()


async def look_now(
    session: AsyncSession,
    settings: Settings,
    *,
    query: str,
    preset_id: str | None = None,
    allow_scan: bool = True,
    correlation_id: str | None = None,
    idempotency_key: str | None = None,
    backend: PerceptionBackend | None = None,
    inject_labels: list[str] | None = None,
) -> SceneResult:
    if not settings.visual_enabled:
        return SceneResult(
            outcome=LookOutcome.DISABLED,
            presence=Presence.UNKNOWN,
            phrase=render("disabled"),
            query=query,
            mocked=True,
        )
    camera = build_camera(settings)
    owned_backend = backend is None
    backend = backend or await build_backend(settings)
    try:
        request = await create_request(
            session,
            trigger=ObservationTrigger.QUERY,
            preset_id=preset_id,
            query=query,
            correlation_id=correlation_id,
        )
        snapshot = await create_snapshot(session, kind=SnapshotKind.QUERY, request_id=request.id)
        frame = await camera.get_latest_frame(correlation_id=correlation_id)
        result = await observe_frame(
            session,
            settings,
            frame,
            backend=backend,
            query=query,
            trigger=ObservationTrigger.QUERY,
            snapshot_id=snapshot.id,
            request_id=request.id,
            correlation_id=correlation_id,
            idempotency_key=idempotency_key,
            inject_labels=inject_labels,
        )
        request.status = ObservationRequestStatus.COMPLETED.value
        snapshot.state = SnapshotState.COMPLETED.value
        snapshot.finished_at = utcnow()
        snapshot.viewpoint_count = 1
        if allow_scan and should_scan(
            _need(result),
            threshold=settings.visual_need_look_threshold,
            hourly_count=await _scans_this_hour(session),
            hourly_cap=settings.visual_scan_hourly_cap,
        ):
            scanned = await request_scan_followup(
                session,
                settings,
                query=query,
                preset_id=preset_id or DEFAULT_PRESET,
                correlation_id=correlation_id,
                prior=result,
            )
            return scanned
        return result
    except Exception as exc:
        await session.rollback()
        return SceneResult(
            outcome=LookOutcome.DEGRADED,
            presence=Presence.UNKNOWN,
            phrase=render("degraded", reason="the camera or perception backend failed"),
            query=query,
            mocked=settings.mock_mode,
            empty_reason=type(exc).__name__,
        )
    finally:
        await camera.aclose()
        if owned_backend:
            await backend.aclose()


async def request_scan_followup(
    session: AsyncSession,
    settings: Settings,
    *,
    query: str,
    preset_id: str,
    correlation_id: str | None,
    prior: SceneResult,
) -> SceneResult:
    assert_preset_only(preset_id)
    camera = build_camera(settings)
    try:
        outcome = await camera.request_scan(
            preset_id=preset_id, reason=query, correlation_id=correlation_id
        )
    finally:
        await camera.aclose()
    if isinstance(outcome, ScanRejected):
        prior.outcome = LookOutcome.SCAN_REJECTED
        prior.reject_reason = outcome.reason
        prior.phrase = render("scan_rejected", reason=outcome.reason)
        prior.may_move_head = False
        return prior
    if not isinstance(outcome, ScanAccepted):
        prior.outcome = LookOutcome.DEGRADED
        prior.phrase = render("degraded", reason="unexpected scan response")
        return prior
    task = await enqueue(
        session,
        task_type=SCAN_TASK,
        payload={
            "preset_id": preset_id,
            "query": query,
            "scan_id": outcome.scan_id,
            "planned_poses": outcome.planned_poses or 3,
            "prior_observation_id": prior.observation_id,
        },
        timeout_seconds=120,
        max_attempts=1,
        correlation_id=correlation_id,
    )
    prior.outcome = LookOutcome.SCAN_QUEUED
    prior.task_id = task.id
    prior.poll_url = f"/api/v1/background-tasks/{task.id}"
    prior.scan_id = outcome.scan_id
    prior.may_move_head = True
    prior.phrase = render("scan_queued")
    return prior


async def execute_scan(
    session: AsyncSession,
    settings: Settings,
    *,
    query: str,
    preset_id: str,
    planned_poses: int,
    correlation_id: str | None = None,
) -> SceneResult:
    assert_preset_only(preset_id)
    camera = build_camera(settings)
    backend = await build_backend(settings)
    try:
        request = await create_request(
            session,
            trigger=ObservationTrigger.SCAN,
            preset_id=preset_id,
            query=query,
            correlation_id=correlation_id,
        )
        snapshot = await create_snapshot(session, kind=SnapshotKind.SCAN, request_id=request.id)
        last: SceneResult | None = None
        poses = max(1, min(planned_poses, 6))
        for _ in range(poses):
            frame = await camera.get_latest_frame(correlation_id=correlation_id)
            last = await observe_frame(
                session,
                settings,
                frame,
                backend=backend,
                query=query,
                trigger=ObservationTrigger.SCAN,
                snapshot_id=snapshot.id,
                request_id=request.id,
                correlation_id=correlation_id,
            )
        request.status = ObservationRequestStatus.COMPLETED.value
        snapshot.state = SnapshotState.COMPLETED.value
        snapshot.finished_at = utcnow()
        snapshot.viewpoint_count = poses
        if last is None:
            return SceneResult(
                outcome=LookOutcome.DEGRADED,
                presence=Presence.UNKNOWN,
                phrase=render("degraded", reason="the scan produced no frames"),
                query=query,
            )
        last.snapshot_id = snapshot.id
        last.scan_outcome = _scan_kind(last)
        last.may_move_head = True
        return last
    finally:
        await camera.aclose()
        await backend.aclose()


def _scan_kind(result: SceneResult) -> ScanOutcomeKind:
    if result.presence is Presence.ABSTAINED:
        return ScanOutcomeKind.NOT_RESOLVED
    if result.presence is Presence.PRESENT:
        return ScanOutcomeKind.RESOLVED_FOUND
    if result.presence is Presence.NOTHING_PRESENT:
        return ScanOutcomeKind.RESOLVED_ABSENT
    if result.presence is Presence.NOTHING_DETECTED:
        return ScanOutcomeKind.DEGRADED
    return ScanOutcomeKind.NOT_RESOLVED


async def find_last_seen(
    session: AsyncSession,
    settings: Settings,
    *,
    query: str,
) -> dict[str, Any]:
    label = normalise_label(query)
    if not label:
        raise ValidationError("query is required")
    row = (
        await session.execute(
            select(ObjectEntity)
            .where(
                ObjectEntity.status == EntityStatus.ACTIVE.value,
                ObjectEntity.display_label == label,
            )
            .order_by(ObjectEntity.last_seen_at.desc())
            .limit(1)
        )
    ).scalar_one_or_none()
    if row is None:
        detection = (
            await session.execute(
                select(ObjectDetection)
                .where(ObjectDetection.label == label)
                .order_by(ObjectDetection.captured_at.desc())
                .limit(1)
            )
        ).scalar_one_or_none()
        if detection is None:
            return {
                "found": False,
                "label": label,
                "phrase": render("never_seen", label=label),
                "may_move_head": False,
                "presence": Presence.UNKNOWN.value,
            }
        observation = await session.get(VisualObservation, detection.observation_id)
        if observation is None:
            return {
                "found": False,
                "label": label,
                "phrase": render("never_seen", label=label),
                "may_move_head": False,
                "presence": Presence.UNKNOWN.value,
            }
        zone = await _zone_name(session, observation.zone_id)
        when = qualitative_when(observation.captured_at, utcnow())
        phrase = (
            render("last_seen_zone", label=label, zone=zone, when=when)
            if zone
            else render("last_seen_no_zone", label=label, when=when)
        )
        return {
            "found": True,
            "label": label,
            "observation_id": observation.id,
            "when": when,
            "zone": zone,
            "phrase": phrase,
            "may_move_head": False,
            "world_frame_status": "unknown",
        }
    zone = await _zone_name(session, row.zone_id)
    when = qualitative_when(row.last_seen_at, utcnow())
    phrase = (
        render("last_seen_zone", label=row.display_label, zone=zone, when=when)
        if zone
        else render("last_seen_no_zone", label=row.display_label, when=when)
    )
    return {
        "found": True,
        "label": row.display_label,
        "entity_id": row.id,
        "observation_id": row.last_seen_observation_id,
        "when": when,
        "zone": zone,
        "phrase": phrase,
        "may_move_head": False,
        "world_frame_status": "unknown",
    }


async def search_visual_memory(
    session: AsyncSession,
    *,
    query: str,
    limit: int = 8,
) -> dict[str, Any]:
    label = normalise_label(query)
    stmt = select(ObjectEntity).where(ObjectEntity.status == EntityStatus.ACTIVE.value)
    if label:
        stmt = stmt.where(ObjectEntity.display_label.contains(label))
    rows = (
        (await session.execute(stmt.order_by(ObjectEntity.last_seen_at.desc()).limit(limit)))
        .scalars()
        .all()
    )
    if not rows:
        return {
            "matches": [],
            "phrase": render("empty_search", query=query or "that"),
            "may_move_head": False,
        }
    items = []
    for row in rows:
        when = qualitative_when(row.last_seen_at, utcnow())
        items.append(
            {
                "entity_id": row.id,
                "label": row.display_label,
                "last_seen_at": row.last_seen_at.isoformat(),
                "when": when,
            }
        )
    top = items[0]
    return {
        "matches": items,
        "phrase": render("search_hit", label=top["label"], when=top["when"]),
        "may_move_head": False,
    }


async def describe_previous_scene(
    session: AsyncSession, *, observation_id: str | None = None
) -> dict[str, Any]:
    if observation_id:
        row = await session.get(VisualObservation, observation_id)
    else:
        row = (
            await session.execute(
                select(VisualObservation).order_by(VisualObservation.captured_at.desc()).limit(1)
            )
        ).scalar_one_or_none()
    if row is None:
        return {
            "found": False,
            "phrase": render("never_seen", label="scene"),
            "may_move_head": False,
        }
    labels = [det.label for det in row.detections]
    presence = Presence.PRESENT if labels else Presence.NOTHING_DETECTED
    phrase = presence_phrase_for_labels(labels)
    return {
        "found": True,
        "observation_id": row.id,
        "captured_at": row.captured_at.isoformat(),
        "labels": labels,
        "presence": presence.value,
        "phrase": phrase,
        "world_frame_status": row.world_frame_status,
        "may_move_head": False,
    }


def presence_phrase_for_labels(labels: list[str]) -> str:
    from vision.phrases import presence_phrase

    if not labels:
        return presence_phrase(Presence.NOTHING_DETECTED, "object")
    return presence_phrase(Presence.PRESENT, labels[0])


async def compare_visual_scenes(
    session: AsyncSession,
    *,
    before_id: str,
    after_id: str,
) -> dict[str, Any]:
    from vision.change import change_summary, detect_changes
    from vision.geometry import Box
    from vision.tracking import DetectionHypothesis

    before = await session.get(VisualObservation, before_id)
    after = await session.get(VisualObservation, after_id)
    if before is None or after is None:
        raise ValidationError("Both scene ids must exist.")

    def as_hyp(obs: VisualObservation) -> list[DetectionHypothesis]:
        out: list[DetectionHypothesis] = []
        for det in obs.detections:
            if det.bbox_x0 is None:
                continue
            out.append(
                DetectionHypothesis(
                    label=det.label,
                    box=Box(det.bbox_x0, det.bbox_y0 or 0, det.bbox_x1 or 1, det.bbox_y1 or 1),
                    score=det.detection_confidence or 0.5,
                    is_person=det.is_person,
                )
            )
        return out

    events = detect_changes(as_hyp(before), as_hyp(after))
    summary = change_summary(events)
    phrase = render("compare_same") if not events else render("compare_changed", summary=summary)
    return {
        "changed": bool(events),
        "events": [
            {"event_type": event.event_type.value, "label": event.label} for event in events
        ],
        "phrase": phrase,
        "may_move_head": False,
    }


async def _zone_name(session: AsyncSession, zone_id: str | None) -> str | None:
    if not zone_id:
        return None
    zone = await session.get(VisualZone, zone_id)
    return None if zone is None else zone.name


async def _scans_this_hour(session: AsyncSession) -> int:
    cutoff = utcnow() - timedelta(hours=1)
    count = (
        await session.execute(
            select(func.count())
            .select_from(ObservationRequest)
            .where(
                ObservationRequest.trigger == ObservationTrigger.SCAN.value,
                ObservationRequest.requested_at >= cutoff,
            )
        )
    ).scalar_one()
    return int(count)


def _need(result: SceneResult) -> Any:
    from vision.need_look import NeedLook

    return NeedLook(total=result.need_look, terms=result.need_look_terms)
