"""Persistence helpers for visual memory writes."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings
from database.base import new_uuid
from database.models.vision import (
    EvidenceAsset,
    ObjectDetection,
    ObjectEntity,
    VisualEmbedding,
    VisualObservation,
    VisualSpatialRelation,
    VisualZone,
)
from shared.errors import ConflictError, ValidationError
from vision.dedup import DedupCandidate, first_duplicate
from vision.enums import (
    FORBIDDEN_METRIC_COLUMNS,
    WORLD_FRAME_STATUS_VALUE,
    EvidenceKind,
    EvidenceState,
    IngestSkipReason,
    ObservationTrigger,
    RetentionClass,
)
from vision.keyframe import FORCE_TRIGGERS, LastKeyframe, evaluate_keyframe
from vision.metrics import collect_disk_metrics
from vision.schemas import (
    DetectionSet,
    DetectionWrite,
    FrameIngest,
    IngestResult,
    SpatialRelationWrite,
    dump_json,
    pose_bucket,
)
from vision.storage import (
    DiskFullError,
    encode_still,
    encode_thumbnail,
    expires_at_for,
    write_evidence_file,
)


def assert_no_metric_columns(table: Any) -> None:
    names = {column.name for column in table.__table__.columns}
    forbidden = names & FORBIDDEN_METRIC_COLUMNS
    if forbidden:
        raise AssertionError(
            f"{table.__tablename__} defines forbidden metric columns: {sorted(forbidden)}"
        )


def normalise_zone_name(name: str) -> str:
    return " ".join(name.strip().lower().split())


async def get_or_create_zone(session: AsyncSession, name: str, **kwargs: Any) -> VisualZone:
    normalised = normalise_zone_name(name)
    existing = (
        await session.execute(select(VisualZone).where(VisualZone.normalised_name == normalised))
    ).scalar_one_or_none()
    if existing is not None:
        return existing
    zone = VisualZone(name=name.strip(), normalised_name=normalised, **kwargs)
    session.add(zone)
    await session.flush()
    return zone


async def insert_evidence_row(
    session: AsyncSession,
    stored: Any,
    *,
    settings: Settings,
    now: datetime,
    zone_id: str | None = None,
    pose_bucket_key: str | None = None,
    model_version: str | None = None,
    source_observation_id: str | None = None,
    idempotency_key: str | None = None,
) -> EvidenceAsset:
    existing = (
        await session.execute(
            select(EvidenceAsset).where(EvidenceAsset.content_hash == stored.content_hash)
        )
    ).scalar_one_or_none()
    if existing is not None:
        existing.reference_count = existing.reference_count + 1
        return existing
    if idempotency_key:
        prior = (
            await session.execute(
                select(EvidenceAsset).where(EvidenceAsset.idempotency_key == idempotency_key)
            )
        ).scalar_one_or_none()
        if prior is not None:
            return prior
    retention = stored.retention_class
    if not isinstance(retention, RetentionClass):
        retention = RetentionClass(retention)
    asset = EvidenceAsset(
        content_hash=stored.content_hash,
        kind=stored.kind.value if hasattr(stored.kind, "value") else stored.kind,
        retention_class=retention.value,
        state=EvidenceState.ACTIVE.value,
        relative_path=stored.relative_path,
        byte_size=stored.byte_size,
        width=stored.width,
        height=stored.height,
        dhash64=stored.dhash64,
        phash64=stored.phash64,
        contains_person=stored.contains_person,
        expires_at=expires_at_for(
            retention, now=now, settings=settings, contains_person=stored.contains_person
        ),
        reference_count=1,
        zone_id=zone_id,
        pose_bucket=pose_bucket_key,
        model_version=model_version,
        observation_id=source_observation_id,
        idempotency_key=idempotency_key,
    )
    session.add(asset)
    await session.flush()
    return asset


async def write_spatial_relation(
    session: AsyncSession, payload: SpatialRelationWrite
) -> VisualSpatialRelation:
    if payload.frame_class.value == "egocentric" and not payload.observed_from_observation_id:
        raise ValidationError("Egocentric relations require observed_from_observation_id.")
    row = VisualSpatialRelation(
        predicate=payload.predicate.value,
        frame_class=payload.frame_class.value,
        subject_entity_id=payload.subject_entity_id,
        object_entity_id=payload.object_entity_id,
        zone_id=payload.object_zone_id,
        observed_from_observation_id=payload.observed_from_observation_id,
        captured_at=payload.captured_at,
        confidence=payload.confidence,
        needs_reobservation=payload.predicate.value == "occludes",
    )
    session.add(row)
    await session.flush()
    return row


def block_person_entity(*, is_person: bool) -> None:
    if is_person:
        raise ValidationError("Person detections cannot create a persistent entity.")


def block_person_embedding(*, is_person: bool) -> None:
    if is_person:
        raise ValidationError("Person detections cannot produce instance embeddings.")


async def create_entity(
    session: AsyncSession,
    *,
    display_label: str,
    captured_at: datetime,
    is_person: bool = False,
    zone_id: str | None = None,
) -> ObjectEntity:
    block_person_entity(is_person=is_person)
    entity = ObjectEntity(
        display_label=display_label,
        is_person=False,
        blocks_embeddings=False,
        first_seen_at=captured_at,
        last_seen_at=captured_at,
        zone_id=zone_id,
    )
    session.add(entity)
    await session.flush()
    return entity


async def create_embedding(
    session: AsyncSession,
    *,
    owner_kind: str,
    owner_id: str,
    space: str,
    model_name: str,
    model_version: str,
    vector: list[float],
    is_person: bool,
) -> VisualEmbedding:
    block_person_embedding(is_person=is_person)
    payload = dump_json(vector)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    row = VisualEmbedding(
        owner_kind=owner_kind,
        owner_id=owner_id,
        space=space,
        model_name=model_name,
        model_version=model_version,
        dim=len(vector),
        vector_sha256=digest,
        sidecar_key=f"{owner_kind}:{owner_id}:{space}:{digest}",
        is_person=False,
    )
    session.add(row)
    await session.flush()
    return row


async def _last_keyframe(
    session: AsyncSession, zone_id: str | None, pose_key: str
) -> LastKeyframe | None:
    stmt = select(VisualObservation).order_by(VisualObservation.captured_at.desc()).limit(1)
    if zone_id:
        stmt = (
            select(VisualObservation)
            .where(VisualObservation.zone_id == zone_id)
            .order_by(VisualObservation.captured_at.desc())
            .limit(1)
        )
    row = (await session.execute(stmt)).scalar_one_or_none()
    if row is None or not row.dhash64 or not row.phash64:
        return None
    detections = (
        await session.execute(
            select(ObjectDetection.label, ObjectDetection.is_person).where(
                ObjectDetection.observation_id == row.id
            )
        )
    ).all()
    labels = [label for label, _ in detections]
    is_person = any(flag for _, flag in detections)
    return LastKeyframe(
        persisted_at=row.captured_at,
        dhash64=row.dhash64,
        phash64=row.phash64,
        detection_set=DetectionSet.from_labels(labels, is_person=is_person),
        zone_id=row.zone_id,
        pose_bucket=row.pose_bucket,
    )


async def _dedup_candidates(
    session: AsyncSession, *, zone_id: str | None, pose_key: str, model_version: str | None
) -> list[DedupCandidate]:
    stmt = (
        select(VisualObservation)
        .where(VisualObservation.pose_bucket == pose_key)
        .order_by(VisualObservation.captured_at.desc())
        .limit(50)
    )
    if zone_id:
        stmt = stmt.where(VisualObservation.zone_id == zone_id)
    if model_version:
        stmt = stmt.where(VisualObservation.model_version == model_version)
    rows = (await session.execute(stmt)).scalars().all()
    out: list[DedupCandidate] = []
    for row in rows:
        if not row.dhash64 or not row.phash64 or not row.content_hash:
            continue
        dets = (
            await session.execute(
                select(ObjectDetection.label, ObjectDetection.is_person).where(
                    ObjectDetection.observation_id == row.id
                )
            )
        ).all()
        out.append(
            DedupCandidate(
                content_hash=row.content_hash or "",
                dhash64=row.dhash64,
                phash64=row.phash64,
                zone_id=row.zone_id,
                pose_bucket=row.pose_bucket,
                model_version=row.model_version,
                captured_at=row.captured_at,
                detection_set=DetectionSet.from_labels(
                    [label for label, _ in dets],
                    is_person=any(flag for _, flag in dets),
                ),
            )
        )
    return out


async def ingest_frame(
    session: AsyncSession,
    settings: Settings,
    frame: FrameIngest,
) -> IngestResult:
    if not settings.visual_enabled:
        return IngestResult(persisted=False, skip_reason=IngestSkipReason.DISABLED.value)

    if frame.detections and not frame.detection_set.labels:
        person = any(
            item.is_person or item.label.lower() in {"person", "human"} for item in frame.detections
        )
        frame.detection_set = DetectionSet.from_labels(
            [item.label for item in frame.detections], is_person=person
        )

    if frame.idempotency_key:
        prior = (
            await session.execute(
                select(VisualObservation).where(
                    VisualObservation.idempotency_key == frame.idempotency_key
                )
            )
        ).scalar_one_or_none()
        if prior is not None:
            return IngestResult(
                persisted=True,
                skip_reason=IngestSkipReason.IDEMPOTENT_REPLAY.value,
                observation_id=prior.id,
                thumbnail_id=prior.thumbnail_evidence_id,
                full_frame_id=prior.full_frame_evidence_id,
                evidence_reused=True,
                content_hash=prior.content_hash,
            )

    if frame.zone_id:
        zone = await session.get(VisualZone, frame.zone_id)
        if zone is not None and not zone.capture_allowed:
            return IngestResult(
                persisted=False, skip_reason=IngestSkipReason.ZONE_CAPTURE_DISALLOWED.value
            )
        if zone is not None and zone.privacy_sensitive:
            frame.contains_person = True

    metrics = collect_disk_metrics(settings)
    if not metrics.capture_allowed:
        reason = (
            IngestSkipReason.BUDGET_EXCEEDED.value
            if metrics.evidence_bytes >= settings.visual_evidence_max_bytes
            else IngestSkipReason.DISK_FULL.value
        )
        return IngestResult(persisted=False, skip_reason=reason)

    encoded = encode_still(frame.image_bytes)
    thumb = encode_thumbnail(encoded.image)
    bucket = pose_bucket(frame.body_yaw_rad, frame.viewpoint_label)
    pose_key = bucket.key()
    now = frame.captured_at

    last = await _last_keyframe(session, frame.zone_id, pose_key)
    verdict = evaluate_keyframe(
        visual_enabled=settings.visual_enabled,
        trigger=frame.trigger,
        is_event=frame.is_event,
        now=now,
        min_persist_interval=timedelta(seconds=settings.visual_min_capture_interval_seconds),
        current_dhash=encoded.dhash64,
        current_phash=encoded.phash64,
        current_detections=frame.detection_set,
        last=last,
    )
    if not verdict.persist:
        return IngestResult(persisted=False, skip_reason=verdict.reason)

    force = frame.is_event or frame.trigger in FORCE_TRIGGERS
    day_start = frame.captured_at.astimezone(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    today_count = (
        await session.execute(
            select(func.count())
            .select_from(VisualObservation)
            .where(VisualObservation.captured_at >= day_start)
        )
    ).scalar_one()
    if int(today_count) >= settings.visual_daily_observation_cap and not force:
        return IngestResult(persisted=False, skip_reason=IngestSkipReason.DAILY_CAP.value)

    incoming = DedupCandidate(
        content_hash=encoded.content_hash,
        dhash64=encoded.dhash64,
        phash64=encoded.phash64,
        zone_id=frame.zone_id,
        pose_bucket=pose_key,
        model_version=frame.model_version,
        captured_at=now,
        detection_set=frame.detection_set,
        produced_event=frame.is_event or frame.trigger is ObservationTrigger.EVENT,
    )
    existing = await _dedup_candidates(
        session, zone_id=frame.zone_id, pose_key=pose_key, model_version=frame.model_version
    )
    dup = first_duplicate(incoming, existing, now=now, event_bypass=force)
    if dup.duplicate:
        return IngestResult(
            persisted=False,
            skip_reason=IngestSkipReason.DUPLICATE.value,
            content_hash=dup.matched_hash,
            evidence_reused=True,
        )

    try:
        stored_thumb = write_evidence_file(
            settings,
            thumb,
            kind=EvidenceKind.THUMBNAIL,
            retention=RetentionClass.THUMBNAIL,
            contains_person=frame.contains_person,
        )
        stored_full = None
        if settings.visual_keep_full_frames:
            stored_full = write_evidence_file(
                settings,
                encoded,
                kind=EvidenceKind.FULL_FRAME,
                retention=RetentionClass.EVIDENCE,
                contains_person=frame.contains_person,
            )
    except DiskFullError:
        return IngestResult(persisted=False, skip_reason=IngestSkipReason.DISK_FULL.value)

    # Files are on disk. Only now insert rows.
    thumb_row = await insert_evidence_row(
        session,
        stored_thumb,
        settings=settings,
        now=now,
        zone_id=frame.zone_id,
        pose_bucket_key=pose_key,
        model_version=frame.model_version,
    )
    full_row = None
    if stored_full is not None:
        full_row = await insert_evidence_row(
            session,
            stored_full,
            settings=settings,
            now=now,
            zone_id=frame.zone_id,
            pose_bucket_key=pose_key,
            model_version=frame.model_version,
        )

    observation_id = new_uuid()
    observation = VisualObservation(
        id=observation_id,
        camera_source_id=frame.camera_source_id,
        request_id=frame.request_id,
        snapshot_id=frame.snapshot_id,
        zone_id=frame.zone_id,
        captured_at=now,
        captured_at_pc=datetime.now(UTC),
        captured_at_pi=frame.captured_at_pi,
        trigger=frame.trigger.value,
        calibration_version=frame.calibration_version,
        t_head_cam=dump_json(frame.T_head_cam) if frame.T_head_cam is not None else None,
        head_pose_4x4=dump_json(frame.head_pose_4x4) if frame.head_pose_4x4 is not None else None,
        head_joints_rad=dump_json(frame.head_joints_rad)
        if frame.head_joints_rad is not None
        else None,
        body_yaw_rad=frame.body_yaw_rad,
        body_yaw_source=frame.body_yaw_source,
        head_pose_is_settled=frame.head_pose_is_settled,
        viewpoint_label=frame.viewpoint_label,
        pose_bucket=pose_key,
        world_frame_status=WORLD_FRAME_STATUS_VALUE,
        width=encoded.width,
        height=encoded.height,
        content_hash=encoded.content_hash,
        dhash64=encoded.dhash64,
        phash64=encoded.phash64,
        blur_flag=frame.blur_flag,
        exposure_flag=frame.exposure_flag,
        quality_score=frame.quality_score,
        quality_flags=dump_json(frame.quality_flags) if frame.quality_flags else None,
        mean_luma=frame.mean_luma,
        laplacian_variance=frame.laplacian_variance,
        camera_health_json=frame.camera_health_json,
        full_frame_evidence_id=full_row.id if full_row is not None else None,
        thumbnail_evidence_id=thumb_row.id,
        model_name=frame.model_name,
        model_version=frame.model_version,
        idempotency_key=frame.idempotency_key,
        correlation_id=frame.correlation_id,
    )
    session.add(observation)
    await session.flush()
    thumb_row.observation_id = observation.id
    if full_row is not None:
        full_row.observation_id = observation.id
    writes: list[DetectionWrite]
    if frame.detections:
        writes = list(frame.detections)
    else:
        writes = [
            DetectionWrite(
                label=label,
                is_person=frame.detection_set.is_person or label in {"person", "human"},
            )
            for label in frame.detection_set.labels
        ]
    for item in writes:
        is_person = item.is_person or item.label.lower() in {"person", "human"}
        box = item.box_xyxy
        session.add(
            ObjectDetection(
                observation_id=observation.id,
                captured_at=now,
                label=item.label,
                is_person=is_person,
                blocks_entity=is_person,
                blocks_embedding=is_person,
                bbox_x0=box[0] if box else None,
                bbox_y0=box[1] if box else None,
                bbox_x1=box[2] if box else None,
                bbox_y1=box[3] if box else None,
                depth_median=item.depth_median,
                occlusion_ratio=item.occlusion_ratio,
                detection_score_raw=item.score,
                detection_confidence=item.score,
                quality_score=item.quality_score,
                model_name=frame.model_name,
                model_version=frame.model_version,
            )
        )
    await session.flush()
    return IngestResult(
        persisted=True,
        observation_id=observation.id,
        thumbnail_id=thumb_row.id,
        full_frame_id=full_row.id if full_row is not None else None,
        evidence_reused=stored_thumb.reused,
        content_hash=encoded.content_hash,
    )


async def require_idempotent_insert(
    session: AsyncSession,
    model: type[Any],
    idempotency_key: str,
    *,
    builder: Any,
) -> Any:
    existing = (
        await session.execute(select(model).where(model.idempotency_key == idempotency_key))
    ).scalar_one_or_none()
    if existing is not None:
        return existing
    row = builder()
    row.idempotency_key = idempotency_key
    session.add(row)
    try:
        await session.flush()
    except Exception as exc:  # pragma: no cover - race with unique constraint
        raise ConflictError("Idempotency key already used.") from exc
    return row
