"""Capture → perceive → persist → track → relate → events."""

from __future__ import annotations

from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings
from database.models.vision import (
    ObjectDetection,
    ObjectEntity,
    ObjectEntityObservation,
    ObjectTrack,
    ObservationRequest,
    SceneSnapshot,
    VisualEvent,
    VisualObservation,
    VisualSpatialRelation,
)
from integrations.reachy.camera.models import CameraFrame
from shared.timeutils import utcnow
from vision.backend import PerceptionBackend, PerceptionOutput
from vision.change import detect_changes, structural_dedup_key
from vision.enums import (
    WORLD_FRAME_STATUS_VALUE,
    AssociationDecision,
    EntityStatus,
    FrameClass,
    LookOutcome,
    ObservationRequestStatus,
    ObservationTrigger,
    Presence,
    ProvenanceStatus,
    SnapshotKind,
    SnapshotState,
    TrackState,
)
from vision.identity import EntityHypothesis, IdentityLink, link_entities
from vision.need_look import compute_need_look
from vision.phrases import presence_phrase, relation_phrase
from vision.repositories import ingest_frame
from vision.results import DetectionView, RelationView, SceneResult
from vision.schemas import DetectionSet, FrameIngest, dump_json
from vision.spatial import infer_relations
from vision.tracking import DetectionHypothesis, TrackHypothesis, associate_detections


async def observe_frame(
    session: AsyncSession,
    settings: Settings,
    frame: CameraFrame,
    *,
    backend: PerceptionBackend,
    query: str,
    trigger: ObservationTrigger,
    snapshot_id: str | None = None,
    request_id: str | None = None,
    zone_id: str | None = None,
    correlation_id: str | None = None,
    idempotency_key: str | None = None,
    inject_labels: list[str] | None = None,
) -> SceneResult:
    queries = inject_labels or _queries(query, settings)
    if inject_labels:
        from vision.perception_mock import perceive_bytes

        mock = perceive_bytes(frame.image_bytes, inject_labels=inject_labels)
        perception = PerceptionOutput(
            detections=mock.detections,
            mocked=True,
            model_name=mock.model_name,
            model_version=mock.model_version,
        )
    else:
        perception = await backend.detect(frame.image_bytes, queries)
    detection_set = DetectionSet.from_labels(
        [item.label for item in perception.detections],
        is_person=any(item.is_person for item in perception.detections),
    )
    ingest = _to_ingest(
        frame,
        trigger=trigger,
        detection_set=detection_set,
        snapshot_id=snapshot_id,
        request_id=request_id,
        zone_id=zone_id,
        correlation_id=correlation_id,
        idempotency_key=idempotency_key,
        model_name=perception.model_name,
        model_version=perception.model_version,
    )
    stored = await ingest_frame(session, settings, ingest)
    if not stored.persisted or stored.observation_id is None:
        presence = _presence(perception, persisted=False)
        phrase = presence_phrase(presence, query or "object")
        return SceneResult(
            outcome=(
                LookOutcome.ANSWERED if stored.skip_reason != "disabled" else LookOutcome.DISABLED
            ),
            presence=presence,
            phrase=phrase,
            query=query,
            mocked=perception.mocked,
            empty_reason=stored.skip_reason or perception.empty_reason,
        )

    observation_id = stored.observation_id
    await _replace_detections(
        session,
        observation_id=observation_id,
        captured_at=ingest.captured_at,
        perception=perception,
        model_name=perception.model_name,
        model_version=perception.model_version,
    )
    tracks, association = associate_detections([], perception.detections)
    await _persist_tracks(session, tracks, snapshot_id=snapshot_id, captured_at=ingest.captured_at)
    existing_entities = await _load_entities(session)
    entities, links = link_entities(
        existing_entities,
        perception.detections,
        n_similar_by_label=association.n_similar_by_label,
    )
    id_map = await _persist_identity(
        session,
        entities,
        links,
        perception.detections,
        observation_id=observation_id,
        captured_at=ingest.captured_at,
        zone_id=zone_id,
    )
    settled = frame.pose is None or frame.pose.head_pose_is_settled is not False
    quality_ok = not frame.is_stale and settled
    relations = infer_relations(perception.detections, quality_ok=quality_ok)
    relation_views = await _persist_relations(
        session,
        relations,
        perception.detections,
        links,
        observation_id=observation_id,
        captured_at=ingest.captured_at,
        zone_id=zone_id,
        id_map=id_map,
    )
    previous = await _previous_detections(session, observation_id, zone_id=zone_id)
    events = detect_changes(previous, perception.detections)
    event_rows = await _persist_events(
        session,
        events,
        observation_id=observation_id,
        zone_id=zone_id,
        thumbnail_id=stored.thumbnail_id,
        correlation_id=correlation_id,
    )
    detection_views = _detection_views(
        perception.detections, association.assignments, links, id_map=id_map
    )
    similar = max(association.n_similar_by_label.values(), default=0)
    need = compute_need_look(
        n_similar=similar,
        detection_confidence=min((d.score for d in perception.detections), default=1.0),
        stale_frame=frame.is_stale,
        pose_unsettled=bool(frame.pose and frame.pose.head_pose_is_settled is False),
        quality_failed=not quality_ok,
    )
    presence = _presence(
        perception,
        persisted=True,
        abstained=any(link.decision is AssociationDecision.ABSTAIN for link in links),
    )
    phrase = presence_phrase(presence, query or "object")
    if relation_views and presence is Presence.PRESENT:
        phrase = relation_views[0].phrase
    evidence_ids = [eid for eid in (stored.thumbnail_id, stored.full_frame_id) if eid]
    return SceneResult(
        outcome=LookOutcome.ANSWERED,
        presence=presence,
        phrase=phrase,
        query=query,
        detections=detection_views,
        relations=relation_views,
        events=event_rows,
        need_look=need.total,
        need_look_terms=need.as_dict(),
        observation_id=observation_id,
        snapshot_id=snapshot_id,
        thumbnail_id=stored.thumbnail_id,
        evidence_ids=evidence_ids,
        mocked=perception.mocked,
        empty_reason=perception.empty_reason,
    )


def _queries(query: str, settings: Settings) -> list[str]:
    text = (query or "").strip()
    defaults = list(settings.visual_default_queries)
    if not text:
        return defaults[:8] or ["object"]
    labels = [text]
    for token in text.replace(",", " ").split():
        if token.lower() not in {item.lower() for item in labels}:
            labels.append(token)
    return labels[:16]


def _to_ingest(
    frame: CameraFrame,
    *,
    trigger: ObservationTrigger,
    detection_set: DetectionSet,
    snapshot_id: str | None,
    request_id: str | None,
    zone_id: str | None,
    correlation_id: str | None,
    idempotency_key: str | None,
    model_name: str | None,
    model_version: str | None,
) -> FrameIngest:
    pose = frame.pose
    captured = frame.timestamps.capture_utc or frame.timestamps.received_utc
    yaw_source = "unknown"
    if pose is not None and pose.body_yaw_source is not None:
        yaw_source = pose.body_yaw_source.value
    pose_matrix = None
    if pose is not None and pose.head_pose_4x4 is not None:
        pose_matrix = [list(row) for row in pose.head_pose_4x4]
    joints = None if pose is None or pose.head_joints_rad is None else list(pose.head_joints_rad)
    return FrameIngest(
        image_bytes=frame.image_bytes,
        captured_at=captured,
        trigger=trigger,
        snapshot_id=snapshot_id,
        request_id=request_id,
        zone_id=zone_id,
        viewpoint_label="fake" if pose is None else "live",
        body_yaw_rad=None if pose is None else pose.body_yaw_rad,
        body_yaw_source=yaw_source,
        head_pose_is_settled=True if pose is None else bool(pose.head_pose_is_settled),
        head_pose_4x4=pose_matrix,
        head_joints_rad=joints,
        world_frame_status=WORLD_FRAME_STATUS_VALUE,
        contains_person=detection_set.is_person,
        detection_set=detection_set,
        idempotency_key=idempotency_key,
        correlation_id=correlation_id or frame.correlation_id,
        model_name=model_name,
        model_version=model_version,
        width=frame.image_width,
        height=frame.image_height,
    )


def _presence(
    perception: PerceptionOutput,
    *,
    persisted: bool,
    abstained: bool = False,
) -> Presence:
    if abstained:
        return Presence.ABSTAINED
    if perception.detections:
        return Presence.PRESENT
    if perception.empty_reason:
        return Presence.NOTHING_DETECTED
    if persisted:
        return Presence.NOTHING_PRESENT
    return Presence.UNKNOWN


async def _replace_detections(
    session: AsyncSession,
    *,
    observation_id: str,
    captured_at: Any,
    perception: PerceptionOutput,
    model_name: str,
    model_version: str,
) -> None:
    await session.execute(
        delete(ObjectDetection).where(ObjectDetection.observation_id == observation_id)
    )
    for item in perception.detections:
        session.add(
            ObjectDetection(
                observation_id=observation_id,
                captured_at=captured_at,
                label=item.label,
                is_person=item.is_person,
                blocks_entity=item.is_person,
                blocks_embedding=item.is_person,
                bbox_x0=item.box.x0,
                bbox_y0=item.box.y0,
                bbox_x1=item.box.x1,
                bbox_y1=item.box.y1,
                depth_median=item.depth_median,
                detection_score_raw=item.score,
                detection_confidence=item.score,
                camera_ray_status=ProvenanceStatus.UNKNOWN.value,
                n_similar_instances=None,
                model_name=model_name,
                model_version=model_version,
            )
        )
    await session.flush()


async def _persist_tracks(
    session: AsyncSession,
    tracks: list[TrackHypothesis],
    *,
    snapshot_id: str | None,
    captured_at: Any,
) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for track in tracks:
        row = ObjectTrack(
            snapshot_id=snapshot_id,
            label=track.label,
            state=track.state or TrackState.CONFIRMED.value,
            first_seen_at=captured_at,
            last_seen_at=captured_at,
            detection_count=track.hits,
        )
        session.add(row)
        await session.flush()
        mapping[track.track_id] = row.id
        track.track_id = row.id
    return mapping


async def _load_entities(session: AsyncSession) -> list[EntityHypothesis]:
    rows = (
        (
            await session.execute(
                select(ObjectEntity).where(ObjectEntity.status == EntityStatus.ACTIVE.value)
            )
        )
        .scalars()
        .all()
    )
    out: list[EntityHypothesis] = []
    for row in rows:
        proto = None
        if row.prototype_embedding:
            import json

            proto = json.loads(row.prototype_embedding)
        out.append(
            EntityHypothesis(
                entity_id=row.id,
                label=row.display_label,
                prototype=proto,
                status=row.status,
                observation_count=row.observation_count,
                is_person=row.is_person,
            )
        )
    return out


async def _persist_identity(
    session: AsyncSession,
    entities: list[EntityHypothesis],
    links: list[IdentityLink],
    detections: list[DetectionHypothesis],
    *,
    observation_id: str,
    captured_at: Any,
    zone_id: str | None,
) -> dict[str, str]:
    by_temp = {entity.entity_id: entity for entity in entities}
    mapping: dict[str, str] = {}
    detection_rows = (
        (
            await session.execute(
                select(ObjectDetection)
                .where(ObjectDetection.observation_id == observation_id)
                .order_by(ObjectDetection.created_at)
            )
        )
        .scalars()
        .all()
    )
    for link in links:
        if link.entity_id is None:
            if link.detection_index < len(detection_rows):
                detection_rows[link.detection_index].n_similar_instances = link.n_similar
                detection_rows[link.detection_index].n_competing_candidates = link.n_competing
                detection_rows[link.detection_index].association_margin = link.margin
            continue
        hypothesis = by_temp[link.entity_id]
        temp_id = link.entity_id
        row = await session.get(ObjectEntity, hypothesis.entity_id)
        if row is None:
            row = ObjectEntity(
                id=hypothesis.entity_id,
                display_label=hypothesis.label,
                is_person=False,
                blocks_embeddings=False,
                first_seen_at=captured_at,
                last_seen_at=captured_at,
                zone_id=zone_id,
                last_seen_observation_id=observation_id,
                observation_count=1,
                prototype_embedding=dump_json(hypothesis.prototype)
                if hypothesis.prototype
                else None,
            )
            session.add(row)
            await session.flush()
            mapping[temp_id] = row.id
            hypothesis.entity_id = row.id
        else:
            mapping[temp_id] = row.id
            row.last_seen_at = captured_at
            row.last_seen_observation_id = observation_id
            row.observation_count = row.observation_count + 1
            row.zone_id = zone_id or row.zone_id
            if hypothesis.prototype:
                row.prototype_embedding = dump_json(hypothesis.prototype)
        if link.detection_index < len(detection_rows):
            det = detection_rows[link.detection_index]
            det.entity_id = row.id
            det.n_similar_instances = link.n_similar
            det.n_competing_candidates = link.n_competing
            det.association_margin = link.margin
            session.add(
                ObjectEntityObservation(
                    entity_id=row.id,
                    detection_id=det.id,
                    observation_id=observation_id,
                    match_score=link.match_score,
                    observed_at=captured_at,
                )
            )
    await session.flush()
    return mapping


async def _persist_relations(
    session: AsyncSession,
    relations: Any,
    detections: list[DetectionHypothesis],
    links: list[IdentityLink],
    *,
    observation_id: str,
    captured_at: Any,
    zone_id: str | None,
    id_map: dict[str, str] | None = None,
) -> list[RelationView]:
    remap = id_map or {}
    entity_by_index: dict[int, str] = {}
    allowed = set(remap.values())
    for link in links:
        if not link.entity_id:
            continue
        mapped = remap.get(link.entity_id, link.entity_id)
        if mapped not in allowed:
            continue
        entity_by_index[link.detection_index] = mapped
    views: list[RelationView] = []
    for rel in relations:
        subject = detections[rel.subject_index]
        obj = detections[rel.object_index] if rel.object_index is not None else None
        phrase = relation_phrase(
            subject.label,
            rel.predicate,
            obj.label if obj else "the scene",
            egocentric=rel.frame_class is FrameClass.EGOCENTRIC,
        )
        session.add(
            VisualSpatialRelation(
                predicate=rel.predicate.value,
                frame_class=rel.frame_class.value,
                subject_entity_id=entity_by_index.get(rel.subject_index),
                object_entity_id=(
                    entity_by_index.get(rel.object_index) if rel.object_index is not None else None
                ),
                observed_from_observation_id=observation_id,
                zone_id=zone_id,
                confidence=rel.confidence,
                needs_reobservation=rel.needs_reobservation,
                captured_at=captured_at,
            )
        )
        views.append(
            RelationView(
                predicate=rel.predicate.value,
                frame_class=rel.frame_class.value,
                subject_label=subject.label,
                object_label=None if obj is None else obj.label,
                phrase=phrase,
                confidence=rel.confidence,
            )
        )
    await session.flush()
    return views


async def _previous_detections(
    session: AsyncSession, observation_id: str, *, zone_id: str | None
) -> list[DetectionHypothesis]:
    stmt = (
        select(VisualObservation)
        .where(VisualObservation.id != observation_id)
        .order_by(VisualObservation.captured_at.desc())
        .limit(1)
    )
    if zone_id:
        stmt = stmt.where(VisualObservation.zone_id == zone_id)
    previous = (await session.execute(stmt)).scalar_one_or_none()
    if previous is None:
        return []
    rows = (
        (
            await session.execute(
                select(ObjectDetection).where(ObjectDetection.observation_id == previous.id)
            )
        )
        .scalars()
        .all()
    )
    from vision.geometry import Box

    out: list[DetectionHypothesis] = []
    for row in rows:
        if row.bbox_x0 is None:
            continue
        out.append(
            DetectionHypothesis(
                label=row.label,
                box=Box(row.bbox_x0, row.bbox_y0 or 0.0, row.bbox_x1 or 1.0, row.bbox_y1 or 1.0),
                score=row.detection_confidence or 0.5,
                is_person=row.is_person,
            )
        )
    return out


async def _persist_events(
    session: AsyncSession,
    events: Any,
    *,
    observation_id: str,
    zone_id: str | None,
    thumbnail_id: str | None,
    correlation_id: str | None,
) -> list[dict[str, Any]]:
    now = utcnow()
    out: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    for event in events:
        key = structural_dedup_key(
            watch_id=None,
            event_type=event.event_type.value,
            label=event.label,
            observation_id=observation_id,
        )
        if key in seen_keys:
            continue
        existing = (
            await session.execute(select(VisualEvent).where(VisualEvent.dedup_key == key))
        ).scalar_one_or_none()
        if existing is not None:
            seen_keys.add(key)
            continue
        seen_keys.add(key)
        row = VisualEvent(
            occurred_after=now,
            occurred_at=now,
            event_type=event.event_type.value,
            zone_id=zone_id,
            observation_id=observation_id,
            after_evidence_id=thumbnail_id,
            payload_json=dump_json({"label": event.label, "confidence": event.confidence}),
            dedup_key=key,
            correlation_id=correlation_id,
        )
        session.add(row)
        out.append(
            {
                "event_type": event.event_type.value,
                "label": event.label,
                "confidence": event.confidence,
                "observation_id": observation_id,
            }
        )
    await session.flush()
    return out


def _detection_views(
    detections: list[DetectionHypothesis],
    associations: Any,
    links: list[IdentityLink],
    *,
    id_map: dict[str, str] | None = None,
) -> list[DetectionView]:
    assoc_by_index = {item.detection_index: item for item in associations}
    link_by_index = {item.detection_index: item for item in links}
    remap = id_map or {}
    views: list[DetectionView] = []
    for index, detection in enumerate(detections):
        assoc = assoc_by_index.get(index)
        link = link_by_index.get(index)
        raw_entity = None if link is None else link.entity_id
        entity_id = None if raw_entity is None else remap.get(raw_entity, raw_entity)
        if entity_id and entity_id.startswith("e-"):
            entity_id = None
        views.append(
            DetectionView(
                label=detection.label,
                score=detection.score,
                box_xyxy=detection.box.as_xyxy(),
                is_person=detection.is_person,
                entity_id=entity_id,
                track_id=None if assoc is None else assoc.track_id,
                association=(
                    (link.decision.value if link else None)
                    or (assoc.decision.value if assoc else "new")
                ),
                n_similar=0 if link is None else link.n_similar,
                depth_median=detection.depth_median,
            )
        )
    return views


async def create_request(
    session: AsyncSession,
    *,
    trigger: ObservationTrigger,
    preset_id: str | None,
    query: str | None,
    correlation_id: str | None,
) -> ObservationRequest:
    row = ObservationRequest(
        trigger=trigger.value,
        status=ObservationRequestStatus.PENDING.value,
        preset_id=preset_id,
        query_text=query,
        requested_at=utcnow(),
        correlation_id=correlation_id,
    )
    session.add(row)
    await session.flush()
    return row


async def create_snapshot(
    session: AsyncSession,
    *,
    kind: SnapshotKind,
    request_id: str | None,
    zone_id: str | None = None,
) -> SceneSnapshot:
    row = SceneSnapshot(
        request_id=request_id,
        zone_id=zone_id,
        kind=kind.value,
        state=SnapshotState.IN_PROGRESS.value,
        started_at=utcnow(),
    )
    session.add(row)
    await session.flush()
    return row


class ObserveNowResult:
    """Shape expected by ``app.api.v1.vision``."""

    def __init__(self, scene: SceneResult, *, request_id: str | None = None) -> None:
        self.presence = scene.presence
        self.phrase = scene.phrase
        self.deferred = scene.outcome is LookOutcome.SCAN_QUEUED
        self.task_id = scene.task_id
        self.poll_url = scene.poll_url
        self.observation_id = scene.observation_id
        self.snapshot_id = scene.snapshot_id
        self.request_id = request_id
        self.detections = [item.as_dict() for item in scene.detections]
        self.relations = [item.as_dict() for item in scene.relations]
        self.need_look = scene.need_look
        self.need_look_terms = scene.need_look_terms
        self.scan_outcome = scene.scan_outcome.value if scene.scan_outcome else None
        self.evidence_id = scene.thumbnail_id
        self.world_frame_status = scene.world_frame_status
        self.may_move_head = scene.may_move_head


async def camera_status(settings: Settings) -> dict[str, Any]:
    from vision.perception import get_camera_status

    payload = await get_camera_status(settings)
    health = payload.get("camera_health") or "unknown"
    return {
        "adapter_version": payload.get("adapter_version") or "v1",
        "camera_health": health,
        "image_width": payload.get("image_width"),
        "image_height": payload.get("image_height"),
        "stream_active": payload.get("stream_active"),
        "time_since_last_frame_ms": payload.get("time_since_last_frame_ms"),
        "mocked": bool(payload.get("mocked", True)),
        "phrase": "Camera status is structured; I am not making a spatial claim.",
        "may_move_head": False,
    }


async def observe_now(
    session: AsyncSession,
    settings: Settings,
    *,
    query: str | None = None,
    zone_name: str | None = None,
    trigger: ObservationTrigger = ObservationTrigger.QUERY,
    allow_scan: bool = False,
    preset_id: str | None = None,
    correlation_id: str | None = None,
    idempotency_key: str | None = None,
    inject_labels: list[str] | None = None,
) -> ObserveNowResult:
    from vision.perception import look_now
    from vision.repositories import get_or_create_zone

    _ = trigger
    if zone_name:
        await get_or_create_zone(session, zone_name)
    text = (query or "").strip() or (inject_labels[0] if inject_labels else "object")
    scene = await look_now(
        session,
        settings,
        query=text,
        preset_id=preset_id,
        allow_scan=allow_scan,
        correlation_id=correlation_id,
        idempotency_key=idempotency_key,
        inject_labels=inject_labels,
    )
    return ObserveNowResult(scene)


async def queue_scan(
    session: AsyncSession,
    settings: Settings,
    *,
    preset_id: str,
    query: str | None,
    zone_id: str | None,
    correlation_id: str | None,
    prior_belief: float,
) -> dict[str, Any]:
    from vision.perception import request_scan_followup

    _ = (zone_id, prior_belief)
    prior = SceneResult(
        outcome=LookOutcome.ANSWERED,
        presence=Presence.UNKNOWN,
        phrase="queued",
        query=query or "object",
        need_look=prior_belief,
    )
    result = await request_scan_followup(
        session,
        settings,
        query=query or "object",
        preset_id=preset_id,
        correlation_id=correlation_id,
        prior=prior,
    )
    return {
        "deferred": result.outcome is LookOutcome.SCAN_QUEUED,
        "task_id": result.task_id,
        "poll_url": result.poll_url,
        "scan_id": result.scan_id,
        "phrase": result.phrase,
        "reject_reason": result.reject_reason,
        "may_move_head": result.may_move_head,
    }
