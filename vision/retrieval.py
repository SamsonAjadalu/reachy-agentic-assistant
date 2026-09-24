"""Synchronous retrieval from visual memory. No GPU, no head motion."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database.models.vision import (
    ObjectDetection,
    ObjectEntity,
    SceneSnapshot,
    VisualObservation,
    VisualSpatialRelation,
    VisualZone,
)
from shared.errors import NotFoundError
from shared.timeutils import utcnow
from vision.enums import EntityStatus, FrameClass, PresenceKind, SpatialPredicate
from vision.phrases import qualitative_when, render, render_last_seen, render_relation


async def find_last_seen(session: AsyncSession, query: str, *, limit: int = 5) -> dict[str, Any]:
    needle = query.strip().lower()
    if not needle:
        return {
            "presence": PresenceKind.UNKNOWN.value,
            "phrase": "Tell me what to look up.",
            "items": [],
        }
    entities = (
        (
            await session.execute(
                select(ObjectEntity)
                .where(
                    ObjectEntity.status == EntityStatus.ACTIVE.value,
                    ObjectEntity.display_label.ilike(f"%{needle}%"),
                )
                .order_by(ObjectEntity.last_seen_at.desc())
                .limit(limit)
            )
        )
        .scalars()
        .all()
    )
    detections = (
        (
            await session.execute(
                select(ObjectDetection)
                .where(ObjectDetection.label.ilike(f"%{needle}%"))
                .order_by(ObjectDetection.captured_at.desc())
                .limit(limit)
            )
        )
        .scalars()
        .all()
    )
    if any((row.n_similar_instances or 0) >= 2 for row in detections[:3]):
        return {
            "presence": PresenceKind.ABSTAINED.value,
            "phrase": render_last_seen(
                label=needle, zone_name=None, when_phrase="", abstained=True
            ),
            "items": [],
            "abstained": True,
        }
    items: list[dict[str, Any]] = []
    for entity in entities:
        zone_name = await _zone_name(session, entity.zone_id)
        when = _ago(entity.last_seen_at)
        items.append(
            {
                "entity_id": entity.id,
                "label": entity.display_label,
                "last_seen_at": entity.last_seen_at.isoformat(),
                "zone": zone_name,
                "observation_id": entity.last_seen_observation_id,
                "phrase": render_last_seen(
                    label=entity.display_label, zone_name=zone_name, when_phrase=when
                ),
            }
        )
    if not items and detections:
        row = detections[0]
        items.append(
            {
                "detection_id": row.id,
                "label": row.label,
                "last_seen_at": row.captured_at.isoformat(),
                "observation_id": row.observation_id,
                "phrase": render_last_seen(
                    label=row.label, zone_name=None, when_phrase=_ago(row.captured_at)
                ),
            }
        )
    if not items:
        return {
            "presence": PresenceKind.UNKNOWN.value,
            "phrase": render("never_seen", label=needle),
            "items": [],
        }
    return {
        "presence": PresenceKind.PRESENT.value,
        "phrase": items[0]["phrase"],
        "items": items,
        "may_move_head": False,
    }


async def search_memory(session: AsyncSession, query: str, *, limit: int = 8) -> dict[str, Any]:
    last = await find_last_seen(session, query, limit=limit)
    relations = (
        (
            await session.execute(
                select(VisualSpatialRelation)
                .order_by(VisualSpatialRelation.captured_at.desc())
                .limit(limit)
            )
        )
        .scalars()
        .all()
    )
    phrases = [
        render_relation(
            subject_label="object",
            predicate=SpatialPredicate(rel.predicate),
            object_label=None,
            frame_class=FrameClass(rel.frame_class),
        )
        for rel in relations[:3]
    ]
    last["relations"] = [
        {
            "predicate": rel.predicate,
            "frame_class": rel.frame_class,
            "confidence": rel.confidence,
            "observation_id": rel.observed_from_observation_id,
        }
        for rel in relations
    ]
    if phrases and last.get("presence") == PresenceKind.UNKNOWN.value:
        last["phrase"] = phrases[0]
    return last


async def describe_scene(session: AsyncSession, snapshot_id: str) -> dict[str, Any]:
    snapshot = await session.get(SceneSnapshot, snapshot_id)
    if snapshot is None:
        raise NotFoundError("Scene snapshot not found.")
    observations = (
        (
            await session.execute(
                select(VisualObservation).where(VisualObservation.snapshot_id == snapshot_id)
            )
        )
        .scalars()
        .all()
    )
    labels: list[str] = []
    for obs in observations:
        dets = (
            await session.execute(
                select(ObjectDetection.label).where(ObjectDetection.observation_id == obs.id)
            )
        ).all()
        labels.extend(label for (label,) in dets)
    labels_text = ", ".join(sorted(set(labels))) if labels else ""
    if labels_text:
        phrase = render("present_count_hedge", label=labels_text)
    else:
        phrase = render("nothing_detected", label="object")
    return {
        "snapshot_id": snapshot.id,
        "state": snapshot.state,
        "kind": snapshot.kind,
        "viewpoint_count": snapshot.viewpoint_count,
        "completeness": snapshot.completeness,
        "labels": sorted(set(labels)),
        "phrase": phrase,
        "may_move_head": False,
    }


async def compare_scenes(session: AsyncSession, left_id: str, right_id: str) -> dict[str, Any]:
    left = await describe_scene(session, left_id)
    right = await describe_scene(session, right_id)
    left_set = set(left["labels"])
    right_set = set(right["labels"])
    appeared = sorted(right_set - left_set)
    disappeared = sorted(left_set - right_set)
    summary = change_summary_labels(appeared, disappeared)
    phrase = (
        render("compare_same")
        if not appeared and not disappeared
        else render("compare_changed", summary=summary)
    )
    return {
        "appeared": appeared,
        "disappeared": disappeared,
        "phrase": phrase,
        "left": left,
        "right": right,
        "may_move_head": False,
    }


async def _zone_name(session: AsyncSession, zone_id: str | None) -> str | None:
    if not zone_id:
        return None
    zone = await session.get(VisualZone, zone_id)
    return zone.name if zone else None


def _ago(when: datetime) -> str:
    return qualitative_when(when, utcnow())


def change_summary_labels(appeared: list[str], disappeared: list[str]) -> str:
    parts: list[str] = []
    if appeared:
        parts.append("now also " + ", ".join(appeared))
    if disappeared:
        parts.append("no longer " + ", ".join(disappeared))
    return "; ".join(parts) if parts else "no meaningful change"
