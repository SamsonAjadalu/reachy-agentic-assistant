"""Keyframe-to-keyframe change events. Dedup is structural, not statistical."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from vision.enums import VisualEventType
from vision.geometry import Box
from vision.identity import IdentityLink
from vision.tracking import DetectionHypothesis

MOVE_CENTER_DELTA = 0.18


@dataclass(frozen=True)
class EventCandidate:
    event_type: VisualEventType
    label: str
    entity_id: str | None
    dedup_key: str
    payload: dict[str, str | int | float | None]


def detect_events(
    *,
    previous: list[DetectionHypothesis],
    current: list[DetectionHypothesis],
    previous_links: list[IdentityLink] | None = None,
    current_links: list[IdentityLink] | None = None,
    occurred_at: datetime,
    zone_id: str | None = None,
) -> list[EventCandidate]:
    prev_labels = _label_multiset(previous)
    curr_labels = _label_multiset(current)
    prev_entities = _entity_set(previous_links)
    curr_entities = _entity_set(current_links)
    events: list[EventCandidate] = []

    for label, count in curr_labels.items():
        delta = count - prev_labels.get(label, 0)
        if delta > 0:
            events.append(
                _event(
                    VisualEventType.APPEARED,
                    label,
                    None,
                    occurred_at,
                    zone_id,
                    {"previous": prev_labels.get(label, 0), "current": count},
                )
            )
    for label, count in prev_labels.items():
        if curr_labels.get(label, 0) < count:
            events.append(
                _event(
                    VisualEventType.DISAPPEARED,
                    label,
                    None,
                    occurred_at,
                    zone_id,
                    {"previous": count, "current": curr_labels.get(label, 0)},
                )
            )

    for entity_id in curr_entities - prev_entities:
        if entity_id:
            events.append(
                _event(VisualEventType.APPEARED, "entity", entity_id, occurred_at, zone_id, {})
            )
    for entity_id in prev_entities - curr_entities:
        if entity_id:
            events.append(
                _event(VisualEventType.DISAPPEARED, "entity", entity_id, occurred_at, zone_id, {})
            )

    moved = _moved(previous, current)
    events.extend(moved)
    return events


def _moved(
    previous: list[DetectionHypothesis], current: list[DetectionHypothesis]
) -> list[EventCandidate]:
    events: list[EventCandidate] = []
    used: set[int] = set()
    for prev in previous:
        if prev.box is None or prev.is_person:
            continue
        best_i = -1
        best_iou = 0.0
        for i, cur in enumerate(current):
            if i in used or cur.box is None or cur.label != prev.label or cur.is_person:
                continue
            iou = prev.box.iou(cur.box)
            if iou > best_iou:
                best_iou = iou
                best_i = i
        if best_i < 0:
            continue
        used.add(best_i)
        cur = current[best_i]
        assert prev.box is not None and cur.box is not None
        if _center_delta(prev.box, cur.box) >= MOVE_CENTER_DELTA and best_iou < 0.5:
            events.append(
                EventCandidate(
                    event_type=VisualEventType.MOVED,
                    label=prev.label,
                    entity_id=None,
                    dedup_key="",
                    payload={"iou": round(best_iou, 3)},
                )
            )
    return events


def _event(
    kind: VisualEventType,
    label: str,
    entity_id: str | None,
    occurred_at: datetime,
    zone_id: str | None,
    extra: dict[str, int | float | str | None],
) -> EventCandidate:
    bucket = occurred_at.replace(second=0, microsecond=0).isoformat()
    key = f"{kind.value}:{entity_id or label}:{zone_id or '-'}:{bucket}"
    return EventCandidate(
        event_type=kind, label=label, entity_id=entity_id, dedup_key=key, payload=extra
    )


def _label_multiset(items: list[DetectionHypothesis]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        if item.is_person:
            continue
        key = item.label.strip().lower()
        counts[key] = counts.get(key, 0) + 1
    return counts


def _entity_set(links: list[IdentityLink] | None) -> set[str]:
    if not links:
        return set()
    return {link.entity_id for link in links if link.entity_id}


def _center_delta(left: Box, right: Box) -> float:
    dx = left.cx - right.cx
    dy = left.cy - right.cy
    return (dx * dx + dy * dy) ** 0.5
