"""Keyframe-to-keyframe change events. Dedup is structural, not statistical."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vision.enums import VisualEventType
from vision.geometry import Box
from vision.hungarian import linear_sum_assignment
from vision.labels import labels_compatible, normalise_label
from vision.tracking import DetectionHypothesis

MOVE_SHIFT = 0.12
IOU_MATCH = 0.3


@dataclass(frozen=True)
class SceneChange:
    event_type: VisualEventType
    label: str
    before_index: int | None
    after_index: int | None
    confidence: float


def detect_changes(
    before: list[DetectionHypothesis],
    after: list[DetectionHypothesis],
) -> list[SceneChange]:
    if not before and not after:
        return []
    if not before:
        return [
            SceneChange(
                VisualEventType.APPEARED,
                normalise_label(item.label),
                None,
                index,
                item.score,
            )
            for index, item in enumerate(after)
            if not item.is_person
        ]
    if not after:
        return [
            SceneChange(
                VisualEventType.DISAPPEARED, normalise_label(item.label), index, None, item.score
            )
            for index, item in enumerate(before)
            if not item.is_person
        ]

    cost = np.full((len(before), len(after)), np.inf, dtype=np.float64)
    for i, left in enumerate(before):
        if left.is_person:
            continue
        for j, right in enumerate(after):
            if right.is_person or not labels_compatible(left.label, right.label):
                continue
            iou = left.box.iou(right.box)
            cost[i, j] = 1.0 - iou
    row_ind, col_ind = linear_sum_assignment(cost)
    matched_before = set()
    matched_after = set()
    events: list[SceneChange] = []
    for i, j in zip((int(a) for a in row_ind), (int(b) for b in col_ind), strict=True):
        pair_cost = float(cost[i, j])
        if pair_cost > (1.0 - IOU_MATCH):
            continue
        matched_before.add(i)
        matched_after.add(j)
        shift = before[i].box.centroid_shift(after[j].box)
        if shift >= MOVE_SHIFT:
            events.append(
                SceneChange(
                    VisualEventType.MOVED,
                    normalise_label(after[j].label),
                    i,
                    j,
                    min(0.9, 0.4 + shift),
                )
            )
        elif _appearance_changed(before[i].box, after[j].box):
            events.append(
                SceneChange(
                    VisualEventType.CHANGED,
                    normalise_label(after[j].label),
                    i,
                    j,
                    0.55,
                )
            )
    for index, item in enumerate(after):
        if item.is_person or index in matched_after:
            continue
        events.append(
            SceneChange(
                VisualEventType.APPEARED, normalise_label(item.label), None, index, item.score
            )
        )
    for index, item in enumerate(before):
        if item.is_person or index in matched_before:
            continue
        events.append(
            SceneChange(
                VisualEventType.DISAPPEARED, normalise_label(item.label), index, None, item.score
            )
        )
    return events


def _appearance_changed(left: Box, right: Box) -> bool:
    area_delta = abs(left.area - right.area)
    return area_delta >= 0.04


def change_summary(events: list[SceneChange]) -> str:
    if not events:
        return "no meaningful change"
    parts = [f"a {event.label} {event.event_type.value}" for event in events[:4]]
    return ", ".join(parts)


def structural_dedup_key(
    *,
    watch_id: str | None,
    event_type: str,
    label: str,
    observation_id: str | None,
) -> str:
    return "|".join(
        [
            watch_id or "-",
            event_type,
            normalise_label(label),
            observation_id or "-",
        ]
    )
