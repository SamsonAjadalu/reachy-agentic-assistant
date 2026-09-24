"""Qualitative spatial relations. Metric distance is unrepresentable."""

from __future__ import annotations

from dataclasses import dataclass

from vision.enums import FrameClass, SpatialPredicate
from vision.labels import normalise_label
from vision.tracking import DetectionHypothesis

Y_OVERLAP_MIN = 0.25
X_OVERLAP_MIN = 0.25
CX_SEPARATION = 0.04
CY_SEPARATION = 0.04
ON_VERTICAL_GAP = 0.12
DEPTH_DELTA = 0.08


@dataclass(frozen=True)
class InferredRelation:
    predicate: SpatialPredicate
    frame_class: FrameClass
    subject_index: int
    object_index: int | None
    confidence: float
    needs_reobservation: bool = False


def infer_relations(
    detections: list[DetectionHypothesis],
    *,
    quality_ok: bool,
) -> list[InferredRelation]:
    """Viewpoint-dependent relations from boxes and ordinal depth.

    A failed quality gate yields no relations (the frame may still persist).
    """
    if not quality_ok or len(detections) < 2:
        return []
    relations: list[InferredRelation] = []
    for i, subject in enumerate(detections):
        if subject.is_person:
            continue
        for j, other in enumerate(detections):
            if i == j or other.is_person:
                continue
            if normalise_label(subject.label) == normalise_label(other.label) and i > j:
                continue
            relations.extend(_pair_relations(i, j, subject, other))
    return relations


def relations_from_detections(
    detections: list[DetectionHypothesis],
    *,
    quality_ok: bool,
) -> list[InferredRelation]:
    return infer_relations(detections, quality_ok=quality_ok)


def _pair_relations(
    i: int,
    j: int,
    subject: DetectionHypothesis,
    other: DetectionHypothesis,
) -> list[InferredRelation]:
    out: list[InferredRelation] = []
    y_overlap = subject.box.y_overlap_fraction(other.box)
    x_overlap = subject.box.x_overlap_fraction(other.box)
    if y_overlap >= Y_OVERLAP_MIN and abs(subject.box.cx - other.box.cx) >= CX_SEPARATION:
        if subject.box.cx < other.box.cx:
            out.append(
                InferredRelation(
                    SpatialPredicate.LEFT_OF,
                    FrameClass.EGOCENTRIC,
                    i,
                    j,
                    min(0.9, 0.55 + y_overlap),
                )
            )
        else:
            out.append(
                InferredRelation(
                    SpatialPredicate.RIGHT_OF,
                    FrameClass.EGOCENTRIC,
                    i,
                    j,
                    min(0.9, 0.55 + y_overlap),
                )
            )
    if x_overlap >= X_OVERLAP_MIN and abs(subject.box.cy - other.box.cy) >= CY_SEPARATION:
        # Image y grows downward; smaller cy is higher in the frame.
        if subject.box.cy < other.box.cy:
            out.append(
                InferredRelation(
                    SpatialPredicate.ABOVE, FrameClass.EGOCENTRIC, i, j, min(0.85, 0.5 + x_overlap)
                )
            )
        else:
            out.append(
                InferredRelation(
                    SpatialPredicate.BELOW, FrameClass.EGOCENTRIC, i, j, min(0.85, 0.5 + x_overlap)
                )
            )
    if (
        x_overlap >= 0.4
        and 0.0 <= (other.box.y0 - subject.box.y1) <= ON_VERTICAL_GAP
        and subject.box.y1 <= other.box.y0 + ON_VERTICAL_GAP
    ):
        out.append(
            InferredRelation(
                SpatialPredicate.ON, FrameClass.EGOCENTRIC, i, j, 0.55, needs_reobservation=True
            )
        )
    if subject.depth_median is not None and other.depth_median is not None:
        delta = other.depth_median - subject.depth_median
        if abs(delta) >= DEPTH_DELTA:
            # Smaller relative depth is treated as closer to the camera.
            if delta > 0:
                out.append(
                    InferredRelation(SpatialPredicate.IN_FRONT_OF, FrameClass.EGOCENTRIC, i, j, 0.6)
                )
            else:
                out.append(
                    InferredRelation(SpatialPredicate.BEHIND, FrameClass.EGOCENTRIC, i, j, 0.6)
                )
    if x_overlap >= 0.3 and y_overlap >= 0.3:
        out.append(
            InferredRelation(SpatialPredicate.CO_VISIBLE_WITH, FrameClass.EGOCENTRIC, i, j, 0.7)
        )
    return out
