"""Within-scan tracking: Hungarian assignment with an explicit no-match.

Identical-looking objects in the same frame are not forced into a unique
track — the system abstains rather than inventing identity.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from vision.enums import AssociationDecision, TrackState
from vision.geometry import Box
from vision.hungarian import linear_sum_assignment
from vision.labels import is_person_label, labels_compatible, normalise_label
from vision.vectors import cosine

DEFAULT_IOU_GATE = 0.3
DEFAULT_COST_GATE = 0.72
DEFAULT_MARGIN = 0.12
DEFAULT_IDENTICAL_COSINE = 0.97


@dataclass
class DetectionHypothesis:
    label: str
    box: Box
    score: float
    is_person: bool = False
    embedding: list[float] | None = None
    depth_median: float | None = None
    detection_id: str | None = None
    query: str | None = None

    @property
    def normalised_label(self) -> str:
        return normalise_label(self.label)


def hypothesis_from_write(
    label: str,
    *,
    box_xyxy: tuple[float, float, float, float],
    score: float = 0.8,
    embedding: list[float] | None = None,
    is_person: bool | None = None,
    depth_median: float | None = None,
) -> DetectionHypothesis:
    person = is_person_label(label) if is_person is None else is_person
    return DetectionHypothesis(
        label=normalise_label(label),
        box=Box.from_xyxy(box_xyxy),
        score=score,
        is_person=person,
        embedding=None if person else embedding,
        depth_median=depth_median,
    )


@dataclass
class TrackHypothesis:
    track_id: str
    label: str
    box: Box
    state: str = TrackState.TENTATIVE.value
    embedding: list[float] | None = None
    hits: int = 1
    misses: int = 0
    is_person: bool = False
    detection_ids: list[str] = field(default_factory=list)

    @property
    def detection_count(self) -> int:
        return self.hits


@dataclass(frozen=True)
class Association:
    detection_index: int
    track_id: str | None
    decision: AssociationDecision
    cost: float | None
    margin: float | None
    n_competing: int
    n_similar: int
    reason: str | None = None


@dataclass
class AssociationResult:
    assignments: list[Association]
    n_similar_by_label: dict[str, int]
    tracks: list[TrackHypothesis] = field(default_factory=list)


def _pair_cost(
    track: TrackHypothesis,
    detection: DetectionHypothesis,
    *,
    identical_cosine: float,
) -> float:
    if track.is_person or detection.is_person:
        if not (track.is_person and detection.is_person):
            return float("inf")
        iou = track.box.iou(detection.box)
        return 1.0 - iou if iou > 0.0 else float("inf")
    if not labels_compatible(track.label, detection.label):
        return float("inf")
    iou = track.box.iou(detection.box)
    embed = cosine(track.embedding, detection.embedding)
    if embed is not None and embed >= identical_cosine:
        # Uses the configured workflow.
        return 0.5 * (1.0 - iou)
    if embed is not None:
        return 0.55 * (1.0 - iou) + 0.45 * (1.0 - max(0.0, embed))
    return 1.0 - iou


def count_similar_instances(
    detections: list[DetectionHypothesis],
    *,
    identical_cosine: float = DEFAULT_IDENTICAL_COSINE,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    by_label: dict[str, list[DetectionHypothesis]] = {}
    for detection in detections:
        if detection.is_person or is_person_label(detection.label):
            continue
        by_label.setdefault(detection.normalised_label, []).append(detection)
    for label, group in by_label.items():
        similar = 0
        for index, left in enumerate(group):
            for right in group[index + 1 :]:
                score = cosine(left.embedding, right.embedding)
                iou = left.box.iou(right.box)
                if (score is not None and score >= identical_cosine) or (
                    score is None and iou < 0.15 and labels_compatible(left.label, right.label)
                ):
                    similar += 1
        # Two boxes of the same open-vocab label in one frame are already a pair.
        counts[label] = max(len(group), similar + (1 if similar else len(group)))
        if len(group) >= 2:
            counts[label] = len(group)
    return counts


def associate_detections(
    tracks: list[TrackHypothesis],
    detections: list[DetectionHypothesis],
    *,
    cost_gate: float = DEFAULT_COST_GATE,
    unmatched_cost: float | None = None,
    margin: float = DEFAULT_MARGIN,
    identical_cosine: float = DEFAULT_IDENTICAL_COSINE,
    next_track_id: int = 1,
) -> tuple[list[TrackHypothesis], AssociationResult]:
    if unmatched_cost is not None:
        cost_gate = unmatched_cost
    similar = count_similar_instances(detections, identical_cosine=identical_cosine)
    if not detections:
        for track in tracks:
            track.misses += 1
            if track.misses >= 2:
                track.state = TrackState.LOST.value
        return tracks, AssociationResult(assignments=[], n_similar_by_label=similar, tracks=tracks)

    if not tracks:
        assignments: list[Association] = []
        updated: list[TrackHypothesis] = []
        for index, detection in enumerate(detections):
            n_similar = similar.get(detection.normalised_label, 1)
            if detection.is_person:
                track_id = f"t-person-{next_track_id}"
                next_track_id += 1
                updated.append(
                    TrackHypothesis(
                        track_id=track_id,
                        label=detection.normalised_label,
                        box=detection.box,
                        is_person=True,
                        detection_ids=[detection.detection_id] if detection.detection_id else [],
                    )
                )
                assignments.append(
                    Association(
                        detection_index=index,
                        track_id=track_id,
                        decision=AssociationDecision.BLOCKED_PERSON,
                        cost=None,
                        margin=None,
                        n_competing=0,
                        n_similar=0,
                        reason="person detections are not persistent identities",
                    )
                )
                continue
            if n_similar >= 2:
                assignments.append(
                    Association(
                        detection_index=index,
                        track_id=None,
                        decision=AssociationDecision.ABSTAIN,
                        cost=None,
                        margin=0.0,
                        n_competing=n_similar,
                        n_similar=n_similar,
                        reason="identical instances; abstaining rather than inventing identity",
                    )
                )
                continue
            track_id = f"t-{next_track_id}"
            next_track_id += 1
            updated.append(
                TrackHypothesis(
                    track_id=track_id,
                    label=detection.normalised_label,
                    box=detection.box,
                    embedding=None if detection.is_person else detection.embedding,
                    detection_ids=[detection.detection_id] if detection.detection_id else [],
                    state=TrackState.CONFIRMED.value,
                )
            )
            assignments.append(
                Association(
                    detection_index=index,
                    track_id=track_id,
                    decision=AssociationDecision.NEW,
                    cost=None,
                    margin=None,
                    n_competing=0,
                    n_similar=n_similar,
                )
            )
        return updated, AssociationResult(
            assignments=assignments, n_similar_by_label=similar, tracks=updated
        )

    cost = np.full((len(detections), len(tracks)), np.inf, dtype=np.float64)
    for d_index, detection in enumerate(detections):
        for t_index, track in enumerate(tracks):
            cost[d_index, t_index] = _pair_cost(track, detection, identical_cosine=identical_cosine)

    row_ind, col_ind = linear_sum_assignment(cost)
    matched_detections = {int(i) for i in row_ind}
    matched_tracks = {int(i) for i in col_ind}
    assignments = []
    for d_index, t_index in zip((int(i) for i in row_ind), (int(i) for i in col_ind), strict=True):
        pair_cost = float(cost[d_index, t_index])
        detection = detections[d_index]
        n_similar = similar.get(detection.normalised_label, 1)
        row = cost[d_index]
        finite = sorted(float(v) for v in row if np.isfinite(v))
        second = finite[1] if len(finite) > 1 else None
        pair_margin = None if second is None else second - pair_cost
        n_competing = sum(1 for v in row if np.isfinite(v) and float(v) <= cost_gate)
        if detection.is_person:
            tracks[t_index].box = detection.box
            tracks[t_index].hits += 1
            tracks[t_index].misses = 0
            assignments.append(
                Association(
                    detection_index=d_index,
                    track_id=tracks[t_index].track_id,
                    decision=AssociationDecision.BLOCKED_PERSON,
                    cost=pair_cost,
                    margin=pair_margin,
                    n_competing=n_competing,
                    n_similar=0,
                    reason="person detections are not persistent identities",
                )
            )
            continue
        if n_similar >= 2 or (pair_margin is not None and pair_margin < margin):
            assignments.append(
                Association(
                    detection_index=d_index,
                    track_id=None,
                    decision=AssociationDecision.ABSTAIN,
                    cost=pair_cost,
                    margin=pair_margin,
                    n_competing=max(n_competing, n_similar),
                    n_similar=n_similar,
                    reason="ambiguous match among identical or near-identical objects",
                )
            )
            continue
        if pair_cost > cost_gate:
            assignments.append(
                Association(
                    detection_index=d_index,
                    track_id=None,
                    decision=AssociationDecision.NO_MATCH,
                    cost=pair_cost,
                    margin=pair_margin,
                    n_competing=n_competing,
                    n_similar=n_similar,
                    reason="cost above no-match gate",
                )
            )
            continue
        track = tracks[t_index]
        track.box = detection.box
        track.hits += 1
        track.misses = 0
        track.state = TrackState.CONFIRMED.value
        if detection.embedding is not None and not detection.is_person:
            track.embedding = detection.embedding
        if detection.detection_id:
            track.detection_ids.append(detection.detection_id)
        assignments.append(
            Association(
                detection_index=d_index,
                track_id=track.track_id,
                decision=AssociationDecision.MATCHED,
                cost=pair_cost,
                margin=pair_margin,
                n_competing=n_competing,
                n_similar=n_similar,
            )
        )

    for d_index, detection in enumerate(detections):
        if d_index in matched_detections:
            continue
        n_similar = similar.get(detection.normalised_label, 1)
        if detection.is_person:
            track_id = f"t-person-{next_track_id}"
            next_track_id += 1
            tracks.append(
                TrackHypothesis(
                    track_id=track_id,
                    label=detection.normalised_label,
                    box=detection.box,
                    is_person=True,
                    detection_ids=[detection.detection_id] if detection.detection_id else [],
                )
            )
            assignments.append(
                Association(
                    detection_index=d_index,
                    track_id=track_id,
                    decision=AssociationDecision.BLOCKED_PERSON,
                    cost=None,
                    margin=None,
                    n_competing=0,
                    n_similar=0,
                )
            )
            continue
        if n_similar >= 2:
            assignments.append(
                Association(
                    detection_index=d_index,
                    track_id=None,
                    decision=AssociationDecision.ABSTAIN,
                    cost=None,
                    margin=0.0,
                    n_competing=n_similar,
                    n_similar=n_similar,
                    reason="identical instances; abstaining rather than inventing identity",
                )
            )
            continue
        track_id = f"t-{next_track_id}"
        next_track_id += 1
        tracks.append(
            TrackHypothesis(
                track_id=track_id,
                label=detection.normalised_label,
                box=detection.box,
                embedding=detection.embedding,
                detection_ids=[detection.detection_id] if detection.detection_id else [],
                state=TrackState.CONFIRMED.value,
            )
        )
        assignments.append(
            Association(
                detection_index=d_index,
                track_id=track_id,
                decision=AssociationDecision.NEW,
                cost=None,
                margin=None,
                n_competing=0,
                n_similar=n_similar,
            )
        )

    for t_index, track in enumerate(tracks):
        if t_index not in matched_tracks:
            track.misses += 1
            if track.misses >= 2:
                track.state = TrackState.LOST.value

    return tracks, AssociationResult(
        assignments=assignments, n_similar_by_label=similar, tracks=tracks
    )


def update_tracks(
    tracks: list[TrackHypothesis],
    detections: list[DetectionHypothesis],
    assignments: AssociationResult
    | list[Association]
    | tuple[list[TrackHypothesis], AssociationResult],
    *,
    next_id: int,
    captured_at: object,
    confirm_after: int = 1,
) -> tuple[list[TrackHypothesis], int]:
    """Confirm or spawn tracks from an association result."""
    _ = (detections, captured_at)
    existing: list[TrackHypothesis]
    links: list[Association]
    if isinstance(assignments, tuple):
        existing, result = assignments
        links = result.assignments
    elif isinstance(assignments, AssociationResult):
        existing = assignments.tracks or tracks
        links = assignments.assignments
    else:
        existing = tracks
        links = assignments
    if not existing:
        existing, result = associate_detections(tracks, detections, next_track_id=next_id)
        links = result.assignments
    for track in existing:
        if track.hits >= confirm_after:
            track.state = TrackState.CONFIRMED.value
    spawned = sum(1 for item in links if item.decision is AssociationDecision.NEW)
    return existing, next_id + spawned
