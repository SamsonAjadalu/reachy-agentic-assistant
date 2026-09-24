"""Structured results for visual tools. Phrases are templates only."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from vision.enums import LookOutcome, Presence, ScanOutcomeKind
from vision.need_look import NeedLook


@dataclass
class DetectionView:
    label: str
    score: float
    box_xyxy: tuple[float, float, float, float]
    is_person: bool
    entity_id: str | None
    track_id: str | None
    association: str
    n_similar: int
    depth_median: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "score": self.score,
            "box_xyxy": list(self.box_xyxy),
            "is_person": self.is_person,
            "entity_id": self.entity_id,
            "track_id": self.track_id,
            "association": self.association,
            "n_similar": self.n_similar,
            "decision": self.association,
            "depth_median": self.depth_median,
            "depth_status": "inferred" if self.depth_median is not None else "unknown",
        }


@dataclass
class RelationView:
    predicate: str
    frame_class: str
    subject_label: str
    object_label: str | None
    phrase: str
    confidence: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "predicate": self.predicate,
            "frame_class": self.frame_class,
            "subject_label": self.subject_label,
            "object_label": self.object_label,
            "phrase": self.phrase,
            "confidence": self.confidence,
        }


@dataclass
class SceneResult:
    outcome: LookOutcome
    presence: Presence
    phrase: str
    query: str
    detections: list[DetectionView] = field(default_factory=list)
    relations: list[RelationView] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    need_look: float = 0.0
    need_look_terms: dict[str, float] = field(default_factory=dict)
    observation_id: str | None = None
    snapshot_id: str | None = None
    thumbnail_id: str | None = None
    evidence_ids: list[str] = field(default_factory=list)
    mocked: bool = True
    may_move_head: bool = False
    task_id: str | None = None
    poll_url: str | None = None
    scan_id: str | None = None
    reject_reason: str | None = None
    scan_outcome: ScanOutcomeKind | None = None
    empty_reason: str | None = None
    world_frame_status: str = "unknown"
    calibration_version: str = "uncalibrated-nominal-sdk"

    def as_dict(self) -> dict[str, Any]:
        return {
            "outcome": self.outcome.value,
            "presence": self.presence.value,
            "phrase": self.phrase,
            "query": self.query,
            "detections": [item.as_dict() for item in self.detections],
            "relations": [item.as_dict() for item in self.relations],
            "events": self.events,
            "need_look": self.need_look,
            "need_look_terms": self.need_look_terms,
            "observation_id": self.observation_id,
            "snapshot_id": self.snapshot_id,
            "thumbnail_id": self.thumbnail_id,
            "evidence_ids": self.evidence_ids,
            "mocked": self.mocked,
            "may_move_head": self.may_move_head,
            "task_id": self.task_id,
            "poll_url": self.poll_url,
            "scan_id": self.scan_id,
            "reject_reason": self.reject_reason,
            "scan_outcome": self.scan_outcome.value if self.scan_outcome else None,
            "empty_reason": self.empty_reason,
            "world_frame_status": self.world_frame_status,
            "calibration_version": self.calibration_version,
            "nothing_detected_vs_nothing_present": {
                "nothing_detected": self.presence is Presence.NOTHING_DETECTED,
                "nothing_present": self.presence is Presence.NOTHING_PRESENT,
            },
        }


def need_from(need: NeedLook) -> tuple[float, dict[str, float]]:
    return need.total, need.as_dict()
