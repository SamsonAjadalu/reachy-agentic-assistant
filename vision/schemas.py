"""Pydantic schemas and JSON helpers used by visual storage and persistence."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from vision.enums import (
    WORLD_FRAME_STATUS_VALUE,
    EvidenceKind,
    FrameClass,
    ObservationTrigger,
    RetentionClass,
    SpatialPredicate,
)


def dump_json(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def load_json(value: str | None, default: Any = None) -> Any:
    if value is None:
        return default
    return json.loads(value)


class DetectionSet(BaseModel):
    """Minimal detection identity used by the keyframe and dedup gates."""

    labels: tuple[str, ...] = ()
    is_person: bool = False

    @classmethod
    def from_labels(
        cls, labels: list[str] | tuple[str, ...], *, is_person: bool = False
    ) -> DetectionSet:
        normalised = tuple(sorted(label.strip().lower() for label in labels if label.strip()))
        return cls(labels=normalised, is_person=is_person)

    def fingerprint(self) -> str:
        marker = "person" if self.is_person else "no-person"
        return f"{marker}|{'+'.join(self.labels)}"


class PoseBucket(BaseModel):
    """Coarse viewpoint used to scope near-duplicate suppression."""

    yaw_bucket_deg: int = 0
    viewpoint_label: str = "unknown"

    def key(self) -> str:
        return f"{self.viewpoint_label}:{self.yaw_bucket_deg}"


def pose_bucket(
    body_yaw_rad: float | None, viewpoint_label: str | None, *, step_deg: int = 15
) -> PoseBucket:
    if body_yaw_rad is None:
        yaw_deg = 0
    else:
        import math

        yaw_deg = int(round(math.degrees(body_yaw_rad) / step_deg) * step_deg)
    label = (viewpoint_label or "unknown").strip() or "unknown"
    return PoseBucket(yaw_bucket_deg=yaw_deg, viewpoint_label=label)


class DetectionWrite(BaseModel):
    label: str
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    box_xyxy: tuple[float, float, float, float] | None = None
    is_person: bool = False
    depth_median: float | None = None
    occlusion_ratio: float | None = Field(default=None, ge=0.0, le=1.0)
    quality_score: float | None = Field(default=None, ge=0.0, le=1.0)
    embedding: list[float] | None = None


class SpatialRelationWrite(BaseModel):
    predicate: SpatialPredicate
    frame_class: FrameClass
    subject_entity_id: str
    object_entity_id: str | None = None
    object_zone_id: str | None = None
    observed_from_observation_id: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    captured_at: datetime

    @model_validator(mode="after")
    def _egocentric_requires_viewpoint(self) -> SpatialRelationWrite:
        if self.frame_class is FrameClass.EGOCENTRIC and not self.observed_from_observation_id:
            raise ValueError(
                "Egocentric spatial relations require observed_from_observation_id; "
                "a viewpoint-dependent claim without a viewpoint is unrepresentable."
            )
        return self


class FrameIngest(BaseModel):
    """One sampled frame offered to the keyframe / storage pipeline."""

    image_bytes: bytes
    captured_at: datetime
    trigger: ObservationTrigger = ObservationTrigger.PASSIVE
    camera_source_id: str | None = None
    zone_id: str | None = None
    request_id: str | None = None
    snapshot_id: str | None = None
    viewpoint_label: str | None = None
    body_yaw_rad: float | None = None
    body_yaw_source: str = "unknown"
    head_pose_is_settled: bool = True
    head_pose_4x4: list[list[float]] | None = None
    head_joints_rad: list[float] | None = None
    T_head_cam: list[list[float]] | None = None
    calibration_version: str = "uncalibrated-nominal-sdk"
    world_frame_status: str = WORLD_FRAME_STATUS_VALUE
    contains_person: bool = False
    is_event: bool = False
    detection_set: DetectionSet = Field(default_factory=DetectionSet)
    detections: list[DetectionWrite] = Field(default_factory=list)
    blur_flag: bool = False
    exposure_flag: bool = False
    quality_score: float | None = None
    quality_flags: list[str] = Field(default_factory=list)
    mean_luma: float | None = None
    laplacian_variance: float | None = None
    camera_health_json: str | None = None
    captured_at_pi: datetime | None = None
    is_stale: bool = False
    idempotency_key: str | None = None
    correlation_id: str | None = None
    model_name: str | None = None
    model_version: str | None = None
    width: int | None = None
    height: int | None = None

    @field_validator("captured_at")
    @classmethod
    def _aware_captured_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("captured_at must be timezone-aware UTC.")
        return value

    @field_validator("world_frame_status")
    @classmethod
    def _world_frame_is_unknown(cls, value: str) -> str:
        if value != WORLD_FRAME_STATUS_VALUE:
            raise ValueError("world_frame_status is always 'unknown' in Phase 1.")
        return value

    @field_validator("head_joints_rad")
    @classmethod
    def _seven_stewart_joints(cls, value: list[float] | None) -> list[float] | None:
        if value is not None and len(value) not in {6, 7}:
            raise ValueError(
                "head_joints_rad must have 6 Stewart joints, optionally plus body yaw."
            )
        return value


class StoredEvidence(BaseModel):
    content_hash: str
    relative_path: str
    kind: EvidenceKind
    retention_class: RetentionClass
    byte_size: int
    width: int
    height: int
    dhash64: str
    phash64: str
    reused: bool = False
    contains_person: bool = False


class IngestResult(BaseModel):
    persisted: bool
    skip_reason: str | None = None
    observation_id: str | None = None
    thumbnail_id: str | None = None
    full_frame_id: str | None = None
    evidence_reused: bool = False
    content_hash: str | None = None
