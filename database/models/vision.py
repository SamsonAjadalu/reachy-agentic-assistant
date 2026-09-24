"""Visual spatial memory ORM models.

JSON lives in ``Text`` columns with a comment; enums are ``String(n)``. There is
no ``sqlalchemy.JSON`` column, no metric position, and no world frame.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Boolean,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database.base import Base, TimestampMixin, UtcDateTime, UUIDPrimaryKeyMixin
from vision.enums import (
    BodyYawSource,
    CalibrationStatus,
    CameraSourceKind,
    ChurnClass,
    DepthSpace,
    DistortionModel,
    EmbeddingOwnerKind,
    EntityStatus,
    EvidenceKind,
    EvidenceState,
    ObservationRequestStatus,
    ObservationTrigger,
    ProvenanceState,
    ProvenanceStatus,
    RetentionClass,
    SnapshotKind,
    SnapshotState,
    TrackState,
    VisualWatchStatus,
    WorldFrameStatus,
)

SOFT_UUID = String(36)


class CameraSource(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "camera_sources"

    name: Mapped[str] = mapped_column(String(80), nullable=False)
    kind: Mapped[str] = mapped_column(
        String(32), nullable=False, default=CameraSourceKind.REACHY_HEAD.value
    )
    device_label: Mapped[str | None] = mapped_column(String(200))
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    last_seen_at: Mapped[datetime | None] = mapped_column(UtcDateTime)
    notes: Mapped[str | None] = mapped_column(Text)

    __table_args__ = (UniqueConstraint("name", name="uq_camera_sources_name"),)


class CameraCalibration(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "camera_calibrations"

    camera_source_id: Mapped[str | None] = mapped_column(
        ForeignKey("camera_sources.id", ondelete="SET NULL"), index=True
    )
    version: Mapped[str] = mapped_column(String(80), nullable=False)
    resolution_width: Mapped[int] = mapped_column(Integer, nullable=False)
    resolution_height: Mapped[int] = mapped_column(Integer, nullable=False)
    distortion_model: Mapped[str] = mapped_column(
        String(32), nullable=False, default=DistortionModel.UNKNOWN.value
    )
    intrinsics_k: Mapped[str | None] = mapped_column(Text, comment="JSON 3x3 matrix, row-major")
    intrinsics_d: Mapped[str | None] = mapped_column(Text, comment="JSON distortion coefficients")
    intrinsics_status: Mapped[str] = mapped_column(
        String(20), nullable=False, default=CalibrationStatus.UNKNOWN.value
    )
    t_head_cam: Mapped[str | None] = mapped_column(
        Text, comment="JSON 4x4 nominal head-from-camera transform"
    )
    t_head_cam_status: Mapped[str] = mapped_column(
        String(20), nullable=False, default=ProvenanceStatus.NOMINAL.value
    )
    accepted: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    accepted_at: Mapped[datetime | None] = mapped_column(UtcDateTime)
    superseded_by_id: Mapped[str | None] = mapped_column(SOFT_UUID)
    reprojection_rms: Mapped[float | None] = mapped_column(Float)
    notes: Mapped[str | None] = mapped_column(Text)

    __table_args__ = (UniqueConstraint("version", name="uq_camera_calibrations_version"),)

    camera_source: Mapped[CameraSource | None] = relationship()


class VisualZone(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "visual_zones"

    name: Mapped[str] = mapped_column(String(200), nullable=False)
    normalised_name: Mapped[str] = mapped_column(String(200), nullable=False)
    privacy_sensitive: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    capture_allowed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    churn_class: Mapped[str] = mapped_column(
        String(20), nullable=False, default=ChurnClass.MEDIUM.value
    )
    description: Mapped[str | None] = mapped_column(Text)

    __table_args__ = (UniqueConstraint("normalised_name", name="uq_visual_zones_normalised_name"),)


class ModelVersion(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Uses the configured workflow."""

    __tablename__ = "model_versions"

    name: Mapped[str] = mapped_column(String(80), nullable=False)
    version: Mapped[str] = mapped_column(String(80), nullable=False)
    task: Mapped[str] = mapped_column(String(40), nullable=False, default="detect")
    weights_sha256: Mapped[str | None] = mapped_column(String(64))
    licence: Mapped[str | None] = mapped_column(String(80))
    retired: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    notes: Mapped[str | None] = mapped_column(Text)

    __table_args__ = (UniqueConstraint("name", "version", name="uq_model_versions_name_version"),)

    runs: Mapped[list[ModelRun]] = relationship(back_populates="model_version")


class ModelRun(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "model_runs"

    model_version_id: Mapped[str] = mapped_column(
        ForeignKey("model_versions.id", ondelete="RESTRICT"), nullable=False
    )
    started_at: Mapped[datetime] = mapped_column(UtcDateTime, nullable=False)
    finished_at: Mapped[datetime | None] = mapped_column(UtcDateTime)
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    device: Mapped[str | None] = mapped_column(String(40))
    params_json: Mapped[str | None] = mapped_column(Text, comment="JSON object")
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="succeeded")
    error_message: Mapped[str | None] = mapped_column(Text)
    observation_id: Mapped[str | None] = mapped_column(SOFT_UUID)

    __table_args__ = (Index("ix_model_runs_version_started_at", "model_version_id", "started_at"),)

    model_version: Mapped[ModelVersion] = relationship(back_populates="runs")


class EvaluationDataset(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "evaluation_datasets"

    name: Mapped[str] = mapped_column(String(120), nullable=False)
    version: Mapped[str] = mapped_column(String(40), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    immutable: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    parent_id: Mapped[str | None] = mapped_column(
        ForeignKey("evaluation_datasets.id", ondelete="SET NULL")
    )
    frame_index_sha256: Mapped[str | None] = mapped_column(String(64))

    __table_args__ = (
        UniqueConstraint("name", "version", name="uq_evaluation_datasets_name_version"),
    )


class EvidenceAsset(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "evidence_assets"

    kind: Mapped[str] = mapped_column(
        String(32), nullable=False, default=EvidenceKind.THUMBNAIL.value
    )
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    relative_path: Mapped[str] = mapped_column(String(400), nullable=False)
    dhash64: Mapped[str | None] = mapped_column(String(16))
    phash64: Mapped[str | None] = mapped_column(String(16))
    width: Mapped[int | None] = mapped_column(Integer)
    height: Mapped[int | None] = mapped_column(Integer)
    byte_size: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    mime_type: Mapped[str] = mapped_column(String(40), nullable=False, default="image/jpeg")
    retention_class: Mapped[str] = mapped_column(
        String(20), nullable=False, default=RetentionClass.THUMBNAIL.value, index=True
    )
    state: Mapped[str] = mapped_column(
        String(20), nullable=False, default=EvidenceState.ACTIVE.value
    )
    expires_at: Mapped[datetime | None] = mapped_column(UtcDateTime)
    contains_person: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    privacy_sensitive: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    reference_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, comment="Hint only; recompute from FKs each sweep"
    )
    zone_id: Mapped[str | None] = mapped_column(
        ForeignKey("visual_zones.id", ondelete="SET NULL"), index=True
    )
    pose_bucket: Mapped[str | None] = mapped_column(String(80), index=True)
    model_version: Mapped[str | None] = mapped_column(String(80))
    observation_id: Mapped[str | None] = mapped_column(SOFT_UUID)
    idempotency_key: Mapped[str | None] = mapped_column(String(200))

    __table_args__ = (
        UniqueConstraint("content_hash", name="uq_evidence_assets_content_hash"),
        UniqueConstraint("idempotency_key", name="uq_evidence_assets_idempotency_key"),
        Index("ix_evidence_assets_state_expires_at", "state", "expires_at"),
        Index("ix_evidence_assets_phash64", "phash64"),
        Index("ix_evidence_assets_dhash64", "dhash64"),
    )

    zone: Mapped[VisualZone | None] = relationship()


class ObservationRequest(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "observation_requests"

    camera_source_id: Mapped[str | None] = mapped_column(
        ForeignKey("camera_sources.id", ondelete="SET NULL"), index=True
    )
    zone_id: Mapped[str | None] = mapped_column(
        ForeignKey("visual_zones.id", ondelete="SET NULL"), index=True
    )
    trigger: Mapped[str] = mapped_column(
        String(32), nullable=False, default=ObservationTrigger.QUERY.value
    )
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, default=ObservationRequestStatus.PENDING.value
    )
    preset_id: Mapped[str | None] = mapped_column(String(80))
    query_text: Mapped[str | None] = mapped_column(String(500))
    reject_reason: Mapped[str | None] = mapped_column(String(80))
    abort_reason: Mapped[str | None] = mapped_column(String(80))
    requested_at: Mapped[datetime] = mapped_column(UtcDateTime, nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(UtcDateTime)
    finished_at: Mapped[datetime | None] = mapped_column(UtcDateTime)
    payload_json: Mapped[str | None] = mapped_column(Text, comment="JSON object")
    idempotency_key: Mapped[str | None] = mapped_column(String(200))
    correlation_id: Mapped[str | None] = mapped_column(String(64), index=True)

    __table_args__ = (
        UniqueConstraint("idempotency_key", name="uq_observation_requests_idempotency_key"),
    )


class SceneSnapshot(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "scene_snapshots"

    request_id: Mapped[str | None] = mapped_column(
        ForeignKey("observation_requests.id", ondelete="SET NULL"), index=True
    )
    zone_id: Mapped[str | None] = mapped_column(
        ForeignKey("visual_zones.id", ondelete="SET NULL"), index=True
    )
    kind: Mapped[str] = mapped_column(String(32), nullable=False, default=SnapshotKind.QUERY.value)
    state: Mapped[str] = mapped_column(
        String(32), nullable=False, default=SnapshotState.PLANNED.value
    )
    abort_reason: Mapped[str | None] = mapped_column(String(80))
    started_at: Mapped[datetime] = mapped_column(UtcDateTime, nullable=False)
    finished_at: Mapped[datetime | None] = mapped_column(UtcDateTime)
    coverage_json: Mapped[str | None] = mapped_column(Text, comment="JSON object")
    viewpoint_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    completeness: Mapped[float | None] = mapped_column(Float)
    idempotency_key: Mapped[str | None] = mapped_column(String(200))

    __table_args__ = (
        UniqueConstraint("idempotency_key", name="uq_scene_snapshots_idempotency_key"),
        Index("ix_scene_snapshots_kind_started_at", "kind", "started_at"),
    )

    request: Mapped[ObservationRequest | None] = relationship()
    zone: Mapped[VisualZone | None] = relationship()


class VisualObservation(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "visual_observations"

    camera_source_id: Mapped[str | None] = mapped_column(
        ForeignKey("camera_sources.id", ondelete="SET NULL"), index=True
    )
    request_id: Mapped[str | None] = mapped_column(
        ForeignKey("observation_requests.id", ondelete="SET NULL"), index=True
    )
    snapshot_id: Mapped[str | None] = mapped_column(
        ForeignKey("scene_snapshots.id", ondelete="SET NULL"), index=True
    )
    zone_id: Mapped[str | None] = mapped_column(ForeignKey("visual_zones.id", ondelete="SET NULL"))
    calibration_id: Mapped[str | None] = mapped_column(
        ForeignKey("camera_calibrations.id", ondelete="SET NULL"), index=True
    )
    captured_at: Mapped[datetime] = mapped_column(UtcDateTime, nullable=False)
    captured_at_pi: Mapped[datetime | None] = mapped_column(UtcDateTime)
    captured_at_pc: Mapped[datetime | None] = mapped_column(UtcDateTime)
    clock_offset_ns: Mapped[int | None] = mapped_column(Integer)
    trigger: Mapped[str] = mapped_column(
        String(32), nullable=False, default=ObservationTrigger.PASSIVE.value
    )
    calibration_version: Mapped[str | None] = mapped_column(String(80))
    intrinsics_k: Mapped[str | None] = mapped_column(
        Text, comment="JSON 3x3, valid at recorded resolution"
    )
    intrinsics_d: Mapped[str | None] = mapped_column(Text, comment="JSON distortion coefficients")
    distortion_model: Mapped[str] = mapped_column(
        String(32), nullable=False, default=DistortionModel.UNKNOWN.value
    )
    intrinsics_status: Mapped[str] = mapped_column(
        String(20), nullable=False, default=ProvenanceStatus.UNKNOWN.value
    )
    t_head_cam: Mapped[str | None] = mapped_column(
        Text, comment="JSON 4x4, nominal, SDK-version-dependent"
    )
    head_pose_4x4: Mapped[str | None] = mapped_column(Text, comment="JSON 4x4 derived pose")
    head_joints_rad: Mapped[str | None] = mapped_column(
        Text, comment="JSON array of 7 measured radians"
    )
    body_yaw_rad: Mapped[float | None] = mapped_column(Float)
    body_yaw_source: Mapped[str] = mapped_column(
        String(20), nullable=False, default=BodyYawSource.UNKNOWN.value
    )
    automatic_body_yaw: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    head_pose_is_settled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    joint_velocity_max_rad_s: Mapped[float | None] = mapped_column(Float)
    pose_capture_skew_ns: Mapped[int | None] = mapped_column(Integer)
    pose_status: Mapped[str] = mapped_column(
        String(20), nullable=False, default=ProvenanceStatus.UNKNOWN.value
    )
    viewpoint_label: Mapped[str | None] = mapped_column(String(80))
    pose_bucket: Mapped[str | None] = mapped_column(String(80))
    world_frame_status: Mapped[str] = mapped_column(
        String(20), nullable=False, default=WorldFrameStatus.UNKNOWN.value
    )
    depth_model: Mapped[str | None] = mapped_column(String(80))
    depth_model_version: Mapped[str | None] = mapped_column(String(80))
    depth_model_licence: Mapped[str | None] = mapped_column(String(80))
    depth_space: Mapped[str] = mapped_column(
        String(32), nullable=False, default=DepthSpace.RELATIVE_NORMALISED.value
    )
    depth_output_kind: Mapped[str | None] = mapped_column(String(32))
    depth_input_space: Mapped[str | None] = mapped_column(String(40))
    depth_normalisation: Mapped[str | None] = mapped_column(String(40))
    depth_norm_lo: Mapped[float | None] = mapped_column(Float)
    depth_norm_hi: Mapped[float | None] = mapped_column(Float)
    sky_fraction: Mapped[float | None] = mapped_column(Float)
    depth_status: Mapped[str] = mapped_column(
        String(20), nullable=False, default=ProvenanceStatus.UNKNOWN.value
    )
    blur_score: Mapped[float | None] = mapped_column(Float)
    blur_flag: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    exposure_flag: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    saturated_fraction: Mapped[float | None] = mapped_column(Float)
    low_texture: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    quality_score: Mapped[float | None] = mapped_column(Float)
    quality_flags: Mapped[str | None] = mapped_column(Text, comment="JSON array")
    mean_luma: Mapped[float | None] = mapped_column(Float)
    laplacian_variance: Mapped[float | None] = mapped_column(Float)
    width: Mapped[int | None] = mapped_column(Integer)
    height: Mapped[int | None] = mapped_column(Integer)
    content_hash: Mapped[str | None] = mapped_column(String(64), index=True)
    dhash64: Mapped[str | None] = mapped_column(String(16))
    phash64: Mapped[str | None] = mapped_column(String(16))
    region_observed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    camera_health_json: Mapped[str | None] = mapped_column(Text, comment="JSON object")
    model_name: Mapped[str | None] = mapped_column(String(80))
    model_version: Mapped[str | None] = mapped_column(String(80))
    model_run_id: Mapped[str | None] = mapped_column(SOFT_UUID)
    full_frame_evidence_id: Mapped[str | None] = mapped_column(
        ForeignKey("evidence_assets.id", ondelete="SET NULL")
    )
    thumbnail_evidence_id: Mapped[str | None] = mapped_column(
        ForeignKey("evidence_assets.id", ondelete="SET NULL")
    )
    idempotency_key: Mapped[str | None] = mapped_column(String(200))
    correlation_id: Mapped[str | None] = mapped_column(String(64), index=True)

    __table_args__ = (
        UniqueConstraint("idempotency_key", name="uq_visual_observations_idempotency_key"),
        Index("ix_visual_observations_captured_at", "captured_at"),
        Index("ix_visual_observations_zone_captured_at", "zone_id", "captured_at"),
        Index("ix_visual_observations_trigger_captured_at", "trigger", "captured_at"),
    )

    camera_source: Mapped[CameraSource | None] = relationship()
    request: Mapped[ObservationRequest | None] = relationship()
    snapshot: Mapped[SceneSnapshot | None] = relationship()
    zone: Mapped[VisualZone | None] = relationship()
    calibration: Mapped[CameraCalibration | None] = relationship()
    detections: Mapped[list[ObjectDetection]] = relationship(
        back_populates="observation", cascade="all, delete-orphan"
    )


class ObjectTrack(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "object_tracks"

    snapshot_id: Mapped[str | None] = mapped_column(
        ForeignKey("scene_snapshots.id", ondelete="SET NULL"), index=True
    )
    label: Mapped[str | None] = mapped_column(String(120))
    state: Mapped[str] = mapped_column(
        String(20), nullable=False, default=TrackState.TENTATIVE.value
    )
    first_seen_at: Mapped[datetime] = mapped_column(UtcDateTime, nullable=False)
    last_seen_at: Mapped[datetime] = mapped_column(UtcDateTime, nullable=False)
    detection_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    __table_args__ = (Index("ix_object_tracks_state_last_seen_at", "state", "last_seen_at"),)


class ObjectEntity(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "object_entities"

    display_label: Mapped[str] = mapped_column(String(200), nullable=False)
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default=EntityStatus.ACTIVE.value
    )
    revision: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    is_person: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        comment="Must stay false; person entities are unrepresentable",
    )
    blocks_embeddings: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    first_seen_at: Mapped[datetime] = mapped_column(UtcDateTime, nullable=False)
    last_seen_at: Mapped[datetime] = mapped_column(UtcDateTime, nullable=False)
    observation_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    zone_id: Mapped[str | None] = mapped_column(ForeignKey("visual_zones.id", ondelete="SET NULL"))
    last_seen_observation_id: Mapped[str | None] = mapped_column(SOFT_UUID)
    canonical_evidence_id: Mapped[str | None] = mapped_column(
        ForeignKey("evidence_assets.id", ondelete="SET NULL")
    )
    prototype_embedding: Mapped[str | None] = mapped_column(
        Text, comment="JSON float32 prototype; empty for persons"
    )
    prototype_vector_sha256: Mapped[str | None] = mapped_column(String(64))
    merged_into_id: Mapped[str | None] = mapped_column(
        ForeignKey("object_entities.id", ondelete="SET NULL"), index=True
    )
    belief: Mapped[float | None] = mapped_column(Float)
    need_look: Mapped[float | None] = mapped_column(Float)
    need_look_terms: Mapped[str | None] = mapped_column(Text, comment="JSON object")
    support_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    contradiction_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    last_confirmed_utc: Mapped[datetime | None] = mapped_column(UtcDateTime)
    churn_class: Mapped[str] = mapped_column(
        String(20), nullable=False, default=ChurnClass.MEDIUM.value
    )
    staleness_half_life_s: Mapped[int | None] = mapped_column(Integer)
    scan_completeness: Mapped[float | None] = mapped_column(Float)
    evidence_frame_ids: Mapped[str | None] = mapped_column(
        Text, comment="JSON array of observation ids"
    )
    provenance_state: Mapped[str] = mapped_column(
        String(20), nullable=False, default=ProvenanceState.CURRENT.value
    )

    __table_args__ = (
        Index("ix_object_entities_status_last_seen_at", "status", "last_seen_at"),
        Index("ix_object_entities_zone_last_seen_at", "zone_id", "last_seen_at"),
    )

    zone: Mapped[VisualZone | None] = relationship()
    merged_into: Mapped[ObjectEntity | None] = relationship(remote_side="ObjectEntity.id")


class ObjectDetection(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "object_detections"

    observation_id: Mapped[str] = mapped_column(
        ForeignKey("visual_observations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    track_id: Mapped[str | None] = mapped_column(
        ForeignKey("object_tracks.id", ondelete="SET NULL"), index=True
    )
    entity_id: Mapped[str | None] = mapped_column(
        ForeignKey("object_entities.id", ondelete="SET NULL")
    )
    captured_at: Mapped[datetime] = mapped_column(UtcDateTime, nullable=False)
    label: Mapped[str] = mapped_column(String(120), nullable=False)
    label_status: Mapped[str] = mapped_column(
        String(20), nullable=False, default=ProvenanceStatus.INFERRED.value
    )
    is_person: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    blocks_entity: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    blocks_embedding: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    bbox_x0: Mapped[float | None] = mapped_column(Float)
    bbox_y0: Mapped[float | None] = mapped_column(Float)
    bbox_x1: Mapped[float | None] = mapped_column(Float)
    bbox_y1: Mapped[float | None] = mapped_column(Float)
    mask_evidence_id: Mapped[str | None] = mapped_column(
        ForeignKey("evidence_assets.id", ondelete="SET NULL")
    )
    crop_evidence_id: Mapped[str | None] = mapped_column(
        ForeignKey("evidence_assets.id", ondelete="SET NULL")
    )
    camera_ray_unit: Mapped[str | None] = mapped_column(
        Text, comment="JSON unit vector in camera frame"
    )
    camera_ray_status: Mapped[str] = mapped_column(
        String(20), nullable=False, default=ProvenanceStatus.UNKNOWN.value
    )
    camera_ray_body: Mapped[str | None] = mapped_column(
        Text, comment="JSON unit vector derived into body frame"
    )
    camera_ray_cone_deg: Mapped[float | None] = mapped_column(Float)
    depth_median: Mapped[float | None] = mapped_column(Float)
    depth_p10: Mapped[float | None] = mapped_column(Float)
    depth_p25: Mapped[float | None] = mapped_column(Float)
    depth_p75: Mapped[float | None] = mapped_column(Float)
    depth_p90: Mapped[float | None] = mapped_column(Float)
    depth_iqr: Mapped[float | None] = mapped_column(Float)
    depth_bottom_edge_median: Mapped[float | None] = mapped_column(Float)
    depth_conf_median: Mapped[float | None] = mapped_column(Float)
    depth_flicker_std: Mapped[float | None] = mapped_column(Float)
    depth_valid_fraction: Mapped[float | None] = mapped_column(Float)
    mask_touches_border: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    mask_border_fraction: Mapped[float | None] = mapped_column(Float)
    mask_radial_norm: Mapped[float | None] = mapped_column(Float)
    mask_area_px: Mapped[int | None] = mapped_column(Integer)
    mask_area_fraction: Mapped[float | None] = mapped_column(Float)
    mask_thinness: Mapped[float | None] = mapped_column(Float)
    occlusion_ratio: Mapped[float | None] = mapped_column(Float)
    detection_score_raw: Mapped[float | None] = mapped_column(Float)
    detection_confidence: Mapped[float | None] = mapped_column(Float)
    confidence_calibration_version: Mapped[str | None] = mapped_column(String(80))
    association_margin: Mapped[float | None] = mapped_column(Float)
    n_competing_candidates: Mapped[int | None] = mapped_column(Integer)
    n_similar_instances: Mapped[int | None] = mapped_column(Integer)
    quality_score: Mapped[float | None] = mapped_column(Float)
    provenance_state: Mapped[str] = mapped_column(
        String(20), nullable=False, default=ProvenanceState.CURRENT.value
    )
    model_name: Mapped[str | None] = mapped_column(String(80))
    model_version: Mapped[str | None] = mapped_column(String(80))
    model_run_id: Mapped[str | None] = mapped_column(SOFT_UUID)

    __table_args__ = (
        Index("ix_object_detections_label_captured_at", "label", "captured_at"),
        Index("ix_object_detections_entity_captured_at", "entity_id", "captured_at"),
    )

    observation: Mapped[VisualObservation] = relationship(back_populates="detections")
    track: Mapped[ObjectTrack | None] = relationship()
    entity: Mapped[ObjectEntity | None] = relationship()


class ObjectEntityObservation(UUIDPrimaryKeyMixin, Base):
    """Uses the configured workflow."""

    __tablename__ = "object_entity_observations"

    entity_id: Mapped[str] = mapped_column(
        ForeignKey("object_entities.id", ondelete="CASCADE"), nullable=False
    )
    detection_id: Mapped[str | None] = mapped_column(
        ForeignKey("object_detections.id", ondelete="SET NULL")
    )
    observation_id: Mapped[str | None] = mapped_column(
        ForeignKey("visual_observations.id", ondelete="SET NULL")
    )
    match_score: Mapped[float | None] = mapped_column(Float)
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    observed_at: Mapped[datetime] = mapped_column(UtcDateTime, nullable=False)

    __table_args__ = (
        Index("ix_object_entity_observations_entity_observed_at", "entity_id", "observed_at"),
    )


class ObjectIdentityRevision(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "object_identity_revisions"

    kind: Mapped[str] = mapped_column(String(20), nullable=False)
    from_entity_id: Mapped[str] = mapped_column(SOFT_UUID, nullable=False)
    to_entity_id: Mapped[str] = mapped_column(SOFT_UUID, nullable=False)
    moved_detection_ids: Mapped[str] = mapped_column(Text, nullable=False, comment="JSON array")
    occurred_at: Mapped[datetime] = mapped_column(UtcDateTime, nullable=False)
    actor: Mapped[str] = mapped_column(String(60), nullable=False, default="system")
    notes: Mapped[str | None] = mapped_column(Text)

    __table_args__ = (
        Index("ix_object_identity_revisions_kind_occurred_at", "kind", "occurred_at"),
    )


class VisualSpatialRelation(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "visual_spatial_relations"

    predicate: Mapped[str] = mapped_column(String(32), nullable=False)
    frame_class: Mapped[str] = mapped_column(String(20), nullable=False)
    subject_entity_id: Mapped[str | None] = mapped_column(
        ForeignKey("object_entities.id", ondelete="SET NULL"), index=True
    )
    object_entity_id: Mapped[str | None] = mapped_column(
        ForeignKey("object_entities.id", ondelete="SET NULL"), index=True
    )
    subject_detection_id: Mapped[str | None] = mapped_column(
        ForeignKey("object_detections.id", ondelete="SET NULL")
    )
    object_detection_id: Mapped[str | None] = mapped_column(
        ForeignKey("object_detections.id", ondelete="SET NULL")
    )
    observed_from_observation_id: Mapped[str | None] = mapped_column(
        ForeignKey("visual_observations.id", ondelete="SET NULL"), index=True
    )
    zone_id: Mapped[str | None] = mapped_column(
        ForeignKey("visual_zones.id", ondelete="SET NULL"), index=True
    )
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    confidence_cap: Mapped[float | None] = mapped_column(Float)
    needs_reobservation: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    captured_at: Mapped[datetime] = mapped_column(UtcDateTime, nullable=False)
    provenance_status: Mapped[str] = mapped_column(
        String(20), nullable=False, default=ProvenanceStatus.INFERRED.value
    )
    model_name: Mapped[str | None] = mapped_column(String(80))
    model_version: Mapped[str | None] = mapped_column(String(80))
    model_run_id: Mapped[str | None] = mapped_column(SOFT_UUID)

    __table_args__ = (
        Index(
            "ix_visual_spatial_relations_predicate_captured_at",
            "predicate",
            "captured_at",
        ),
    )


class VisualWatch(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "visual_watches"

    query_json: Mapped[str] = mapped_column(Text, nullable=False, comment="JSON object")
    trigger: Mapped[str] = mapped_column(String(32), nullable=False, default="schedule")
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default=VisualWatchStatus.ACTIVE.value
    )
    zone_id: Mapped[str | None] = mapped_column(
        ForeignKey("visual_zones.id", ondelete="SET NULL"), index=True
    )
    entity_id: Mapped[str | None] = mapped_column(
        ForeignKey("object_entities.id", ondelete="SET NULL"), index=True
    )
    next_check_at: Mapped[datetime | None] = mapped_column(UtcDateTime)
    cooldown_seconds: Mapped[int] = mapped_column(Integer, nullable=False, default=3600)
    quiet_hours_json: Mapped[str | None] = mapped_column(Text, comment="JSON object")
    baseline_evidence_id: Mapped[str | None] = mapped_column(
        ForeignKey("evidence_assets.id", ondelete="SET NULL")
    )
    last_fired_at: Mapped[datetime | None] = mapped_column(UtcDateTime)
    idempotency_key: Mapped[str | None] = mapped_column(String(200))

    __table_args__ = (
        UniqueConstraint("idempotency_key", name="uq_visual_watches_idempotency_key"),
        Index("ix_visual_watches_status_next_check_at", "status", "next_check_at"),
    )


class VisualEvent(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "visual_events"

    occurred_after: Mapped[datetime] = mapped_column(UtcDateTime, nullable=False)
    occurred_at: Mapped[datetime] = mapped_column(UtcDateTime, nullable=False)
    event_type: Mapped[str] = mapped_column(String(32), nullable=False)
    entity_id: Mapped[str | None] = mapped_column(
        ForeignKey("object_entities.id", ondelete="SET NULL"), index=True
    )
    zone_id: Mapped[str | None] = mapped_column(
        ForeignKey("visual_zones.id", ondelete="SET NULL"), index=True
    )
    watch_id: Mapped[str | None] = mapped_column(
        ForeignKey("visual_watches.id", ondelete="SET NULL"), index=True
    )
    observation_id: Mapped[str | None] = mapped_column(
        ForeignKey("visual_observations.id", ondelete="SET NULL")
    )
    before_evidence_id: Mapped[str | None] = mapped_column(
        ForeignKey("evidence_assets.id", ondelete="SET NULL")
    )
    after_evidence_id: Mapped[str | None] = mapped_column(
        ForeignKey("evidence_assets.id", ondelete="SET NULL")
    )
    payload_json: Mapped[str | None] = mapped_column(Text, comment="JSON object")
    dedup_key: Mapped[str | None] = mapped_column(String(200))
    idempotency_key: Mapped[str | None] = mapped_column(String(200))
    correlation_id: Mapped[str | None] = mapped_column(String(64), index=True)

    __table_args__ = (
        UniqueConstraint("dedup_key", name="uq_visual_events_dedup_key"),
        UniqueConstraint("idempotency_key", name="uq_visual_events_idempotency_key"),
        Index("ix_visual_events_occurred_at", "occurred_at"),
        Index("ix_visual_events_type_occurred_at", "event_type", "occurred_at"),
    )


class VisualEmbedding(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Metadata only. Vectors live in a sidecar; person owners are refused at write time."""

    __tablename__ = "visual_embeddings"

    owner_kind: Mapped[str] = mapped_column(
        String(32), nullable=False, default=EmbeddingOwnerKind.ENTITY.value
    )
    owner_id: Mapped[str] = mapped_column(String(36), nullable=False)
    space: Mapped[str] = mapped_column(String(40), nullable=False)
    model_name: Mapped[str] = mapped_column(String(80), nullable=False)
    model_version: Mapped[str] = mapped_column(String(80), nullable=False)
    dim: Mapped[int] = mapped_column(Integer, nullable=False)
    dtype: Mapped[str] = mapped_column(String(20), nullable=False, default="float32")
    vector_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    sidecar_key: Mapped[str | None] = mapped_column(String(200))
    is_person: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    __table_args__ = (
        UniqueConstraint(
            "owner_kind",
            "owner_id",
            "space",
            "model_name",
            "model_version",
            name="uq_visual_embeddings_owner_space_model",
        ),
    )


class EvaluationSample(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "evaluation_samples"

    dataset_id: Mapped[str] = mapped_column(
        ForeignKey("evaluation_datasets.id", ondelete="CASCADE"), nullable=False, index=True
    )
    suite: Mapped[str] = mapped_column(String(80), nullable=False)
    external_key: Mapped[str] = mapped_column(String(200), nullable=False)
    evidence_id: Mapped[str] = mapped_column(
        ForeignKey("evidence_assets.id", ondelete="RESTRICT"), nullable=False
    )
    split: Mapped[str] = mapped_column(String(20), nullable=False, default="test")
    ground_truth_json: Mapped[str] = mapped_column(Text, nullable=False, comment="JSON object")
    annotation_version: Mapped[str | None] = mapped_column(String(40))

    __table_args__ = (
        UniqueConstraint("suite", "external_key", name="uq_evaluation_samples_suite_external_key"),
    )

    dataset: Mapped[EvaluationDataset] = relationship()
    evidence: Mapped[EvidenceAsset] = relationship()


class EvaluationResult(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "evaluation_results"

    dataset_id: Mapped[str | None] = mapped_column(
        ForeignKey("evaluation_datasets.id", ondelete="SET NULL"), index=True
    )
    sample_id: Mapped[str | None] = mapped_column(
        ForeignKey("evaluation_samples.id", ondelete="SET NULL"), index=True
    )
    model_version_id: Mapped[str | None] = mapped_column(
        ForeignKey("model_versions.id", ondelete="RESTRICT"), index=True
    )
    model_run_id: Mapped[str | None] = mapped_column(SOFT_UUID)
    metric_name: Mapped[str] = mapped_column(String(80), nullable=False)
    metric_value: Mapped[float | None] = mapped_column(Float)
    metrics_json: Mapped[str | None] = mapped_column(Text, comment="JSON object")
    notes: Mapped[str | None] = mapped_column(Text)
    idempotency_key: Mapped[str | None] = mapped_column(String(200))

    __table_args__ = (
        Index("ix_evaluation_results_dataset_created_at", "dataset_id", "created_at"),
        UniqueConstraint("idempotency_key", name="uq_evaluation_results_idempotency_key"),
    )


class VisualDeletionReceipt(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "visual_deletion_receipts"

    scope: Mapped[str] = mapped_column(String(40), nullable=False)
    requested_at: Mapped[datetime] = mapped_column(UtcDateTime, nullable=False)
    counts_json: Mapped[str] = mapped_column(Text, nullable=False, comment="JSON object")
    manifest_digest: Mapped[str] = mapped_column(String(64), nullable=False)
    verified_empty: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    dry_run: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    approval_id: Mapped[str | None] = mapped_column(SOFT_UUID)

    __table_args__ = (Index("ix_visual_deletion_receipts_requested_at", "requested_at"),)
