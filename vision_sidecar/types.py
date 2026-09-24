"""Typed job API for FastAPI → sidecar HTTP.

Boxes are normalised xyxy in ``[0, 1]``. No metric distance field exists.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class JobKind(StrEnum):
    DETECT = "detect"
    SEGMENT = "segment"
    DEPTH = "depth"
    EMBED = "embed"
    SCENE_EMBED = "scene_embed"


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class FaultKind(StrEnum):
    CAMERA_UNAVAILABLE = "camera_unavailable"
    DECODE_ERROR = "decode_error"
    TIMEOUT = "timeout"
    OOM = "oom"
    EMPTY = "empty"
    STALE_FRAME = "stale_frame"
    DEPTH_SWAP = "depth_swap"


class FramePayload(BaseModel):
    image_b64: str | None = None
    image_path: str | None = None
    content_sha256: str | None = None

    @model_validator(mode="after")
    def _one_source(self) -> FramePayload:
        if bool(self.image_b64) == bool(self.image_path):
            raise ValueError("Provide exactly one of image_b64 or image_path.")
        return self


class DetectRequest(BaseModel):
    frame: FramePayload
    queries: list[str] = Field(min_length=1, max_length=32)
    max_detections: int = Field(default=32, ge=1, le=128)
    fault: FaultKind | None = None


class BoxPrompt(BaseModel):
    box_xyxy: tuple[float, float, float, float]
    label: str | None = None


class PointPrompt(BaseModel):
    xy: tuple[float, float]
    label: int = 1


class SegmentRequest(BaseModel):
    frame: FramePayload
    boxes: list[BoxPrompt] = Field(default_factory=list)
    points: list[PointPrompt] = Field(default_factory=list)
    fault: FaultKind | None = None

    @model_validator(mode="after")
    def _needs_prompt(self) -> SegmentRequest:
        if not self.boxes and not self.points:
            raise ValueError("Segmentation needs at least one box or point prompt.")
        return self


class DepthRequest(BaseModel):
    frame: FramePayload
    include_preview: bool = True
    fault: FaultKind | None = None


class EmbedRequest(BaseModel):
    crops: list[FramePayload] = Field(default_factory=list)
    texts: list[str] = Field(default_factory=list)
    fault: FaultKind | None = None

    @model_validator(mode="after")
    def _needs_input(self) -> EmbedRequest:
        if not self.crops and not self.texts:
            raise ValueError("embed needs at least one crop or text.")
        return self


class SceneEmbedRequest(BaseModel):
    frame: FramePayload
    fault: FaultKind | None = None


class JobSubmitRequest(BaseModel):
    kind: JobKind
    detect: DetectRequest | None = None
    segment: SegmentRequest | None = None
    depth: DepthRequest | None = None
    embed: EmbedRequest | None = None
    scene_embed: SceneEmbedRequest | None = None
    wait_ms: int = Field(default=0, ge=0, le=120_000)

    @model_validator(mode="after")
    def _payload_matches_kind(self) -> JobSubmitRequest:
        payload = {
            JobKind.DETECT: self.detect,
            JobKind.SEGMENT: self.segment,
            JobKind.DEPTH: self.depth,
            JobKind.EMBED: self.embed,
            JobKind.SCENE_EMBED: self.scene_embed,
        }
        chosen = payload[self.kind]
        others = [value for key, value in payload.items() if key != self.kind and value is not None]
        if chosen is None:
            raise ValueError(f"Missing payload for kind {self.kind.value!r}.")
        if others:
            raise ValueError("Submit exactly one job payload matching kind.")
        return self


class Detection(BaseModel):
    label: str
    score: float = Field(ge=0.0, le=1.0)
    box_xyxy: tuple[float, float, float, float]
    query: str
    provenance: Literal["inferred"] = "inferred"

    @field_validator("box_xyxy")
    @classmethod
    def _unit_box(
        cls, value: tuple[float, float, float, float]
    ) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = value
        if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
            raise ValueError("box_xyxy must be normalised xyxy in [0, 1] with positive area.")
        return value


class Mask(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    box_xyxy: tuple[float, float, float, float]
    area_fraction: float = Field(ge=0.0, le=1.0)
    rle_counts: list[int] = Field(default_factory=list)
    provenance: Literal["inferred"] = "inferred"


class DepthMap(BaseModel):
    width: int
    height: int
    relative_min: float
    relative_max: float
    relative_median: float
    has_confidence: bool
    preview_png_b64: str | None = None
    ordinal_ready: bool = True
    provenance: Literal["inferred"] = "inferred"
    licence: str
    model_name: str
    model_version: str


class EmbeddingVector(BaseModel):
    values: list[float]
    dim: int
    space: str
    model_name: str
    model_version: str
    provenance: Literal["inferred"] = "inferred"

    @model_validator(mode="after")
    def _dim_matches(self) -> EmbeddingVector:
        if len(self.values) != self.dim:
            raise ValueError("embedding dim does not match values length")
        return self


class ProviderMeta(BaseModel):
    provider_name: str
    model_name: str
    model_version: str
    licence: str
    device: str
    mocked: bool


class DetectResult(BaseModel):
    detections: list[Detection]
    empty_reason: str | None = None
    meta: ProviderMeta


class SegmentResult(BaseModel):
    masks: list[Mask]
    meta: ProviderMeta


class DepthResult(BaseModel):
    depth: DepthMap
    meta: ProviderMeta


class EmbedResult(BaseModel):
    vectors: list[EmbeddingVector]
    meta: ProviderMeta


class JobPublic(BaseModel):
    job_id: str
    kind: JobKind
    status: JobStatus
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    provider_name: str | None = None
    model_name: str | None = None
    model_version: str | None = None
    latency_ms: float | None = None
    degraded: bool = False
    degraded_reason: str | None = None
    correlation_id: str | None = None
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
