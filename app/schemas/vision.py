"""Visual memory request and response bodies."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import Field

from app.schemas.common import ApiModel


class CameraStatusOut(ApiModel):
    adapter_version: str
    camera_health: str
    image_width: int | None = None
    image_height: int | None = None
    stream_active: bool | None = None
    time_since_last_frame_ms: int | None = None
    mocked: bool = True
    phrase: str = "Camera status is structured; I am not making a spatial claim."
    may_move_head: bool = False
    world_frame_status: str = "unknown"


class LookNowRequest(ApiModel):
    query: str | None = Field(default=None, max_length=200)
    zone: str | None = Field(default=None, max_length=80)
    preset_id: str | None = Field(default=None, max_length=80)
    allow_scan: bool = False
    inject_labels: list[str] = Field(default_factory=list, max_length=16)


class ObserveOut(ApiModel):
    presence: str
    phrase: str
    deferred: bool = False
    task_id: str | None = None
    poll_url: str | None = None
    observation_id: str | None = None
    snapshot_id: str | None = None
    request_id: str | None = None
    detections: list[dict[str, Any]] = Field(default_factory=list)
    relations: list[dict[str, Any]] = Field(default_factory=list)
    need_look: float | None = None
    need_look_terms: dict[str, float] | None = None
    scan_outcome: str | None = None
    evidence_id: str | None = None
    world_frame_status: str = "unknown"
    may_move_head: bool = False
    presence_is_not_absence: bool = True


class LastSeenOut(ApiModel):
    presence: str
    phrase: str
    items: list[dict[str, Any]] = Field(default_factory=list)
    abstained: bool = False
    may_move_head: bool = False
    relations: list[dict[str, Any]] = Field(default_factory=list)


class SceneOut(ApiModel):
    snapshot_id: str
    state: str
    kind: str
    viewpoint_count: int
    completeness: float | None = None
    labels: list[str] = Field(default_factory=list)
    phrase: str
    may_move_head: bool = False


class CompareRequest(ApiModel):
    left_snapshot_id: str
    right_snapshot_id: str


class CompareOut(ApiModel):
    appeared: list[str] = Field(default_factory=list)
    disappeared: list[str] = Field(default_factory=list)
    phrase: str
    left: dict[str, Any]
    right: dict[str, Any]
    may_move_head: bool = False


class WatchCreate(ApiModel):
    label: str = Field(min_length=1, max_length=120)
    event: str = "change"
    zone: str | None = None
    cooldown_seconds: int = Field(default=3600, ge=60, le=86400)
    quiet_hours: dict[str, str] | None = None


class WatchOut(ApiModel):
    id: str
    status: str
    trigger: str
    query: dict[str, Any]
    next_check_at: datetime | None = None
    cooldown_seconds: int
    last_fired_at: datetime | None = None
    phrase: str = "Watch recorded. I will notify you on Telegram if it fires."
    may_move_head: bool = False


class WatchList(ApiModel):
    items: list[WatchOut]
    total: int
    limit: int
    offset: int
    phrase: str = ""
    may_move_head: bool = False


class ScanRequest(ApiModel):
    preset_id: str = Field(min_length=1, max_length=80)
    query: str | None = Field(default=None, max_length=200)
    zone: str | None = None
