"""Versioned shapes for Pi camera frames, health, and bounded scans.

Field optionality follows the workstation draft contract. Nothing here is a Pi
measurement; values marked optional stay optional until a Pi report agrees.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

ADAPTER_VERSION = "v1"
ADAPTER_RELEASE = "1.0.0-agreed-pi-5b44bfc"
ADAPTER_CONTRACT = "reachy-mini-visual-memory-api"
ADAPTER_CONTRACT_STATUS = "AGREED"

CORRELATION_HEADER = "X-Correlation-ID"
FRAME_METADATA_HEADER = "X-Reachy-Frame-Metadata"

# Draft defaults aligned with the Pi camera HTTP adapter. Paths are
# configurable because the Pi report has not landed.
DEFAULT_FRAME_PATH = "/api/v1/camera/frame"
DEFAULT_STATUS_PATH = "/api/v1/camera/status"
DEFAULT_SCAN_PATH = "/api/v1/scan"
DEFAULT_SCAN_STATUS_PATH = "/api/v1/scan/{scan_id}"
DEFAULT_SCAN_CANCEL_PATH = "/api/v1/scan/{scan_id}/cancel"

STILL_MIME_TYPES = frozenset(
    {
        "image/jpeg",
        "image/jpg",
        "image/png",
        "image/webp",
    }
)

# Proposed identifiers from the draft preset table. Not an allowlist: the Pi
# validates its own set. The client only requires an identifier string.
PROPOSED_PRESET_IDS = frozenset(
    {
        "CLOSE_LOOK",
        "PARALLAX_LEFT_RIGHT",
        "REGION_SWEEP",
        "BODY_SWEEP",
        "VERIFY_ABSENCE",
    }
)

# Keys that must never appear on a scan request. Joint targets are not
# expressible through PiCameraPort.request_scan; this set is a wire-level net.
FORBIDDEN_SCAN_KEYS = frozenset(
    {
        "joints",
        "joint",
        "joint_targets",
        "joint_target",
        "joint_positions",
        "q",
        "pose",
        "poses",
        "head_pose",
        "head_pose_4x4",
        "head_joints_rad",
        "yaw",
        "body_yaw",
        "body_yaw_rad",
        "target",
        "targets",
        "trajectory",
        "positions",
        "angles",
        "pan",
        "tilt",
    }
)


class CameraHealth(StrEnum):
    """Draft enumeration from ADR-0001. Unknown Pi strings are not coerced."""

    OK = "ok"
    STALLED = "stalled"
    BUSY = "busy"
    DECODE_ERROR = "decode_error"
    UNAVAILABLE = "unavailable"


class BodyYawSource(StrEnum):
    MEASURED = "measured"
    COMMANDED = "commanded"
    UNKNOWN = "unknown"


class ScanRejectReason(StrEnum):
    """Draft 409 reasons. Rejection is a normal outcome, not an adapter error."""

    VOICE_ACTIVE = "voice_active"
    BUSY_WITH_MOTION = "busy_with_motion"
    MOTORS_DISABLED = "motors_disabled"
    LOW_BATTERY = "low_battery"
    UNSAFE_POSE = "unsafe_pose"
    RATE_LIMITED = "rate_limited"
    USER_DISABLED = "user_disabled"


HeadPose4x4 = tuple[
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
]


@dataclass(frozen=True, slots=True)
class ObservationTimestamps:
    """Clocks for one observation.

    ``received_*`` are always set by this adapter. Pi capture clocks are
    optional until the contract is agreed.
    """

    received_utc: datetime
    received_monotonic_ns: int
    capture_utc: datetime | None = None
    capture_monotonic_ns: int | None = None
    frame_age_ms: int | None = None


@dataclass(frozen=True, slots=True)
class CameraPose:
    """Pose accompanying a frame. Entirely optional: missing pose still stores the still."""

    head_pose_4x4: HeadPose4x4 | None = None
    head_joints_rad: tuple[float, ...] | None = None
    body_yaw_rad: float | None = None
    body_yaw_source: BodyYawSource | None = None
    automatic_body_yaw: bool | None = None
    head_pose_is_settled: bool | None = None
    joint_velocity_max_rad_s: float | None = None
    pose_capture_skew_ns: int | None = None
    imu: dict[str, float] | None = None


@dataclass(frozen=True, slots=True)
class CameraFrame:
    image_bytes: bytes
    mime_type: str
    timestamps: ObservationTimestamps
    correlation_id: str
    adapter_version: str = ADAPTER_VERSION
    frame_id: int | None = None
    camera_source_id: str | None = None
    camera_health: CameraHealth | None = None
    is_stale: bool = False
    image_width: int | None = None
    image_height: int | None = None
    pose: CameraPose | None = None
    camera_specs_name: str | None = None
    sdk_version: str | None = None
    daemon_version: str | None = None
    focus_position: float | None = None
    stream_transport: str | None = None


@dataclass(frozen=True, slots=True)
class CameraStatus:
    correlation_id: str
    adapter_version: str = ADAPTER_VERSION
    camera_health: CameraHealth | None = None
    image_width: int | None = None
    image_height: int | None = None
    focus_position: float | None = None
    stream_active: bool | None = None
    time_since_last_frame_ms: int | None = None


@dataclass(frozen=True, slots=True)
class ScanAccepted:
    scan_id: str
    correlation_id: str
    adapter_version: str = ADAPTER_VERSION
    planned_poses: int | None = None
    estimated_duration_ms: int | None = None


@dataclass(frozen=True, slots=True)
class ScanRejected:
    """Normal outcome: the Pi refused to move. Domain code must hedge, not fail."""

    reason: str
    correlation_id: str
    adapter_version: str = ADAPTER_VERSION
    retry_after_ms: int | None = None

    @property
    def known_reason(self) -> ScanRejectReason | None:
        try:
            return ScanRejectReason(self.reason)
        except ValueError:
            return None


@dataclass(frozen=True, slots=True)
class ScanCancelResult:
    scan_id: str
    correlation_id: str
    cancelled: bool
    adapter_version: str = ADAPTER_VERSION


@dataclass(frozen=True, slots=True)
class ScanStatus:
    scan_id: str
    correlation_id: str
    state: str
    preset_id: str | None = None
    planned_poses: int | None = None
    estimated_duration_ms: int | None = None
    current_pose_index: int | None = None
    completed_poses: int | None = None
    frame_ids: tuple[int, ...] = ()
    cancel_requested: bool | None = None
    cancel_reason: str | None = None
    error: str | None = None
    adapter_version: str = ADAPTER_VERSION


ScanOutcome = ScanAccepted | ScanRejected
