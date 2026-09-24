"""Camera port factory: in-process fake (MOCK_MODE) or live Pi adapter."""

from __future__ import annotations

import time
import uuid
from io import BytesIO

from PIL import Image

from app.config import Settings
from integrations.reachy.camera.fake import sample_still_jpeg
from integrations.reachy.camera.models import (
    ADAPTER_VERSION,
    FORBIDDEN_SCAN_KEYS,
    PROPOSED_PRESET_IDS,
    BodyYawSource,
    CameraFrame,
    CameraHealth,
    CameraPose,
    CameraStatus,
    ObservationTimestamps,
    ScanAccepted,
    ScanCancelResult,
    ScanOutcome,
    ScanRejected,
    ScanStatus,
)
from integrations.reachy.camera.port import PiCameraPort
from integrations.reachy.camera.settings import PiCameraSettings
from shared.errors import ValidationError
from shared.timeutils import utcnow

IDENTITY_HEAD_POSE = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)


class InProcessFakeCamera:
    """Deterministic PiCameraPort for MOCK_MODE. No network, no joint targets."""

    adapter_version = ADAPTER_VERSION

    def __init__(
        self,
        *,
        reject_reason: str | None = None,
        planned_poses: int = 3,
        stale: bool = False,
        health: CameraHealth = CameraHealth.OK,
    ) -> None:
        self.reject_reason = reject_reason
        self.planned_poses = planned_poses
        self.stale = stale
        self.health = health
        self.frame_index = 0
        self.accepted_ids: list[str] = []
        self.cancelled_ids: list[str] = []
        self.last_preset: str | None = None
        self.scan_calls = 0
        self.frame_calls = 0

    def _still(self) -> bytes:
        color = (
            32 + (self.frame_index * 17) % 80,
            64 + (self.frame_index * 11) % 80,
            96 + (self.frame_index * 5) % 80,
        )
        buffer = BytesIO()
        Image.new("RGB", (64, 48), color=color).save(buffer, format="JPEG", quality=80)
        return buffer.getvalue()

    async def get_latest_frame(self, *, correlation_id: str | None = None) -> CameraFrame:
        self.frame_calls += 1
        cid = correlation_id or uuid.uuid4().hex
        image = self._still()
        yaw = 0.05 * self.frame_index
        self.frame_index += 1
        return CameraFrame(
            image_bytes=image,
            mime_type="image/jpeg",
            timestamps=ObservationTimestamps(
                received_utc=utcnow(),
                received_monotonic_ns=time.monotonic_ns(),
                capture_utc=utcnow(),
                capture_monotonic_ns=time.monotonic_ns(),
                frame_age_ms=800 if self.stale else 12,
            ),
            correlation_id=cid,
            camera_health=self.health,
            is_stale=self.stale,
            image_width=64,
            image_height=48,
            pose=CameraPose(
                head_pose_4x4=IDENTITY_HEAD_POSE,
                head_joints_rad=(0.0,) * 7,
                body_yaw_rad=yaw,
                body_yaw_source=BodyYawSource.UNKNOWN,
                automatic_body_yaw=False,
                head_pose_is_settled=True,
                joint_velocity_max_rad_s=0.0,
                pose_capture_skew_ns=0,
            ),
        )

    async def get_status(self, *, correlation_id: str | None = None) -> CameraStatus:
        return CameraStatus(
            correlation_id=correlation_id or uuid.uuid4().hex,
            camera_health=self.health,
            image_width=64,
            image_height=48,
            stream_active=False,
            time_since_last_frame_ms=12,
        )

    async def health_check(self, *, correlation_id: str | None = None) -> CameraStatus:
        return await self.get_status(correlation_id=correlation_id)

    async def request_scan(
        self,
        *,
        preset_id: str,
        reason: str | None = None,
        correlation_id: str | None = None,
    ) -> ScanOutcome:
        _ = reason
        self.scan_calls += 1
        assert_preset_only(preset_id)
        cid = correlation_id or uuid.uuid4().hex
        if self.reject_reason:
            return ScanRejected(reason=self.reject_reason, correlation_id=cid, retry_after_ms=5000)
        scan_id = uuid.uuid4().hex
        self.accepted_ids.append(scan_id)
        self.last_preset = preset_id
        return ScanAccepted(
            scan_id=scan_id,
            correlation_id=cid,
            planned_poses=self.planned_poses,
            estimated_duration_ms=8000,
        )

    async def cancel_scan(
        self, *, scan_id: str, correlation_id: str | None = None
    ) -> ScanCancelResult:
        self.cancelled_ids.append(scan_id)
        return ScanCancelResult(
            scan_id=scan_id,
            correlation_id=correlation_id or uuid.uuid4().hex,
            cancelled=True,
        )

    async def get_scan_status(
        self, *, scan_id: str, correlation_id: str | None = None
    ) -> ScanStatus:
        return ScanStatus(
            scan_id=scan_id,
            correlation_id=correlation_id or uuid.uuid4().hex,
            state="queued",
            preset_id=self.last_preset,
            planned_poses=self.planned_poses,
        )

    async def aclose(self) -> None:
        return None


def assert_preset_only(preset_id: str) -> None:
    if not preset_id or not preset_id.strip():
        raise ValidationError("preset_id is required; raw poses are not accepted.")
    lowered = preset_id.strip()
    if lowered.lower() in FORBIDDEN_SCAN_KEYS:
        raise ValidationError("Joint targets are not expressible. Request a preset_id.")
    if any(ch.isspace() for ch in lowered):
        raise ValidationError("preset_id must be a single identifier.")


def pi_settings_from_app(settings: Settings) -> PiCameraSettings:
    token = settings.reachy_camera_bearer_token()
    return PiCameraSettings(
        enabled=settings.reachy_camera_enabled,
        base_url=settings.reachy_camera_url.rstrip("/"),
        token=token,
        timeout_seconds=settings.reachy_camera_timeout_seconds,
        connect_timeout_seconds=settings.reachy_camera_connect_timeout_seconds,
        max_body_bytes=settings.reachy_camera_max_body_bytes,
        stale_frame_ms=settings.reachy_camera_stale_frame_ms,
        frame_path=settings.reachy_camera_frame_path,
        frame_method=settings.reachy_camera_frame_method,
        status_path=settings.reachy_camera_status_path,
        scan_path=settings.reachy_camera_scan_path,
        scan_status_path=settings.reachy_camera_scan_status_path,
        scan_cancel_path=settings.reachy_camera_scan_cancel_path,
        get_attempts=settings.reachy_camera_get_attempts,
    )


def build_camera(settings: Settings) -> PiCameraPort:
    if settings.mock_mode or not settings.reachy_camera_enabled:
        return InProcessFakeCamera()
    from integrations.reachy.camera.client import PiCameraClientV1

    return PiCameraClientV1(pi_settings_from_app(settings))


def jpeg_placeholder() -> bytes:
    return sample_still_jpeg(width=64, height=48)


def known_preset(preset_id: str) -> bool:
    return preset_id in PROPOSED_PRESET_IDS
