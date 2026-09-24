"""In-process fake Pi camera for MOCK_MODE. Implements PiCameraPort."""

from __future__ import annotations

import uuid
from io import BytesIO

from PIL import Image

from integrations.reachy.camera.models import (
    ADAPTER_VERSION,
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
from shared.timeutils import utcnow


def _jpeg(*, color: tuple[int, int, int], width: int = 64, height: int = 48) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (width, height), color=color).save(buffer, format="JPEG", quality=85)
    return buffer.getvalue()


class LocalFakeCamera:
    """Deterministic stills + preset-only scans. No joint targets."""

    adapter_version = ADAPTER_VERSION

    def __init__(
        self,
        *,
        reject_reason: str | None = None,
        health: CameraHealth = CameraHealth.OK,
    ) -> None:
        self.reject_reason = reject_reason
        self.health = health
        self.frame_index = 0
        self.accepted_ids: list[str] = []
        self.last_preset: str | None = None

    async def get_latest_frame(self, *, correlation_id: str | None = None) -> CameraFrame:
        self.frame_index += 1
        shade = 40 + (self.frame_index * 17) % 180
        now = utcnow()
        cid = correlation_id or uuid.uuid4().hex
        return CameraFrame(
            image_bytes=_jpeg(color=(shade, 80, 120)),
            mime_type="image/jpeg",
            timestamps=ObservationTimestamps(
                received_utc=now,
                received_monotonic_ns=self.frame_index * 10_000_000,
                capture_utc=now,
                frame_age_ms=12,
            ),
            correlation_id=cid,
            camera_health=self.health,
            is_stale=self.health is not CameraHealth.OK,
            image_width=64,
            image_height=48,
            pose=CameraPose(head_pose_is_settled=True, body_yaw_rad=0.0),
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
        cid = correlation_id or uuid.uuid4().hex
        self.last_preset = preset_id
        if self.reject_reason:
            return ScanRejected(reason=self.reject_reason, correlation_id=cid, retry_after_ms=4000)
        scan_id = uuid.uuid4().hex
        self.accepted_ids.append(scan_id)
        return ScanAccepted(scan_id=scan_id, correlation_id=cid, planned_poses=3)

    async def cancel_scan(
        self, *, scan_id: str, correlation_id: str | None = None
    ) -> ScanCancelResult:
        return ScanCancelResult(
            scan_id=scan_id,
            correlation_id=correlation_id or uuid.uuid4().hex,
            cancelled=True,
        )

    async def get_scan_status(
        self, *, scan_id: str, correlation_id: str | None = None
    ) -> ScanStatus:
        cid = correlation_id or uuid.uuid4().hex
        return ScanStatus(
            scan_id=scan_id,
            correlation_id=cid,
            state="queued",
            preset_id=self.last_preset,
            planned_poses=3,
        )

    async def aclose(self) -> None:
        return None


def as_port(camera: LocalFakeCamera) -> PiCameraPort:
    return camera
