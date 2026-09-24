"""Live Reachy Pi camera contract tests (optional, real hardware).

Skipped unless ``REACHY_PI_LIVE=1`` and ``REACHY_VISION_TOKEN`` (or camera/text fallback)
Uses the configured workflow.
"""

from __future__ import annotations

import os

import pytest

from integrations.reachy.camera.client import PiCameraClientV1
from integrations.reachy.camera.errors import PiCameraAuthError
from integrations.reachy.camera.settings import PiCameraSettings

pytestmark = pytest.mark.skipif(
    os.environ.get("REACHY_PI_LIVE", "").lower() not in {"1", "true", "yes"},
    reason="Set REACHY_PI_LIVE=1 to run against the real Pi",
)


def _live_settings() -> PiCameraSettings:
    settings = PiCameraSettings.from_env()
    if not settings.token:
        pytest.skip("Pi vision token not configured")
    return PiCameraSettings(
        enabled=True,
        base_url=settings.base_url,
        token=settings.token,
        timeout_seconds=settings.timeout_seconds,
        connect_timeout_seconds=settings.connect_timeout_seconds,
        max_body_bytes=settings.max_body_bytes,
        stale_frame_ms=settings.stale_frame_ms,
        frame_path=settings.frame_path,
        frame_method=settings.frame_method,
        status_path=settings.status_path,
        scan_path=settings.scan_path,
        scan_status_path=settings.scan_status_path,
        scan_cancel_path=settings.scan_cancel_path,
        get_attempts=settings.get_attempts,
    )


class TestLivePiCameraContract:
    async def test_status_and_frame_dimensions(self) -> None:
        async with PiCameraClientV1(_live_settings()) as client:
            status = await client.get_status(correlation_id="live-contract-status")
            frame = await client.get_latest_frame(correlation_id="live-contract-frame")
        assert status.camera_health is not None
        assert status.image_width == 1280
        assert status.image_height == 720
        assert frame.image_width == 1280
        assert frame.image_height == 720
        assert frame.mime_type == "image/jpeg"
        assert len(frame.image_bytes) > 1000
        assert frame.frame_id is not None
        assert frame.pose is not None

    async def test_auth_failure(self) -> None:
        base = _live_settings()
        bad = PiCameraSettings(
            enabled=True,
            base_url=base.base_url,
            token="invalid-token-for-live-test",
            timeout_seconds=base.timeout_seconds,
            connect_timeout_seconds=base.connect_timeout_seconds,
            max_body_bytes=base.max_body_bytes,
            stale_frame_ms=base.stale_frame_ms,
            frame_path=base.frame_path,
            frame_method=base.frame_method,
            status_path=base.status_path,
            scan_path=base.scan_path,
            scan_status_path=base.scan_status_path,
            scan_cancel_path=base.scan_cancel_path,
            get_attempts=base.get_attempts,
        )
        async with PiCameraClientV1(bad) as client:
            with pytest.raises(PiCameraAuthError):
                await client.get_status()
