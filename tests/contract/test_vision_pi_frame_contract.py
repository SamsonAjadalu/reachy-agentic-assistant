"""Contract tests: domain code talks to PiCameraPort, not URLs."""

from __future__ import annotations

import inspect
from datetime import UTC

import pytest

from integrations.reachy.camera.errors import (
    PiCameraOversizedResponse,
    PiCameraTimeout,
    PiCameraUnsupportedMedia,
)
from integrations.reachy.camera.fake import FakePiCameraState, open_fake_pi_client
from integrations.reachy.camera.models import CameraFrame, CameraHealth, CameraStatus
from integrations.reachy.camera.port import PiCameraPort
from integrations.reachy.camera.settings import PiCameraSettings

TOKEN = "pi-camera-contract-token"


def _settings(**overrides: object) -> PiCameraSettings:
    values: dict[str, object] = {
        "enabled": True,
        "base_url": "http://pi.test",
        "token": TOKEN,
        "timeout_seconds": 1.0,
        "connect_timeout_seconds": 0.4,
        "max_body_bytes": 32 * 1024,
        "stale_frame_ms": 1000,
    }
    values.update(overrides)
    return PiCameraSettings(**values)  # type: ignore[arg-type]


async def capture_observation(port: PiCameraPort, *, correlation_id: str) -> CameraFrame:
    """Domain-shaped helper: latest still through the port."""
    return await port.get_latest_frame(correlation_id=correlation_id)


async def poll_camera(port: PiCameraPort, *, correlation_id: str) -> CameraStatus:
    return await port.health_check(correlation_id=correlation_id)


class TestPiCameraPortContract:
    def test_domain_helpers_do_not_mention_urls(self) -> None:
        for helper in (capture_observation, poll_camera):
            source = inspect.getsource(helper)
            assert "http" not in source.lower()
            assert "/api/" not in source

    async def test_client_satisfies_port(self) -> None:
        state = FakePiCameraState(token=TOKEN)
        async with open_fake_pi_client(state, _settings()) as client:
            assert isinstance(client, PiCameraPort)
            assert client.adapter_version == "v1"

    async def test_ok_frame_and_pose_via_port(self) -> None:
        state = FakePiCameraState(token=TOKEN)
        async with open_fake_pi_client(state, _settings()) as client:
            port: PiCameraPort = client
            frame = await capture_observation(port, correlation_id="obs-1")
        assert frame.camera_health is CameraHealth.OK
        assert frame.pose is not None
        assert frame.pose.body_yaw_source is not None
        assert frame.timestamps.received_utc.tzinfo is UTC or (
            frame.timestamps.received_utc.utcoffset() is not None
        )
        assert frame.timestamps.capture_utc is not None
        assert frame.timestamps.capture_monotonic_ns is not None
        assert frame.image_width == 8
        assert frame.image_height == 8

    async def test_stale_frame_via_port(self) -> None:
        state = FakePiCameraState(token=TOKEN, frame_age_ms=12_000)
        async with open_fake_pi_client(state, _settings()) as client:
            frame = await capture_observation(client, correlation_id="obs-stale")
        assert frame.is_stale is True
        assert frame.image_bytes

    async def test_missing_pose_via_port(self) -> None:
        state = FakePiCameraState(token=TOKEN, include_pose=False)
        async with open_fake_pi_client(state, _settings()) as client:
            frame = await capture_observation(client, correlation_id="obs-nopose")
        assert frame.pose is None
        assert frame.image_bytes

    async def test_health_check_via_port(self) -> None:
        state = FakePiCameraState(token=TOKEN, camera_health="unavailable")
        async with open_fake_pi_client(state, _settings()) as client:
            status = await poll_camera(client, correlation_id="obs-health")
        assert status.camera_health is CameraHealth.UNAVAILABLE
        assert status.stream_active is False

    async def test_timeout_via_port(self) -> None:
        state = FakePiCameraState(token=TOKEN, hang_frame=True, hang_seconds=2.0)
        async with open_fake_pi_client(state, _settings(timeout_seconds=0.15)) as client:
            with pytest.raises(PiCameraTimeout):
                await capture_observation(client, correlation_id="obs-timeout")

    async def test_oversized_via_port(self) -> None:
        state = FakePiCameraState(token=TOKEN, oversized_frame=True)
        async with open_fake_pi_client(state, _settings(max_body_bytes=2048)) as client:
            with pytest.raises(PiCameraOversizedResponse):
                await capture_observation(client, correlation_id="obs-big")

    async def test_malformed_mime_via_port(self) -> None:
        state = FakePiCameraState(token=TOKEN, malformed_mime=True)
        async with open_fake_pi_client(state, _settings()) as client:
            with pytest.raises(PiCameraUnsupportedMedia):
                await capture_observation(client, correlation_id="obs-video")
