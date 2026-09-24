"""Unit tests for PiCameraClientV1 against the fake Pi and respx."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
import respx

from integrations.reachy.camera.client import PiCameraClientV1
from integrations.reachy.camera.errors import (
    PiCameraAuthError,
    PiCameraInvalidRequest,
    PiCameraOversizedResponse,
    PiCameraTimeout,
    PiCameraUnavailable,
    PiCameraUnsupportedMedia,
)
from integrations.reachy.camera.fake import FakePiCameraState, open_fake_pi_client
from integrations.reachy.camera.models import (
    CORRELATION_HEADER,
    FORBIDDEN_SCAN_KEYS,
    CameraHealth,
    ScanAccepted,
)
from integrations.reachy.camera.settings import PiCameraSettings

TOKEN = "pi-camera-unit-test-token"


def _settings(**overrides: object) -> PiCameraSettings:
    values: dict[str, object] = {
        "enabled": True,
        "base_url": "http://pi.test",
        "token": TOKEN,
        "timeout_seconds": 1.0,
        "connect_timeout_seconds": 0.4,
        "max_body_bytes": 32 * 1024,
        "stale_frame_ms": 1000,
        "get_attempts": 2,
    }
    values.update(overrides)
    return PiCameraSettings(**values)  # type: ignore[arg-type]


class TestPiCameraClientV1:
    async def test_ok_frame_includes_pose_and_auth(self) -> None:
        state = FakePiCameraState(token=TOKEN)
        async with open_fake_pi_client(state, _settings()) as client:
            frame = await client.get_latest_frame(correlation_id="corr-frame-1")
        assert frame.camera_health is CameraHealth.OK
        assert frame.mime_type == "image/jpeg"
        assert frame.image_bytes[:3] == b"\xff\xd8\xff"
        assert frame.pose is not None
        assert frame.pose.head_pose_4x4 is not None
        assert frame.timestamps.capture_utc is not None
        assert frame.timestamps.capture_monotonic_ns == state.capture_monotonic_ns
        assert frame.timestamps.received_utc.tzinfo is not None
        assert frame.is_stale is False
        assert state.last_correlation_id == "corr-frame-1"

    async def test_stale_frame_is_returned_not_dropped(self) -> None:
        state = FakePiCameraState(token=TOKEN, frame_age_ms=15_000)
        async with open_fake_pi_client(state, _settings()) as client:
            frame = await client.get_latest_frame()
        assert frame.is_stale is True
        assert frame.image_bytes
        assert frame.timestamps.frame_age_ms == 15_000

    async def test_frame_id_from_fake_pi(self) -> None:
        state = FakePiCameraState(token=TOKEN)
        async with open_fake_pi_client(state, _settings()) as client:
            frame = await client.get_latest_frame(correlation_id="frame-id-1")
        assert frame.frame_id is not None
        assert frame.frame_id >= 101
        assert frame.camera_source_id == "fake-pi-camera"

    async def test_stalled_health_marks_stale(self) -> None:
        state = FakePiCameraState(token=TOKEN, camera_health="stalled", frame_age_ms=10)
        async with open_fake_pi_client(state, _settings()) as client:
            frame = await client.get_latest_frame()
        assert frame.camera_health is CameraHealth.STALLED
        assert frame.is_stale is True

    async def test_missing_pose_still_returns_frame(self) -> None:
        state = FakePiCameraState(token=TOKEN, include_pose=False)
        async with open_fake_pi_client(state, _settings()) as client:
            frame = await client.get_latest_frame()
        assert frame.pose is None
        assert frame.image_bytes
        assert frame.mime_type == "image/jpeg"

    async def test_unknown_health_is_left_unset(self) -> None:
        state = FakePiCameraState(token=TOKEN, camera_health="lens-cap-mystery")
        async with open_fake_pi_client(state, _settings()) as client:
            frame = await client.get_latest_frame()
        assert frame.camera_health is None
        assert frame.image_bytes

    async def test_timeout_is_mapped(self) -> None:
        state = FakePiCameraState(token=TOKEN, hang_frame=True, hang_seconds=2.0)
        async with open_fake_pi_client(state, _settings(timeout_seconds=0.15)) as client:
            with pytest.raises(PiCameraTimeout) as exc_info:
                await client.get_latest_frame(correlation_id="corr-timeout")
        assert exc_info.value.correlation_id == "corr-timeout"
        assert TOKEN not in str(exc_info.value)
        assert TOKEN not in json.dumps(exc_info.value.to_dict())

    async def test_oversized_body_is_rejected(self) -> None:
        state = FakePiCameraState(token=TOKEN, oversized_frame=True)
        async with open_fake_pi_client(state, _settings(max_body_bytes=2048)) as client:
            with pytest.raises(PiCameraOversizedResponse):
                await client.get_latest_frame()

    async def test_video_mime_is_rejected(self) -> None:
        state = FakePiCameraState(token=TOKEN, malformed_mime=True)
        async with open_fake_pi_client(state, _settings()) as client:
            with pytest.raises(PiCameraUnsupportedMedia):
                await client.get_latest_frame()

    async def test_unauthorized(self) -> None:
        state = FakePiCameraState(token="other-token")
        async with open_fake_pi_client(state, _settings()) as client:
            with pytest.raises(PiCameraAuthError):
                await client.get_latest_frame()

    async def test_disabled_client(self) -> None:
        client = PiCameraClientV1(_settings(enabled=False))
        try:
            with pytest.raises(PiCameraUnavailable) as exc_info:
                await client.get_latest_frame()
            assert exc_info.value.code == "disabled"
        finally:
            await client.aclose()

    async def test_json_encoding_is_accepted(self) -> None:
        state = FakePiCameraState(token=TOKEN, encoding="json")
        async with open_fake_pi_client(state, _settings()) as client:
            frame = await client.get_latest_frame()
        assert frame.mime_type == "image/jpeg"
        assert frame.pose is not None

    async def test_header_encoding_is_accepted(self) -> None:
        state = FakePiCameraState(token=TOKEN, encoding="header")
        async with open_fake_pi_client(state, _settings()) as client:
            frame = await client.get_latest_frame()
        assert frame.timestamps.capture_monotonic_ns == state.capture_monotonic_ns

    async def test_in_flight_request_can_be_cancelled(self) -> None:
        state = FakePiCameraState(token=TOKEN, hang_frame=True, hang_seconds=2.0)
        async with open_fake_pi_client(state, _settings(timeout_seconds=2.0)) as client:
            task = asyncio.create_task(client.get_latest_frame())
            await asyncio.sleep(0.05)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    async def test_health_check_uses_status(self) -> None:
        state = FakePiCameraState(token=TOKEN, camera_health="busy")
        async with open_fake_pi_client(state, _settings()) as client:
            status = await client.health_check(correlation_id="corr-health")
        assert status.camera_health is CameraHealth.BUSY
        assert state.status_calls == 1
        assert state.last_correlation_id == "corr-health"

    def test_from_env_reads_placeholders(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("REACHY_CAMERA_ENABLED", "true")
        monkeypatch.setenv("REACHY_CAMERA_URL", "http://example.invalid:7861")
        monkeypatch.setenv("REACHY_CAMERA_TOKEN", "env-camera-token-value")
        monkeypatch.setenv("REACHY_CAMERA_FRAME_PATH", "/custom/frame")
        monkeypatch.delenv("REACHY_TEXT_TURN_TOKEN", raising=False)
        settings = PiCameraSettings.from_env()
        assert settings.enabled is True
        assert settings.base_url == "http://example.invalid:7861"
        assert settings.token == "env-camera-token-value"
        assert settings.frame_path == "/custom/frame"


class TestRetries:
    @respx.mock
    async def test_get_status_retries_once_on_503(self) -> None:
        route = respx.get("http://pi.test/api/v1/camera/status").mock(
            side_effect=[
                httpx.Response(503, json={"error": "busy"}),
                httpx.Response(200, json={"camera_health": "ok", "stream_active": False}),
            ]
        )
        client = PiCameraClientV1(_settings())
        try:
            status = await client.get_status(correlation_id="corr-retry")
        finally:
            await client.aclose()
        assert status.camera_health is CameraHealth.OK
        assert route.call_count == 2
        assert route.calls.last.request.headers[CORRELATION_HEADER] == "corr-retry"
        assert route.calls.last.request.headers["authorization"] == f"Bearer {TOKEN}"

    @respx.mock
    async def test_frame_post_does_not_retry(self) -> None:
        route = respx.post("http://pi.test/api/v1/camera/frame").mock(
            return_value=httpx.Response(503, json={"error": "busy"})
        )
        client = PiCameraClientV1(_settings())
        try:
            with pytest.raises(PiCameraUnavailable):
                await client.get_latest_frame()
        finally:
            await client.aclose()
        assert route.call_count == 1

    @respx.mock
    async def test_connect_error_is_unavailable(self) -> None:
        respx.post("http://pi.test/api/v1/camera/frame").mock(
            side_effect=httpx.ConnectError("connection refused")
        )
        client = PiCameraClientV1(_settings())
        try:
            with pytest.raises(PiCameraUnavailable):
                await client.get_latest_frame()
        finally:
            await client.aclose()


class TestScanPayloadSafety:
    async def test_invalid_preset_never_hits_the_wire(self) -> None:
        state = FakePiCameraState(token=TOKEN)
        async with open_fake_pi_client(state, _settings()) as client:
            with pytest.raises(PiCameraInvalidRequest):
                await client.request_scan(preset_id='{"joints": [0.1]}')
        assert state.scan_calls == 0

    async def test_scan_body_has_no_joint_keys(self) -> None:
        state = FakePiCameraState(token=TOKEN)
        async with open_fake_pi_client(state, _settings()) as client:
            result = await client.request_scan(
                preset_id="PARALLAX_LEFT_RIGHT",
                reason="occluded",
                correlation_id="corr-scan",
            )
        assert isinstance(result, ScanAccepted)
        assert state.last_scan_body is not None
        assert FORBIDDEN_SCAN_KEYS.isdisjoint(state.last_scan_body)
        assert set(state.last_scan_body) <= {"preset_id", "correlation_id", "reason"}
        assert state.last_scan_body["preset_id"] == "PARALLAX_LEFT_RIGHT"
