"""Unit tests for bounded Pi scan requests (preset_id only)."""

from __future__ import annotations

import httpx
import pytest
import respx

from integrations.reachy.camera.client import PiCameraClientV1
from integrations.reachy.camera.errors import (
    PiCameraInvalidRequest,
    PiCameraTimeout,
    PiCameraUnavailable,
)
from integrations.reachy.camera.fake import FakePiCameraState, open_fake_pi_client
from integrations.reachy.camera.models import (
    FORBIDDEN_SCAN_KEYS,
    ScanAccepted,
    ScanRejected,
    ScanRejectReason,
)
from integrations.reachy.camera.settings import PiCameraSettings

TOKEN = "pi-camera-scan-test-token"


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


class TestVisionPiScan:
    async def test_accept_returns_scan_id(self) -> None:
        state = FakePiCameraState(token=TOKEN)
        async with open_fake_pi_client(state, _settings()) as client:
            result = await client.request_scan(
                preset_id="CLOSE_LOOK",
                reason="low_margin",
                correlation_id="scan-ok",
            )
        assert isinstance(result, ScanAccepted)
        assert result.scan_id
        assert result.planned_poses == 3
        assert result.estimated_duration_ms == 8000
        assert state.last_scan_body is not None
        assert FORBIDDEN_SCAN_KEYS.isdisjoint(state.last_scan_body)
        assert "joints" not in state.last_scan_body
        assert "pose" not in state.last_scan_body

    async def test_voice_active_is_rejected_not_raised(self) -> None:
        state = FakePiCameraState(token=TOKEN, reject_scan_reason="voice_active")
        async with open_fake_pi_client(state, _settings()) as client:
            result = await client.request_scan(preset_id="PARALLAX_LEFT_RIGHT")
        assert isinstance(result, ScanRejected)
        assert result.known_reason is ScanRejectReason.VOICE_ACTIVE
        assert result.retry_after_ms == 5000

    async def test_busy_motion_is_rejected_not_raised(self) -> None:
        state = FakePiCameraState(
            token=TOKEN,
            reject_scan_reason="busy_with_motion",
            retry_after_ms=2500,
        )
        async with open_fake_pi_client(state, _settings()) as client:
            result = await client.request_scan(preset_id="REGION_SWEEP")
        assert isinstance(result, ScanRejected)
        assert result.known_reason is ScanRejectReason.BUSY_WITH_MOTION
        assert result.retry_after_ms == 2500

    async def test_cancel_is_idempotent(self) -> None:
        state = FakePiCameraState(token=TOKEN)
        async with open_fake_pi_client(state, _settings()) as client:
            accepted = await client.request_scan(preset_id="VERIFY_ABSENCE")
            assert isinstance(accepted, ScanAccepted)
            first = await client.cancel_scan(scan_id=accepted.scan_id, correlation_id="c1")
            second = await client.cancel_scan(scan_id=accepted.scan_id, correlation_id="c2")
        assert first.cancelled is True
        assert second.cancelled is True
        assert state.cancelled_ids == [accepted.scan_id, accepted.scan_id]

    async def test_joint_like_preset_is_rejected_locally(self) -> None:
        state = FakePiCameraState(token=TOKEN)
        async with open_fake_pi_client(state, _settings()) as client:
            with pytest.raises(PiCameraInvalidRequest):
                await client.request_scan(preset_id="0.1,0.2,0.3")
            with pytest.raises(PiCameraInvalidRequest):
                await client.request_scan(preset_id="")
        assert state.scan_calls == 0

    @respx.mock
    async def test_scan_post_is_not_retried(self) -> None:
        route = respx.post("http://pi.test/api/v1/scan").mock(
            return_value=httpx.Response(503, json={"error": "busy"})
        )
        client = PiCameraClientV1(_settings())
        try:
            with pytest.raises(PiCameraUnavailable):
                await client.request_scan(preset_id="BODY_SWEEP")
        finally:
            await client.aclose()
        assert route.call_count == 1

    @respx.mock
    async def test_scan_timeout_mapped(self) -> None:
        respx.post("http://pi.test/api/v1/scan").mock(side_effect=httpx.TimeoutException("slow"))
        client = PiCameraClientV1(_settings())
        try:
            with pytest.raises(PiCameraTimeout) as exc_info:
                await client.request_scan(preset_id="CLOSE_LOOK", correlation_id="scan-to")
        finally:
            await client.aclose()
        assert exc_info.value.correlation_id == "scan-to"
        assert TOKEN not in str(exc_info.value)

    async def test_scan_status_after_accept(self) -> None:
        state = FakePiCameraState(token=TOKEN)
        async with open_fake_pi_client(state, _settings()) as client:
            accepted = await client.request_scan(
                preset_id="CLOSE_LOOK",
                correlation_id="scan-status-1",
            )
            assert isinstance(accepted, ScanAccepted)
            status = await client.get_scan_status(
                scan_id=accepted.scan_id,
                correlation_id="scan-status-2",
            )
        assert status.state == "queued"
        assert status.preset_id == "CLOSE_LOOK"
        assert status.scan_id == accepted.scan_id
