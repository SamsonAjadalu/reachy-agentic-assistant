"""Contract tests for the authenticated Pi visual-memory routes."""

from __future__ import annotations
import json
from types import SimpleNamespace
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from reachy_mini_conversation_app.console import LocalStream
from reachy_mini_conversation_app.camera_runtime import CONTRACT_VERSION, FrameSnapshot
from reachy_mini_conversation_app.visual_memory_api import FRAME_METADATA_HEADER, VisualMemoryAPI


def _snapshot(frame_id: int = 9) -> FrameSnapshot:
    return FrameSnapshot(
        image_bytes=b"\xff\xd8\xfftest-jpeg\xff\xd9",
        frame_id=frame_id,
        capture_utc=datetime.now(timezone.utc),
        capture_monotonic_ns=__import__("time").monotonic_ns(),
        image_width=320,
        image_height=240,
        pose=None,
        camera_source_id="reachy-mini:test",
        camera_specs_name="reachy_mini_wireless",
        sdk_version="1.9.0",
        daemon_version="1.9.0",
        stream_transport="local",
    )


class _Sampler:
    """Deterministic cache-only sampler stub."""

    def __init__(self) -> None:
        self.calls = 0

    def get_snapshot(self, **_kwargs: object) -> FrameSnapshot:
        self.calls += 1
        return _snapshot(self.calls)

    def status(self) -> dict[str, object]:
        return {
            "camera_health": "ok",
            "stream_active": True,
            "image_width": 320,
            "image_height": 240,
            "time_since_last_frame_ms": 3,
            "focus_position": None,
        }


class _ScanManager:
    """Minimal ticket stub used to test the wire contract."""

    def __init__(self) -> None:
        self.cancelled: list[str] = []
        self.ticket = {
            "scan_id": "scan_test",
            "state": "queued",
            "planned_poses": 3,
            "estimated_duration_ms": 3000,
        }

    def summary(self) -> dict[str, object]:
        return {"enabled": True, "state": "idle"}

    def start_scan(self, *, preset_id: str, correlation_id: str) -> SimpleNamespace:
        assert preset_id == "REGION_SWEEP"
        return SimpleNamespace(scan_id="scan_test", as_dict=lambda: dict(self.ticket))

    def status(self, scan_id: str) -> dict[str, object] | None:
        return dict(self.ticket) if scan_id == "scan_test" else None

    def cancel(self, scan_id: str) -> bool:
        self.cancelled.append(scan_id)
        return scan_id == "scan_test"


class _Movement:
    """Read-only motion status stub."""

    def get_status(self) -> dict[str, object]:
        return {"worker_running": True, "is_listening": False, "is_speaking": False}


def _client(monkeypatch, *, sampler: _Sampler | None = None) -> tuple[TestClient, _Sampler, _ScanManager]:
    monkeypatch.setenv("REACHY_VISION_TOKEN", "vision-test-token")
    app = FastAPI()
    resolved_sampler = sampler or _Sampler()
    scans = _ScanManager()
    VisualMemoryAPI(resolved_sampler, scans, _Movement()).mount(app)
    return TestClient(app), resolved_sampler, scans


def _headers(request_id: str = "req-1") -> dict[str, str]:
    return {"Authorization": "Bearer vision-test-token", "X-Correlation-ID": request_id}


def test_status_requires_separate_vision_auth(monkeypatch) -> None:
    """Vision routes reject missing/wrong auth without consulting text-turn auth."""
    client, _, _ = _client(monkeypatch)
    assert client.get("/api/v1/camera/status").status_code == 401
    response = client.get(
        "/api/v1/camera/status",
        headers={"Authorization": "Bearer wrong", "X-Correlation-ID": "req-1"},
    )
    assert response.status_code == 401
    assert response.json()["contract_version"] == CONTRACT_VERSION


def test_status_and_binary_frame_match_workstation_contract(monkeypatch) -> None:
    """The workstation route family receives binary JPEG plus its metadata header."""
    client, _, _ = _client(monkeypatch)
    status = client.get("/api/v1/camera/status", headers=_headers())
    assert status.status_code == 200
    assert status.json()["camera_health"] == "ok"
    assert status.json()["contract_version"] == CONTRACT_VERSION

    response = client.post("/api/v1/camera/frame", headers=_headers("req-frame"))
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/jpeg")
    assert len(response.content) <= 512 * 1024
    metadata = json.loads(response.headers[FRAME_METADATA_HEADER])
    assert metadata["frame_id"] == 1
    assert metadata["camera_source_id"] == "reachy-mini:test"
    assert metadata["contract_version"] == CONTRACT_VERSION
    assert metadata["frame_age_ms"] >= 0


def test_frame_request_validation_and_idempotency(monkeypatch) -> None:
    """Invalid bounds fail and repeated idempotency keys return identical bytes."""
    client, sampler, _ = _client(monkeypatch)
    invalid = client.post(
        "/api/v1/camera/frame",
        headers=_headers("req-invalid"),
        json={"max_age_ms": 5001},
    )
    assert invalid.status_code == 422
    assert invalid.json()["error"]["code"] == "invalid_request"

    headers = {**_headers("req-idem"), "Idempotency-Key": "capture-1"}
    first = client.post("/api/v1/camera/frame", headers=headers)
    second = client.post("/api/v1/camera/frame", headers=headers)
    assert first.content == second.content
    assert sampler.calls == 1


def test_scan_start_status_cancel_and_bounds(monkeypatch) -> None:
    """Only preset identifiers are accepted and cancellation is idempotent."""
    client, _, scans = _client(monkeypatch)
    invalid = client.post(
        "/api/v1/scan",
        headers=_headers("req-scan-bad"),
        json={"preset_id": "REGION_SWEEP", "yaw": 1.0},
    )
    assert invalid.status_code == 422

    accepted = client.post(
        "/api/v1/scan",
        headers={**_headers("req-scan"), "Idempotency-Key": "scan-request-1"},
        json={"preset_id": "REGION_SWEEP", "correlation_id": "req-scan"},
    )
    assert accepted.status_code == 202
    assert accepted.json()["scan_id"] == "scan_test"
    status = client.get("/api/v1/scan/scan_test", headers=_headers("req-status"))
    assert status.status_code == 200
    cancelled = client.post("/api/v1/scan/scan_test/cancel", headers=_headers("req-cancel"))
    assert cancelled.status_code == 202
    assert scans.cancelled == ["scan_test"]


def test_capture_does_not_change_mic_and_text_turn_still_works(monkeypatch) -> None:
    """Camera retrieval preserves mute state and the existing text route behavior."""
    monkeypatch.setenv("REACHY_VISION_TOKEN", "vision-test-token")
    monkeypatch.setenv("REACHY_TEXT_TURN_TOKEN", "text-test-token")
    app = FastAPI()
    handler = MagicMock()
    handler.inject_text_turn = AsyncMock(
        return_value={"ok": True, "turn_id": "turn-1", "assistant_text": "Still connected."}
    )
    robot = SimpleNamespace(media=SimpleNamespace(audio=None, backend=None))
    stream = LocalStream(handler, robot, settings_app=app)
    stream.mount_text_turn_endpoint(app)
    stream._set_mic_muted(True, source="test")
    VisualMemoryAPI(_Sampler(), _ScanManager(), _Movement()).mount(app)
    client = TestClient(app)

    frame = client.post("/api/v1/camera/frame", headers=_headers("req-mic"))
    text = client.post(
        "/api/v1/text-turn",
        headers={"Authorization": "Bearer text-test-token"},
        json={"turn_id": "turn-1", "text": "Regression check"},
    )
    assert frame.status_code == 200
    assert text.status_code == 200
    assert text.json()["assistant_text"] == "Still connected."
    assert stream._mic_muted is True
