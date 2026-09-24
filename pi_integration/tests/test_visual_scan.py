"""Safety and arbitration tests for bounded visual scans."""

from __future__ import annotations
import time
from types import SimpleNamespace
from datetime import datetime, timezone

import numpy as np
import pytest

import reachy_mini_conversation_app.visual_scan as scan_module
from reachy_mini_conversation_app.visual_scan import ScanRejectedError, VisualScanManager
from reachy_mini_conversation_app.camera_runtime import FrameSnapshot


class _Movement:
    def __init__(self) -> None:
        self.lease: str | None = None
        self.callback = None
        self.moves: list[tuple[str, object]] = []
        self.cancelled: list[str] = []
        self.released: list[str] = []
        self.listening = False
        self.speaking = False
        self.tracking = False

    def get_status(self) -> dict[str, object]:
        return {
            "is_listening": self.listening,
            "is_speaking": self.speaking,
            "head_tracking": self.tracking,
            "last_commanded_pose": {"antennas": [0.0, 0.0], "body_yaw": 0.0},
        }

    def try_acquire_scan_lease(self, scan_id: str, callback) -> tuple[bool, str | None]:
        if self.listening or self.speaking or self.tracking or self.lease is not None:
            return False, "busy_with_motion"
        self.lease = scan_id
        self.callback = callback
        return True, None

    def queue_scan_move(self, scan_id: str, move: object) -> bool:
        if self.lease != scan_id:
            return False
        self.moves.append((scan_id, move))
        return True

    def cancel_scan_moves(self, scan_id: str) -> None:
        self.cancelled.append(scan_id)

    def release_scan_lease(self, scan_id: str) -> None:
        self.released.append(scan_id)
        if self.lease == scan_id:
            self.lease = None

    def preempt(self, reason: str) -> None:
        assert self.callback is not None
        self.callback(reason)


class _Sampler:
    def __init__(self) -> None:
        self.frame_id = 0

    def status(self) -> dict[str, object]:
        return {"camera_health": "ok"}

    def get_snapshot(self, **_kwargs: object) -> FrameSnapshot:
        self.frame_id += 1
        return FrameSnapshot(
            image_bytes=b"\xff\xd8\xff\xd9",
            frame_id=self.frame_id,
            capture_utc=datetime.now(timezone.utc),
            capture_monotonic_ns=time.monotonic_ns(),
            image_width=2,
            image_height=2,
            pose=None,
            camera_source_id="test",
            camera_specs_name=None,
            sdk_version=None,
            daemon_version=None,
            stream_transport="local",
        )


def _manager(monkeypatch, *, enabled: bool = True) -> tuple[VisualScanManager, _Movement]:
    monkeypatch.setattr(scan_module, "SCAN_MOVE_DURATION_S", 0.01)
    monkeypatch.setattr(scan_module, "SCAN_SETTLE_S", 0.005)
    robot = SimpleNamespace(
        get_current_head_pose=lambda: np.eye(4),
        client=SimpleNamespace(
            _last_status=SimpleNamespace(
                backend_status=SimpleNamespace(
                    error=None,
                    motor_control_mode=SimpleNamespace(value="enabled"),
                )
            )
        ),
    )
    movement = _Movement()
    return VisualScanManager(robot, movement, _Sampler(), enabled=enabled), movement


def _wait_terminal(manager: VisualScanManager, scan_id: str) -> dict[str, object]:
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        status = manager.status(scan_id)
        assert status is not None
        if status["state"] in {"completed", "cancelled", "failed"}:
            return status
        time.sleep(0.005)
    raise AssertionError("scan did not terminate within test deadline")


def test_scan_is_bounded_serialized_and_returns_to_start(monkeypatch) -> None:
    """A preset uses one lease and queues a bounded restore through the same owner."""
    manager, movement = _manager(monkeypatch)
    record = manager.start_scan(preset_id="REGION_SWEEP", correlation_id="req-1")
    result = _wait_terminal(manager, record.scan_id)

    assert result["state"] == "completed"
    assert result["planned_poses"] == 3
    assert result["completed_poses"] == 3
    assert len(result["frame_ids"]) == 3
    assert len(movement.moves) == 4  # three observations plus one queued restore
    assert {owner for owner, _ in movement.moves} == {record.scan_id}
    assert movement.released == [record.scan_id]


def test_scan_rejects_disabled_unknown_and_busy_states(monkeypatch) -> None:
    """Disabled, arbitrary, and conversation-busy requests fail before motion."""
    disabled, _ = _manager(monkeypatch, enabled=False)
    with pytest.raises(ScanRejectedError, match="user_disabled"):
        disabled.start_scan(preset_id="CLOSE_LOOK", correlation_id="req")

    manager, movement = _manager(monkeypatch)
    with pytest.raises(ScanRejectedError, match="unsafe_pose"):
        manager.start_scan(preset_id="ARBITRARY_JOINTS", correlation_id="req")
    movement.listening = True
    with pytest.raises(ScanRejectedError, match="busy_with_motion"):
        manager.start_scan(preset_id="CLOSE_LOOK", correlation_id="req")


def test_scan_cancellation_and_voice_preemption_win(monkeypatch) -> None:
    """A listening transition cancels the low-priority scan without changing it."""
    manager, movement = _manager(monkeypatch)
    monkeypatch.setattr(scan_module, "SCAN_MOVE_DURATION_S", 0.1)
    monkeypatch.setattr(scan_module, "SCAN_SETTLE_S", 0.05)
    record = manager.start_scan(preset_id="REGION_SWEEP", correlation_id="req")
    deadline = time.monotonic() + 1.0
    while not movement.moves and time.monotonic() < deadline:
        time.sleep(0.005)
    movement.listening = True
    movement.preempt("listening_started")
    time.sleep(0.05)
    movement.listening = False
    result = _wait_terminal(manager, record.scan_id)

    assert result["state"] == "cancelled"
    assert result["cancel_reason"] == "listening_started"
    assert record.scan_id in movement.cancelled
    assert result["restore_state"] == "completed"
