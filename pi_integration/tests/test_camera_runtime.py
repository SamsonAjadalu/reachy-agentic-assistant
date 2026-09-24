"""Tests for the single-reader camera sampler."""

from __future__ import annotations
import time
from types import SimpleNamespace
from dataclasses import replace

import numpy as np
import pytest

from reachy_mini_conversation_app.camera_runtime import (
    CONTRACT_VERSION,
    StaleFrameError,
    SharedCameraSampler,
    CameraUnavailableError,
)


class _ScriptedCamera:
    def __init__(self, frames: list[np.ndarray | None]) -> None:
        self._frames = list(frames)
        self.calls = 0
        self.camera_specs = SimpleNamespace(name="reachy_mini_wireless")

    def read(self) -> np.ndarray | None:
        self.calls += 1
        if self._frames:
            return self._frames.pop(0)
        return None


class _Movement:
    def get_status(self) -> dict[str, object]:
        return {"last_commanded_pose": {"body_yaw": 0.25}}


class _Robot:
    def __init__(self, camera: _ScriptedCamera | None) -> None:
        self.media = SimpleNamespace(camera=camera, backend=SimpleNamespace(value="local"))
        self.client = SimpleNamespace(_last_status=SimpleNamespace(version="1.9.0"))
        self.imu = {
            "accelerometer": [0.0, 0.0, 9.8],
            "gyroscope": [0.0, 0.0, 0.0],
            "quaternion": [1.0, 0.0, 0.0, 0.0],
            "temperature": 25.0,
        }

    def get_current_head_pose(self) -> np.ndarray:
        return np.eye(4)

    def get_current_joint_positions(self) -> tuple[list[float], list[float]]:
        return [0.0] * 7, [0.0, 0.0]


def _frame(width: int = 320, height: int = 240) -> np.ndarray:
    return np.full((height, width, 3), (10, 80, 200), dtype=np.uint8)


def test_first_frame_warmup_and_metadata() -> None:
    """Warmup tolerates empty pulls and publishes measured metadata."""
    camera = _ScriptedCamera([None, None, _frame()])
    sampler = SharedCameraSampler(_Robot(camera), _Movement())
    sampler.start()
    try:
        assert sampler.status()["sampler_state"] in {"warming_up", "healthy"}
        snapshot = sampler.get_snapshot(max_age_ms=1000, wait_timeout_s=1.0)
        metadata = snapshot.metadata(camera_health="ok")
        assert snapshot.frame_id == 1
        assert snapshot.image_bytes.startswith(b"\xff\xd8\xff")
        assert (snapshot.image_width, snapshot.image_height) == (320, 240)
        assert metadata["contract_version"] == CONTRACT_VERSION
        assert metadata["camera_source_id"] == "reachy-mini:reachy_mini_wireless"
        assert metadata["body_yaw_source"] == "commanded"
        assert metadata["body_yaw_rad"] == 0.25
        assert camera.calls >= 3
    finally:
        sampler.stop()


def test_absent_frame_is_structured_unavailable() -> None:
    """A missing SDK camera returns a bounded unavailable failure."""
    sampler = SharedCameraSampler(_Robot(None), _Movement())
    sampler.start()
    try:
        with pytest.raises(CameraUnavailableError) as raised:
            sampler.get_snapshot(max_age_ms=100, wait_timeout_s=0.05)
        assert raised.value.code == "camera_unavailable"
        assert raised.value.retryable is True
    finally:
        sampler.stop()


def test_stale_frame_is_rejected() -> None:
    """The sampler serves frames within the caller's age bound."""
    sampler = SharedCameraSampler(_Robot(_ScriptedCamera([_frame()])), _Movement())
    sampler.start()
    try:
        snapshot = sampler.get_snapshot(wait_timeout_s=1.0)
        with sampler._condition:
            sampler._latest = replace(snapshot, capture_monotonic_ns=time.monotonic_ns() - 2_000_000_000)
        with pytest.raises(StaleFrameError) as raised:
            sampler.get_snapshot(max_age_ms=100, wait_timeout_s=0.0)
        assert raised.value.details["age_ms"] >= 1900
    finally:
        sampler.stop()


def test_sampler_recovers_by_retrying_existing_camera() -> None:
    """Recovery uses retry/backoff with the existing camera handle."""
    camera = _ScriptedCamera([None, None, None, None, _frame()])
    sampler = SharedCameraSampler(_Robot(camera), _Movement())
    sampler.start()
    try:
        snapshot = sampler.get_snapshot(wait_timeout_s=2.0)
        status = sampler.status()
        assert snapshot.frame_id == 1
        assert status["camera_health"] == "ok"
        assert status["pull_timeouts"] >= 4
        assert not hasattr(camera, "open")
        assert not hasattr(camera, "close")
    finally:
        sampler.stop()


def test_encoded_frame_obeys_resolution_and_size_bounds() -> None:
    """Only a bounded JPEG is retained, even for a large noisy source frame."""
    random_frame = np.random.default_rng(42).integers(0, 256, size=(1080, 1920, 3), dtype=np.uint8)
    sampler = SharedCameraSampler(_Robot(_ScriptedCamera([random_frame])), _Movement())
    sampler.start()
    try:
        snapshot = sampler.get_snapshot(wait_timeout_s=3.0)
        assert snapshot.image_width <= 1280
        assert snapshot.image_height <= 720
        assert len(snapshot.image_bytes) <= 512 * 1024
    finally:
        sampler.stop()


def test_restart_requires_new_frame_and_does_not_reuse_old_cache() -> None:
    """Stopping and starting the sampler clears the previous run's still."""
    camera = _ScriptedCamera([_frame(), None, _frame()])
    sampler = SharedCameraSampler(_Robot(camera), _Movement())
    sampler.start()
    first = sampler.get_snapshot(wait_timeout_s=1.0)
    sampler.stop()
    sampler.start()
    try:
        with pytest.raises(CameraUnavailableError):
            sampler.get_snapshot(wait_timeout_s=0.01)
        second = sampler.get_snapshot(wait_timeout_s=1.0)
        assert second.frame_id == 1
        assert second.capture_monotonic_ns > first.capture_monotonic_ns
    finally:
        sampler.stop()
