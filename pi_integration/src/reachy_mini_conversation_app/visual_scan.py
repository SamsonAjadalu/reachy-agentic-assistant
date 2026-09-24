"""Bounded, preset-only visual scans serialized by ``MovementManager``."""

from __future__ import annotations
import time
import uuid
import logging
import threading
from typing import Any, Literal
from datetime import datetime, timezone
from collections import deque
from dataclasses import field, dataclass

import numpy as np

from reachy_mini.utils import create_head_pose
from reachy_mini.utils.interpolation import compose_world_offset
from reachy_mini_conversation_app.camera_runtime import CameraFrameError, SharedCameraSampler
from reachy_mini_conversation_app.dance_emotion_moves import GotoQueueMove


logger = logging.getLogger(__name__)

MAX_SCAN_POSES = 5
MAX_SCAN_DURATION_S = 20.0
SCAN_MOVE_DURATION_S = 0.6
SCAN_SETTLE_S = 0.2
SCAN_RATE_LIMIT_PER_HOUR = 6

ScanState = Literal["queued", "running", "restoring", "completed", "cancelling", "cancelled", "failed"]

# Relative head-frame intents. These are disabled by default until attended
# physical validation sets REACHY_VISION_SCANS_ENABLED=1.
_PRESET_OFFSETS: dict[str, tuple[tuple[float, float, float, float, float, float], ...]] = {
    "CLOSE_LOOK": ((0.0, 0.0, 0.0, 0.0, 0.0, 0.0),),
    "PARALLAX_LEFT_RIGHT": (
        (0.0, 0.008, 0.0, 0.0, 0.0, 0.0),
        (0.0, -0.008, 0.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    ),
    "REGION_SWEEP": (
        (0.0, 0.0, 0.0, 0.0, 0.0, -10.0),
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 0.0, 0.0, 10.0),
    ),
    "VERIFY_ABSENCE": (
        (0.0, 0.0, 0.0, 0.0, 0.0, -8.0),
        (0.0, 0.0, 0.0, 0.0, 0.0, 8.0),
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    ),
}


class ScanRejectedError(RuntimeError):
    """Normal safety rejection returned to the workstation adapter."""

    def __init__(self, reason: str, *, retry_after_ms: int | None = None) -> None:
        """Store the stable rejection reason and optional retry hint."""
        super().__init__(reason)
        self.reason = reason
        self.retry_after_ms = retry_after_ms


@dataclass(slots=True)
class ScanRecord:
    """Mutable scan ticket guarded by ``VisualScanManager._lock``."""

    scan_id: str
    correlation_id: str
    preset_id: str
    planned_poses: int
    estimated_duration_ms: int
    state: ScanState = "queued"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: datetime | None = None
    finished_at: datetime | None = None
    current_pose_index: int | None = None
    completed_poses: int = 0
    frame_ids: list[int] = field(default_factory=list)
    cancel_requested: bool = False
    cancel_reason: str | None = None
    restore_state: str = "pending"
    error: str | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event, repr=False)
    thread: threading.Thread | None = field(default=None, repr=False)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe status copy."""
        return {
            "scan_id": self.scan_id,
            "correlation_id": self.correlation_id,
            "preset_id": self.preset_id,
            "state": self.state,
            "planned_poses": self.planned_poses,
            "estimated_duration_ms": self.estimated_duration_ms,
            "current_pose_index": self.current_pose_index,
            "completed_poses": self.completed_poses,
            "frame_ids": list(self.frame_ids),
            "cancel_requested": self.cancel_requested,
            "cancel_reason": self.cancel_reason,
            "restore_state": self.restore_state,
            "error": self.error,
            "created_at": _iso(self.created_at),
            "started_at": _iso(self.started_at),
            "finished_at": _iso(self.finished_at),
        }


class VisualScanManager:
    """Execute one low-priority scan using only the existing movement queue."""

    def __init__(
        self,
        robot: Any,
        movement_manager: Any,
        camera_sampler: SharedCameraSampler,
        *,
        enabled: bool = False,
        rate_limit_per_hour: int = SCAN_RATE_LIMIT_PER_HOUR,
    ) -> None:
        """Configure preset execution around the existing motion manager."""
        self._robot = robot
        self._movement_manager = movement_manager
        self._camera_sampler = camera_sampler
        self._enabled = enabled
        self._rate_limit_per_hour = max(1, min(rate_limit_per_hour, 30))
        self._lock = threading.Lock()
        self._records: dict[str, ScanRecord] = {}
        self._active_scan_id: str | None = None
        self._started_times: deque[float] = deque()

    @property
    def preset_ids(self) -> tuple[str, ...]:
        """Return the immutable network allowlist."""
        return tuple(_PRESET_OFFSETS)

    def start_scan(self, *, preset_id: str, correlation_id: str) -> ScanRecord:
        """Validate and start a preset scan, returning immediately with a ticket."""
        if not self._enabled:
            raise ScanRejectedError("user_disabled")
        offsets = _PRESET_OFFSETS.get(preset_id)
        if offsets is None or not 1 <= len(offsets) <= MAX_SCAN_POSES:
            raise ScanRejectedError("unsafe_pose")
        self._check_motor_state()
        camera_health = self._camera_sampler.status()["camera_health"]
        if camera_health not in {"ok", "busy"}:
            raise ScanRejectedError("unsafe_pose", retry_after_ms=1000)

        with self._lock:
            self._prune_rate_limit_locked()
            if self._active_scan_id is not None:
                raise ScanRejectedError("busy_with_motion", retry_after_ms=1000)
            if len(self._started_times) >= self._rate_limit_per_hour:
                raise ScanRejectedError("rate_limited", retry_after_ms=60_000)
            scan_id = f"scan_{uuid.uuid4().hex}"

        acquired, reason = self._movement_manager.try_acquire_scan_lease(
            scan_id,
            lambda preempt_reason: self.preempt_active_scan(scan_id, preempt_reason),
        )
        if not acquired:
            raise ScanRejectedError(reason or "busy_with_motion", retry_after_ms=1000)

        estimated_ms = int(len(offsets) * (SCAN_MOVE_DURATION_S + SCAN_SETTLE_S) * 1000 + 800)
        record = ScanRecord(
            scan_id=scan_id,
            correlation_id=correlation_id,
            preset_id=preset_id,
            planned_poses=len(offsets),
            estimated_duration_ms=min(estimated_ms, int(MAX_SCAN_DURATION_S * 1000)),
        )
        with self._lock:
            if self._active_scan_id is not None:
                self._movement_manager.release_scan_lease(scan_id)
                raise ScanRejectedError("busy_with_motion", retry_after_ms=1000)
            self._records[scan_id] = record
            self._active_scan_id = scan_id
            self._started_times.append(time.monotonic())
            self._prune_records_locked()
            thread = threading.Thread(target=self._run_scan, args=(record, offsets), daemon=True, name=scan_id)
            record.thread = thread
            thread.start()
        return record

    def status(self, scan_id: str) -> dict[str, Any] | None:
        """Return a safe copy of one scan ticket."""
        with self._lock:
            record = self._records.get(scan_id)
            return record.as_dict() if record is not None else None

    def cancel(self, scan_id: str, reason: str = "workstation_cancelled") -> bool:
        """Idempotently request cancellation of an active or completed scan."""
        with self._lock:
            record = self._records.get(scan_id)
            if record is None:
                return False
            if record.state in {"completed", "cancelled", "failed"}:
                return True
            record.cancel_requested = True
            record.cancel_reason = reason[:64]
            record.state = "cancelling"
            record.cancel_event.set()
        self._movement_manager.cancel_scan_moves(scan_id)
        return True

    def preempt_active_scan(self, scan_id: str, reason: str) -> None:
        """Cancel immediately when voice, tracking, or normal motion takes priority."""
        self.cancel(scan_id, reason=reason)

    def close(self, timeout_s: float = 3.0) -> None:
        """Cancel and boundedly join the active scan during application shutdown."""
        with self._lock:
            scan_id = self._active_scan_id
            record = self._records.get(scan_id) if scan_id else None
        if scan_id is not None:
            self.cancel(scan_id, reason="application_shutdown")
        if record is not None and record.thread is not None and record.thread is not threading.current_thread():
            record.thread.join(timeout=max(0.0, timeout_s))

    def summary(self) -> dict[str, Any]:
        """Return scan availability and active ticket for camera health."""
        with self._lock:
            active = self._records.get(self._active_scan_id) if self._active_scan_id else None
            return {
                "enabled": self._enabled,
                "active_scan_id": self._active_scan_id,
                "state": active.state if active else "idle",
                "preset_ids": list(_PRESET_OFFSETS),
                "max_poses": MAX_SCAN_POSES,
                "max_duration_ms": int(MAX_SCAN_DURATION_S * 1000),
            }

    def _run_scan(
        self,
        record: ScanRecord,
        offsets: tuple[tuple[float, float, float, float, float, float], ...],
    ) -> None:
        deadline = time.monotonic() + MAX_SCAN_DURATION_S
        start_pose: np.ndarray[Any, Any] | None = None
        start_antennas = (0.0, 0.0)
        start_body_yaw = 0.0
        current_pose: np.ndarray[Any, Any] | None = None
        try:
            start_pose = self._validated_current_pose()
            current_pose = start_pose
            movement_status = self._movement_manager.get_status()
            commanded = movement_status.get("last_commanded_pose", {})
            if isinstance(commanded, dict):
                raw_antennas = commanded.get("antennas")
                if isinstance(raw_antennas, list | tuple) and len(raw_antennas) == 2:
                    start_antennas = (float(raw_antennas[0]), float(raw_antennas[1]))
                raw_yaw = commanded.get("body_yaw")
                if isinstance(raw_yaw, int | float):
                    start_body_yaw = float(raw_yaw)

            with self._lock:
                record.state = "running"
                record.started_at = datetime.now(timezone.utc)

            for index, offset_values in enumerate(offsets):
                if self._must_stop(record, deadline):
                    break
                offset = create_head_pose(*offset_values, degrees=True, mm=False)
                target = compose_world_offset(start_pose, offset)
                move = GotoQueueMove(
                    target_head_pose=target,
                    start_head_pose=current_pose,
                    target_antennas=start_antennas,
                    start_antennas=start_antennas,
                    target_body_yaw=start_body_yaw,
                    start_body_yaw=start_body_yaw,
                    duration=SCAN_MOVE_DURATION_S,
                )
                if not self._movement_manager.queue_scan_move(record.scan_id, move):
                    self.cancel(record.scan_id, reason="lease_lost")
                    break
                with self._lock:
                    record.current_pose_index = index
                if not self._wait_interruptibly(record, SCAN_MOVE_DURATION_S + SCAN_SETTLE_S, deadline):
                    break
                settled_ns = time.monotonic_ns()
                try:
                    snapshot = self._camera_sampler.get_snapshot(
                        max_age_ms=500,
                        wait_timeout_s=min(1.0, max(0.0, deadline - time.monotonic())),
                        after_monotonic_ns=settled_ns,
                    )
                except CameraFrameError as exc:
                    raise RuntimeError(exc.code) from exc
                with self._lock:
                    record.completed_poses += 1
                    record.frame_ids.append(snapshot.frame_id)
                current_pose = target

            cancelled = record.cancel_event.is_set() or time.monotonic() >= deadline
            if time.monotonic() >= deadline and not record.cancel_reason:
                record.cancel_reason = "scan_timeout"
                record.cancel_requested = True
            self._restore(record, start_pose, current_pose, start_antennas, start_body_yaw, deadline)
            with self._lock:
                record.state = "cancelled" if cancelled else "completed"
        except Exception as exc:
            logger.warning("Visual scan %s failed safely: %s", record.scan_id, type(exc).__name__)
            if start_pose is not None:
                self._restore(
                    record,
                    start_pose,
                    current_pose if current_pose is not None else start_pose,
                    start_antennas,
                    start_body_yaw,
                    deadline,
                )
            with self._lock:
                record.state = "failed"
                record.error = type(exc).__name__
        finally:
            self._movement_manager.cancel_scan_moves(record.scan_id)
            self._movement_manager.release_scan_lease(record.scan_id)
            with self._lock:
                record.finished_at = datetime.now(timezone.utc)
                record.current_pose_index = None
                if self._active_scan_id == record.scan_id:
                    self._active_scan_id = None

    def _restore(
        self,
        record: ScanRecord,
        start_pose: np.ndarray[Any, Any],
        current_pose: np.ndarray[Any, Any],
        antennas: tuple[float, float],
        body_yaw: float,
        deadline: float,
    ) -> None:
        with self._lock:
            record.state = "restoring"
            record.restore_state = "waiting_for_voice"
        while time.monotonic() < deadline:
            status = self._movement_manager.get_status()
            if not status["is_listening"] and not status["is_speaking"]:
                break
            time.sleep(0.05)
        else:
            with self._lock:
                record.restore_state = "deferred_voice_active"
            return

        restore = GotoQueueMove(
            target_head_pose=start_pose,
            start_head_pose=current_pose,
            target_antennas=antennas,
            start_antennas=antennas,
            target_body_yaw=body_yaw,
            start_body_yaw=body_yaw,
            duration=SCAN_MOVE_DURATION_S,
        )
        if not self._movement_manager.queue_scan_move(record.scan_id, restore):
            with self._lock:
                record.restore_state = "lease_lost"
            return
        with self._lock:
            record.restore_state = "running"
        end = min(deadline, time.monotonic() + SCAN_MOVE_DURATION_S)
        while time.monotonic() < end:
            status = self._movement_manager.get_status()
            if status["is_listening"] or status["is_speaking"]:
                self._movement_manager.cancel_scan_moves(record.scan_id)
                with self._lock:
                    record.restore_state = "preempted_by_voice"
                return
            time.sleep(0.05)
        with self._lock:
            record.restore_state = "completed"

    def _wait_interruptibly(self, record: ScanRecord, duration_s: float, deadline: float) -> bool:
        end = min(deadline, time.monotonic() + duration_s)
        while time.monotonic() < end:
            if record.cancel_event.wait(timeout=min(0.05, max(0.0, end - time.monotonic()))):
                return False
            status = self._movement_manager.get_status()
            if status["is_listening"] or status["is_speaking"] or status["head_tracking"]:
                self.cancel(record.scan_id, reason="voice_started" if status["is_listening"] else "motion_priority")
                return False
        return time.monotonic() < deadline

    @staticmethod
    def _must_stop(record: ScanRecord, deadline: float) -> bool:
        return record.cancel_event.is_set() or time.monotonic() >= deadline

    def _validated_current_pose(self) -> np.ndarray[Any, Any]:
        raw = np.asarray(self._robot.get_current_head_pose(), dtype=np.float64)
        if raw.shape != (4, 4) or not np.isfinite(raw).all():
            raise ScanRejectedError("unsafe_pose")
        return raw.copy()

    def _check_motor_state(self) -> None:
        status = getattr(getattr(self._robot, "client", None), "_last_status", None)
        backend = getattr(status, "backend_status", None)
        error = getattr(backend, "error", None)
        if error:
            raise ScanRejectedError("unsafe_pose")
        mode = getattr(getattr(backend, "motor_control_mode", None), "value", None)
        if mode is not None and mode != "enabled":
            raise ScanRejectedError("motors_disabled")

    def _prune_rate_limit_locked(self) -> None:
        cutoff = time.monotonic() - 3600.0
        while self._started_times and self._started_times[0] < cutoff:
            self._started_times.popleft()

    def _prune_records_locked(self) -> None:
        if len(self._records) <= 32:
            return
        terminal = [key for key, value in self._records.items() if value.state in {"completed", "cancelled", "failed"}]
        for key in terminal[: len(self._records) - 32]:
            self._records.pop(key, None)


def _iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.isoformat().replace("+00:00", "Z")
