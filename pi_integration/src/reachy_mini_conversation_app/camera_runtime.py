"""Single-reader camera sampler shared by tools and the LAN vision API."""

from __future__ import annotations
import time
import logging
import threading
from io import BytesIO
from typing import Any, Literal
from datetime import datetime, timezone
from collections import deque
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version

import numpy as np
from PIL import Image


logger = logging.getLogger(__name__)

CONTRACT_VERSION = "1.0.0"
DEFAULT_STALE_AFTER_MS = 500
DEFAULT_MAX_JPEG_BYTES = 512 * 1024
DEFAULT_MAX_WIDTH = 1280
DEFAULT_MAX_HEIGHT = 720


class CameraFrameError(RuntimeError):
    """Base class for structured camera-frame failures."""

    def __init__(self, code: str, message: str, *, retryable: bool, details: dict[str, Any] | None = None) -> None:
        """Store the stable wire code and bounded diagnostic fields."""
        super().__init__(message)
        self.code = code
        self.retryable = retryable
        self.details = details or {}


class CameraUnavailableError(CameraFrameError):
    """Raised when no sampled frame has ever become available."""

    def __init__(self, details: dict[str, Any] | None = None) -> None:
        """Build a retryable unavailable error."""
        super().__init__(
            "camera_unavailable",
            "No frame is available from the camera sampler.",
            retryable=True,
            details=details,
        )


class StaleFrameError(CameraFrameError):
    """Raised when the newest sampled frame is older than the caller permits."""

    def __init__(self, *, age_ms: int, max_age_ms: int) -> None:
        """Build a retryable stale-frame error with measured age."""
        super().__init__(
            "stale_frame",
            "The latest camera frame is stale.",
            retryable=True,
            details={"age_ms": age_ms, "max_age_ms": max_age_ms},
        )


@dataclass(frozen=True, slots=True)
class PoseSnapshot:
    """Best-effort cached robot state sampled immediately after a frame pull."""

    sample_monotonic_ns: int
    pose_capture_skew_ns: int
    head_pose_4x4: tuple[tuple[float, ...], ...] | None
    head_joints_rad: tuple[float, ...] | None
    antenna_joints_rad: tuple[float, ...] | None
    body_yaw_rad: float | None
    body_yaw_source: Literal["commanded", "unknown"]
    head_pose_is_settled: bool | None
    joint_velocity_max_rad_s: float | None
    imu: tuple[tuple[str, Any], ...] | None

    def as_metadata(self) -> dict[str, Any]:
        """Return a JSON-safe copy of the pose metadata."""
        payload: dict[str, Any] = {
            "pose_sample_monotonic_ns": self.sample_monotonic_ns,
            "pose_capture_skew_ns": self.pose_capture_skew_ns,
            "head_pose_4x4": [list(row) for row in self.head_pose_4x4] if self.head_pose_4x4 else None,
            "head_joints_rad": list(self.head_joints_rad) if self.head_joints_rad else None,
            "antenna_joints_rad": list(self.antenna_joints_rad) if self.antenna_joints_rad else None,
            "body_yaw_rad": self.body_yaw_rad,
            "body_yaw_source": self.body_yaw_source,
            "automatic_body_yaw": None,
            "head_pose_is_settled": self.head_pose_is_settled,
            "joint_velocity_max_rad_s": self.joint_velocity_max_rad_s,
        }
        if self.imu is not None:
            payload["imu"] = dict(self.imu)
        return payload


@dataclass(frozen=True, slots=True)
class FrameSnapshot:
    """Immutable encoded frame and its capture-time metadata."""

    image_bytes: bytes
    frame_id: int
    capture_utc: datetime
    capture_monotonic_ns: int
    image_width: int
    image_height: int
    pose: PoseSnapshot | None
    camera_source_id: str
    camera_specs_name: str | None
    sdk_version: str | None
    daemon_version: str | None
    stream_transport: str | None

    def age_ms(self, monotonic_ns: int | None = None) -> int:
        """Return non-negative frame age against the local monotonic clock."""
        now_ns = time.monotonic_ns() if monotonic_ns is None else monotonic_ns
        return max(0, int((now_ns - self.capture_monotonic_ns) / 1_000_000))

    def metadata(self, *, camera_health: str, monotonic_ns: int | None = None) -> dict[str, Any]:
        """Return the versioned metadata expected by the workstation adapter."""
        payload: dict[str, Any] = {
            "contract_version": CONTRACT_VERSION,
            "frame_id": self.frame_id,
            "camera_source_id": self.camera_source_id,
            "capture_utc": self.capture_utc.isoformat().replace("+00:00", "Z"),
            "capture_monotonic_ns": self.capture_monotonic_ns,
            "frame_age_ms": self.age_ms(monotonic_ns),
            "image_width": self.image_width,
            "image_height": self.image_height,
            "image_mime": "image/jpeg",
            "image_bytes": len(self.image_bytes),
            "camera_health": camera_health,
            "camera_specs_name": self.camera_specs_name,
            "sdk_version": self.sdk_version,
            "daemon_version": self.daemon_version,
            "focus_position": None,
            "stream_transport": self.stream_transport,
        }
        if self.pose is not None:
            payload.update(self.pose.as_metadata())
        return payload


class SharedCameraSampler:
    """Own the application's only calls to ``media.camera.read``.

    Recovery intentionally consists only of bounded retries against the current
    SDK media object. If the normal SDK lifecycle replaces ``media.camera``, the
    Uses the configured workflow.
    acquires media itself.
    """

    def __init__(
        self,
        robot: Any,
        movement_manager: Any,
        *,
        enabled: bool = True,
        stale_after_ms: int = DEFAULT_STALE_AFTER_MS,
        max_jpeg_bytes: int = DEFAULT_MAX_JPEG_BYTES,
        max_width: int = DEFAULT_MAX_WIDTH,
        max_height: int = DEFAULT_MAX_HEIGHT,
    ) -> None:
        """Configure the single reader and bounded JPEG cache."""
        self._robot = robot
        self._movement_manager = movement_manager
        self._enabled = enabled
        self._stale_after_ms = stale_after_ms
        self._max_jpeg_bytes = max_jpeg_bytes
        self._max_width = max_width
        self._max_height = max_height
        self._condition = threading.Condition()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._latest: FrameSnapshot | None = None
        self._state = "disabled" if not enabled else "not_started"
        self._last_error: str | None = None
        self._frames_ok = 0
        self._pull_timeouts = 0
        self._errors = 0
        self._started_monotonic_ns: int | None = None
        self._success_times_ns: deque[int] = deque(maxlen=30)
        self._previous_joint_sample: tuple[int, tuple[float, ...]] | None = None

    def start(self) -> None:
        """Start the bounded sampler thread once."""
        with self._condition:
            if not self._enabled:
                self._state = "disabled"
                return
            if self._thread is not None and self._thread.is_alive():
                return
            if self._state == "stopped":
                # Uses the configured workflow.
                # captured by a previous application run.
                self._latest = None
                self._frames_ok = 0
                self._pull_timeouts = 0
                self._errors = 0
                self._last_error = None
                self._success_times_ns.clear()
            self._stop_event.clear()
            self._started_monotonic_ns = time.monotonic_ns()
            self._state = "warming_up"
            self._thread = threading.Thread(target=self._run, name="camera-sampler", daemon=True)
            self._thread.start()

    def stop(self, timeout_s: float = 2.0) -> None:
        """Stop sampling without closing or otherwise changing SDK media."""
        self._stop_event.set()
        with self._condition:
            self._condition.notify_all()
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(0.0, timeout_s))
        with self._condition:
            if self._state != "disabled":
                self._state = "stopped"
            self._thread = None
            self._condition.notify_all()

    def get_snapshot(
        self,
        *,
        max_age_ms: int = 1000,
        wait_timeout_s: float = 1.0,
        after_monotonic_ns: int | None = None,
    ) -> FrameSnapshot:
        """Return a fresh immutable snapshot, waiting only on the cache condition."""
        deadline = time.monotonic() + max(0.0, wait_timeout_s)
        with self._condition:
            while True:
                latest = self._latest
                if latest is not None:
                    age_ms = latest.age_ms()
                    after_ok = after_monotonic_ns is None or latest.capture_monotonic_ns >= after_monotonic_ns
                    if after_ok and age_ms <= max_age_ms:
                        return latest
                remaining = deadline - time.monotonic()
                if remaining <= 0 or self._stop_event.is_set() or not self._enabled:
                    break
                self._condition.wait(timeout=min(remaining, 0.1))

            if latest is None:
                raise CameraUnavailableError(self._error_details_locked())
            raise StaleFrameError(age_ms=latest.age_ms(), max_age_ms=max_age_ms)

    def status(self) -> dict[str, Any]:
        """Return a cheap, side-effect-free sampler health snapshot."""
        with self._condition:
            latest = self._latest
            raw_state = self._state
            age_ms = latest.age_ms() if latest is not None else None
            if raw_state == "warming_up":
                health = "busy"
            elif raw_state in {"disabled", "not_started", "stopped", "unavailable"}:
                health = "unavailable"
            elif raw_state == "decode_error":
                health = "decode_error"
            elif latest is None or (age_ms is not None and age_ms > self._stale_after_ms):
                health = "stalled"
            else:
                health = "ok"
            return {
                "contract_version": CONTRACT_VERSION,
                "camera_health": health,
                "sampler_state": raw_state,
                "stream_active": self._thread is not None and self._thread.is_alive(),
                "frame_id": latest.frame_id if latest else None,
                "image_width": latest.image_width if latest else None,
                "image_height": latest.image_height if latest else None,
                "time_since_last_frame_ms": age_ms,
                "focus_position": None,
                "frames_ok": self._frames_ok,
                "pull_timeouts": self._pull_timeouts,
                "errors": self._errors,
                "observed_fps": self._observed_fps_locked(),
                "stale_after_ms": self._stale_after_ms,
                "last_error": self._last_error,
                "camera_source_id": latest.camera_source_id if latest else self._camera_source_id(),
                "stream_transport": latest.stream_transport if latest else self._stream_transport(),
            }

    def _run(self) -> None:
        missing_backoff_s = 0.1
        consecutive_empty = 0
        while not self._stop_event.is_set():
            camera = getattr(getattr(self._robot, "media", None), "camera", None)
            if camera is None:
                self._record_failure("unavailable", "SDK media camera is not initialized")
                if self._stop_event.wait(missing_backoff_s):
                    break
                missing_backoff_s = min(2.0, missing_backoff_s * 2.0)
                continue

            missing_backoff_s = 0.1
            try:
                frame = camera.read()
            except Exception as exc:
                self._record_failure("decode_error", type(exc).__name__)
                if self._stop_event.wait(min(1.0, 0.05 * (2 ** min(self._errors, 4)))):
                    break
                continue

            if frame is None:
                consecutive_empty += 1
                with self._condition:
                    self._pull_timeouts += 1
                    if self._latest is None and self._warmup_expired_locked():
                        self._state = "unavailable"
                    elif self._latest is not None and self._latest.age_ms() > self._stale_after_ms:
                        self._state = "stalled"
                    self._condition.notify_all()
                delay = min(0.5, 0.02 * (2 ** min(max(0, consecutive_empty - 3), 5)))
                if self._stop_event.wait(delay):
                    break
                continue

            consecutive_empty = 0
            try:
                captured_mono = time.monotonic_ns()
                captured_utc = datetime.now(timezone.utc)
                pose = self._sample_pose(captured_mono)
                image_bytes, width, height = self._encode_jpeg(frame)
                source_id = self._camera_source_id(camera)
                snapshot = FrameSnapshot(
                    image_bytes=image_bytes,
                    frame_id=self._frames_ok + 1,
                    capture_utc=captured_utc,
                    capture_monotonic_ns=captured_mono,
                    image_width=width,
                    image_height=height,
                    pose=pose,
                    camera_source_id=source_id,
                    camera_specs_name=self._camera_specs_name(camera),
                    sdk_version=self._sdk_version(),
                    daemon_version=self._daemon_version(),
                    stream_transport=self._stream_transport(),
                )
            except Exception as exc:
                self._record_failure("decode_error", type(exc).__name__)
                if self._stop_event.wait(0.1):
                    break
                continue

            with self._condition:
                self._latest = snapshot
                self._frames_ok += 1
                self._success_times_ns.append(captured_mono)
                self._state = "healthy"
                self._last_error = None
                self._condition.notify_all()
            self._stop_event.wait(0.02)

    def _sample_pose(self, capture_monotonic_ns: int) -> PoseSnapshot | None:
        try:
            head_pose_raw = self._robot.get_current_head_pose()
            head_joints_raw, antenna_joints_raw = self._robot.get_current_joint_positions()
            sample_ns = time.monotonic_ns()
            head_pose = tuple(tuple(float(value) for value in row) for row in np.asarray(head_pose_raw).tolist())
            head_joints = tuple(float(value) for value in head_joints_raw)
            antenna_joints = tuple(float(value) for value in antenna_joints_raw)
            velocity = self._joint_velocity(sample_ns, head_joints)
            movement_status = self._movement_manager.get_status()
            commanded = movement_status.get("last_commanded_pose", {})
            raw_body_yaw = commanded.get("body_yaw") if isinstance(commanded, dict) else None
            body_yaw = float(raw_body_yaw) if isinstance(raw_body_yaw, int | float) else None
            imu_raw = getattr(self._robot, "imu", None)
            imu = self._freeze_imu(imu_raw)
            return PoseSnapshot(
                sample_monotonic_ns=sample_ns,
                pose_capture_skew_ns=sample_ns - capture_monotonic_ns,
                head_pose_4x4=head_pose,
                head_joints_rad=head_joints,
                antenna_joints_rad=antenna_joints,
                body_yaw_rad=body_yaw,
                body_yaw_source="commanded" if body_yaw is not None else "unknown",
                head_pose_is_settled=None if velocity is None else velocity <= 0.05,
                joint_velocity_max_rad_s=velocity,
                imu=imu,
            )
        except Exception as exc:
            logger.debug("Camera pose sample unavailable: %s", type(exc).__name__)
            return None

    def _joint_velocity(self, sample_ns: int, joints: tuple[float, ...]) -> float | None:
        previous = self._previous_joint_sample
        self._previous_joint_sample = (sample_ns, joints)
        if previous is None or len(previous[1]) != len(joints):
            return None
        elapsed_s = (sample_ns - previous[0]) / 1_000_000_000
        if elapsed_s <= 0:
            return None
        return max(abs(current - prior) / elapsed_s for current, prior in zip(joints, previous[1], strict=True))

    @staticmethod
    def _freeze_imu(value: Any) -> tuple[tuple[str, Any], ...] | None:
        if not isinstance(value, dict):
            return None
        frozen: list[tuple[str, Any]] = []
        for key in ("accelerometer", "gyroscope", "quaternion", "temperature"):
            item = value.get(key)
            if isinstance(item, list | tuple):
                frozen.append((key, tuple(float(component) for component in item)))
            elif isinstance(item, int | float):
                frozen.append((key, float(item)))
        return tuple(frozen) or None

    def _encode_jpeg(self, frame: Any) -> tuple[bytes, int, int]:
        array = np.asarray(frame)
        if array.ndim != 3 or array.shape[2] != 3 or array.shape[0] <= 0 or array.shape[1] <= 0:
            raise ValueError("camera frame must have shape (height, width, 3)")
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        image = Image.fromarray(array[:, :, ::-1].copy())
        image.thumbnail((self._max_width, self._max_height), Image.Resampling.LANCZOS)
        quality = 80
        for _ in range(8):
            buffer = BytesIO()
            image.save(buffer, format="JPEG", quality=quality, optimize=True)
            data = buffer.getvalue()
            if len(data) <= self._max_jpeg_bytes:
                return bytes(data), image.width, image.height
            if quality > 45:
                quality -= 10
                continue
            next_width = max(160, int(image.width * 0.8))
            next_height = max(120, int(image.height * 0.8))
            if (next_width, next_height) == image.size:
                break
            image = image.resize((next_width, next_height), Image.Resampling.LANCZOS)
        raise ValueError("encoded camera frame exceeds the configured byte limit")

    def _record_failure(self, state: str, error: str) -> None:
        with self._condition:
            self._errors += 1
            self._state = state
            self._last_error = error[:120]
            self._condition.notify_all()

    def _warmup_expired_locked(self) -> bool:
        if self._started_monotonic_ns is None:
            return True
        return time.monotonic_ns() - self._started_monotonic_ns > 2_500_000_000

    def _error_details_locked(self) -> dict[str, Any]:
        return {
            "sampler_state": self._state,
            "pull_timeouts": self._pull_timeouts,
            "errors": self._errors,
            "last_error": self._last_error,
        }

    def _observed_fps_locked(self) -> float | None:
        if len(self._success_times_ns) < 2:
            return None
        elapsed_s = (self._success_times_ns[-1] - self._success_times_ns[0]) / 1_000_000_000
        if elapsed_s <= 0:
            return None
        return round((len(self._success_times_ns) - 1) / elapsed_s, 2)

    @staticmethod
    def _camera_specs_name(camera: Any) -> str | None:
        specs = getattr(camera, "camera_specs", None)
        name = getattr(specs, "name", None)
        return str(name) if name else None

    def _camera_source_id(self, camera: Any | None = None) -> str:
        resolved = camera or getattr(getattr(self._robot, "media", None), "camera", None)
        specs_name = self._camera_specs_name(resolved)
        return f"reachy-mini:{specs_name or 'camera'}"

    def _stream_transport(self) -> str | None:
        backend = getattr(getattr(self._robot, "media", None), "backend", None)
        value = getattr(backend, "value", backend)
        return str(value) if value is not None else None

    @staticmethod
    def _sdk_version() -> str | None:
        try:
            return version("reachy-mini")
        except PackageNotFoundError:
            return None

    def _daemon_version(self) -> str | None:
        status = getattr(getattr(self._robot, "client", None), "_last_status", None)
        value = getattr(status, "version", None)
        return str(value) if value else None
