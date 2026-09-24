"""Environment-backed settings for the Pi camera adapter.

These names are not yet on ``app.config.Settings`` (owned by another module).
``PiCameraSettings.from_env`` reads ``os.environ`` so this adapter can ship
without touching ``app/config.py``. A later change should promote the same
names onto Settings.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from integrations.reachy.camera.models import (
    DEFAULT_FRAME_PATH,
    DEFAULT_SCAN_CANCEL_PATH,
    DEFAULT_SCAN_PATH,
    DEFAULT_SCAN_STATUS_PATH,
    DEFAULT_STATUS_PATH,
)

# Names another agent should add to app.config.Settings / .env.
ENV_ENABLED = "REACHY_CAMERA_ENABLED"
ENV_URL = "REACHY_CAMERA_URL"
ENV_VISION_TOKEN = "REACHY_VISION_TOKEN"  # noqa: S105  # Pi vision bridge (preferred)
ENV_TOKEN = "REACHY_CAMERA_TOKEN"  # noqa: S105  # legacy alias on workstation
ENV_TOKEN_FALLBACK = "REACHY_TEXT_TURN_TOKEN"  # noqa: S105
ENV_TIMEOUT_SECONDS = "REACHY_CAMERA_TIMEOUT_SECONDS"
ENV_CONNECT_TIMEOUT_SECONDS = "REACHY_CAMERA_CONNECT_TIMEOUT_SECONDS"
ENV_MAX_BODY_BYTES = "REACHY_CAMERA_MAX_BODY_BYTES"
ENV_STALE_FRAME_MS = "REACHY_CAMERA_STALE_FRAME_MS"
ENV_FRAME_PATH = "REACHY_CAMERA_FRAME_PATH"
ENV_FRAME_METHOD = "REACHY_CAMERA_FRAME_METHOD"
ENV_STATUS_PATH = "REACHY_CAMERA_STATUS_PATH"
ENV_SCAN_PATH = "REACHY_CAMERA_SCAN_PATH"
ENV_SCAN_STATUS_PATH = "REACHY_CAMERA_SCAN_STATUS_PATH"
ENV_SCAN_CANCEL_PATH = "REACHY_CAMERA_SCAN_CANCEL_PATH"
ENV_GET_ATTEMPTS = "REACHY_CAMERA_GET_ATTEMPTS"

DEFAULT_BASE_URL = "http://reachy-mini.local:7861"
DEFAULT_TIMEOUT_SECONDS = 10.0
DEFAULT_CONNECT_TIMEOUT_SECONDS = 5.0
DEFAULT_MAX_BODY_BYTES = 5 * 1024 * 1024
DEFAULT_STALE_FRAME_MS = 1000
DEFAULT_GET_ATTEMPTS = 2
DEFAULT_FRAME_METHOD = "POST"


def _env(name: str) -> str | None:
    raw = os.environ.get(name)
    if raw is None:
        return None
    stripped = raw.strip()
    return stripped if stripped else None


def _env_bool(name: str, default: bool) -> bool:
    raw = _env(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = _env(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(frozen=True, slots=True)
class PiCameraSettings:
    enabled: bool = False
    base_url: str = DEFAULT_BASE_URL
    token: str = ""
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    connect_timeout_seconds: float = DEFAULT_CONNECT_TIMEOUT_SECONDS
    max_body_bytes: int = DEFAULT_MAX_BODY_BYTES
    stale_frame_ms: int = DEFAULT_STALE_FRAME_MS
    frame_path: str = DEFAULT_FRAME_PATH
    frame_method: str = DEFAULT_FRAME_METHOD
    status_path: str = DEFAULT_STATUS_PATH
    scan_path: str = DEFAULT_SCAN_PATH
    scan_status_path: str = DEFAULT_SCAN_STATUS_PATH
    scan_cancel_path: str = DEFAULT_SCAN_CANCEL_PATH
    get_attempts: int = DEFAULT_GET_ATTEMPTS

    @classmethod
    def from_env(cls) -> PiCameraSettings:
        token = _env(ENV_VISION_TOKEN) or _env(ENV_TOKEN) or _env(ENV_TOKEN_FALLBACK) or ""
        method = (_env(ENV_FRAME_METHOD) or DEFAULT_FRAME_METHOD).upper()
        if method not in {"GET", "POST"}:
            method = DEFAULT_FRAME_METHOD
        timeout = max(0.05, _env_float(ENV_TIMEOUT_SECONDS, DEFAULT_TIMEOUT_SECONDS))
        connect = max(
            0.05, _env_float(ENV_CONNECT_TIMEOUT_SECONDS, DEFAULT_CONNECT_TIMEOUT_SECONDS)
        )
        attempts = max(1, _env_int(ENV_GET_ATTEMPTS, DEFAULT_GET_ATTEMPTS))
        max_body = max(1024, _env_int(ENV_MAX_BODY_BYTES, DEFAULT_MAX_BODY_BYTES))
        stale_ms = max(0, _env_int(ENV_STALE_FRAME_MS, DEFAULT_STALE_FRAME_MS))
        return cls(
            enabled=_env_bool(ENV_ENABLED, False),
            base_url=(_env(ENV_URL) or DEFAULT_BASE_URL).rstrip("/"),
            token=token,
            timeout_seconds=timeout,
            connect_timeout_seconds=min(connect, timeout),
            max_body_bytes=max_body,
            stale_frame_ms=stale_ms,
            frame_path=_env(ENV_FRAME_PATH) or DEFAULT_FRAME_PATH,
            frame_method=method,
            status_path=_env(ENV_STATUS_PATH) or DEFAULT_STATUS_PATH,
            scan_path=_env(ENV_SCAN_PATH) or DEFAULT_SCAN_PATH,
            scan_status_path=_env(ENV_SCAN_STATUS_PATH) or DEFAULT_SCAN_STATUS_PATH,
            scan_cancel_path=_env(ENV_SCAN_CANCEL_PATH) or DEFAULT_SCAN_CANCEL_PATH,
            get_attempts=attempts,
        )
