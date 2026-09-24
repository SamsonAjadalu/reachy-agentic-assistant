"""Authenticated visual-memory routes mounted on the existing LAN listener."""

from __future__ import annotations
import os
import re
import json
import time
import uuid
import asyncio
import hashlib
import secrets
import threading
from typing import Any
from collections import OrderedDict, deque
from dataclasses import dataclass

from fastapi import FastAPI, Request
from pydantic import Field, BaseModel, ConfigDict, ValidationError
from fastapi.responses import Response, JSONResponse

from reachy_mini_conversation_app.visual_scan import ScanRejectedError, VisualScanManager
from reachy_mini_conversation_app.camera_runtime import (
    CONTRACT_VERSION,
    FrameSnapshot,
    CameraFrameError,
    SharedCameraSampler,
)


VISION_TOKEN_ENV = "REACHY_VISION_TOKEN"  # noqa: S105 - environment variable name
CORRELATION_HEADER = "X-Correlation-ID"
REQUEST_ID_HEADER = "X-Request-ID"
FRAME_METADATA_HEADER = "X-Reachy-Frame-Metadata"

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_MAX_REQUEST_BYTES = 4096
_IDEMPOTENCY_TTL_S = 600.0


class FrameRequest(BaseModel):
    """Strict bounded frame request; an empty body uses compatible defaults."""

    model_config = ConfigDict(extra="forbid")

    correlation_id: str | None = Field(default=None, max_length=128)
    request_id: str | None = Field(default=None, max_length=128)
    max_age_ms: int = Field(default=1000, ge=0, le=1000)
    wait_timeout_ms: int = Field(default=1000, ge=0, le=2000)


class ScanRequest(BaseModel):
    """Preset-only request matching the workstation camera client."""

    model_config = ConfigDict(extra="forbid")

    preset_id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z][A-Za-z0-9_]{0,63}$")
    correlation_id: str | None = Field(default=None, max_length=128)
    request_id: str | None = Field(default=None, max_length=128)
    reason: str | None = Field(default=None, max_length=256)


class CancelRequest(BaseModel):
    """Strict scan cancellation body."""

    model_config = ConfigDict(extra="forbid")

    correlation_id: str | None = Field(default=None, max_length=128)
    request_id: str | None = Field(default=None, max_length=128)


@dataclass(frozen=True, slots=True)
class _FrameIdempotencyEntry:
    body_hash: str
    snapshot: FrameSnapshot
    expires_at: float


@dataclass(frozen=True, slots=True)
class _ScanIdempotencyEntry:
    body_hash: str
    scan_id: str
    expires_at: float


class _SlidingWindowLimiter:
    """Small process-local limiter for the authenticated single-client API."""

    def __init__(self) -> None:
        """Initialize independent in-memory request windows."""
        self._lock = threading.Lock()
        self._events: dict[str, deque[float]] = {}

    def allow(self, key: str, *, limit: int, window_s: float) -> bool:
        now = time.monotonic()
        with self._lock:
            events = self._events.setdefault(key, deque())
            cutoff = now - window_s
            while events and events[0] < cutoff:
                events.popleft()
            if len(events) >= limit:
                return False
            events.append(now)
            return True


class VisualMemoryAPI:
    """Mount and own camera/scan API state without creating another server."""

    def __init__(self, sampler: SharedCameraSampler, scan_manager: VisualScanManager, movement_manager: Any) -> None:
        """Keep references to the app-owned camera, scan, and motion state."""
        self._sampler = sampler
        self._scan_manager = scan_manager
        self._movement_manager = movement_manager
        self._limiter = _SlidingWindowLimiter()
        self._idempotency_lock = threading.Lock()
        self._frame_idempotency: OrderedDict[str, _FrameIdempotencyEntry] = OrderedDict()
        self._scan_idempotency: OrderedDict[str, _ScanIdempotencyEntry] = OrderedDict()

    def mount(self, app: FastAPI) -> None:
        """Mount routes on the caller's existing 7861 FastAPI application."""

        @app.get("/api/v1/camera/status")
        async def camera_status(request: Request) -> JSONResponse:
            auth_error = self._authenticate(request)
            request_id, id_error = self._request_id(request, None)
            if auth_error is not None:
                return self._error(auth_error[0], auth_error[1], request_id, retryable=auth_error[2])
            if id_error is not None:
                return self._error(422, "invalid_request", request_id, retryable=False, details=id_error)
            if not self._limiter.allow("camera_status", limit=120, window_s=60.0):
                return self._error(429, "rate_limited", request_id, retryable=True)
            status = self._sampler.status()
            status.update(
                {
                    "ok": True,
                    "contract_version": CONTRACT_VERSION,
                    "correlation_id": request_id,
                    "request_id": request_id,
                    "scan": self._scan_manager.summary(),
                    "motion": self._movement_manager.get_status(),
                }
            )
            return JSONResponse(status, headers=self._response_id_headers(request_id))

        @app.post("/api/v1/camera/frame")
        async def camera_frame(request: Request) -> Response:
            auth_error = self._authenticate(request)
            if auth_error is not None:
                request_id, _ = self._request_id(request, None)
                return self._error(auth_error[0], auth_error[1], request_id, retryable=auth_error[2])
            payload, parse_error = await self._parse_body(request, FrameRequest, allow_empty=True)
            request_id, id_error = self._request_id(request, payload)
            if parse_error is not None or id_error is not None:
                return self._error(
                    422,
                    "invalid_request",
                    request_id,
                    retryable=False,
                    details=parse_error or id_error,
                )
            assert isinstance(payload, FrameRequest)
            if not self._limiter.allow("camera_frame", limit=10, window_s=1.0):
                return self._error(429, "rate_limited", request_id, retryable=True)
            idempotency_key, key_error = self._idempotency_key(request)
            if key_error is not None:
                return self._error(422, "invalid_request", request_id, retryable=False, details=key_error)
            body_hash = self._body_hash(payload)
            cached = self._get_frame_idempotency(idempotency_key, body_hash)
            if cached == "conflict":
                return self._error(409, "idempotency_conflict", request_id, retryable=False)
            try:
                snapshot = (
                    cached
                    if isinstance(cached, FrameSnapshot)
                    else await asyncio.wait_for(
                        asyncio.to_thread(
                            self._sampler.get_snapshot,
                            max_age_ms=payload.max_age_ms,
                            wait_timeout_s=payload.wait_timeout_ms / 1000.0,
                        ),
                        timeout=2.1,
                    )
                )
            except CameraFrameError as exc:
                status_code = 409 if exc.code == "stale_frame" else 503
                return self._error(
                    status_code,
                    exc.code,
                    request_id,
                    retryable=exc.retryable,
                    message=str(exc),
                    details=exc.details,
                )
            except TimeoutError:
                return self._error(504, "request_timeout", request_id, retryable=True)
            if idempotency_key is not None and not isinstance(cached, FrameSnapshot):
                self._store_frame_idempotency(idempotency_key, body_hash, snapshot)
            metadata = snapshot.metadata(camera_health="ok")
            metadata.update({"correlation_id": request_id, "request_id": request_id})
            headers = self._response_id_headers(request_id)
            headers.update(
                {
                    FRAME_METADATA_HEADER: json.dumps(metadata, separators=(",", ":")),
                    "X-Contract-Version": CONTRACT_VERSION,
                    "X-Frame-ID": str(snapshot.frame_id),
                    "X-Frame-Age-Ms": str(snapshot.age_ms()),
                    "X-Camera-Source-ID": snapshot.camera_source_id,
                    "Cache-Control": "no-store",
                }
            )
            return Response(snapshot.image_bytes, media_type="image/jpeg", headers=headers)

        @app.post("/api/v1/scan")
        async def start_scan(request: Request) -> JSONResponse:
            auth_error = self._authenticate(request)
            if auth_error is not None:
                request_id, _ = self._request_id(request, None)
                return self._error(auth_error[0], auth_error[1], request_id, retryable=auth_error[2])
            payload, parse_error = await self._parse_body(request, ScanRequest)
            request_id, id_error = self._request_id(request, payload)
            if parse_error is not None or id_error is not None:
                return self._error(
                    422, "invalid_request", request_id, retryable=False, details=parse_error or id_error
                )
            assert isinstance(payload, ScanRequest)
            if not self._limiter.allow("scan_start", limit=12, window_s=60.0):
                return self._scan_rejection("rate_limited", request_id, retry_after_ms=60_000, status_code=429)
            idempotency_key, key_error = self._idempotency_key(request)
            if key_error is not None:
                return self._error(422, "invalid_request", request_id, retryable=False, details=key_error)
            body_hash = self._body_hash(payload)
            cached = self._get_scan_idempotency(idempotency_key, body_hash)
            if cached == "conflict":
                return self._error(409, "idempotency_conflict", request_id, retryable=False)
            if isinstance(cached, str):
                status = self._scan_manager.status(cached)
                if status is not None:
                    return JSONResponse(
                        self._scan_accepted(status, request_id),
                        status_code=202,
                        headers=self._response_id_headers(request_id),
                    )
            try:
                record = await asyncio.to_thread(
                    self._scan_manager.start_scan,
                    preset_id=payload.preset_id,
                    correlation_id=request_id,
                )
            except ScanRejectedError as exc:
                return self._scan_rejection(exc.reason, request_id, retry_after_ms=exc.retry_after_ms)
            if idempotency_key is not None:
                self._store_scan_idempotency(idempotency_key, body_hash, record.scan_id)
            return JSONResponse(
                self._scan_accepted(record.as_dict(), request_id),
                status_code=202,
                headers=self._response_id_headers(request_id),
            )

        @app.get("/api/v1/scan/{scan_id}")
        async def scan_status(scan_id: str, request: Request) -> JSONResponse:
            auth_error = self._authenticate(request)
            request_id, id_error = self._request_id(request, None)
            if auth_error is not None:
                return self._error(auth_error[0], auth_error[1], request_id, retryable=auth_error[2])
            if id_error is not None or not _ID_RE.fullmatch(scan_id):
                return self._error(422, "invalid_request", request_id, retryable=False)
            status = self._scan_manager.status(scan_id)
            if status is None:
                return self._error(404, "scan_not_found", request_id, retryable=False)
            payload = {"ok": True, "contract_version": CONTRACT_VERSION, "request_id": request_id, **status}
            return JSONResponse(payload, headers=self._response_id_headers(request_id))

        @app.post("/api/v1/scan/{scan_id}/cancel")
        async def cancel_scan(scan_id: str, request: Request) -> JSONResponse:
            auth_error = self._authenticate(request)
            if auth_error is not None:
                request_id, _ = self._request_id(request, None)
                return self._error(auth_error[0], auth_error[1], request_id, retryable=auth_error[2])
            payload, parse_error = await self._parse_body(request, CancelRequest, allow_empty=True)
            request_id, id_error = self._request_id(request, payload)
            if parse_error is not None or id_error is not None or not _ID_RE.fullmatch(scan_id):
                return self._error(
                    422, "invalid_request", request_id, retryable=False, details=parse_error or id_error
                )
            if not self._limiter.allow("scan_cancel", limit=60, window_s=60.0):
                return self._error(429, "rate_limited", request_id, retryable=True)
            cancelled = await asyncio.to_thread(self._scan_manager.cancel, scan_id)
            if not cancelled:
                return self._error(404, "scan_not_found", request_id, retryable=False)
            return JSONResponse(
                {
                    "ok": True,
                    "contract_version": CONTRACT_VERSION,
                    "request_id": request_id,
                    "correlation_id": request_id,
                    "scan_id": scan_id,
                    "cancelled": True,
                },
                status_code=202,
                headers=self._response_id_headers(request_id),
            )

    @staticmethod
    def _authenticate(request: Request) -> tuple[int, str, bool] | None:
        configured = os.getenv(VISION_TOKEN_ENV, "").strip()
        if not configured:
            return 503, "service_not_configured", True
        authorization = request.headers.get("authorization", "")
        scheme, separator, supplied = authorization.partition(" ")
        if (
            scheme.lower() != "bearer"
            or not separator
            or not supplied
            or not secrets.compare_digest(supplied, configured)
        ):
            return 401, "unauthorized", False
        return None

    async def _parse_body(
        self,
        request: Request,
        model: type[BaseModel],
        *,
        allow_empty: bool = False,
    ) -> tuple[BaseModel | None, dict[str, Any] | None]:
        body = await request.body()
        if len(body) > _MAX_REQUEST_BYTES:
            return None, {"max_request_bytes": _MAX_REQUEST_BYTES}
        if not body:
            if not allow_empty:
                return None, {"body": "required"}
            body = b"{}"
        try:
            raw = json.loads(body)
            if not isinstance(raw, dict):
                raise ValueError
            return model.model_validate(raw), None
        except (json.JSONDecodeError, ValueError, ValidationError) as exc:
            if isinstance(exc, ValidationError):
                fields = [".".join(str(item) for item in error["loc"]) for error in exc.errors()[:8]]
                return None, {"fields": fields}
            return None, {"body": "invalid_json_object"}

    @staticmethod
    def _request_id(request: Request, payload: BaseModel | None) -> tuple[str, dict[str, Any] | None]:
        header_correlation = request.headers.get(CORRELATION_HEADER)
        header_request = request.headers.get(REQUEST_ID_HEADER)
        body_correlation = getattr(payload, "correlation_id", None) if payload is not None else None
        body_request = getattr(payload, "request_id", None) if payload is not None else None
        candidates = [
            value.strip() for value in (header_correlation, header_request, body_correlation, body_request) if value
        ]
        request_id = candidates[0] if candidates else uuid.uuid4().hex
        if not _ID_RE.fullmatch(request_id):
            return uuid.uuid4().hex, {"request_id": "invalid"}
        if any(value != request_id for value in candidates[1:]):
            return request_id, {"request_id": "mismatch"}
        return request_id, None

    @staticmethod
    def _idempotency_key(request: Request) -> tuple[str | None, dict[str, Any] | None]:
        raw = request.headers.get("idempotency-key")
        if raw is None:
            return None, None
        key = raw.strip()
        if not _ID_RE.fullmatch(key):
            return None, {"idempotency_key": "invalid"}
        return key, None

    @staticmethod
    def _body_hash(payload: BaseModel) -> str:
        canonical = json.dumps(
            payload.model_dump(exclude={"correlation_id", "request_id"}), sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _get_frame_idempotency(self, key: str | None, body_hash: str) -> FrameSnapshot | str | None:
        if key is None:
            return None
        with self._idempotency_lock:
            self._prune_idempotency_locked()
            entry = self._frame_idempotency.get(key)
            if entry is None:
                return None
            if entry.body_hash != body_hash:
                return "conflict"
            self._frame_idempotency.move_to_end(key)
            return entry.snapshot

    def _store_frame_idempotency(self, key: str, body_hash: str, snapshot: FrameSnapshot) -> None:
        with self._idempotency_lock:
            self._frame_idempotency[key] = _FrameIdempotencyEntry(
                body_hash=body_hash,
                snapshot=snapshot,
                expires_at=time.monotonic() + _IDEMPOTENCY_TTL_S,
            )
            self._frame_idempotency.move_to_end(key)
            while (
                len(self._frame_idempotency) > 8
                or sum(len(entry.snapshot.image_bytes) for entry in self._frame_idempotency.values()) > 4 * 1024 * 1024
            ):
                self._frame_idempotency.popitem(last=False)

    def _get_scan_idempotency(self, key: str | None, body_hash: str) -> str | None:
        if key is None:
            return None
        with self._idempotency_lock:
            self._prune_idempotency_locked()
            entry = self._scan_idempotency.get(key)
            if entry is None:
                return None
            if entry.body_hash != body_hash:
                return "conflict"
            self._scan_idempotency.move_to_end(key)
            return entry.scan_id

    def _store_scan_idempotency(self, key: str, body_hash: str, scan_id: str) -> None:
        with self._idempotency_lock:
            self._scan_idempotency[key] = _ScanIdempotencyEntry(
                body_hash=body_hash,
                scan_id=scan_id,
                expires_at=time.monotonic() + _IDEMPOTENCY_TTL_S,
            )
            self._scan_idempotency.move_to_end(key)
            while len(self._scan_idempotency) > 64:
                self._scan_idempotency.popitem(last=False)

    def _prune_idempotency_locked(self) -> None:
        now = time.monotonic()
        self._frame_idempotency = OrderedDict(
            (key, entry) for key, entry in self._frame_idempotency.items() if entry.expires_at > now
        )
        self._scan_idempotency = OrderedDict(
            (key, entry) for key, entry in self._scan_idempotency.items() if entry.expires_at > now
        )

    @staticmethod
    def _scan_accepted(status: dict[str, Any], request_id: str) -> dict[str, Any]:
        return {
            "ok": True,
            "contract_version": CONTRACT_VERSION,
            "request_id": request_id,
            "correlation_id": request_id,
            "scan_id": status["scan_id"],
            "state": status["state"],
            "planned_poses": status["planned_poses"],
            "estimated_duration_ms": status["estimated_duration_ms"],
            "status_url": f"/api/v1/scan/{status['scan_id']}",
        }

    def _scan_rejection(
        self,
        reason: str,
        request_id: str,
        *,
        retry_after_ms: int | None = None,
        status_code: int = 409,
    ) -> JSONResponse:
        body = {
            "ok": False,
            "contract_version": CONTRACT_VERSION,
            "request_id": request_id,
            "correlation_id": request_id,
            "reason": reason,
            "retry_after_ms": retry_after_ms,
            "error": {"code": reason, "message": reason.replace("_", " "), "retryable": True, "details": {}},
        }
        return JSONResponse(body, status_code=status_code, headers=self._response_id_headers(request_id))

    def _error(
        self,
        status_code: int,
        code: str,
        request_id: str,
        *,
        retryable: bool,
        message: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> JSONResponse:
        body = {
            "ok": False,
            "contract_version": CONTRACT_VERSION,
            "request_id": request_id,
            "correlation_id": request_id,
            "error": {
                "code": code,
                "message": message or code.replace("_", " "),
                "retryable": retryable,
                "details": details or {},
            },
        }
        headers = self._response_id_headers(request_id)
        if status_code == 429:
            headers["Retry-After"] = "1"
        return JSONResponse(body, status_code=status_code, headers=headers)

    @staticmethod
    def _response_id_headers(request_id: str) -> dict[str, str]:
        return {
            CORRELATION_HEADER: request_id,
            REQUEST_ID_HEADER: request_id,
            "X-Contract-Version": CONTRACT_VERSION,
        }
