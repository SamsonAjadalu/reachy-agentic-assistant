"""In-process fake Reachy Pi camera HTTP surface for tests.

This is a test double for the *draft* contract. It is not evidence of Pi
Uses the configured workflow.
"""

from __future__ import annotations

import asyncio
import base64
import json
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any

import httpx
from PIL import Image
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from integrations.reachy.camera.client import PiCameraClientV1
from integrations.reachy.camera.models import (
    DEFAULT_FRAME_PATH,
    DEFAULT_SCAN_CANCEL_PATH,
    DEFAULT_SCAN_PATH,
    DEFAULT_SCAN_STATUS_PATH,
    DEFAULT_STATUS_PATH,
    FORBIDDEN_SCAN_KEYS,
)
from integrations.reachy.camera.settings import PiCameraSettings

IDENTITY_HEAD_POSE = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)


def sample_still_jpeg(*, width: int = 8, height: int = 8) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (width, height), color=(32, 64, 96)).save(buffer, format="JPEG", quality=80)
    return buffer.getvalue()


@dataclass
class FakePiCameraState:
    token: str
    require_auth: bool = True
    hang_frame: bool = False
    hang_status: bool = False
    hang_seconds: float = 2.0
    oversized_frame: bool = False
    malformed_mime: bool = False
    include_pose: bool = True
    frame_age_ms: int = 15
    camera_health: str = "ok"
    reject_scan_reason: str | None = None
    retry_after_ms: int = 5000
    extra_frame_bytes: int = 0
    encoding: str = "multipart"
    capture_utc: str = "2026-09-02T20:00:00Z"
    capture_monotonic_ns: int = 1_234_567_890_00
    planned_poses: int = 3
    estimated_duration_ms: int = 8000
    last_scan_body: dict[str, Any] | None = None
    cancelled_ids: list[str] = field(default_factory=list)
    accepted_ids: list[str] = field(default_factory=list)
    scan_records: dict[str, dict[str, Any]] = field(default_factory=dict)
    next_frame_id: int = 100
    frame_calls: int = 0
    status_calls: int = 0
    scan_calls: int = 0
    last_correlation_id: str | None = None


def build_fake_pi_app(state: FakePiCameraState) -> Starlette:
    async def _unauthorized() -> JSONResponse:
        return JSONResponse({"error": "unauthorized"}, status_code=401)

    async def _check_auth(request: Request) -> JSONResponse | None:
        state.last_correlation_id = request.headers.get("x-correlation-id")
        if not state.require_auth:
            return None
        header = request.headers.get("authorization", "")
        if header != f"Bearer {state.token}":
            return await _unauthorized()
        return None

    async def camera_frame(request: Request) -> Response:
        denied = await _check_auth(request)
        if denied is not None:
            return denied
        state.frame_calls += 1
        if state.hang_frame:
            await asyncio.sleep(state.hang_seconds)
        if state.malformed_mime:
            return Response(b"\x00\x00\x00\x18ftypmp42", media_type="video/mp4")
        if state.oversized_frame:
            pad = b"\x00" * max(state.extra_frame_bytes, 64 * 1024)
            return Response(pad, media_type="application/octet-stream")

        image = sample_still_jpeg()
        if state.extra_frame_bytes:
            image = image + (b"\x00" * state.extra_frame_bytes)
        metadata = _frame_metadata(state)
        state.next_frame_id += 1
        metadata["frame_id"] = state.next_frame_id
        metadata["camera_source_id"] = "fake-pi-camera"
        headers = {
            "X-Reachy-Frame-Metadata": json.dumps(metadata),
            "X-Frame-ID": str(state.next_frame_id),
            "X-Frame-Age-Ms": str(state.frame_age_ms),
            "X-Camera-Source-ID": "fake-pi-camera",
        }
        if state.encoding == "json":
            body = {**metadata, "image_base64": base64.b64encode(image).decode("ascii")}
            body["image_mime"] = "image/jpeg"
            return JSONResponse(body)
        if state.encoding == "header":
            return Response(image, media_type="image/jpeg", headers=headers)
        return _multipart_frame(metadata, image)

    async def camera_status(request: Request) -> Response:
        denied = await _check_auth(request)
        if denied is not None:
            return denied
        state.status_calls += 1
        if state.hang_status:
            await asyncio.sleep(state.hang_seconds)
        payload: dict[str, Any] = {
            "camera_health": state.camera_health,
            "image_width": 8,
            "image_height": 8,
            "stream_active": False,
            "time_since_last_frame_ms": state.frame_age_ms,
        }
        return JSONResponse(payload)

    async def start_scan(request: Request) -> Response:
        denied = await _check_auth(request)
        if denied is not None:
            return denied
        state.scan_calls += 1
        try:
            body = await request.json()
        except (json.JSONDecodeError, ValueError):
            return JSONResponse({"error": "invalid json"}, status_code=400)
        if not isinstance(body, dict):
            return JSONResponse({"error": "invalid json"}, status_code=400)
        state.last_scan_body = body
        forbidden = FORBIDDEN_SCAN_KEYS.intersection(body)
        if forbidden:
            return JSONResponse(
                {"error": "joint targets are not accepted", "fields": sorted(forbidden)},
                status_code=400,
            )
        if "preset_id" not in body or not isinstance(body.get("preset_id"), str):
            return JSONResponse({"error": "preset_id required"}, status_code=400)
        if state.reject_scan_reason:
            return JSONResponse(
                {
                    "reason": state.reject_scan_reason,
                    "retry_after_ms": state.retry_after_ms,
                },
                status_code=409,
            )
        scan_id = uuid.uuid4().hex
        state.accepted_ids.append(scan_id)
        preset_id = (
            body.get("preset_id") if isinstance(body.get("preset_id"), str) else "CLOSE_LOOK"
        )
        correlation_id = (
            body.get("correlation_id") if isinstance(body.get("correlation_id"), str) else ""
        )
        state.scan_records[scan_id] = {
            "scan_id": scan_id,
            "correlation_id": correlation_id,
            "preset_id": preset_id,
            "state": "queued",
            "planned_poses": state.planned_poses,
            "estimated_duration_ms": state.estimated_duration_ms,
            "current_pose_index": 0,
            "completed_poses": 0,
            "frame_ids": [],
            "cancel_requested": False,
            "cancel_reason": None,
            "error": None,
        }
        return JSONResponse(
            {
                "scan_id": scan_id,
                "planned_poses": state.planned_poses,
                "estimated_duration_ms": state.estimated_duration_ms,
            },
            status_code=202,
        )

    async def scan_status(request: Request) -> Response:
        denied = await _check_auth(request)
        if denied is not None:
            return denied
        scan_id = request.path_params["scan_id"]
        record = state.scan_records.get(scan_id)
        if record is None:
            return JSONResponse({"error": "scan_not_found"}, status_code=404)
        return JSONResponse({"ok": True, **record})

    async def cancel_scan(request: Request) -> Response:
        denied = await _check_auth(request)
        if denied is not None:
            return denied
        scan_id = request.path_params["scan_id"]
        state.cancelled_ids.append(scan_id)
        record = state.scan_records.get(scan_id)
        if record is not None:
            record["cancel_requested"] = True
            record["state"] = "cancelled"
        return JSONResponse({"scan_id": scan_id, "cancelled": True})

    routes = [
        Route(DEFAULT_FRAME_PATH, camera_frame, methods=["POST", "GET"]),
        Route(DEFAULT_STATUS_PATH, camera_status, methods=["GET"]),
        Route(DEFAULT_SCAN_PATH, start_scan, methods=["POST"]),
        Route(DEFAULT_SCAN_STATUS_PATH, scan_status, methods=["GET"]),
        Route(DEFAULT_SCAN_CANCEL_PATH, cancel_scan, methods=["POST"]),
    ]
    app = Starlette(routes=routes)
    app.state.pi = state
    return app


def _frame_metadata(state: FakePiCameraState) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "capture_utc": state.capture_utc,
        "capture_monotonic_ns": state.capture_monotonic_ns,
        "frame_age_ms": state.frame_age_ms,
        "camera_health": state.camera_health,
        "image_width": 8,
        "image_height": 8,
    }
    if state.include_pose:
        metadata.update(
            {
                "head_pose_4x4": [list(row) for row in IDENTITY_HEAD_POSE],
                "head_joints_rad": [0.0] * 7,
                "body_yaw_rad": 0.0,
                "body_yaw_source": "unknown",
                "automatic_body_yaw": False,
                "head_pose_is_settled": True,
                "joint_velocity_max_rad_s": 0.0,
                "pose_capture_skew_ns": 0,
            }
        )
    return metadata


def _multipart_frame(metadata: dict[str, Any], image: bytes) -> Response:
    boundary = "fake-pi-frame-boundary"
    meta_bytes = json.dumps(metadata).encode("utf-8")
    chunks = [
        f"--{boundary}\r\n".encode(),
        b'Content-Disposition: form-data; name="metadata"\r\n',
        b"Content-Type: application/json\r\n\r\n",
        meta_bytes,
        b"\r\n",
        f"--{boundary}\r\n".encode(),
        b'Content-Disposition: form-data; name="image"; filename="frame.jpg"\r\n',
        b"Content-Type: image/jpeg\r\n\r\n",
        image,
        b"\r\n",
        f"--{boundary}--\r\n".encode(),
    ]
    return Response(b"".join(chunks), media_type=f"multipart/form-data; boundary={boundary}")


@asynccontextmanager
async def open_fake_pi_client(
    state: FakePiCameraState,
    settings: PiCameraSettings | None = None,
) -> AsyncIterator[PiCameraClientV1]:
    cfg = settings or PiCameraSettings(
        enabled=True,
        base_url="http://pi.test",
        token=state.token,
        timeout_seconds=1.0,
        connect_timeout_seconds=0.5,
        max_body_bytes=32 * 1024,
        stale_frame_ms=1000,
    )
    app = build_fake_pi_app(state)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url=cfg.base_url,
        timeout=httpx.Timeout(cfg.timeout_seconds, connect=cfg.connect_timeout_seconds),
    ) as http:
        client = PiCameraClientV1(cfg, client=http)
        yield client
