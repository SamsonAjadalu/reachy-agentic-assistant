"""Parse proposed Pi frame encodings. Wire shapes are PENDING-PI."""

from __future__ import annotations

import base64
import json
from datetime import datetime
from io import BytesIO
from typing import Any

from integrations.reachy.camera.errors import (
    PiCameraMalformedResponse,
    PiCameraUnsupportedMedia,
)
from integrations.reachy.camera.models import (
    STILL_MIME_TYPES,
    BodyYawSource,
    CameraHealth,
    CameraPose,
    HeadPose4x4,
)
from shared.errors import ValidationError
from shared.timeutils import parse_iso8601


def normalize_mime(value: str | None) -> str:
    if not value:
        return ""
    return value.split(";", 1)[0].strip().lower()


def sniff_still_mime(data: bytes) -> str | None:
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def assert_still_image(data: bytes, declared_mime: str | None) -> str:
    sniffed = sniff_still_mime(data)
    declared = normalize_mime(declared_mime)
    if declared.startswith("video/") or declared == "image/gif":
        raise PiCameraUnsupportedMedia(
            "Pi camera response was not a still image.",
            details={"declared_mime": declared},
        )
    mime = sniffed or declared
    if mime == "image/jpg":
        mime = "image/jpeg"
    if mime not in STILL_MIME_TYPES or sniffed is None:
        raise PiCameraUnsupportedMedia(
            "Pi camera response was not a still image.",
            details={"declared_mime": declared or None, "sniffed_mime": sniffed},
        )
    return mime


def still_dimensions(data: bytes) -> tuple[int, int] | None:
    try:
        from PIL import Image
    except ImportError:
        return None
    try:
        with Image.open(BytesIO(data)) as image:
            width, height = image.size
    except OSError:
        return None
    if width <= 0 or height <= 0:
        return None
    return int(width), int(height)


def parse_json_object(body: bytes, *, correlation_id: str | None) -> dict[str, Any]:
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PiCameraMalformedResponse(
            "Pi camera returned non-JSON.",
            correlation_id=correlation_id,
        ) from exc
    if not isinstance(payload, dict):
        raise PiCameraMalformedResponse(
            "Pi camera JSON was not an object.",
            correlation_id=correlation_id,
        )
    return payload


def unwrap_payload(payload: dict[str, Any]) -> dict[str, Any]:
    inner = payload.get("data")
    if isinstance(inner, dict) and (
        "camera_health" in inner or "image_base64" in inner or "image" in inner
    ):
        return inner
    return payload


def as_int(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.strip():
        try:
            return int(value.strip())
        except ValueError:
            return None
    return None


def as_float(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str) and value.strip():
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def as_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


def as_str(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def parse_capture_utc(value: object) -> datetime | None:
    text = as_str(value)
    if text is None:
        return None
    try:
        return parse_iso8601(text)
    except ValidationError:
        return None


def parse_health(value: object) -> CameraHealth | None:
    text = as_str(value)
    if text is None:
        return None
    try:
        return CameraHealth(text)
    except ValueError:
        return None


def parse_body_yaw_source(value: object) -> BodyYawSource | None:
    text = as_str(value)
    if text is None:
        return None
    try:
        return BodyYawSource(text)
    except ValueError:
        return None


def parse_head_pose_4x4(value: object) -> HeadPose4x4 | None:
    if not isinstance(value, list | tuple) or len(value) != 4:
        return None
    rows: list[tuple[float, float, float, float]] = []
    for row in value:
        if not isinstance(row, list | tuple) or len(row) != 4:
            return None
        try:
            parsed = (float(row[0]), float(row[1]), float(row[2]), float(row[3]))
        except (TypeError, ValueError):
            return None
        rows.append(parsed)
    return rows[0], rows[1], rows[2], rows[3]


def parse_head_joints(value: object) -> tuple[float, ...] | None:
    if not isinstance(value, list | tuple) or not value:
        return None
    joints: list[float] = []
    for item in value:
        try:
            joints.append(float(item))
        except (TypeError, ValueError):
            return None
    return tuple(joints)


def parse_imu(value: object) -> dict[str, float] | None:
    if not isinstance(value, dict) or not value:
        return None
    parsed: dict[str, float] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            continue
        number = as_float(item)
        if number is not None:
            parsed[key] = number
    return parsed or None


def parse_pose(metadata: dict[str, Any]) -> CameraPose | None:
    pose = CameraPose(
        head_pose_4x4=parse_head_pose_4x4(metadata.get("head_pose_4x4")),
        head_joints_rad=parse_head_joints(metadata.get("head_joints_rad")),
        body_yaw_rad=as_float(metadata.get("body_yaw_rad")),
        body_yaw_source=parse_body_yaw_source(metadata.get("body_yaw_source")),
        automatic_body_yaw=as_bool(metadata.get("automatic_body_yaw")),
        head_pose_is_settled=as_bool(metadata.get("head_pose_is_settled")),
        joint_velocity_max_rad_s=as_float(metadata.get("joint_velocity_max_rad_s")),
        pose_capture_skew_ns=as_int(metadata.get("pose_capture_skew_ns")),
        imu=parse_imu(metadata.get("imu")),
    )
    if (
        pose.head_pose_4x4 is None
        and pose.head_joints_rad is None
        and pose.body_yaw_rad is None
        and pose.body_yaw_source is None
        and pose.automatic_body_yaw is None
        and pose.head_pose_is_settled is None
        and pose.joint_velocity_max_rad_s is None
        and pose.pose_capture_skew_ns is None
        and pose.imu is None
    ):
        return None
    return pose


def _header_map(raw: bytes) -> dict[str, str]:
    text = raw.decode("latin-1", errors="replace")
    headers: dict[str, str] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        headers[key.strip().lower()] = value.strip()
    return headers


def _disposition_name(header: str) -> str | None:
    for item in header.split(";"):
        item = item.strip()
        if item.lower().startswith("name="):
            return item.split("=", 1)[1].strip().strip('"')
    return None


def parse_multipart_parts(content_type: str, body: bytes) -> dict[str, tuple[str, bytes]]:
    boundary: str | None = None
    for param in content_type.split(";"):
        item = param.strip()
        if item.lower().startswith("boundary="):
            boundary = item.split("=", 1)[1].strip().strip('"')
            break
    if not boundary:
        raise PiCameraMalformedResponse("multipart response missing boundary.")
    delimiter = b"--" + boundary.encode("ascii", errors="strict")
    parts: dict[str, tuple[str, bytes]] = {}
    for chunk in body.split(delimiter):
        if not chunk or chunk.startswith(b"--"):
            continue
        if chunk.startswith(b"\r\n"):
            chunk = chunk[2:]
        elif chunk.startswith(b"\n"):
            chunk = chunk[1:]
        if b"\r\n\r\n" in chunk:
            header_bytes, _, data = chunk.partition(b"\r\n\r\n")
        elif b"\n\n" in chunk:
            header_bytes, _, data = chunk.partition(b"\n\n")
        else:
            continue
        if data.endswith(b"\r\n"):
            data = data[:-2]
        elif data.endswith(b"\n"):
            data = data[:-1]
        headers = _header_map(header_bytes)
        name = _disposition_name(headers.get("content-disposition", ""))
        if not name:
            continue
        ctype = normalize_mime(headers.get("content-type")) or "application/octet-stream"
        parts[name] = (ctype, data)
    return parts


def extract_frame_payload(
    *,
    content_type: str | None,
    body: bytes,
    headers: dict[str, str],
    correlation_id: str | None,
) -> tuple[bytes, str, dict[str, Any]]:
    """Return (image_bytes, mime, metadata). Proposed encodings, PENDING-PI."""
    mime = normalize_mime(content_type)
    lowered = {key.lower(): value for key, value in headers.items()}
    meta_header = lowered.get("x-reachy-frame-metadata") or lowered.get("x-pi-frame-metadata")

    if mime.startswith("multipart/"):
        parts = parse_multipart_parts(content_type or "", body)
        meta_part = parts.get("metadata")
        image_part = parts.get("image") or parts.get("frame")
        if image_part is None:
            raise PiCameraMalformedResponse(
                "multipart frame response had no image part.",
                correlation_id=correlation_id,
            )
        metadata: dict[str, Any] = {}
        if meta_part is not None:
            metadata = parse_json_object(meta_part[1], correlation_id=correlation_id)
        still_mime = assert_still_image(image_part[1], image_part[0] or metadata.get("image_mime"))
        return image_part[1], still_mime, metadata

    if mime == "application/json":
        payload = unwrap_payload(parse_json_object(body, correlation_id=correlation_id))
        raw_b64 = payload.get("image_base64") or payload.get("image")
        if not isinstance(raw_b64, str) or not raw_b64:
            raise PiCameraMalformedResponse(
                "JSON frame response had no image bytes.",
                correlation_id=correlation_id,
            )
        try:
            image = base64.b64decode(raw_b64, validate=False)
        except (ValueError, TypeError) as exc:
            raise PiCameraMalformedResponse(
                "JSON frame response had invalid image_base64.",
                correlation_id=correlation_id,
            ) from exc
        declared = as_str(payload.get("image_mime")) or as_str(payload.get("mime_type"))
        still_mime = assert_still_image(image, declared)
        return image, still_mime, payload

    if mime in STILL_MIME_TYPES or mime in {"application/octet-stream", ""}:
        metadata = {}
        if meta_header:
            try:
                parsed = json.loads(meta_header)
            except json.JSONDecodeError as exc:
                raise PiCameraMalformedResponse(
                    "Frame metadata header was not JSON.",
                    correlation_id=correlation_id,
                ) from exc
            if not isinstance(parsed, dict):
                raise PiCameraMalformedResponse(
                    "Frame metadata header was not an object.",
                    correlation_id=correlation_id,
                )
            metadata = parsed
        still_mime = assert_still_image(body, mime or metadata.get("image_mime"))
        return body, still_mime, metadata

    raise PiCameraUnsupportedMedia(
        "Pi camera response was not a still image.",
        correlation_id=correlation_id,
        details={"declared_mime": mime or None},
    )
