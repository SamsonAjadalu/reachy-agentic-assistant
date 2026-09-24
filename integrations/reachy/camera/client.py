"""Production async HTTP client for the draft Pi camera API (adapter v1)."""

from __future__ import annotations

import asyncio
import re
import time
import uuid
from typing import Any
from urllib.parse import quote

import httpx

from app.logging_config import correlation_id_var, get_logger
from integrations.reachy.camera.errors import (
    PiCameraAuthError,
    PiCameraError,
    PiCameraInvalidRequest,
    PiCameraMalformedResponse,
    PiCameraOversizedResponse,
    PiCameraStaleFrame,
    PiCameraTimeout,
    PiCameraUnavailable,
    PiCameraUnsupportedMedia,
)
from integrations.reachy.camera.models import (
    ADAPTER_VERSION,
    CORRELATION_HEADER,
    FORBIDDEN_SCAN_KEYS,
    CameraFrame,
    CameraHealth,
    CameraStatus,
    ObservationTimestamps,
    ScanAccepted,
    ScanCancelResult,
    ScanOutcome,
    ScanRejected,
    ScanStatus,
)
from integrations.reachy.camera.parse import (
    as_bool,
    as_float,
    as_int,
    as_str,
    extract_frame_payload,
    parse_capture_utc,
    parse_health,
    parse_json_object,
    parse_pose,
    still_dimensions,
    unwrap_payload,
)
from integrations.reachy.camera.settings import PiCameraSettings
from security.redaction import register_secret
from shared.timeutils import utcnow

logger = get_logger(__name__)

_PRESET_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")
_SCAN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_MAX_REASON_CHARS = 256


class PiCameraClientV1:
    """Authenticated still-frame and bounded-scan client. Implements ``PiCameraPort``."""

    adapter_version = ADAPTER_VERSION

    def __init__(
        self,
        settings: PiCameraSettings | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._settings = settings or PiCameraSettings.from_env()
        self._token = self._settings.token
        if self._token:
            register_secret(self._token)
        timeout = httpx.Timeout(
            self._settings.timeout_seconds,
            connect=min(self._settings.connect_timeout_seconds, self._settings.timeout_seconds),
        )
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=timeout)

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> PiCameraClientV1:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.aclose()

    async def get_latest_frame(self, *, correlation_id: str | None = None) -> CameraFrame:
        cid = self._require_ready(correlation_id)
        received_mono = time.monotonic_ns()
        received_utc = utcnow()
        frame_body: dict[str, Any] | None = None
        if self._settings.frame_method == "POST":
            max_age = min(1000, max(0, self._settings.stale_frame_ms))
            frame_body = {
                "correlation_id": cid,
                "max_age_ms": max_age,
                "wait_timeout_ms": min(2000, max(0, int(self._settings.timeout_seconds * 1000))),
            }
        status_code, headers, body = await self._request(
            self._settings.frame_method,
            self._settings.frame_path,
            correlation_id=cid,
            json_body=frame_body,
            retry=False,
        )
        self._raise_http_error(status_code, body, headers, correlation_id=cid)
        metadata = _merge_frame_headers(headers, {})
        try:
            image, mime, parsed_meta = extract_frame_payload(
                content_type=headers.get("content-type"),
                body=body,
                headers=headers,
                correlation_id=cid,
            )
            metadata = _merge_frame_headers(headers, parsed_meta)
        except (PiCameraUnsupportedMedia, PiCameraMalformedResponse) as exc:
            if exc.correlation_id is None:
                exc.correlation_id = cid
            raise

        timestamps = ObservationTimestamps(
            received_utc=received_utc,
            received_monotonic_ns=received_mono,
            capture_utc=parse_capture_utc(metadata.get("capture_utc")),
            capture_monotonic_ns=as_int(metadata.get("capture_monotonic_ns")),
            frame_age_ms=as_int(metadata.get("frame_age_ms")),
        )
        health = parse_health(metadata.get("camera_health"))
        is_stale = _frame_is_stale(
            frame_age_ms=timestamps.frame_age_ms,
            health=health,
            stale_frame_ms=self._settings.stale_frame_ms,
        )
        width = as_int(metadata.get("image_width"))
        height = as_int(metadata.get("image_height"))
        if width is None or height is None:
            dims = still_dimensions(image)
            if dims is not None:
                width = width if width is not None else dims[0]
                height = height if height is not None else dims[1]

        frame = CameraFrame(
            image_bytes=image,
            mime_type=mime,
            timestamps=timestamps,
            correlation_id=cid,
            frame_id=as_int(metadata.get("frame_id")),
            camera_source_id=as_str(metadata.get("camera_source_id")),
            camera_health=health,
            is_stale=is_stale,
            image_width=width,
            image_height=height,
            pose=parse_pose(metadata),
            camera_specs_name=as_str(metadata.get("camera_specs_name")),
            sdk_version=as_str(metadata.get("sdk_version")),
            daemon_version=as_str(metadata.get("daemon_version")),
            focus_position=as_float(metadata.get("focus_position")),
            stream_transport=as_str(metadata.get("stream_transport")),
        )
        if is_stale:
            logger.warning(
                "Pi camera returned a stale frame",
                extra={
                    "correlation_id": cid,
                    "frame_age_ms": timestamps.frame_age_ms,
                    "camera_health": health.value if health else None,
                    "stale_frame_ms": self._settings.stale_frame_ms,
                },
            )
        return frame

    async def get_status(self, *, correlation_id: str | None = None) -> CameraStatus:
        cid = self._require_ready(correlation_id)
        status_code, headers, body = await self._request(
            "GET",
            self._settings.status_path,
            correlation_id=cid,
            retry=True,
        )
        self._raise_http_error(status_code, body, headers, correlation_id=cid)
        payload = unwrap_payload(parse_json_object(body, correlation_id=cid))
        return CameraStatus(
            correlation_id=cid,
            camera_health=parse_health(payload.get("camera_health")),
            image_width=as_int(payload.get("image_width")),
            image_height=as_int(payload.get("image_height")),
            focus_position=as_float(payload.get("focus_position")),
            stream_active=as_bool(payload.get("stream_active")),
            time_since_last_frame_ms=as_int(payload.get("time_since_last_frame_ms")),
        )

    async def health_check(self, *, correlation_id: str | None = None) -> CameraStatus:
        return await self.get_status(correlation_id=correlation_id)

    async def request_scan(
        self,
        *,
        preset_id: str,
        reason: str | None = None,
        correlation_id: str | None = None,
    ) -> ScanOutcome:
        cid = self._require_ready(correlation_id)
        if not _PRESET_ID_RE.fullmatch(preset_id):
            raise PiCameraInvalidRequest(
                "preset_id must be an identifier; joint targets are not allowed.",
                code="invalid_preset_id",
                correlation_id=cid,
            )
        payload: dict[str, Any] = {"preset_id": preset_id, "correlation_id": cid}
        if reason is not None and reason.strip():
            payload["reason"] = reason.strip()[:_MAX_REASON_CHARS]
        _assert_scan_payload_safe(payload)

        status_code, headers, body = await self._request(
            "POST",
            self._settings.scan_path,
            correlation_id=cid,
            json_body=payload,
            retry=False,
        )
        if status_code in {409, 429}:
            rejected = _parse_scan_rejected(body, correlation_id=cid)
            if status_code == 429 and rejected.reason == "unknown":
                rejected = ScanRejected(
                    reason="rate_limited",
                    correlation_id=cid,
                    retry_after_ms=rejected.retry_after_ms,
                )
            logger.info(
                "Pi rejected a scan request",
                extra={
                    "correlation_id": cid,
                    "preset_id": preset_id,
                    "reason": rejected.reason,
                    "retry_after_ms": rejected.retry_after_ms,
                },
            )
            return rejected
        self._raise_http_error(status_code, body, headers, correlation_id=cid)
        if status_code not in {200, 202}:
            raise PiCameraError(
                "Pi camera scan returned an unexpected status.",
                code="unexpected_status",
                correlation_id=cid,
                status_code=status_code,
            )
        parsed = unwrap_payload(parse_json_object(body, correlation_id=cid))
        scan_id = as_str(parsed.get("scan_id"))
        if not scan_id:
            raise PiCameraMalformedResponse(
                "Pi accepted a scan without scan_id.",
                correlation_id=cid,
                status_code=status_code,
            )
        return ScanAccepted(
            scan_id=scan_id,
            correlation_id=cid,
            planned_poses=as_int(parsed.get("planned_poses")),
            estimated_duration_ms=as_int(parsed.get("estimated_duration_ms")),
        )

    async def cancel_scan(
        self,
        *,
        scan_id: str,
        correlation_id: str | None = None,
    ) -> ScanCancelResult:
        cid = self._require_ready(correlation_id)
        if not _SCAN_ID_RE.fullmatch(scan_id):
            raise PiCameraError(
                "scan_id is not a valid identifier.",
                code="invalid_scan_id",
                correlation_id=cid,
            )
        path = self._cancel_path(scan_id)
        payload = {"correlation_id": cid}
        _assert_scan_payload_safe(payload)
        status_code, headers, body = await self._request(
            "POST",
            path,
            correlation_id=cid,
            json_body=payload,
            retry=False,
        )
        if status_code in {200, 202, 204, 404}:
            cancelled = status_code != 404
            if status_code == 404:
                # Idempotent: a missing scan is already not running.
                cancelled = True
            return ScanCancelResult(scan_id=scan_id, correlation_id=cid, cancelled=cancelled)
        self._raise_http_error(status_code, body, headers, correlation_id=cid)
        return ScanCancelResult(scan_id=scan_id, correlation_id=cid, cancelled=True)

    async def get_scan_status(
        self,
        *,
        scan_id: str,
        correlation_id: str | None = None,
    ) -> ScanStatus:
        cid = self._require_ready(correlation_id)
        if not _SCAN_ID_RE.fullmatch(scan_id):
            raise PiCameraError(
                "scan_id is not a valid identifier.",
                code="invalid_scan_id",
                correlation_id=cid,
            )
        path = self._scan_status_path(scan_id)
        status_code, headers, body = await self._request(
            "GET",
            path,
            correlation_id=cid,
            retry=False,
        )
        self._raise_http_error(status_code, body, headers, correlation_id=cid)
        payload = unwrap_payload(parse_json_object(body, correlation_id=cid))
        state = as_str(payload.get("state"))
        if not state:
            raise PiCameraMalformedResponse(
                "Pi scan status had no state.",
                correlation_id=cid,
                status_code=status_code,
            )
        raw_ids = payload.get("frame_ids")
        frame_ids: tuple[int, ...] = ()
        if isinstance(raw_ids, list):
            parsed_ids = [as_int(item) for item in raw_ids]
            frame_ids = tuple(item for item in parsed_ids if item is not None)
        return ScanStatus(
            scan_id=scan_id,
            correlation_id=cid,
            state=state,
            preset_id=as_str(payload.get("preset_id")),
            planned_poses=as_int(payload.get("planned_poses")),
            estimated_duration_ms=as_int(payload.get("estimated_duration_ms")),
            current_pose_index=as_int(payload.get("current_pose_index")),
            completed_poses=as_int(payload.get("completed_poses")),
            frame_ids=frame_ids,
            cancel_requested=as_bool(payload.get("cancel_requested")),
            cancel_reason=as_str(payload.get("cancel_reason")),
            error=as_str(payload.get("error")),
        )

    def _scan_status_path(self, scan_id: str) -> str:
        encoded = quote(scan_id, safe="")
        template = self._settings.scan_status_path
        if "{scan_id}" in template:
            return template.format(scan_id=encoded)
        return f"{template.rstrip('/')}/{encoded}"

    def _cancel_path(self, scan_id: str) -> str:
        encoded = quote(scan_id, safe="")
        template = self._settings.scan_cancel_path
        if "{scan_id}" in template:
            return template.format(scan_id=encoded)
        return f"{template.rstrip('/')}/{encoded}"

    def _require_ready(self, correlation_id: str | None) -> str:
        cid = (correlation_id or correlation_id_var.get() or uuid.uuid4().hex).strip()
        if not cid:
            cid = uuid.uuid4().hex
        if not self._settings.enabled:
            raise PiCameraUnavailable(
                "Pi camera client is not enabled.",
                code="disabled",
                correlation_id=cid,
            )
        if not self._token:
            raise PiCameraUnavailable(
                "Pi camera token is not configured.",
                code="missing_token",
                correlation_id=cid,
            )
        return cid

    async def _request(
        self,
        method: str,
        path: str,
        *,
        correlation_id: str,
        json_body: dict[str, Any] | None = None,
        retry: bool,
    ) -> tuple[int, dict[str, str], bytes]:
        url = _join_url(self._settings.base_url, path)
        headers = {
            "Authorization": f"Bearer {self._token}",
            CORRELATION_HEADER: correlation_id,
            "Accept": "multipart/form-data, application/json, image/jpeg, image/png, image/webp",
        }
        attempts = self._settings.get_attempts if retry else 1
        token = correlation_id_var.set(correlation_id)
        last_error: Exception | None = None
        try:
            async with asyncio.timeout(self._settings.timeout_seconds):
                for attempt in range(attempts):
                    request_kwargs: dict[str, Any] = {"headers": headers}
                    if json_body is not None:
                        request_kwargs["json"] = json_body
                    request = self._client.build_request(method, url, **request_kwargs)
                    try:
                        response = await self._client.send(request, stream=True)
                    except asyncio.CancelledError:
                        raise
                    except httpx.TimeoutException as exc:
                        last_error = exc
                        if retry and attempt + 1 < attempts:
                            logger.warning(
                                "Pi camera GET timed out; retrying",
                                extra={
                                    "correlation_id": correlation_id,
                                    "method": method,
                                    "path": path,
                                    "attempt": attempt + 1,
                                },
                            )
                            continue
                        logger.warning(
                            "Pi camera request timed out",
                            extra={
                                "correlation_id": correlation_id,
                                "method": method,
                                "path": path,
                                "timeout_seconds": self._settings.timeout_seconds,
                            },
                        )
                        raise PiCameraTimeout(
                            "Pi camera request timed out.",
                            correlation_id=correlation_id,
                        ) from exc
                    except httpx.HTTPError as exc:
                        last_error = exc
                        if retry and attempt + 1 < attempts:
                            logger.warning(
                                "Pi camera transport failed; retrying GET",
                                extra={
                                    "correlation_id": correlation_id,
                                    "method": method,
                                    "path": path,
                                    "attempt": attempt + 1,
                                    "error": type(exc).__name__,
                                },
                            )
                            continue
                        logger.warning(
                            "Pi camera transport failed",
                            extra={
                                "correlation_id": correlation_id,
                                "method": method,
                                "path": path,
                                "error": type(exc).__name__,
                            },
                        )
                        raise PiCameraUnavailable(
                            "Pi camera endpoint is unavailable.",
                            correlation_id=correlation_id,
                        ) from exc

                    try:
                        body = await _read_bounded(
                            response,
                            self._settings.max_body_bytes,
                            correlation_id=correlation_id,
                        )
                    finally:
                        await response.aclose()

                    header_map = {k.lower(): v for k, v in response.headers.items()}
                    if retry and 500 <= response.status_code < 600 and attempt + 1 < attempts:
                        logger.warning(
                            "Pi camera GET returned a server error; retrying",
                            extra={
                                "correlation_id": correlation_id,
                                "method": method,
                                "path": path,
                                "status_code": response.status_code,
                                "attempt": attempt + 1,
                            },
                        )
                        continue
                    return response.status_code, header_map, body
        except asyncio.CancelledError:
            raise
        except TimeoutError as exc:
            logger.warning(
                "Pi camera request exceeded the wait bound",
                extra={
                    "correlation_id": correlation_id,
                    "method": method,
                    "path": path,
                    "timeout_seconds": self._settings.timeout_seconds,
                },
            )
            raise PiCameraTimeout(
                "Pi camera request timed out.",
                correlation_id=correlation_id,
            ) from exc
        finally:
            correlation_id_var.reset(token)

        if last_error is not None:
            if isinstance(last_error, httpx.TimeoutException):
                raise PiCameraTimeout(
                    "Pi camera request timed out.",
                    correlation_id=correlation_id,
                ) from last_error
            raise PiCameraUnavailable(
                "Pi camera endpoint is unavailable.",
                correlation_id=correlation_id,
            ) from last_error
        raise PiCameraUnavailable(
            "Pi camera endpoint is unavailable.",
            correlation_id=correlation_id,
        )

    def _raise_http_error(
        self,
        status_code: int,
        body: bytes,
        headers: dict[str, str],
        *,
        correlation_id: str,
    ) -> None:
        if 200 <= status_code < 300:
            return
        message, code, details = _error_fields(body)
        if status_code in {401, 403}:
            raise PiCameraAuthError(
                "Pi camera rejected the request.",
                correlation_id=correlation_id,
                status_code=status_code,
                details=details,
            )
        if status_code == 404:
            raise PiCameraUnavailable(
                "Pi camera endpoint was not found.",
                code="not_found",
                correlation_id=correlation_id,
                status_code=status_code,
                details=details,
            )
        if status_code in {408, 504}:
            raise PiCameraTimeout(
                message or "Pi camera request timed out.",
                correlation_id=correlation_id,
                status_code=status_code,
                details=details,
            )
        if status_code == 429:
            retry_after = as_int(headers.get("retry-after"))
            if retry_after is not None:
                details = {**details, "retry_after": retry_after}
            raise PiCameraError(
                message or "Pi camera rate-limited the request.",
                code=code or "rate_limited",
                correlation_id=correlation_id,
                status_code=status_code,
                details=details,
            )
        if status_code == 409 and (code == "stale_frame" or "stale" in (message or "").lower()):
            raise PiCameraStaleFrame(
                message or "Pi camera frame was too stale.",
                correlation_id=correlation_id,
                status_code=status_code,
                details=details,
            )
        if status_code >= 500:
            raise PiCameraUnavailable(
                message or "Pi camera endpoint is unavailable.",
                correlation_id=correlation_id,
                status_code=status_code,
                details=details,
            )
        raise PiCameraError(
            message or "Pi camera request failed.",
            code=code or "request_failed",
            correlation_id=correlation_id,
            status_code=status_code,
            details=details,
        )


def _join_url(base: str, path: str) -> str:
    return base.rstrip("/") + "/" + path.lstrip("/")


def _merge_frame_headers(headers: dict[str, str], metadata: dict[str, Any]) -> dict[str, Any]:
    merged = dict(metadata)
    frame_id = as_int(headers.get("x-frame-id"))
    if frame_id is not None:
        merged.setdefault("frame_id", frame_id)
    frame_age = as_int(headers.get("x-frame-age-ms"))
    if frame_age is not None:
        merged.setdefault("frame_age_ms", frame_age)
    camera_source = as_str(headers.get("x-camera-source-id"))
    if camera_source is not None:
        merged.setdefault("camera_source_id", camera_source)
    return merged


def _frame_is_stale(
    *,
    frame_age_ms: int | None,
    health: CameraHealth | None,
    stale_frame_ms: int,
) -> bool:
    if health is CameraHealth.STALLED:
        return True
    if frame_age_ms is not None and frame_age_ms > stale_frame_ms:
        return True
    return False


def _assert_scan_payload_safe(payload: dict[str, Any]) -> None:
    extra = FORBIDDEN_SCAN_KEYS.intersection(payload)
    if extra:
        raise PiCameraError(
            "Scan payload contains camera fields; motion targets use the motion API.",
            code="forbidden_scan_field",
            details={"fields": sorted(extra)},
        )


def _parse_scan_rejected(body: bytes, *, correlation_id: str) -> ScanRejected:
    payload: dict[str, Any]
    try:
        payload = unwrap_payload(parse_json_object(body, correlation_id=correlation_id))
    except PiCameraMalformedResponse:
        payload = {}
    reason = as_str(payload.get("reason")) or as_str(payload.get("error")) or "unknown"
    if isinstance(payload.get("error"), dict):
        inner = payload["error"]
        reason = as_str(inner.get("code")) or as_str(inner.get("message")) or reason
    return ScanRejected(
        reason=reason,
        correlation_id=correlation_id,
        retry_after_ms=as_int(payload.get("retry_after_ms")),
    )


def _error_fields(body: bytes) -> tuple[str, str | None, dict[str, Any]]:
    if not body:
        return "", None, {}
    try:
        payload = json_object_or_empty(body)
    except PiCameraMalformedResponse:
        return "", None, {}
    message = (
        as_str(payload.get("message"))
        or as_str(payload.get("detail"))
        or as_str(payload.get("error"))
    )
    code = as_str(payload.get("code"))
    details: dict[str, Any] = {}
    error = payload.get("error")
    if isinstance(error, dict):
        code = as_str(error.get("code")) or code
        message = as_str(error.get("message")) or message
        raw_details = error.get("details")
        if isinstance(raw_details, dict):
            details = raw_details
    if isinstance(message, str) and len(message) > 200:
        message = message[:200]
    return message or "", code, details


def json_object_or_empty(body: bytes) -> dict[str, Any]:
    return parse_json_object(body, correlation_id=None)


async def _read_bounded(
    response: httpx.Response,
    max_bytes: int,
    *,
    correlation_id: str,
) -> bytes:
    content_length = response.headers.get("content-length")
    if content_length is not None:
        try:
            length: int | None = int(content_length)
        except ValueError:
            length = None
        if length is not None and length > max_bytes:
            raise PiCameraOversizedResponse(
                "Pi camera response exceeded the size limit.",
                correlation_id=correlation_id,
                details={"content_length": length, "max_body_bytes": max_bytes},
            )
    chunks: list[bytes] = []
    total = 0
    async for chunk in response.aiter_bytes(chunk_size=65_536):
        total += len(chunk)
        if total > max_bytes:
            raise PiCameraOversizedResponse(
                "Pi camera response exceeded the size limit.",
                correlation_id=correlation_id,
                details={"bytes_read": total, "max_body_bytes": max_bytes},
            )
        chunks.append(chunk)
    return b"".join(chunks)
