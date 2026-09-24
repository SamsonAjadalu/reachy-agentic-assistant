"""Structured errors for the Pi camera HTTP adapter."""

from __future__ import annotations

from typing import Any

from security.redaction import redact_value


class PiCameraError(Exception):
    """Uses the configured workflow."""

    def __init__(
        self,
        message: str,
        *,
        code: str,
        correlation_id: str | None = None,
        status_code: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.correlation_id = correlation_id
        self.status_code = status_code
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "correlation_id": self.correlation_id,
            "status_code": self.status_code,
            "details": redact_value(self.details),
        }


class PiCameraTimeout(PiCameraError):
    def __init__(
        self,
        message: str = "Pi camera request timed out.",
        *,
        code: str = "timeout",
        **kwargs: Any,
    ) -> None:
        super().__init__(message, code=code, **kwargs)


class PiCameraUnavailable(PiCameraError):
    def __init__(
        self,
        message: str = "Pi camera endpoint is unavailable.",
        *,
        code: str = "unavailable",
        **kwargs: Any,
    ) -> None:
        super().__init__(message, code=code, **kwargs)


class PiCameraAuthError(PiCameraError):
    def __init__(
        self,
        message: str = "Pi camera rejected the request.",
        *,
        code: str = "unauthorized",
        **kwargs: Any,
    ) -> None:
        super().__init__(message, code=code, **kwargs)


class PiCameraMalformedResponse(PiCameraError):
    def __init__(
        self,
        message: str = "Pi camera returned a malformed response.",
        *,
        code: str = "malformed_response",
        **kwargs: Any,
    ) -> None:
        super().__init__(message, code=code, **kwargs)


class PiCameraOversizedResponse(PiCameraError):
    def __init__(
        self,
        message: str = "Pi camera response exceeded the size limit.",
        *,
        code: str = "oversized_response",
        **kwargs: Any,
    ) -> None:
        super().__init__(message, code=code, **kwargs)


class PiCameraUnsupportedMedia(PiCameraError):
    def __init__(
        self,
        message: str = "Pi camera response was not a still image.",
        *,
        code: str = "unsupported_media",
        **kwargs: Any,
    ) -> None:
        super().__init__(message, code=code, **kwargs)


class PiCameraInvalidRequest(PiCameraError):
    def __init__(
        self,
        message: str = "Pi camera request was invalid.",
        *,
        code: str = "invalid_request",
        **kwargs: Any,
    ) -> None:
        super().__init__(message, code=code, **kwargs)


class PiCameraStaleFrame(PiCameraError):
    """Pi refused to return a frame older than the caller's max_age_ms."""

    def __init__(
        self,
        message: str = "Pi camera frame was too stale.",
        *,
        code: str = "stale_frame",
        **kwargs: Any,
    ) -> None:
        super().__init__(message, code=code, **kwargs)
