"""Typed errors raised by the personal assistant HTTP clients."""

from __future__ import annotations

from typing import Any


class ApiError(Exception):
    """Base class for client-side API failures."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        code: str | None = None,
        correlation_id: str | None = None,
        request_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.code = code
        self.correlation_id = correlation_id
        self.request_id = request_id
        self.details = details or {}


class AuthError(ApiError):
    """Missing or invalid bearer token."""


class NotFoundError(ApiError):
    """The requested resource does not exist."""


class ValidationError(ApiError):
    """Request body or query parameters failed server-side validation."""


class ServiceUnavailable(ApiError):
    """The service or an integration is temporarily unavailable."""


class TimeoutError(ApiError):
    """The request timed out before a response arrived."""


class RateLimitError(ApiError):
    """The caller exceeded the configured rate limit."""

    def __init__(
        self,
        message: str,
        *,
        retry_after: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.retry_after = retry_after


class ApprovalRequired(ApiError):
    """An approval-gated action could not proceed without owner consent."""

    def __init__(
        self,
        message: str,
        *,
        approval_id: str | None = None,
        action_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.approval_id = approval_id
        self.action_id = action_id
