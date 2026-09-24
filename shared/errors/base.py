"""Error hierarchy for the personal assistant."""

from __future__ import annotations

from typing import Any


class AssistantError(Exception):
    """Base class for all deliberate application errors."""

    code: str = "internal_error"
    http_status: int = 500

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        http_status: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        if code is not None:
            self.code = code
        if http_status is not None:
            self.http_status = http_status
        self.details: dict[str, Any] = details or {}

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.details:
            payload["details"] = self.details
        return payload


class ValidationError(AssistantError):
    code = "validation_error"
    http_status = 422


class NotFoundError(AssistantError):
    code = "not_found"
    http_status = 404


class ConflictError(AssistantError):
    code = "conflict"
    http_status = 409


class PermissionDeniedError(AssistantError):
    code = "permission_denied"
    http_status = 403


class ConfigurationError(AssistantError):
    code = "configuration_error"
    http_status = 500


class IntegrationDisabledError(AssistantError):
    """Raised when a caller uses an integration that is switched off in config."""

    code = "integration_disabled"
    http_status = 503


class ExternalServiceError(AssistantError):
    code = "external_service_error"
    http_status = 502


class ProviderTimeoutError(ExternalServiceError):
    code = "provider_timeout"
    http_status = 504


class ProviderRateLimitError(ExternalServiceError):
    code = "provider_rate_limited"
    http_status = 429


class IdempotencyConflictError(ConflictError):
    """Same idempotency key replayed with a different request body."""

    code = "idempotency_conflict"


class ApprovalError(AssistantError):
    """An approval-gated action could not proceed."""

    code = "approval_error"
    http_status = 409


class SecurityViolationError(AssistantError):
    """A request tripped an explicit security control.

    Deliberately vague to the caller; the specific reason is logged server-side.
    """

    code = "security_violation"
    http_status = 400
