"""Typed application errors.

Every error carries a stable machine-readable ``code`` so clients (including the
Reachy tool adapters) can branch without string matching, and an HTTP status for
the API layer. Error messages are considered model-visible: never interpolate a
credential, token or raw provider response into one.
"""

from shared.errors.base import (
    ApprovalError,
    AssistantError,
    ConfigurationError,
    ConflictError,
    ExternalServiceError,
    IdempotencyConflictError,
    IntegrationDisabledError,
    NotFoundError,
    PermissionDeniedError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    SecurityViolationError,
    ValidationError,
)

__all__ = [
    "ApprovalError",
    "AssistantError",
    "ConfigurationError",
    "ConflictError",
    "ExternalServiceError",
    "IdempotencyConflictError",
    "IntegrationDisabledError",
    "NotFoundError",
    "PermissionDeniedError",
    "ProviderRateLimitError",
    "ProviderTimeoutError",
    "SecurityViolationError",
    "ValidationError",
]
