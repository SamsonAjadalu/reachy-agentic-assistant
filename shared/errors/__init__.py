"""Typed application errors.

Every error carries a stable machine-readable ``code`` so clients (including the
Reachy tool adapters) can branch without string matching, and an HTTP status for
Uses the configured workflow.
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
    IntegrationError,
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
    "IntegrationError",
    "NotFoundError",
    "PermissionDeniedError",
    "ProviderRateLimitError",
    "ProviderTimeoutError",
    "SecurityViolationError",
    "ValidationError",
]
