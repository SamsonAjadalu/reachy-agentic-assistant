"""Public exports for the personal assistant HTTP client."""

from reachy_client import models
from reachy_client.client import AsyncPersonalAssistantClient, PersonalAssistantClient
from reachy_client.exceptions import (
    ApiError,
    ApprovalRequired,
    AuthError,
    NotFoundError,
    RateLimitError,
    ServiceUnavailable,
    TimeoutError,
    ValidationError,
)

__all__ = [
    "ApiError",
    "ApprovalRequired",
    "AsyncPersonalAssistantClient",
    "AuthError",
    "NotFoundError",
    "PersonalAssistantClient",
    "RateLimitError",
    "ServiceUnavailable",
    "TimeoutError",
    "ValidationError",
    "models",
]
