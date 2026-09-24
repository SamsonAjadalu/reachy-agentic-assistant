"""Bearer token authentication.

Uses the configured workflow.
error message, and a failure returns the same generic message whether the header
was absent, malformed or simply wrong.
"""

from __future__ import annotations

import secrets

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.config import Settings, get_settings
from app.logging_config import get_logger
from shared.errors import ConfigurationError, PermissionDeniedError

logger = get_logger(__name__)

# auto_error=False so a missing header produces our envelope, not FastAPI's.
_bearer = HTTPBearer(auto_error=False, description="Static bearer token issued to Reachy clients.")


class Principal:
    """The authenticated caller.

    A single shared token today. The type exists so per-client tokens can be
    introduced later without changing every endpoint signature.
    """

    def __init__(self, name: str, source: str) -> None:
        self.name = name
        self.source = source

    def __repr__(self) -> str:
        return f"Principal(name={self.name!r})"


async def require_token(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    settings: Settings = Depends(get_settings),
) -> Principal:
    expected = settings.pa_api_token.get_secret_value()
    if not expected:
        raise ConfigurationError(
            "PA_API_TOKEN is not configured, so no client can be authenticated. "
            "Run scripts/generate_api_token.py."
        )

    presented = credentials.credentials if credentials is not None else ""
    scheme_ok = credentials is not None and credentials.scheme.lower() == "bearer"

    # Always compare, even when the scheme is wrong, so timing does not reveal
    # which part of the header was rejected.
    token_ok = secrets.compare_digest(presented.encode("utf-8"), expected.encode("utf-8"))

    if not (scheme_ok and token_ok):
        client = request.client.host if request.client else "unknown"
        logger.warning(
            "Rejected unauthenticated request",
            extra={"client": client, "http_path": request.url.path},
        )
        raise PermissionDeniedError(
            "A valid bearer token is required.", code="unauthorized", http_status=401
        )

    return Principal(name="reachy-client", source="static_token")


CurrentPrincipal = Depends(require_token)
