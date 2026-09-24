"""Uses the configured workflow."""

from __future__ import annotations

import secrets

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.logging_config import get_logger
from shared.errors import ConfigurationError, PermissionDeniedError
from vision_sidecar.config import SidecarSettings

logger = get_logger(__name__)

_bearer = HTTPBearer(auto_error=False, description="Sidecar bearer token.")


def get_settings_from_app(request: Request) -> SidecarSettings:
    return request.app.state.settings


async def require_token(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    settings: SidecarSettings = Depends(get_settings_from_app),
) -> str:
    expected = settings.auth_token
    if not expected:
        raise ConfigurationError(
            "VISION_SIDECAR_TOKEN (or PA_API_TOKEN) is not configured, "
            "so no client can be authenticated."
        )
    presented = credentials.credentials if credentials is not None else ""
    scheme_ok = credentials is not None and credentials.scheme.lower() == "bearer"
    token_ok = secrets.compare_digest(presented.encode("utf-8"), expected.encode("utf-8"))
    if not (scheme_ok and token_ok):
        client = request.client.host if request.client else "unknown"
        logger.warning(
            "Rejected unauthenticated sidecar request",
            extra={"client": client, "http_path": request.url.path},
        )
        raise PermissionDeniedError(
            "A valid bearer token is required.", code="unauthorized", http_status=401
        )
    return "sidecar-client"
