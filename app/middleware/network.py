"""Network-level access control.

Runs before authentication so a host outside the allowed CIDRs never reaches the
token comparison at all. This is defence in depth, not the primary control: a
LAN peer that steals the bearer token still gets in, which is why the token
matters and why the deployment guidance is Tailscale rather than a wide LAN bind.
"""

from __future__ import annotations

import ipaddress
from collections.abc import Awaitable, Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from app.logging_config import get_logger

logger = get_logger(__name__)

Handler = Callable[[Request], Awaitable[Response]]


class NetworkAllowlistMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: object, allowed_networks: list[str], exempt_paths: frozenset[str]):
        super().__init__(app)  # type: ignore[arg-type]
        self.networks = [ipaddress.ip_network(entry, strict=False) for entry in allowed_networks]
        self.exempt_paths = exempt_paths

    def is_allowed(self, host: str) -> bool:
        if not self.networks:
            return True
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            return False
        # A v4 client arriving over a dual-stack socket appears as ::ffff:x.x.x.x.
        if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
            address = address.ipv4_mapped
        return any(address in network for network in self.networks)

    async def dispatch(self, request: Request, call_next: Handler) -> Response:
        if request.url.path in self.exempt_paths:
            return await call_next(request)

        host = request.client.host if request.client else ""
        if not self.is_allowed(host):
            logger.warning("Blocked request from disallowed network", extra={"client": host})
            return JSONResponse(
                status_code=403,
                content={
                    "ok": False,
                    "error": {
                        "code": "network_not_allowed",
                        "message": "This client address is not permitted to reach the assistant.",
                    },
                    "request_id": getattr(request.state, "request_id", None),
                },
            )
        return await call_next(request)
