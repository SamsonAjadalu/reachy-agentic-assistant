"""Request context, size limits and rate limiting."""

from __future__ import annotations

import time
import uuid
from collections import defaultdict, deque
from collections.abc import Awaitable, Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from app.logging_config import correlation_id_var, get_logger, request_id_var

logger = get_logger(__name__)

REQUEST_ID_HEADER = "X-Request-ID"
CORRELATION_ID_HEADER = "X-Correlation-ID"

Handler = Callable[[Request], Awaitable[Response]]


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Assign a request id, honour an inbound correlation id, and time the call.

    The correlation id is what lets a Reachy voice turn be traced through the
    API, a background task and a Telegram approval as one logical operation.
    """

    async def dispatch(self, request: Request, call_next: Handler) -> Response:
        request_id = request.headers.get(REQUEST_ID_HEADER) or uuid.uuid4().hex
        correlation_id = request.headers.get(CORRELATION_ID_HEADER) or request_id

        request_token = request_id_var.set(request_id)
        correlation_token = correlation_id_var.set(correlation_id)
        request.state.request_id = request_id
        request.state.correlation_id = correlation_id

        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            duration_ms = (time.perf_counter() - started) * 1000
            logger.exception(
                "Unhandled error while processing request",
                extra={
                    "http_method": request.method,
                    "http_path": request.url.path,
                    "duration_ms": round(duration_ms, 2),
                },
            )
            raise
        finally:
            request_id_var.reset(request_token)
            correlation_id_var.reset(correlation_token)

        duration_ms = (time.perf_counter() - started) * 1000
        response.headers[REQUEST_ID_HEADER] = request_id
        response.headers[CORRELATION_ID_HEADER] = correlation_id
        logger.info(
            "request",
            extra={
                "http_method": request.method,
                "http_path": request.url.path,
                "http_status": response.status_code,
                "duration_ms": round(duration_ms, 2),
                "request_id": request_id,
                "correlation_id": correlation_id,
            },
        )
        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Conservative headers.

    The API serves JSON to a robot and a CLI, so a restrictive CSP costs nothing
    and blocks the browser-based attack surface entirely.
    """

    async def dispatch(self, request: Request, call_next: Handler) -> Response:
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault("Cache-Control", "no-store")
        response.headers.setdefault(
            "Content-Security-Policy",
            "default-src 'none'; frame-ancestors 'none'; base-uri 'none'",
        )
        return response


class BodySizeLimitMiddleware(BaseHTTPMiddleware):
    """Reject oversized bodies before they are buffered.

    Image uploads use a larger ceiling enforced by their own endpoint; this is
    the blanket limit for everything else.
    """

    def __init__(self, app: object, max_bytes: int, exempt_prefixes: tuple[str, ...] = ()) -> None:
        super().__init__(app)  # type: ignore[arg-type]
        self.max_bytes = max_bytes
        self.exempt_prefixes = exempt_prefixes

    async def dispatch(self, request: Request, call_next: Handler) -> Response:
        if not request.url.path.startswith(self.exempt_prefixes):
            declared = request.headers.get("content-length")
            if declared is not None:
                try:
                    if int(declared) > self.max_bytes:
                        return _too_large(self.max_bytes, request)
                except ValueError:
                    return _too_large(self.max_bytes, request)
        return await call_next(request)


def _too_large(limit: int, request: Request) -> JSONResponse:
    return JSONResponse(
        status_code=413,
        content={
            "ok": False,
            "error": {
                "code": "request_too_large",
                "message": f"Request body exceeds the {limit} byte limit.",
            },
            "request_id": getattr(request.state, "request_id", None),
        },
    )


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Fixed-window-free sliding limiter, per client address.

    The purpose is not to stop an attacker - the bearer token does that - but to
    stop a misbehaving tool loop from hammering Gmail or the cluster.
    """

    def __init__(self, app: object, requests_per_minute: int, exempt_paths: frozenset[str]) -> None:
        super().__init__(app)  # type: ignore[arg-type]
        self.limit = requests_per_minute
        self.exempt_paths = exempt_paths
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    async def dispatch(self, request: Request, call_next: Handler) -> Response:
        if self.limit <= 0 or request.url.path in self.exempt_paths:
            return await call_next(request)

        client = request.client.host if request.client else "unknown"
        now = time.monotonic()
        window = self._hits[client]
        cutoff = now - 60.0
        while window and window[0] < cutoff:
            window.popleft()

        if len(window) >= self.limit:
            retry_after = max(1, int(60 - (now - window[0])))
            logger.warning("Rate limit exceeded", extra={"client": client, "limit": self.limit})
            return JSONResponse(
                status_code=429,
                headers={"Retry-After": str(retry_after)},
                content={
                    "ok": False,
                    "error": {
                        "code": "rate_limited",
                        "message": (
                            f"Too many requests. The limit is {self.limit} per minute; "
                            f"retry in {retry_after}s."
                        ),
                    },
                    "request_id": getattr(request.state, "request_id", None),
                },
            )

        window.append(now)
        # Bound memory if many distinct clients appear.
        if len(self._hits) > 512:
            for key in [k for k, v in self._hits.items() if not v]:
                del self._hits[key]
        return await call_next(request)
