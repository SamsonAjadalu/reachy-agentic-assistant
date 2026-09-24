"""Exception handlers producing the standard error envelope.

Nothing raised inside the application reaches the client as a stack trace or a
raw provider message: unexpected exceptions are logged in full and reported to
the caller as a generic internal error carrying only the request id.
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import JSONResponse

from app.logging_config import get_logger
from security.redaction import redact_text, redact_value
from shared.errors import AssistantError

logger = get_logger(__name__)


def _envelope(
    request: Request, status: int, code: str, message: str, details: object | None = None
) -> JSONResponse:
    body: dict[str, object] = {
        "ok": False,
        "error": {"code": code, "message": redact_text(message)},
        "request_id": getattr(request.state, "request_id", None),
    }
    if details:
        body["error"]["details"] = redact_value(details)  # type: ignore[index]
    return JSONResponse(status_code=status, content=body)


async def assistant_error_handler(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, AssistantError)
    log = logger.warning if exc.http_status < 500 else logger.error
    log(
        "Handled application error",
        extra={
            "error_code": exc.code,
            "http_status": exc.http_status,
            "http_path": request.url.path,
        },
    )
    return _envelope(request, exc.http_status, exc.code, exc.message, exc.details or None)


async def validation_error_handler(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, RequestValidationError)
    problems = [
        {
            "field": ".".join(str(part) for part in error.get("loc", ()) if part != "body"),
            "message": error.get("msg", "invalid value"),
            "type": error.get("type", "value_error"),
        }
        for error in exc.errors()
    ]
    return _envelope(
        request,
        422,
        "validation_error",
        "The request body or query parameters failed validation.",
        {"problems": problems},
    )


async def http_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, StarletteHTTPException)
    codes = {
        400: "bad_request",
        401: "unauthorized",
        403: "forbidden",
        404: "not_found",
        405: "method_not_allowed",
        409: "conflict",
        413: "request_too_large",
        415: "unsupported_media_type",
        429: "rate_limited",
    }
    code = codes.get(exc.status_code, f"http_{exc.status_code}")
    message = exc.detail if isinstance(exc.detail, str) else "Request could not be completed."
    return _envelope(request, exc.status_code, code, message)


async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception(
        "Unhandled exception",
        extra={"http_path": request.url.path, "exception_type": type(exc).__name__},
    )
    return _envelope(
        request,
        500,
        "internal_error",
        "The assistant hit an unexpected error. The request id identifies the server log entry.",
    )


def register_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(AssistantError, assistant_error_handler)
    app.add_exception_handler(RequestValidationError, validation_error_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(Exception, unhandled_error_handler)
