"""HTTP transport helpers shared by sync and async clients."""

from __future__ import annotations

import uuid
from collections.abc import Callable, Mapping
from typing import Any, TypeVar

import httpx

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

CORRELATION_HEADER = "X-Correlation-ID"
IDEMPOTENCY_HEADER = "Idempotency-Key"

T = TypeVar("T")


def build_headers(
    token: str,
    *,
    correlation_id: str | None = None,
    idempotency_key: str | None = None,
    extra: Mapping[str, str] | None = None,
) -> dict[str, str]:
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
        CORRELATION_HEADER: correlation_id or uuid.uuid4().hex,
    }
    if idempotency_key:
        headers[IDEMPOTENCY_HEADER] = idempotency_key
    if extra:
        headers.update(extra)
    return headers


def should_retry(method: str, idempotency_key: str | None, status_code: int | None) -> bool:
    if method.upper() != "GET" and not (method.upper() == "POST" and idempotency_key):
        return False
    if status_code is not None and 500 <= status_code < 600:
        return True
    return status_code is None


def map_error(
    response: httpx.Response,
    *,
    correlation_id: str,
) -> ApiError:
    request_id: str | None = None
    code: str | None = None
    message = f"HTTP {response.status_code}"
    details: dict[str, Any] = {}

    try:
        payload = response.json()
        if isinstance(payload, dict):
            request_id = payload.get("request_id")
            error = payload.get("error")
            if isinstance(error, dict):
                code = error.get("code")
                message = str(error.get("message", message))
                raw_details = error.get("details")
                if isinstance(raw_details, dict):
                    details = raw_details
    except Exception:
        message = response.text or message

    status_code = response.status_code
    if status_code == 401:
        return AuthError(
            message,
            status_code=status_code,
            code=code,
            correlation_id=correlation_id,
            request_id=request_id,
            details=details,
        )
    if status_code == 404:
        return NotFoundError(
            message,
            status_code=status_code,
            code=code,
            correlation_id=correlation_id,
            request_id=request_id,
            details=details,
        )
    if status_code == 422:
        return ValidationError(
            message,
            status_code=status_code,
            code=code,
            correlation_id=correlation_id,
            request_id=request_id,
            details=details,
        )
    if status_code == 429:
        retry_after: int | None = None
        raw = response.headers.get("Retry-After")
        if raw and raw.isdigit():
            retry_after = int(raw)
        return RateLimitError(
            message,
            retry_after=retry_after,
            status_code=status_code,
            code=code,
            correlation_id=correlation_id,
            request_id=request_id,
            details=details,
        )
    if status_code == 504:
        return TimeoutError(
            message,
            status_code=status_code,
            code=code,
            correlation_id=correlation_id,
            request_id=request_id,
            details=details,
        )
    if status_code == 503 or 500 <= status_code < 600:
        return ServiceUnavailable(
            message,
            status_code=status_code,
            code=code,
            correlation_id=correlation_id,
            request_id=request_id,
            details=details,
        )
    if status_code == 409 and code in {"approval_error", "conflict"}:
        return ApprovalRequired(
            message,
            approval_id=str(details.get("approval_id")) if details.get("approval_id") else None,
            action_id=str(details.get("action_id")) if details.get("action_id") else None,
            status_code=status_code,
            code=code,
            correlation_id=correlation_id,
            request_id=request_id,
            details=details,
        )
    return ApiError(
        message,
        status_code=status_code,
        code=code,
        correlation_id=correlation_id,
        request_id=request_id,
        details=details,
    )


def raise_for_status(response: httpx.Response, *, correlation_id: str) -> None:
    if response.is_success:
        return
    raise map_error(response, correlation_id=correlation_id)


def request_with_retry(
    send: Callable[[], httpx.Response],
    *,
    method: str,
    idempotency_key: str | None,
) -> httpx.Response:
    last_exc: Exception | None = None
    status_code: int | None = None

    for attempt in range(2):
        try:
            response = send()
            status_code = response.status_code
            if response.is_success or not should_retry(method, idempotency_key, status_code):
                return response
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            last_exc = exc
            if not should_retry(method, idempotency_key, None) or attempt == 1:
                raise TimeoutError(
                    str(exc),
                    correlation_id="",
                ) from exc
            continue

    if last_exc is not None:
        raise TimeoutError(str(last_exc)) from last_exc
    assert status_code is not None
    return response
