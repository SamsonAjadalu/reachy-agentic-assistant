"""Authenticated HTTP access to the Google REST APIs.

The official client library is synchronous and builds its own thread pool; this
service is asyncio end to end, so the handful of endpoints it needs are called
directly over httpx. That also makes every request assertable in tests with
respx instead of monkeypatching a discovery document.

Shared behaviour lives here so no individual service re-implements it:

* one 401 triggers exactly one token refresh and one retry, and a second 401 is
  reported as an authorisation problem rather than retried forever;
* 429 and 5xx back off with jitter and honour ``Retry-After``;
* every page of a list endpoint is bounded, so a large mailbox cannot turn one
  question into thousands of requests.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import AsyncIterator
from typing import Any

import httpx

from app.config import Settings, get_settings
from app.logging_config import get_logger
from integrations.google.oauth import TokenProvider
from shared.errors import IntegrationError, ProviderRateLimitError, ProviderTimeoutError

logger = get_logger(__name__)

GMAIL_BASE = "https://gmail.googleapis.com/gmail/v1"
CALENDAR_BASE = "https://www.googleapis.com/calendar/v3"
PEOPLE_BASE = "https://people.googleapis.com/v1"
DRIVE_BASE = "https://www.googleapis.com/drive/v3"

RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})
MAX_ATTEMPTS = 4
MAX_PAGES = 20
REQUEST_TIMEOUT = httpx.Timeout(30.0, connect=10.0)


class GoogleClient:
    """Thin authenticated wrapper over the Google REST endpoints."""

    def __init__(
        self,
        settings: Settings | None = None,
        token_provider: TokenProvider | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._tokens = token_provider or TokenProvider(self._settings)
        self._owns_client = http_client is None
        self._http = http_client or httpx.AsyncClient(timeout=REQUEST_TIMEOUT)

    @property
    def tokens(self) -> TokenProvider:
        return self._tokens

    async def aclose(self) -> None:
        if self._owns_client:
            await self._http.aclose()

    async def __aenter__(self) -> GoogleClient:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    async def request(
        self,
        method: str,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
        expect_json: bool = True,
    ) -> Any:
        token = await self._tokens.get_access_token(self._http)
        refreshed = False

        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                response = await self._http.request(
                    method,
                    url,
                    params=params,
                    json=json_body,
                    headers={"Authorization": f"Bearer {token}"},
                )
            except httpx.TimeoutException as exc:
                if attempt == MAX_ATTEMPTS:
                    raise ProviderTimeoutError(
                        "Google did not respond in time.", details={"integration": "google"}
                    ) from exc
                await self._sleep(attempt, None)
                continue
            except httpx.HTTPError as exc:
                raise IntegrationError(
                    f"Could not reach Google ({type(exc).__name__}).", integration="google"
                ) from exc

            if response.status_code == httpx.codes.UNAUTHORIZED and not refreshed:
                # The cached access token expired early or was revoked mid-flight.
                logger.info("Google returned 401; refreshing the access token")
                self._tokens.invalidate()
                token = await self._tokens.get_access_token(self._http)
                refreshed = True
                continue

            if response.status_code in RETRYABLE_STATUS and attempt < MAX_ATTEMPTS:
                await self._sleep(attempt, response)
                continue

            return self._finish(response, expect_json)

        raise IntegrationError("Google request failed after retries.", integration="google")

    def _finish(self, response: httpx.Response, expect_json: bool) -> Any:
        if response.status_code == httpx.codes.TOO_MANY_REQUESTS:
            raise ProviderRateLimitError(
                "Google is rate limiting this account; try again shortly.",
                details={"integration": "google"},
            )
        if response.status_code >= httpx.codes.BAD_REQUEST:
            raise IntegrationError(
                f"Google rejected the request: {_describe(response)}",
                integration="google",
                http_status=502 if response.status_code >= 500 else 400,
            )
        if not expect_json:
            return response.text
        if not response.content:
            return None
        try:
            return response.json()
        except ValueError as exc:
            raise IntegrationError(
                "Google returned a response that was not JSON.", integration="google"
            ) from exc

    async def _sleep(self, attempt: int, response: httpx.Response | None) -> None:
        """Back off with jitter, preferring the server's own Retry-After."""
        delay: float | None = None
        if response is not None:
            header = response.headers.get("retry-after")
            if header and header.isdigit():
                delay = float(header)
        if delay is None:
            delay = min(2**attempt, 16) + random.uniform(0, 0.5)  # noqa: S311 - jitter, not crypto
        logger.info("Backing off a Google call", extra={"attempt": attempt, "delay": delay})
        await asyncio.sleep(delay)

    async def paginate(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        items_key: str,
        limit: int,
        page_param: str = "pageToken",
        page_size_param: str = "pageSize",
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield items across pages, stopping at ``limit`` or ``MAX_PAGES``."""
        collected = 0
        page_token: str | None = None

        for _ in range(MAX_PAGES):
            page_params = dict(params or {})
            remaining = limit - collected
            page_params[page_size_param] = min(remaining, 100)
            if page_token:
                page_params[page_param] = page_token

            body = await self.request("GET", url, params=page_params)
            for item in body.get(items_key) or []:
                yield item
                collected += 1
                if collected >= limit:
                    return

            page_token = body.get("nextPageToken")
            if not page_token:
                return


def _describe(response: httpx.Response) -> str:
    """A short, safe description of a failure.

    Google error bodies echo request parameters, which for Gmail can include
    message content, so only the status and the top-level message are used.
    """
    try:
        error = response.json().get("error")
    except ValueError:
        error = None
    if isinstance(error, dict):
        message = error.get("message") or error.get("status")
    elif isinstance(error, str):
        message = error
    else:
        message = None
    return f"HTTP {response.status_code}: {message or 'no detail'}"
