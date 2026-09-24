"""Unit tests for the typed personal assistant HTTP client."""

from __future__ import annotations

import httpx
import pytest
import respx

from reachy_client._transport import CORRELATION_HEADER, IDEMPOTENCY_HEADER, build_headers
from reachy_client.client import PersonalAssistantClient
from reachy_client.exceptions import (
    AuthError,
    NotFoundError,
    RateLimitError,
    ServiceUnavailable,
    TimeoutError,
    ValidationError,
)

BASE = "http://pa.test"
TOKEN = "test-token-0123456789abcdef0123456789abcdef"


@pytest.fixture
def client() -> PersonalAssistantClient:
    return PersonalAssistantClient(BASE, TOKEN)


class TestHeaders:
    def test_builds_bearer_and_correlation(self) -> None:
        headers = build_headers(TOKEN, correlation_id="abc123")
        assert headers["Authorization"] == f"Bearer {TOKEN}"
        assert headers[CORRELATION_HEADER] == "abc123"

    def test_idempotency_key_is_optional(self) -> None:
        headers = build_headers(TOKEN)
        assert IDEMPOTENCY_HEADER not in headers
        with_key = build_headers(TOKEN, idempotency_key="idem-1")
        assert with_key[IDEMPOTENCY_HEADER] == "idem-1"


class TestExceptionMapping:
    @respx.mock
    def test_401_maps_to_auth_error(self, client: PersonalAssistantClient) -> None:
        respx.get(f"{BASE}/api/v1/ping").mock(
            return_value=httpx.Response(
                401,
                json={
                    "ok": False,
                    "error": {"code": "unauthorized", "message": "bad token"},
                    "request_id": "req-1",
                },
            )
        )
        with pytest.raises(AuthError) as exc_info:
            client.ping()
        assert exc_info.value.status_code == 401
        assert exc_info.value.request_id == "req-1"

    @respx.mock
    def test_404_maps_to_not_found(self, client: PersonalAssistantClient) -> None:
        from reachy_client.models import TaskOut

        respx.get(f"{BASE}/api/v1/tasks/missing").mock(
            return_value=httpx.Response(
                404,
                json={"ok": False, "error": {"code": "not_found", "message": "gone"}},
            )
        )
        with pytest.raises(NotFoundError):
            client._get("/api/v1/tasks/missing", TaskOut)

    @respx.mock
    def test_422_maps_to_validation_error(self, client: PersonalAssistantClient) -> None:
        respx.get(f"{BASE}/api/v1/ping").mock(
            return_value=httpx.Response(
                422,
                json={"ok": False, "error": {"code": "validation_error", "message": "bad"}},
            )
        )
        with pytest.raises(ValidationError):
            client.ping()

    @respx.mock
    def test_429_maps_to_rate_limit(self, client: PersonalAssistantClient) -> None:
        respx.get(f"{BASE}/api/v1/ping").mock(
            return_value=httpx.Response(
                429,
                headers={"Retry-After": "12"},
                json={"ok": False, "error": {"code": "rate_limited", "message": "slow down"}},
            )
        )
        with pytest.raises(RateLimitError) as exc_info:
            client.ping()
        assert exc_info.value.retry_after == 12


class TestRetry:
    @respx.mock
    def test_get_retries_once_on_503(self, client: PersonalAssistantClient) -> None:
        route = respx.get(f"{BASE}/api/v1/ping").mock(
            side_effect=[
                httpx.Response(503, json={"ok": False, "error": {"message": "busy"}}),
                httpx.Response(
                    200,
                    json={
                        "pong": True,
                        "message": "ok",
                        "server_time_utc": "2026-01-01T00:00:00Z",
                        "server_time_local": "2026-01-01T00:00:00-05:00",
                        "timezone": "America/Toronto",
                    },
                ),
            ]
        )
        result = client.ping()
        assert result.pong is True
        assert route.call_count == 2

    @respx.mock
    def test_post_without_idempotency_does_not_retry(self, client: PersonalAssistantClient) -> None:
        route = respx.post(f"{BASE}/api/v1/tasks/t1/complete").mock(
            return_value=httpx.Response(503, json={"ok": False, "error": {"message": "busy"}})
        )
        with pytest.raises(ServiceUnavailable):
            client.complete_task("t1")
        assert route.call_count == 1

    @respx.mock
    def test_post_with_idempotency_retries(self, client: PersonalAssistantClient) -> None:
        from reachy_client.models import TaskCreate

        route = respx.post(f"{BASE}/api/v1/tasks").mock(
            side_effect=[
                httpx.Response(503, json={"ok": False, "error": {"message": "busy"}}),
                httpx.Response(
                    201,
                    json={
                        "id": "t1",
                        "description": "x",
                        "status": "open",
                        "priority": "normal",
                        "created_at": "2026-01-01T00:00:00Z",
                    },
                ),
            ]
        )
        task = client.create_task(TaskCreate(description="x"), idempotency_key="k1")
        assert task.id == "t1"
        assert route.call_count == 2


class TestCorrelationId:
    @respx.mock
    def test_sends_correlation_header(self, client: PersonalAssistantClient) -> None:
        route = respx.get(f"{BASE}/api/v1/ping").mock(
            return_value=httpx.Response(
                200,
                json={
                    "pong": True,
                    "message": "ok",
                    "server_time_utc": "2026-01-01T00:00:00Z",
                    "server_time_local": "2026-01-01T00:00:00-05:00",
                    "timezone": "America/Toronto",
                },
            )
        )
        client.ping()
        assert CORRELATION_HEADER in route.calls.last.request.headers

    @respx.mock
    def test_timeout_carries_correlation_id(self, client: PersonalAssistantClient) -> None:
        respx.get(f"{BASE}/api/v1/ping").mock(side_effect=httpx.ConnectError("refused"))
        with pytest.raises(TimeoutError) as exc_info:
            client.ping()
        assert exc_info.value.correlation_id

    @respx.mock
    def test_connect_timeout_is_bounded_client_error(self, client: PersonalAssistantClient) -> None:
        route = respx.get(f"{BASE}/api/v1/ping").mock(
            side_effect=httpx.ConnectTimeout("connect timed out")
        )
        with pytest.raises(TimeoutError):
            client.ping()
        assert route.call_count == 2
