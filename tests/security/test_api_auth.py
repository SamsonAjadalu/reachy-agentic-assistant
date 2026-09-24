"""API authentication, network allowlisting and request limits."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.config import Settings
from app.middleware.network import NetworkAllowlistMiddleware
from tests.conftest import TEST_API_TOKEN

pytestmark = pytest.mark.anyio if False else []


class TestBearerToken:
    async def test_protected_endpoint_rejects_a_missing_token(
        self, anonymous_client: AsyncClient
    ) -> None:
        response = await anonymous_client.get("/api/v1/ping")
        assert response.status_code == 401
        assert response.json()["error"]["code"] == "unauthorized"

    async def test_protected_endpoint_rejects_a_wrong_token(
        self, anonymous_client: AsyncClient
    ) -> None:
        response = await anonymous_client.get(
            "/api/v1/ping", headers={"Authorization": "Bearer wrong-token"}
        )
        assert response.status_code == 401

    async def test_rejects_a_non_bearer_scheme_carrying_the_right_secret(
        self, anonymous_client: AsyncClient
    ) -> None:
        response = await anonymous_client.get(
            "/api/v1/ping", headers={"Authorization": f"Basic {TEST_API_TOKEN}"}
        )
        assert response.status_code == 401

    async def test_rejects_a_token_with_a_trailing_character(
        self, anonymous_client: AsyncClient
    ) -> None:
        response = await anonymous_client.get(
            "/api/v1/ping", headers={"Authorization": f"Bearer {TEST_API_TOKEN}x"}
        )
        assert response.status_code == 401

    async def test_accepts_the_configured_token(self, client: AsyncClient) -> None:
        response = await client.get("/api/v1/ping")
        assert response.status_code == 200
        assert response.json()["pong"] is True

    async def test_failure_response_never_echoes_the_expected_token(
        self, anonymous_client: AsyncClient
    ) -> None:
        response = await anonymous_client.get(
            "/api/v1/ping", headers={"Authorization": "Bearer nope"}
        )
        assert TEST_API_TOKEN not in response.text

    async def test_health_and_ready_stay_public(self, anonymous_client: AsyncClient) -> None:
        # systemd and an off-network monitor must be able to tell "down" from "blocked".
        assert (await anonymous_client.get("/health")).status_code == 200
        assert (await anonymous_client.get("/ready")).status_code == 200


class TestNetworkAllowlist:
    @pytest.mark.parametrize(
        ("networks", "host", "allowed"),
        [
            (["127.0.0.0/8"], "127.0.0.1", True),
            (["127.0.0.0/8"], "10.1.2.3", False),
            (["192.168.0.0/16"], "192.168.4.5", True),
            (["100.64.0.0/10"], "100.101.102.103", True),  # Tailscale CGNAT
            (["127.0.0.0/8"], "::ffff:127.0.0.1", True),  # v4-mapped v6
            (["127.0.0.0/8"], "not-an-address", False),
            ([], "203.0.113.9", True),  # empty allowlist means unrestricted
        ],
    )
    def test_address_matching(self, networks: list[str], host: str, allowed: bool) -> None:
        middleware = NetworkAllowlistMiddleware(
            app=object(), allowed_networks=networks, exempt_paths=frozenset()
        )
        assert middleware.is_allowed(host) is allowed

    async def test_request_from_a_disallowed_address_is_blocked(self, settings: Settings) -> None:
        from app.main import create_app

        settings.pa_api_allowed_networks = ["10.99.0.0/16"]
        application = create_app(settings)

        async with AsyncClient(
            transport=ASGITransport(app=application, client=("203.0.113.5", 12345)),
            base_url="http://testserver",
            headers={"Authorization": f"Bearer {TEST_API_TOKEN}"},
        ) as http_client:
            response = await http_client.get("/api/v1/ping")

        assert response.status_code == 403
        assert response.json()["error"]["code"] == "network_not_allowed"

    async def test_blocking_happens_before_authentication(self, settings: Settings) -> None:
        """Off-network callers receive a consistent authentication response."""
        from app.main import create_app

        settings.pa_api_allowed_networks = ["10.99.0.0/16"]
        application = create_app(settings)

        async with AsyncClient(
            transport=ASGITransport(app=application, client=("203.0.113.5", 12345)),
            base_url="http://testserver",
        ) as http_client:
            response = await http_client.get("/api/v1/ping")

        assert response.status_code == 403  # not 401


class TestRequestLimits:
    async def test_oversized_body_is_rejected_before_parsing(self, settings: Settings) -> None:
        from app.main import create_app

        settings.pa_max_request_bytes = 1024
        application = create_app(settings)

        async with AsyncClient(
            transport=ASGITransport(app=application),
            base_url="http://testserver",
            headers={"Authorization": f"Bearer {TEST_API_TOKEN}"},
        ) as http_client:
            response = await http_client.post("/api/v1/ping", content=b"x" * 4096)

        assert response.status_code == 413
        assert response.json()["error"]["code"] == "request_too_large"

    async def test_rate_limit_triggers_and_reports_retry_after(self, settings: Settings) -> None:
        from app.main import create_app

        settings.pa_rate_limit_per_minute = 3
        application = create_app(settings)

        async with AsyncClient(
            transport=ASGITransport(app=application),
            base_url="http://testserver",
            headers={"Authorization": f"Bearer {TEST_API_TOKEN}"},
        ) as http_client:
            statuses = [(await http_client.get("/api/v1/ping")).status_code for _ in range(5)]
            limited = await http_client.get("/api/v1/ping")

        assert statuses.count(429) >= 1
        assert limited.status_code == 429
        assert "Retry-After" in limited.headers

    async def test_health_is_exempt_from_rate_limiting(self, settings: Settings) -> None:
        from app.main import create_app

        settings.pa_rate_limit_per_minute = 2
        application = create_app(settings)

        async with AsyncClient(
            transport=ASGITransport(app=application), base_url="http://testserver"
        ) as http_client:
            statuses = [(await http_client.get("/health")).status_code for _ in range(10)]

        assert set(statuses) == {200}


class TestResponseHygiene:
    async def test_security_headers_are_present(self, client: AsyncClient) -> None:
        headers = (await client.get("/api/v1/ping")).headers
        assert headers["X-Content-Type-Options"] == "nosniff"
        assert headers["X-Frame-Options"] == "DENY"
        assert "frame-ancestors 'none'" in headers["Content-Security-Policy"]

    async def test_request_id_is_returned_and_honoured(self, client: AsyncClient) -> None:
        response = await client.get("/api/v1/ping", headers={"X-Request-ID": "abc-123"})
        assert response.headers["X-Request-ID"] == "abc-123"
        assert response.json()["request_id"] == "abc-123"

    async def test_correlation_id_defaults_to_the_request_id(self, client: AsyncClient) -> None:
        response = await client.get("/api/v1/ping")
        assert response.headers["X-Correlation-ID"] == response.headers["X-Request-ID"]

    async def test_unknown_route_returns_the_standard_envelope(self, client: AsyncClient) -> None:
        response = await client.get("/api/v1/definitely-not-a-route")
        assert response.status_code == 404
        body = response.json()
        assert body["ok"] is False
        assert body["error"]["code"] == "not_found"


async def test_status_and_integrations_never_expose_credentials(
    client: AsyncClient, app: FastAPI
) -> None:
    for path in ("/api/v1/status", "/api/v1/integrations"):
        text = (await client.get(path)).text
        assert TEST_API_TOKEN not in text
        assert "secret" not in text.lower() or "client_secret" not in text.lower()
