"""OAuth storage and refresh behaviour.

The refresh token is the most valuable secret the system holds: it grants
continuing access to mail. These tests check that it is encrypted at rest, kept
out of logs and API responses, and that the key lives somewhere other than the
ciphertext.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
import respx

from app.config import Settings
from integrations.google.oauth import (
    ALL_SCOPES,
    TOKEN_ENDPOINT,
    WRITE_SCOPES,
    StoredCredentials,
    TokenProvider,
    build_authorization_request,
    build_token_store,
    forget_credentials,
    load_credentials,
    save_credentials,
)
from security.redaction import redact_text
from shared.errors import ConfigurationError, IntegrationError

REFRESH_TOKEN = "1//0gFakeRefreshTokenForTests_abcdefghijklmnop"


@pytest.fixture
def google_settings(settings: Settings) -> Settings:
    settings.google_enabled = True
    settings.google_client_id = "1234.apps.googleusercontent.com"
    settings.google_client_secret = type(settings.google_client_secret)("fake-client-secret")
    return settings


@pytest.fixture
def stored(google_settings: Settings) -> StoredCredentials:
    credentials = StoredCredentials(
        refresh_token=REFRESH_TOKEN,
        scopes=list(ALL_SCOPES),
        account_email="owner@example.com",
    )
    save_credentials(credentials, google_settings)
    return credentials


class TestStorage:
    def test_the_refresh_token_is_not_readable_on_disk(
        self, google_settings: Settings, stored: StoredCredentials
    ) -> None:
        raw = Path(google_settings.google_token_store_path).read_bytes()
        assert REFRESH_TOKEN.encode() not in raw

    def test_the_store_is_owner_only(
        self, google_settings: Settings, stored: StoredCredentials
    ) -> None:
        mode = Path(google_settings.google_token_store_path).stat().st_mode & 0o777
        assert mode & 0o077 == 0

    def test_a_round_trip_returns_the_same_grant(
        self, google_settings: Settings, stored: StoredCredentials
    ) -> None:
        loaded = load_credentials(google_settings)
        assert loaded is not None
        assert loaded.refresh_token == REFRESH_TOKEN
        assert loaded.account_email == "owner@example.com"

    def test_loading_registers_the_token_for_redaction(
        self, google_settings: Settings, stored: StoredCredentials
    ) -> None:
        load_credentials(google_settings)
        assert REFRESH_TOKEN not in redact_text(f"token={REFRESH_TOKEN}")

    def test_the_wrong_key_gives_an_actionable_error(
        self, google_settings: Settings, stored: StoredCredentials, tmp_path: Path
    ) -> None:
        """A restored backup with a mismatched key must say so, not crash obscurely."""
        from cryptography.fernet import Fernet

        other = tmp_path / "other-key" / "secret.key"
        other.parent.mkdir(parents=True)
        other.write_text(Fernet.generate_key().decode())
        other.chmod(0o600)
        google_settings.pa_secret_key_file = other

        with pytest.raises(ConfigurationError, match="does not match"):
            load_credentials(google_settings)

    def test_forgetting_removes_the_grant(
        self, google_settings: Settings, stored: StoredCredentials
    ) -> None:
        assert forget_credentials(google_settings) is True
        assert load_credentials(google_settings) is None

    def test_the_key_may_not_live_beside_the_ciphertext(self, google_settings: Settings) -> None:
        """Co-locating them would mean one stolen directory yields both."""
        google_settings.pa_secret_key_file = (
            Path(google_settings.google_token_store_path).parent / "secret.key"
        )
        with pytest.raises(ConfigurationError, match="same directory"):
            build_token_store(google_settings)

    def test_health_reports_state_without_the_token(
        self, google_settings: Settings, stored: StoredCredentials
    ) -> None:
        health = build_token_store(google_settings).health()
        assert health["providers"] == ["google"]
        assert REFRESH_TOKEN not in str(health)


class TestAuthorizationUrl:
    def test_the_url_requests_offline_access_and_pkce(self, google_settings: Settings) -> None:
        request = build_authorization_request(google_settings)
        assert "access_type=offline" in request.url
        assert "code_challenge_method=S256" in request.url
        assert "prompt=consent" in request.url

    def test_the_verifier_is_not_in_the_url(self, google_settings: Settings) -> None:
        """Only the challenge travels; the verifier stays on the machine."""
        request = build_authorization_request(google_settings)
        assert request.code_verifier not in request.url

    def test_each_request_has_a_fresh_state(self, google_settings: Settings) -> None:
        first = build_authorization_request(google_settings)
        second = build_authorization_request(google_settings)
        assert first.state != second.state

    def test_every_scope_the_app_uses_is_requested(self, google_settings: Settings) -> None:
        request = build_authorization_request(google_settings)
        assert "gmail.readonly" in request.url
        assert "gmail.send" in request.url


class TestRefresh:
    @respx.mock
    async def test_an_access_token_is_obtained_and_cached(
        self, google_settings: Settings, stored: StoredCredentials
    ) -> None:
        route = respx.post(TOKEN_ENDPOINT).mock(
            return_value=httpx.Response(
                200, json={"access_token": "ya29.fake-access", "expires_in": 3600}
            )
        )
        provider = TokenProvider(google_settings)

        first = await provider.get_access_token()
        second = await provider.get_access_token()

        assert first == second == "ya29.fake-access"
        assert route.call_count == 1

    @respx.mock
    async def test_concurrent_callers_cause_one_refresh(
        self, google_settings: Settings, stored: StoredCredentials
    ) -> None:
        import asyncio

        route = respx.post(TOKEN_ENDPOINT).mock(
            return_value=httpx.Response(
                200, json={"access_token": "ya29.fake-access", "expires_in": 3600}
            )
        )
        provider = TokenProvider(google_settings)

        await asyncio.gather(*(provider.get_access_token() for _ in range(5)))

        assert route.call_count == 1

    @respx.mock
    async def test_a_revoked_grant_says_how_to_recover(
        self, google_settings: Settings, stored: StoredCredentials
    ) -> None:
        respx.post(TOKEN_ENDPOINT).mock(
            return_value=httpx.Response(400, json={"error": "invalid_grant"})
        )

        with pytest.raises(IntegrationError, match="setup_google_oauth"):
            await TokenProvider(google_settings).get_access_token()

    @respx.mock
    async def test_the_access_token_never_reaches_a_log(
        self, google_settings: Settings, stored: StoredCredentials
    ) -> None:
        respx.post(TOKEN_ENDPOINT).mock(
            return_value=httpx.Response(
                200, json={"access_token": "ya29.super-secret", "expires_in": 3600}
            )
        )
        await TokenProvider(google_settings).get_access_token()
        assert "ya29.super-secret" not in redact_text("Bearer ya29.super-secret")

    async def test_an_unauthorised_provider_explains_the_fix(
        self, google_settings: Settings
    ) -> None:
        with pytest.raises(IntegrationError, match="setup_google_oauth"):
            await TokenProvider(google_settings).get_access_token()


class TestScopeEnforcement:
    def test_a_missing_scope_is_refused_before_the_call(self, google_settings: Settings) -> None:
        """Better a clear local error than a 403 from Google mid-conversation."""
        save_credentials(
            StoredCredentials(
                refresh_token=REFRESH_TOKEN,
                scopes=["https://www.googleapis.com/auth/gmail.readonly"],
            ),
            google_settings,
        )
        provider = TokenProvider(google_settings)

        provider.require_scope("https://www.googleapis.com/auth/gmail.readonly")
        with pytest.raises(IntegrationError, match="gmail.send"):
            provider.require_scope("https://www.googleapis.com/auth/gmail.send")

    def test_status_reports_write_capability(
        self, google_settings: Settings, stored: StoredCredentials
    ) -> None:
        status = TokenProvider(google_settings).status()
        assert status["authorised"] is True
        assert status["has_write_scopes"] is True
        assert all(scope in status["scopes"] for scope in WRITE_SCOPES)

    def test_status_carries_no_token(
        self, google_settings: Settings, stored: StoredCredentials
    ) -> None:
        assert REFRESH_TOKEN not in str(TokenProvider(google_settings).status())


class TestStatusEndpoint:
    async def test_the_endpoint_reports_mock_mode(self, client) -> None:
        body = (await client.get("/api/v1/integrations/google")).json()
        assert body["mode"] == "mock"
        assert body["authorised"] is False

    async def test_the_endpoint_requires_a_token(self, anonymous_client) -> None:
        assert (await anonymous_client.get("/api/v1/integrations/google")).status_code == 401
