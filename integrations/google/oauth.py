"""Google OAuth 2.0 for an installed application.

One consent grant covers Gmail, Calendar, Contacts and Drive: the assistant asks
for every scope it will ever need at setup time, because a mid-conversation
re-consent is impossible on a headless machine.

The refresh token is written to the Fernet-encrypted store, never to the
database and never to a log. Access tokens live in memory only, are refreshed
about a minute before expiry, and a concurrent burst of API calls triggers one
refresh rather than several.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import secrets
import urllib.parse
from dataclasses import dataclass
from datetime import datetime, timedelta

import httpx

from app.config import Settings, get_settings
from app.logging_config import get_logger
from security.keyring import resolve_secret_key
from security.redaction import register_secret
from security.token_store import EncryptedTokenStore
from shared.errors import ConfigurationError, IntegrationError
from shared.timeutils import utcnow

logger = get_logger(__name__)

AUTH_ENDPOINT = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_ENDPOINT = "https://oauth2.googleapis.com/token"  # noqa: S105 - a URL, not a secret
REVOKE_ENDPOINT = "https://oauth2.googleapis.com/revoke"
USERINFO_ENDPOINT = "https://www.googleapis.com/oauth2/v3/userinfo"

STORE_KEY = "google"
REFRESH_MARGIN = timedelta(seconds=90)

# Read scopes are requested unconditionally; the two write scopes are what make
# approval gating necessary, and are listed separately so SECURITY.md can point
# at exactly which grants allow an outbound side effect.
READ_SCOPES = (
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/calendar.readonly",
    "https://www.googleapis.com/auth/contacts.readonly",
    "https://www.googleapis.com/auth/drive.readonly",
    "openid",
    "email",
)
WRITE_SCOPES = (
    # compose creates drafts; send transmits an existing draft; modify covers
    # reversible mailbox hygiene (archive / labels). None of these skip the
    # matching server-side gate for irreversible sends.
    "https://www.googleapis.com/auth/gmail.compose",
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/calendar.events",
)
ALL_SCOPES = READ_SCOPES + WRITE_SCOPES


@dataclass(frozen=True)
class AuthorizationRequest:
    url: str
    state: str
    code_verifier: str
    redirect_uri: str


@dataclass
class StoredCredentials:
    refresh_token: str
    scopes: list[str]
    account_email: str | None = None
    obtained_at: str | None = None

    def to_payload(self) -> dict[str, object]:
        return {
            "refresh_token": self.refresh_token,
            "scopes": self.scopes,
            "account_email": self.account_email,
            "obtained_at": self.obtained_at or utcnow().isoformat(),
        }

    def has_scope(self, scope: str) -> bool:
        return scope in self.scopes


def build_token_store(settings: Settings | None = None) -> EncryptedTokenStore:
    settings = settings or get_settings()
    if settings.google_token_store_path is None:
        raise ConfigurationError("GOOGLE_TOKEN_STORE_PATH is not configured.")
    return EncryptedTokenStore(settings.google_token_store_path, resolve_secret_key(settings))


def load_credentials(settings: Settings | None = None) -> StoredCredentials | None:
    settings = settings or get_settings()
    entry = build_token_store(settings).get(STORE_KEY)
    if not entry or not entry.get("refresh_token"):
        return None
    refresh_token = str(entry["refresh_token"])
    register_secret(refresh_token)
    return StoredCredentials(
        refresh_token=refresh_token,
        scopes=list(entry.get("scopes") or []),
        account_email=entry.get("account_email"),
        obtained_at=entry.get("obtained_at"),
    )


def save_credentials(credentials: StoredCredentials, settings: Settings | None = None) -> None:
    register_secret(credentials.refresh_token)
    build_token_store(settings).put(STORE_KEY, credentials.to_payload())


def forget_credentials(settings: Settings | None = None) -> bool:
    return build_token_store(settings).delete(STORE_KEY)


def build_authorization_request(
    settings: Settings | None = None, scopes: tuple[str, ...] = ALL_SCOPES
) -> AuthorizationRequest:
    """Assemble the consent URL with PKCE.

    PKCE matters even for a "confidential" desktop client: the client secret in
    a locally installed app is not really secret, and without the verifier an
    intercepted authorization code would be redeemable.
    """
    settings = settings or get_settings()
    if not settings.google_client_id:
        raise ConfigurationError("GOOGLE_CLIENT_ID is not configured.")

    verifier = base64.urlsafe_b64encode(secrets.token_bytes(64)).decode("ascii").rstrip("=")
    challenge = (
        base64.urlsafe_b64encode(hashlib.sha256(verifier.encode("ascii")).digest())
        .decode("ascii")
        .rstrip("=")
    )
    state = secrets.token_urlsafe(24)

    query = {
        "client_id": settings.google_client_id,
        "redirect_uri": settings.google_redirect_uri,
        "response_type": "code",
        "scope": " ".join(scopes),
        "state": state,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        # Google only returns a refresh token on the first consent unless the
        # prompt is forced, which makes re-running setup silently useless.
        "access_type": "offline",
        "prompt": "consent",
    }
    return AuthorizationRequest(
        url=f"{AUTH_ENDPOINT}?{urllib.parse.urlencode(query)}",
        state=state,
        code_verifier=verifier,
        redirect_uri=settings.google_redirect_uri,
    )


async def exchange_code(
    code: str, request: AuthorizationRequest, settings: Settings | None = None
) -> StoredCredentials:
    settings = settings or get_settings()
    payload = {
        "code": code,
        "client_id": settings.google_client_id,
        "client_secret": settings.google_client_secret.get_secret_value(),
        "redirect_uri": request.redirect_uri,
        "grant_type": "authorization_code",
        "code_verifier": request.code_verifier,
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(TOKEN_ENDPOINT, data=payload)
        body = _token_response(response)

        refresh_token = body.get("refresh_token")
        if not refresh_token:
            raise IntegrationError(
                "Google did not return a refresh token. Revoke the app's access at "
                "https://myaccount.google.com/permissions and run setup again.",
                integration="google",
            )

        access_token = str(body.get("access_token", ""))
        email = await _fetch_email(client, access_token) if access_token else None

    return StoredCredentials(
        refresh_token=str(refresh_token),
        scopes=str(body.get("scope", "")).split() or list(ALL_SCOPES),
        account_email=email,
        obtained_at=utcnow().isoformat(),
    )


async def _fetch_email(client: httpx.AsyncClient, access_token: str) -> str | None:
    try:
        response = await client.get(
            USERINFO_ENDPOINT, headers={"Authorization": f"Bearer {access_token}"}
        )
        if response.status_code == httpx.codes.OK:
            value = response.json().get("email")
            return str(value) if value else None
    except httpx.HTTPError:
        pass
    return None


def _as_int(value: object, *, default: int) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return default


def _token_response(response: httpx.Response) -> dict[str, object]:
    """Parse a token endpoint reply, keeping the secret out of the error path."""
    try:
        body = response.json()
    except ValueError:
        body = {}
    if response.status_code != httpx.codes.OK:
        detail = body.get("error_description") or body.get("error") or "unknown error"
        raise IntegrationError(
            f"Google token endpoint refused the request: {detail}", integration="google"
        )
    if not isinstance(body, dict):
        raise IntegrationError(
            "Google token endpoint returned an unexpected body.", integration="google"
        )
    return body


class TokenProvider:
    """Supplies a valid access token, refreshing as needed.

    Shared by every Google service so one refresh serves all of them. The lock
    means a burst of concurrent calls after expiry performs a single refresh
    instead of one per caller, which Google would rate-limit.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._lock = asyncio.Lock()
        self._access_token: str | None = None
        self._expires_at: datetime | None = None
        self._credentials: StoredCredentials | None = None
        self.refresh_count = 0

    @property
    def credentials(self) -> StoredCredentials | None:
        if self._credentials is None:
            self._credentials = load_credentials(self._settings)
        return self._credentials

    def is_authorised(self) -> bool:
        return self.credentials is not None

    def require_scope(self, scope: str) -> None:
        credentials = self.credentials
        if credentials is None:
            raise IntegrationError(
                "Google is not authorised. Run scripts/setup_google_oauth.py.",
                integration="google",
            )
        if not credentials.has_scope(scope):
            raise IntegrationError(
                f"The stored Google grant does not include {scope}. Re-run "
                "scripts/setup_google_oauth.py to consent to the additional scope.",
                integration="google",
            )

    def invalidate(self) -> None:
        """Drop the cached access token so the next call refreshes."""
        self._access_token = None
        self._expires_at = None

    async def get_access_token(self, client: httpx.AsyncClient | None = None) -> str:
        now = utcnow()
        if self._access_token and self._expires_at and now < self._expires_at - REFRESH_MARGIN:
            return self._access_token

        async with self._lock:
            # Another caller may have refreshed while this one waited.
            if self._access_token and self._expires_at and now < self._expires_at - REFRESH_MARGIN:
                return self._access_token
            return await self._refresh(client)

    async def _refresh(self, client: httpx.AsyncClient | None) -> str:
        credentials = self.credentials
        if credentials is None:
            raise IntegrationError(
                "Google is not authorised. Run scripts/setup_google_oauth.py.",
                integration="google",
            )

        payload = {
            "client_id": self._settings.google_client_id,
            "client_secret": self._settings.google_client_secret.get_secret_value(),
            "refresh_token": credentials.refresh_token,
            "grant_type": "refresh_token",
        }

        owns_client = client is None
        http = client or httpx.AsyncClient(timeout=30.0)
        try:
            response = await http.post(TOKEN_ENDPOINT, data=payload)
            body = _token_response(response)
        except IntegrationError as exc:
            if "invalid_grant" in str(exc):
                raise IntegrationError(
                    "The stored Google refresh token was revoked or expired. Run "
                    "scripts/setup_google_oauth.py to re-authorise.",
                    integration="google",
                ) from exc
            raise
        finally:
            if owns_client:
                await http.aclose()

        access_token = str(body.get("access_token", ""))
        if not access_token:
            raise IntegrationError("Google returned no access token.", integration="google")

        register_secret(access_token)
        expires_in = _as_int(body.get("expires_in"), default=3600)
        self._access_token = access_token
        self._expires_at = utcnow() + timedelta(seconds=expires_in)
        self.refresh_count += 1
        logger.info("Refreshed the Google access token", extra={"expires_in": expires_in})
        return access_token

    def status(self) -> dict[str, object]:
        """Non-sensitive summary for the diagnostics endpoint."""
        credentials = self.credentials
        return {
            "authorised": credentials is not None,
            "account_email": credentials.account_email if credentials else None,
            "scopes": credentials.scopes if credentials else [],
            "has_write_scopes": bool(
                credentials and all(credentials.has_scope(s) for s in WRITE_SCOPES)
            ),
            "access_token_cached": self._access_token is not None,
        }
