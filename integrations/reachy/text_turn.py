"""HTTP client for the Reachy conversation app's text-turn endpoint."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from app.config import Settings, get_settings
from app.logging_config import get_logger
from security.redaction import register_secret

logger = get_logger(__name__)


class TextTurnError(RuntimeError):
    """Reachy text-turn failed in a way the user should see a generic message for."""


class TextTurnTimeout(TextTurnError):
    """The Pi did not answer within the configured deadline."""


class TextTurnUnavailable(TextTurnError):
    """The Pi endpoint could not be reached."""


class TextTurnMalformedResponse(TextTurnError):
    """Reachy returned a body without assistant_text."""


@dataclass(frozen=True)
class TextTurnResult:
    assistant_text: str


class ReachyTextTurnClient:
    """Forwards plain user text to the active Reachy HF session."""

    def __init__(self, settings: Settings | None = None, client: httpx.AsyncClient | None = None):
        self._settings = settings or get_settings()
        self._url = self._settings.reachy_text_turn_url
        self._token = self._settings.reachy_text_turn_token.get_secret_value()
        if self._token:
            register_secret(self._token)
        timeout = httpx.Timeout(
            self._settings.reachy_text_turn_timeout_seconds,
            connect=min(10.0, self._settings.reachy_text_turn_timeout_seconds),
        )
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=timeout)

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def submit_turn(self, *, text: str, turn_id: str) -> str:
        """Send one user turn and return Reachy's assistant_text."""
        if not self._settings.reachy_text_turn_enabled:
            raise TextTurnUnavailable("Reachy text chat is not enabled.")
        if not self._token:
            raise TextTurnUnavailable("Reachy text-turn token is not configured.")
        if not text.strip():
            raise TextTurnError("Message text is empty.")

        headers = {"Authorization": f"Bearer {self._token}"}
        payload = {"text": text, "turn_id": turn_id}

        try:
            response = await self._client.post(self._url, json=payload, headers=headers)
        except httpx.TimeoutException as exc:
            logger.warning(
                "Reachy text-turn timed out",
                extra={
                    "turn_id": turn_id,
                    "timeout_seconds": self._settings.reachy_text_turn_timeout_seconds,
                },
            )
            raise TextTurnTimeout("Reachy did not answer in time.") from exc
        except httpx.HTTPError as exc:
            logger.warning(
                "Reachy text-turn transport failed",
                extra={"turn_id": turn_id, "error": type(exc).__name__},
            )
            raise TextTurnUnavailable("Reachy is unavailable right now.") from exc

        return _parse_response(response, turn_id=turn_id)


def _parse_response(response: httpx.Response, *, turn_id: str) -> str:
    try:
        body: dict[str, Any] = response.json()
    except ValueError as exc:
        logger.warning(
            "Reachy text-turn returned non-JSON",
            extra={"turn_id": turn_id, "status_code": response.status_code},
        )
        raise TextTurnMalformedResponse("Reachy returned an invalid response.") from exc

    if response.status_code >= 400 or body.get("ok") is False:
        error = str(body.get("error") or body.get("detail") or "request failed")
        logger.warning(
            "Reachy text-turn rejected the request",
            extra={"turn_id": turn_id, "status_code": response.status_code, "error": error[:120]},
        )
        if response.status_code in {401, 403}:
            raise TextTurnUnavailable("Reachy rejected the request.")
        raise TextTurnError("Reachy could not process that message.")

    assistant_text = body.get("assistant_text")
    if not isinstance(assistant_text, str) or not assistant_text.strip():
        logger.warning(
            "Reachy text-turn response missing assistant_text",
            extra={"turn_id": turn_id, "status_code": response.status_code},
        )
        raise TextTurnMalformedResponse("Reachy returned an empty reply.")

    return assistant_text.strip()
