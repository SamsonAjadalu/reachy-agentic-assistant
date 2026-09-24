"""Telegram Bot API channel.

Talks to the HTTP API directly with httpx rather than through a framework: the
assistant only needs sendMessage, editMessageText, answerCallbackQuery,
sendPhoto and getUpdates, and a small client keeps the retry and redaction
behaviour under our control.

Two rules shape this module:

Uses the configured workflow.
  Telegram puts the token in the URL path, so every outgoing URL is redacted
  before it can reach a logger.
* Outbound text is escaped for MarkdownV2. Message bodies frequently contain
  untrusted content (an email subject, a filename), and unescaped text would at
  best fail to send and at worst let that content forge formatting.
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any

import httpx

from app.config import Settings, get_settings
from app.logging_config import get_logger
from notifications.base import ApprovalRequest, DeliveryResult, OutboundMessage
from security.redaction import register_secret

logger = get_logger(__name__)

API_ROOT = "https://api.telegram.org"
RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})
MAX_ATTEMPTS = 3
MAX_MESSAGE_CHARS = 4096

# Telegram requires every one of these to be escaped in MarkdownV2 text.
_MARKDOWN_V2_SPECIALS = r"_*[]()~`>#+-=|{}.!"
_MARKDOWN_V2_RE = re.compile(f"([{re.escape(_MARKDOWN_V2_SPECIALS)}])")

_SEVERITY_PREFIX = {
    "info": "",
    "warning": "Warning: ",
    "critical": "Critical: ",
}


def escape_markdown_v2(text: str) -> str:
    return _MARKDOWN_V2_RE.sub(r"\\\1", text)


def truncate(text: str, limit: int = MAX_MESSAGE_CHARS) -> str:
    """Trim to Telegram's hard message limit, leaving room for the marker."""
    if len(text) <= limit:
        return text
    marker = "\n[truncated]"
    return text[: limit - len(marker)] + marker


class TelegramError(RuntimeError):
    """Uses the configured workflow."""


class TelegramChannel:
    """Notification channel backed by the Telegram Bot API."""

    name = "telegram"

    def __init__(self, settings: Settings | None = None, client: httpx.AsyncClient | None = None):
        self._settings = settings or get_settings()
        self._token = self._settings.telegram_bot_token.get_secret_value()
        if self._token:
            register_secret(self._token)
        self._chat_ids = list(self._settings.telegram_allowed_chat_ids)
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=httpx.Timeout(20.0, connect=10.0))

    @property
    def primary_chat_id(self) -> int | None:
        return self._chat_ids[0] if self._chat_ids else None

    def is_allowed_chat(self, chat_id: int | None) -> bool:
        return chat_id is not None and chat_id in self._chat_ids

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> TelegramChannel:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    # ------------------------------------------------------------- transport
    async def call(self, method: str, payload: dict[str, Any]) -> dict[str, Any]:
        """POST to a Bot API method, retrying only what is safe to retry.

        Telegram signals rate limiting with 429 plus ``retry_after``; obeying it
        is the difference between a brief pause and the bot being throttled for
        far longer.
        """
        if not self._token:
            raise TelegramError("Telegram bot token is not configured")

        url = f"{API_ROOT}/bot{self._token}/{method}"
        last_error = "unknown error"

        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                response = await self._client.post(url, json=payload)
            except httpx.HTTPError as exc:
                last_error = type(exc).__name__
                logger.warning(
                    "Telegram request failed",
                    extra={"method": method, "attempt": attempt, "error": last_error},
                )
            else:
                if response.status_code == httpx.codes.OK:
                    body = response.json()
                    if body.get("ok"):
                        result = body.get("result")
                        return result if isinstance(result, dict) else {"result": result}
                    last_error = str(body.get("description", "api returned ok=false"))
                    # A 200 with ok=false is a permanent rejection of this payload.
                    raise TelegramError(f"Telegram rejected {method}: {last_error}")

                last_error = f"HTTP {response.status_code}"
                if response.status_code not in RETRYABLE_STATUS:
                    detail = _describe_error(response)
                    raise TelegramError(f"Telegram rejected {method}: {detail}")

                delay = _retry_after(response) or min(2**attempt, 10)
                logger.warning(
                    "Telegram call retrying",
                    extra={"method": method, "attempt": attempt, "delay_seconds": delay},
                )
                if attempt < MAX_ATTEMPTS:
                    await asyncio.sleep(delay)
                    continue

            if attempt < MAX_ATTEMPTS:
                await asyncio.sleep(min(2**attempt, 10))

        raise TelegramError(f"Telegram call {method} failed after {MAX_ATTEMPTS} attempts")

    # -------------------------------------------------------------- sending
    async def send(self, message: OutboundMessage) -> DeliveryResult:
        chat_id = self.primary_chat_id
        if chat_id is None:
            return DeliveryResult(delivered=False, suppressed_reason="no_allowed_chat")

        prefix = _SEVERITY_PREFIX.get(message.severity, "")
        text = truncate(
            f"*{escape_markdown_v2(prefix + message.title)}*\n\n{escape_markdown_v2(message.body)}"
        )

        try:
            if message.attachment_path:
                result = await self._send_photo(chat_id, Path(message.attachment_path), text)
            else:
                result = await self.call(
                    "sendMessage",
                    {
                        "chat_id": chat_id,
                        "text": text,
                        "parse_mode": "MarkdownV2",
                        "disable_notification": message.severity == "info",
                    },
                )
        except TelegramError as exc:
            return DeliveryResult(delivered=False, error=str(exc))

        return DeliveryResult(delivered=True, provider_message_id=str(result.get("message_id", "")))

    async def _send_photo(self, chat_id: int, path: Path, caption: str) -> dict[str, Any]:
        try:
            content = await asyncio.to_thread(path.read_bytes)
        except OSError as exc:
            raise TelegramError("Attachment could not be read") from exc

        url = f"{API_ROOT}/bot{self._token}/sendPhoto"
        response = await self._client.post(
            url,
            data={
                "chat_id": str(chat_id),
                "caption": truncate(caption, 1024),
                "parse_mode": "MarkdownV2",
            },
            files={"photo": (path.name, content, "application/octet-stream")},
        )
        if response.status_code != httpx.codes.OK:
            raise TelegramError(f"Telegram rejected sendPhoto: {_describe_error(response)}")
        body = response.json()
        if not body.get("ok"):
            raise TelegramError("Telegram rejected sendPhoto")
        result: dict[str, Any] = body["result"]
        return result

    # ------------------------------------------------------------- approvals
    async def request_approval(self, request: ApprovalRequest) -> DeliveryResult:
        """Send the prompt with two inline buttons.

        The callback data carries only an opaque token. Telegram caps callback
        data at 64 bytes and echoes it back to anyone who can reach the bot, so
        Uses the configured workflow.
        """
        chat_id = self.primary_chat_id
        if chat_id is None:
            return DeliveryResult(delivered=False, suppressed_reason="no_allowed_chat")

        approve = f"ok:{request.callback_token}"
        reject = f"no:{request.callback_token}"
        if max(len(approve), len(reject)) > 64:
            return DeliveryResult(delivered=False, error="callback_token_too_long")

        text = truncate(
            f"*{escape_markdown_v2(request.title)}*\n\n"
            f"{escape_markdown_v2(request.preview)}\n\n"
            f"_{escape_markdown_v2(f'Expires in {request.expires_in_seconds // 60} minutes')}_"
        )

        try:
            result = await self.call(
                "sendMessage",
                {
                    "chat_id": chat_id,
                    "text": text,
                    "parse_mode": "MarkdownV2",
                    "reply_markup": {
                        "inline_keyboard": [
                            [
                                {"text": request.approve_label, "callback_data": approve},
                                {"text": request.reject_label, "callback_data": reject},
                            ]
                        ]
                    },
                },
            )
        except TelegramError as exc:
            return DeliveryResult(delivered=False, error=str(exc))

        return DeliveryResult(delivered=True, provider_message_id=str(result.get("message_id", "")))

    async def resolve_approval_message(
        self, provider_message_id: str, outcome: str, detail: str
    ) -> None:
        """Strip the buttons and stamp the outcome onto the original message."""
        chat_id = self.primary_chat_id
        if chat_id is None or not provider_message_id:
            return
        try:
            await self.call(
                "editMessageText",
                {
                    "chat_id": chat_id,
                    "message_id": int(provider_message_id),
                    "text": truncate(
                        f"*{escape_markdown_v2(outcome.upper())}*\n\n{escape_markdown_v2(detail)}"
                    ),
                    "parse_mode": "MarkdownV2",
                    "reply_markup": {"inline_keyboard": []},
                },
            )
        except (TelegramError, ValueError) as exc:
            # The decision is already recorded; a failed edit is cosmetic.
            logger.warning("Could not update approval message", extra={"error": str(exc)})

    async def answer_callback(self, callback_id: str, text: str) -> None:
        try:
            await self.call(
                "answerCallbackQuery",
                {"callback_query_id": callback_id, "text": text[:200]},
            )
        except TelegramError as exc:
            logger.warning("Could not answer callback query", extra={"error": str(exc)})

    # ---------------------------------------------------------------- health
    async def health(self) -> dict[str, Any]:
        if not self._token:
            return {"ok": False, "detail": "token not configured"}
        try:
            me = await self.call("getMe", {})
        except TelegramError as exc:
            return {"ok": False, "detail": str(exc)}
        return {
            "ok": True,
            "bot_username": me.get("username"),
            "allowed_chats": len(self._chat_ids),
        }

    async def get_updates(self, offset: int | None, timeout_seconds: int) -> list[dict[str, Any]]:
        payload: dict[str, Any] = {
            "timeout": timeout_seconds,
            "allowed_updates": ["message", "callback_query"],
        }
        if offset is not None:
            payload["offset"] = offset

        url = f"{API_ROOT}/bot{self._token}/getUpdates"
        response = await self._client.post(
            url, json=payload, timeout=httpx.Timeout(timeout_seconds + 15)
        )
        if response.status_code != httpx.codes.OK:
            raise TelegramError(f"getUpdates failed: {_describe_error(response)}")
        body = response.json()
        if not body.get("ok"):
            raise TelegramError("getUpdates returned ok=false")
        updates: list[dict[str, Any]] = body.get("result", [])
        return updates


def _retry_after(response: httpx.Response) -> float | None:
    header = response.headers.get("retry-after")
    if header and header.isdigit():
        return float(header)
    try:
        parameters = response.json().get("parameters") or {}
    except (json.JSONDecodeError, ValueError):
        return None
    value = parameters.get("retry_after")
    return float(value) if isinstance(value, int | float) else None


def _describe_error(response: httpx.Response) -> str:
    """Summarise a failure without echoing a URL that contains the token."""
    try:
        description = response.json().get("description")
    except (json.JSONDecodeError, ValueError):
        description = None
    return f"HTTP {response.status_code}: {description or 'no description'}"
