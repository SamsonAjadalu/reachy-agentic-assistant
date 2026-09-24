"""Long-polling listener for Telegram callbacks and commands.

Polling rather than webhooks, deliberately: the workstation sits behind a home
router with no inbound path from the public internet, and a webhook would mean
exposing a port and terminating TLS for one bot.

Everything arriving here is untrusted. An update is acted on only when its chat
Uses the configured workflow.
interpreted as an instruction, only matched against a fixed command table or
forwarded verbatim to the Reachy Pi text-turn endpoint when enabled.
"""

from __future__ import annotations

import asyncio
import contextlib
import uuid
from collections import deque
from typing import Any

from app.config import Settings, get_settings
from app.logging_config import correlation_id_var, get_logger
from approvals.state_machine import resolve_by_token
from database.session import session_scope
from integrations.reachy.text_turn import (
    ReachyTextTurnClient,
    TextTurnError,
    TextTurnMalformedResponse,
    TextTurnTimeout,
    TextTurnUnavailable,
)
from integrations.telegram.real import TelegramChannel, escape_markdown_v2, truncate
from shared.timeutils import utcnow

logger = get_logger(__name__)

POLL_TIMEOUT_SECONDS = 25
ERROR_BACKOFF_SECONDS = 5
MAX_ERROR_BACKOFF_SECONDS = 120
MAX_REMEMBERED_UPDATE_IDS = 2048

APPROVE_PREFIX = "ok:"
REJECT_PREFIX = "no:"

_DECISION_REPLY = {
    "approved": "Done.",
    "rejected": "Cancelled.",
    "already_resolved": "That was already answered.",
    "expired": "That request expired.",
    "unknown_or_expired": "That request is no longer available.",
    "not_authorised": "Not permitted.",
}

_REACHY_FAILURE_REPLY = {
    "timeout": "Reachy is taking too long to answer. Please try again in a moment.",
    "unavailable": (
        "Reachy is unavailable right now. Check that the Pi conversation app is running."
    ),
    "malformed": "Reachy sent back an unreadable reply. Please try again.",
    "error": "Reachy could not answer that message. Please try again.",
}


class TelegramListener:
    """Consumes updates and applies approval decisions."""

    def __init__(
        self,
        settings: Settings | None = None,
        channel: TelegramChannel | None = None,
        reachy_client: ReachyTextTurnClient | None = None,
    ):
        self._settings = settings or get_settings()
        self._channel = channel or TelegramChannel(self._settings)
        self._reachy_client = reachy_client
        self._offset: int | None = None
        self._task: asyncio.Task[None] | None = None
        self._stopping = asyncio.Event()
        self._processed_update_ids: set[int] = set()
        self._processed_update_order: deque[int] = deque(maxlen=MAX_REMEMBERED_UPDATE_IDS)
        self.updates_seen = 0
        self.updates_rejected = 0

    async def start(self) -> None:
        if self._task is not None:
            return
        self._stopping.clear()
        self._task = asyncio.create_task(self._run(), name="telegram-listener")
        logger.info("Telegram listener started")

    async def stop(self) -> None:
        self._stopping.set()
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        if self._reachy_client is not None:
            await self._reachy_client.aclose()
        await self._channel.aclose()
        logger.info("Telegram listener stopped")

    async def _run(self) -> None:
        backoff = ERROR_BACKOFF_SECONDS
        while not self._stopping.is_set():
            try:
                updates = await self._channel.get_updates(self._offset, POLL_TIMEOUT_SECONDS)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # the poll loop must outlive any single failure
                logger.warning(
                    "Telegram polling failed; backing off",
                    extra={"error": type(exc).__name__, "backoff_seconds": backoff},
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, MAX_ERROR_BACKOFF_SECONDS)
                continue

            backoff = ERROR_BACKOFF_SECONDS
            for update in updates:
                # Advance the offset even for updates we refuse, otherwise one
                # bad update would be replayed forever.
                self._offset = int(update["update_id"]) + 1
                try:
                    await self.handle_update(update)
                except Exception:
                    logger.exception("Failed to handle a Telegram update")

    def _remember_update_id(self, update_id: int) -> None:
        if update_id in self._processed_update_ids:
            return
        self._processed_update_ids.add(update_id)
        self._processed_update_order.append(update_id)
        while len(self._processed_update_order) > MAX_REMEMBERED_UPDATE_IDS:
            old = self._processed_update_order.popleft()
            self._processed_update_ids.discard(old)

    async def handle_update(self, update: dict[str, Any]) -> dict[str, Any]:
        """Route one update. Returns a small summary for tests and diagnostics."""
        update_id = update.get("update_id")
        if isinstance(update_id, int):
            if update_id in self._processed_update_ids:
                return {"handled": False, "reason": "duplicate_update"}
            self._remember_update_id(update_id)

        self.updates_seen += 1
        if "callback_query" in update:
            return await self._handle_callback(update["callback_query"])
        if "message" in update:
            return await self._handle_message(update["message"])
        return {"handled": False, "reason": "unsupported_update"}

    async def _handle_callback(self, callback: dict[str, Any]) -> dict[str, Any]:
        chat_id = _chat_id_of(callback.get("message", {}))
        data = str(callback.get("data") or "")
        callback_id = str(callback.get("id") or "")

        if not self._channel.is_allowed_chat(chat_id):
            self.updates_rejected += 1
            logger.warning("Callback from a chat outside the allowlist", extra={"chat": chat_id})
            await self._channel.answer_callback(callback_id, "Not permitted.")
            return {"handled": False, "reason": "not_authorised"}

        if data.startswith(APPROVE_PREFIX):
            approve, token = True, data[len(APPROVE_PREFIX) :]
        elif data.startswith(REJECT_PREFIX):
            approve, token = False, data[len(REJECT_PREFIX) :]
        else:
            self.updates_rejected += 1
            await self._channel.answer_callback(callback_id, "Unrecognised action.")
            return {"handled": False, "reason": "unrecognised_callback"}

        token_context = correlation_id_var.set(f"tg-{callback_id}")
        try:
            async with session_scope() as session:
                result = await resolve_by_token(
                    session,
                    token,
                    approve=approve,
                    chat_id=chat_id,
                    settings=self._settings,
                )
        finally:
            correlation_id_var.reset(token_context)

        reply_key = str(result.get("decision") or result.get("reason") or "")
        await self._channel.answer_callback(
            callback_id, _DECISION_REPLY.get(reply_key, "Received.")
        )
        return {"handled": True, **result}

    async def _handle_message(self, message: dict[str, Any]) -> dict[str, Any]:
        chat_id = _chat_id_of(message)
        text = str(message.get("text") or "").strip()

        if not self._channel.is_allowed_chat(chat_id):
            self.updates_rejected += 1
            logger.warning(
                "Message from a chat outside the allowlist",
                extra={"chat": chat_id, "length": len(text)},
            )
            return {"handled": False, "reason": "not_authorised"}

        command = text.split()[0].lower().split("@")[0] if text else ""
        handler = _COMMANDS.get(command)
        if handler is not None:
            reply = await handler(self)
            await self._channel.call(
                "sendMessage",
                {
                    "chat_id": chat_id,
                    "text": escape_markdown_v2(reply),
                    "parse_mode": "MarkdownV2",
                },
            )
            return {"handled": True, "command": command}

        if not text:
            return {"handled": False, "reason": "no_text"}

        if not self._settings.reachy_text_turn_enabled:
            return {"handled": False, "reason": "no_command"}

        assert chat_id is not None
        return await self._handle_reachy_chat(message, chat_id=chat_id, text=text)

    async def _handle_reachy_chat(
        self, message: dict[str, Any], *, chat_id: int, text: str
    ) -> dict[str, Any]:
        turn_id = str(uuid.uuid4())
        message_id = message.get("message_id")
        token_context = correlation_id_var.set(turn_id)
        failure_key = "error"
        assistant_text = ""

        try:
            client = self._reachy_client or ReachyTextTurnClient(self._settings)
            assistant_text = await client.submit_turn(text=text, turn_id=turn_id)
        except TextTurnTimeout:
            failure_key = "timeout"
        except TextTurnUnavailable:
            failure_key = "unavailable"
        except TextTurnMalformedResponse:
            failure_key = "malformed"
        except TextTurnError:
            failure_key = "error"
        except Exception:
            logger.exception("Unexpected Reachy text-turn failure", extra={"turn_id": turn_id})
            failure_key = "error"
        finally:
            correlation_id_var.reset(token_context)

        reply = assistant_text or _REACHY_FAILURE_REPLY[failure_key]
        payload: dict[str, Any] = {
            "chat_id": chat_id,
            "text": escape_markdown_v2(truncate(reply)),
            "parse_mode": "MarkdownV2",
        }
        if isinstance(message_id, int):
            payload["reply_to_message_id"] = message_id

        await self._channel.call("sendMessage", payload)
        return {
            "handled": True,
            "reason": "reachy_chat" if assistant_text else f"reachy_{failure_key}",
            "turn_id": turn_id,
        }


async def _command_status(listener: TelegramListener) -> str:
    from sqlalchemy import func, select

    from database.models import Approval, BackgroundTask, Reminder
    from shared.enums import ApprovalStatus, BackgroundTaskStatus, ReminderStatus

    async with session_scope() as session:
        pending_approvals = await session.scalar(
            select(func.count())
            .select_from(Approval)
            .where(Approval.status == ApprovalStatus.PENDING.value)
        )
        active_tasks = await session.scalar(
            select(func.count())
            .select_from(BackgroundTask)
            .where(
                BackgroundTask.status.in_(
                    [BackgroundTaskStatus.QUEUED.value, BackgroundTaskStatus.RUNNING.value]
                )
            )
        )
        upcoming = await session.scalar(
            select(func.count())
            .select_from(Reminder)
            .where(
                Reminder.status.in_([ReminderStatus.ACTIVE.value, ReminderStatus.SNOOZED.value]),
                Reminder.trigger_at > utcnow(),
            )
        )

    return (
        "Assistant status\n"
        f"Pending approvals: {pending_approvals or 0}\n"
        f"Active background tasks: {active_tasks or 0}\n"
        f"Upcoming reminders: {upcoming or 0}"
    )


async def _command_pending(listener: TelegramListener) -> str:
    from sqlalchemy import select

    from database.models import PendingAction
    from shared.enums import PendingActionStatus

    async with session_scope() as session:
        rows = (
            await session.scalars(
                select(PendingAction)
                .where(PendingAction.status == PendingActionStatus.PENDING.value)
                .order_by(PendingAction.created_at.desc())
                .limit(10)
            )
        ).all()

    if not rows:
        return "Nothing is waiting for approval."
    lines = [f"- {row.summary} ({row.action_type})" for row in rows]
    return "Waiting for approval:\n" + "\n".join(lines)


async def _command_help(listener: TelegramListener) -> str:
    reachy_line = (
        "Plain text is forwarded to Reachy when text-turn is enabled.\n"
        if listener._settings.reachy_text_turn_enabled
        else ""
    )
    return (
        "Commands\n"
        "/status - queue and reminder counts\n"
        "/pending - actions waiting for approval\n"
        "/help - this list\n\n"
        f"{reachy_line}"
        "Approvals are answered with the buttons on each request."
    )


_COMMANDS = {
    "/status": _command_status,
    "/pending": _command_pending,
    "/help": _command_help,
    "/start": _command_help,
}


def _chat_id_of(message: dict[str, Any]) -> int | None:
    chat = message.get("chat") or {}
    value = chat.get("id")
    return int(value) if isinstance(value, int) else None
