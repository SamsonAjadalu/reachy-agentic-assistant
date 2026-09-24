"""Telegram ↔ Reachy text-turn integration tests."""

from __future__ import annotations

from typing import Any

import httpx
import pytest
import respx

from app.config import Settings
from approvals.state_machine import (
    clear_executors,
    register_executor,
    request_approval,
)
from integrations.reachy.text_turn import (
    TextTurnMalformedResponse,
    TextTurnTimeout,
    TextTurnUnavailable,
)
from integrations.telegram.listener import TelegramListener
from integrations.telegram.real import API_ROOT, TelegramChannel
from shared.enums import RiskLevel

FAKE_TOKEN = "7654321098:AAFakeTokenValueForTestsOnly-xyz123"
CHAT_ID = 424242


def api(method: str) -> str:
    return f"{API_ROOT}/bot{FAKE_TOKEN}/{method}"


def ok(result: Any) -> httpx.Response:
    return httpx.Response(200, json={"ok": True, "result": result})


class FakeReachyClient:
    def __init__(
        self,
        *,
        reply: str = "Tomorrow you have lab meeting at 2pm.",
        error: Exception | None = None,
    ) -> None:
        self.reply = reply
        self.error = error
        self.calls: list[dict[str, str]] = []

    async def submit_turn(self, *, text: str, turn_id: str) -> str:
        self.calls.append({"text": text, "turn_id": turn_id})
        if self.error is not None:
            raise self.error
        return self.reply

    async def aclose(self) -> None:
        return None


@pytest.fixture
def settings(settings: Settings) -> Settings:
    settings.telegram_enabled = True
    settings.telegram_bot_token = type(settings.telegram_bot_token)(FAKE_TOKEN)
    settings.telegram_allowed_chat_ids = [CHAT_ID]
    settings.reachy_text_turn_enabled = True
    settings.reachy_text_turn_url = "http://reachy.test/api/v1/text-turn"
    settings.reachy_text_turn_token = type(settings.reachy_text_turn_token)("reachy-test-token")
    return settings


@pytest.fixture
def channel(settings: Settings) -> TelegramChannel:
    return TelegramChannel(settings)


@pytest.fixture
def reachy() -> FakeReachyClient:
    return FakeReachyClient()


@pytest.fixture
def listener(
    settings: Settings, channel: TelegramChannel, reachy: FakeReachyClient
) -> TelegramListener:
    return TelegramListener(settings, channel, reachy_client=reachy)


class TestTelegramReachyChat:
    @respx.mock
    async def test_normal_message_forwards_to_reachy_and_replies(
        self, listener: TelegramListener, reachy: FakeReachyClient
    ) -> None:
        route = respx.post(api("sendMessage")).mock(return_value=ok({"message_id": 42}))

        result = await listener.handle_update(
            {
                "update_id": 100,
                "message": {
                    "chat": {"id": CHAT_ID},
                    "message_id": 17,
                    "text": "What is on my calendar tomorrow?",
                },
            }
        )

        assert result["handled"] is True
        assert result["reason"] == "reachy_chat"
        assert len(reachy.calls) == 1
        assert reachy.calls[0]["text"] == "What is on my calendar tomorrow?"
        assert reachy.calls[0]["turn_id"] == result["turn_id"]
        body = route.calls.last.request.content.decode()
        assert "lab meeting" in body
        assert '"reply_to_message_id": 17' in body or '"reply_to_message_id":17' in body

    @respx.mock
    async def test_duplicate_update_is_ignored(
        self, listener: TelegramListener, reachy: FakeReachyClient
    ) -> None:
        respx.post(api("sendMessage")).mock(return_value=ok({"message_id": 42}))
        update = {
            "update_id": 101,
            "message": {"chat": {"id": CHAT_ID}, "message_id": 18, "text": "Hello"},
        }

        first = await listener.handle_update(update)
        second = await listener.handle_update(update)

        assert first["handled"] is True
        assert second == {"handled": False, "reason": "duplicate_update"}
        assert len(reachy.calls) == 1

    @respx.mock
    async def test_unauthorized_chat_is_rejected(
        self, listener: TelegramListener, reachy: FakeReachyClient
    ) -> None:
        route = respx.post(api("sendMessage")).mock(return_value=ok({"message_id": 42}))

        result = await listener.handle_update(
            {
                "update_id": 102,
                "message": {"chat": {"id": 999999}, "message_id": 19, "text": "Hello"},
            }
        )

        assert result["reason"] == "not_authorised"
        assert reachy.calls == []
        assert route.call_count == 0
        assert listener.updates_rejected == 1

    @respx.mock
    async def test_pi_timeout_returns_clear_failure(
        self, settings: Settings, channel: TelegramChannel
    ) -> None:
        reachy = FakeReachyClient(error=TextTurnTimeout("timed out"))
        listener = TelegramListener(settings, channel, reachy_client=reachy)
        route = respx.post(api("sendMessage")).mock(return_value=ok({"message_id": 43}))

        result = await listener.handle_update(
            {
                "update_id": 103,
                "message": {"chat": {"id": CHAT_ID}, "message_id": 20, "text": "Hello"},
            }
        )

        assert result["reason"] == "reachy_timeout"
        assert "too long" in route.calls.last.request.content.decode()

    @respx.mock
    async def test_pi_unavailable_returns_clear_failure(
        self, settings: Settings, channel: TelegramChannel
    ) -> None:
        reachy = FakeReachyClient(error=TextTurnUnavailable("down"))
        listener = TelegramListener(settings, channel, reachy_client=reachy)
        route = respx.post(api("sendMessage")).mock(return_value=ok({"message_id": 44}))

        result = await listener.handle_update(
            {
                "update_id": 104,
                "message": {"chat": {"id": CHAT_ID}, "message_id": 21, "text": "Hello"},
            }
        )

        assert result["reason"] == "reachy_unavailable"
        assert "unavailable" in route.calls.last.request.content.decode()

    @respx.mock
    async def test_malformed_response_returns_clear_failure(
        self, settings: Settings, channel: TelegramChannel
    ) -> None:
        reachy = FakeReachyClient(error=TextTurnMalformedResponse("bad"))
        listener = TelegramListener(settings, channel, reachy_client=reachy)
        route = respx.post(api("sendMessage")).mock(return_value=ok({"message_id": 45}))

        result = await listener.handle_update(
            {
                "update_id": 105,
                "message": {"chat": {"id": CHAT_ID}, "message_id": 22, "text": "Hello"},
            }
        )

        assert result["reason"] == "reachy_malformed"
        assert "unreadable" in route.calls.last.request.content.decode()


class TestApprovalCallbacksStillWork:
    @pytest.fixture(autouse=True)
    def _executors(self):
        clear_executors()
        yield
        clear_executors()

    @pytest.fixture
    def performed(self) -> list[dict[str, Any]]:
        calls: list[dict[str, Any]] = []

        @register_executor("test.action")
        async def _run(session, payload: dict[str, Any], settings) -> dict[str, Any]:
            calls.append(payload)
            return {"ok": True}

        return calls

    async def _pending(self, session, settings: Settings) -> str:
        _, approval = await request_approval(
            session,
            action_type="test.action",
            summary="Do the thing",
            payload={"value": 1},
            preview_text="Does the thing",
            risk_level=RiskLevel.EXTERNAL_WRITE,
            settings=settings,
        )
        await session.commit()
        return approval.callback_token

    @respx.mock
    async def test_approval_callback_still_executes(
        self,
        session,
        settings: Settings,
        channel: TelegramChannel,
        reachy: FakeReachyClient,
        performed: list,
    ) -> None:
        respx.post(api("sendMessage")).mock(return_value=ok({"message_id": 1}))
        respx.post(api("editMessageText")).mock(return_value=ok({"message_id": 1}))
        respx.post(api("answerCallbackQuery")).mock(return_value=ok(True))
        token = await self._pending(session, settings)
        listener = TelegramListener(settings, channel, reachy_client=reachy)

        result = await listener.handle_update(
            {
                "update_id": 200,
                "callback_query": {
                    "id": "cb1",
                    "data": f"ok:{token}",
                    "message": {"chat": {"id": CHAT_ID}},
                },
            }
        )

        assert result["decision"] == "approved"
        assert len(performed) == 1
        assert reachy.calls == []
