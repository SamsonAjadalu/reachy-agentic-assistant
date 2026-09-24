"""Unit tests for the Reachy text-turn HTTP client."""

from __future__ import annotations

import httpx
import pytest
import respx

from app.config import Settings
from integrations.reachy.text_turn import (
    ReachyTextTurnClient,
    TextTurnMalformedResponse,
    TextTurnTimeout,
    TextTurnUnavailable,
)


@pytest.fixture
def settings(settings: Settings) -> Settings:
    settings.reachy_text_turn_enabled = True
    settings.reachy_text_turn_url = "http://reachy.test/api/v1/text-turn"
    settings.reachy_text_turn_token = type(settings.reachy_text_turn_token)("reachy-test-token")
    settings.reachy_text_turn_timeout_seconds = 5
    return settings


class TestReachyTextTurnClient:
    @respx.mock
    async def test_success_returns_assistant_text(self, settings: Settings) -> None:
        route = respx.post("http://reachy.test/api/v1/text-turn").mock(
            return_value=httpx.Response(
                200,
                json={"ok": True, "assistant_text": "Tomorrow you have lab meeting."},
            )
        )
        client = ReachyTextTurnClient(settings)
        try:
            text = await client.submit_turn(text="calendar tomorrow", turn_id="turn-1")
        finally:
            await client.aclose()

        assert text == "Tomorrow you have lab meeting."
        body = route.calls.last.request.read()
        assert b'"text":"calendar tomorrow"' in body
        assert b'"turn_id":"turn-1"' in body
        assert route.calls.last.request.headers["authorization"] == "Bearer reachy-test-token"

    @respx.mock
    async def test_timeout_is_mapped(self, settings: Settings) -> None:
        respx.post("http://reachy.test/api/v1/text-turn").mock(
            side_effect=httpx.TimeoutException("timed out")
        )
        client = ReachyTextTurnClient(settings)
        try:
            with pytest.raises(TextTurnTimeout):
                await client.submit_turn(text="hello", turn_id="turn-2")
        finally:
            await client.aclose()

    @respx.mock
    async def test_connect_error_is_unavailable(self, settings: Settings) -> None:
        respx.post("http://reachy.test/api/v1/text-turn").mock(
            side_effect=httpx.ConnectError("connection refused")
        )
        client = ReachyTextTurnClient(settings)
        try:
            with pytest.raises(TextTurnUnavailable):
                await client.submit_turn(text="hello", turn_id="turn-3")
        finally:
            await client.aclose()

    @respx.mock
    async def test_missing_assistant_text_is_malformed(self, settings: Settings) -> None:
        respx.post("http://reachy.test/api/v1/text-turn").mock(
            return_value=httpx.Response(200, json={"ok": True})
        )
        client = ReachyTextTurnClient(settings)
        try:
            with pytest.raises(TextTurnMalformedResponse):
                await client.submit_turn(text="hello", turn_id="turn-4")
        finally:
            await client.aclose()

    @respx.mock
    async def test_unauthorized_is_unavailable(self, settings: Settings) -> None:
        respx.post("http://reachy.test/api/v1/text-turn").mock(
            return_value=httpx.Response(401, json={"ok": False, "error": "unauthorized"})
        )
        client = ReachyTextTurnClient(settings)
        try:
            with pytest.raises(TextTurnUnavailable):
                await client.submit_turn(text="hello", turn_id="turn-5")
        finally:
            await client.aclose()
