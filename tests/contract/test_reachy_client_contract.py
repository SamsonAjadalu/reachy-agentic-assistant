"""Contract tests: async client against the real FastAPI app via ASGI transport."""

from __future__ import annotations

from datetime import timedelta

import httpx
import pytest
from httpx import ASGITransport

from reachy_client.client import AsyncPersonalAssistantClient
from reachy_client.models import (
    CreateDraftRequest,
    CreateReplyDraftRequest,
    Delay,
    ReminderCreate,
    ReminderSnooze,
    ScheduleInput,
    SendDraftRequest,
    TaskCreate,
    TaskUpdate,
)
from shared.timeutils import utcnow
from tests.conftest import TEST_API_TOKEN


@pytest.fixture
async def pa_client(app) -> AsyncPersonalAssistantClient:
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as http:
        client = AsyncPersonalAssistantClient("http://testserver", TEST_API_TOKEN)
        client._client = http
        yield client


class TestSystem:
    async def test_ping(self, pa_client: AsyncPersonalAssistantClient) -> None:
        result = await pa_client.ping()
        assert result.pong is True
        assert "reachable" in result.message.lower()

    async def test_status(self, pa_client: AsyncPersonalAssistantClient) -> None:
        result = await pa_client.status()
        assert result.mock_mode is True
        assert result.database["ok"] is True


class TestRemindersAndTasks:
    async def test_create_and_list_reminder(self, pa_client: AsyncPersonalAssistantClient) -> None:
        trigger = utcnow() + timedelta(hours=2)
        created = await pa_client.create_reminder(
            ReminderCreate(text="Contract test", schedule=ScheduleInput(at=trigger)),
            idempotency_key="contract-reminder-1",
        )
        assert created.text == "Contract test"

        listed = await pa_client.list_reminders(upcoming_only=True, limit=10)
        assert any(item.id == created.id for item in listed.items)

        snoozed = await pa_client.snooze_reminder(
            created.id, ReminderSnooze(delay=Delay(minutes=15))
        )
        assert snoozed.snooze_count == 1

    async def test_create_list_complete_task(self, pa_client: AsyncPersonalAssistantClient) -> None:
        created = await pa_client.create_task(
            TaskCreate(description="Contract task"),
            idempotency_key="contract-task-1",
        )
        listed = await pa_client.list_tasks(status="open", limit=20)
        assert any(item.id == created.id for item in listed.items)

        updated = await pa_client.update_task(
            created.id, TaskUpdate(description="Updated contract task")
        )
        assert updated.description == "Updated contract task"

        completed = await pa_client.complete_task(created.id)
        assert completed.status in {"completed", "open"}


class TestIntegrations:
    async def test_weather(self, pa_client: AsyncPersonalAssistantClient) -> None:
        result = await pa_client.get_weather()
        assert result.report.location
        assert result.advice.summary

    async def test_documents_search(self, pa_client: AsyncPersonalAssistantClient) -> None:
        result = await pa_client.search_documents("anything", limit=5)
        assert result.query == "anything"

    async def test_workstation_status(self, pa_client: AsyncPersonalAssistantClient) -> None:
        result = await pa_client.get_workstation_status()
        assert result.status.hostname
        assert result.spoken_summary

    async def test_pending_notifications(self, pa_client: AsyncPersonalAssistantClient) -> None:
        result = await pa_client.list_pending_notifications()
        assert result.total >= 0

    async def test_gmail_search_mock(self, pa_client: AsyncPersonalAssistantClient) -> None:
        result = await pa_client.search_gmail("in:inbox", limit=3)
        assert result.total >= 0

    async def test_calendar_events(self, pa_client: AsyncPersonalAssistantClient) -> None:
        result = await pa_client.get_calendar_events(days=3)
        assert result.total >= 0

    async def test_contacts_search(self, pa_client: AsyncPersonalAssistantClient) -> None:
        result = await pa_client.search_contacts("test")
        assert result.total >= 0

    async def test_notion_search(self, pa_client: AsyncPersonalAssistantClient) -> None:
        result = await pa_client.search_notion("Manipulation")
        assert result.total >= 0
        if result.items:
            content = await pa_client.read_notion_page(result.items[0].id, max_chars=1000)
            assert content.page.id == result.items[0].id
            assert content.content_is_untrusted is True

    async def test_wardrobe(self, pa_client: AsyncPersonalAssistantClient) -> None:
        items = await pa_client.search_wardrobe(limit=5)
        assert items.total >= 0
        rec = await pa_client.recommend_outfit(limit=1)
        assert rec.considered >= 0

    async def test_action_status_lookup(self, pa_client: AsyncPersonalAssistantClient) -> None:
        draft = await pa_client.create_gmail_draft(
            CreateDraftRequest(
                to=["nathan.chen@example.com"],
                subject="Contract status",
                body="Status lookup test.",
            )
        )
        ticket = await pa_client.send_existing_draft(
            SendDraftRequest(draft_id=draft.draft.id, content_hash=draft.draft.content_hash)
        )
        status = await pa_client.get_action_status(ticket.action_id)
        assert status.status == "awaiting_approval"
        assert status.action_type == "google.gmail.send_draft"
        assert status.approval_id == ticket.approval_id

    async def test_gmail_reply_draft(self, pa_client: AsyncPersonalAssistantClient) -> None:
        draft = await pa_client.create_gmail_reply_draft(
            "msg-003",
            CreateReplyDraftRequest(body="Thursday at 8:30 works."),
        )
        assert draft.draft.thread_id == "thread-003"
        assert draft.draft.subject.startswith("Re:")
        assert draft.draft.in_reply_to == "<msg-003@example.com>"
