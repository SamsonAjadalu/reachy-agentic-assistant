"""Gmail threaded reply drafts end to end against the mock provider."""

from __future__ import annotations

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings
from approvals.state_machine import clear_executors, resolve_by_token
from database.models import Approval
from integrations.google.executors import register_google_executors
from integrations.google.mock import MockGmailService
from integrations.registry import reset_providers, set_provider_override


@pytest.fixture(autouse=True)
def gmail_provider():
    gmail = MockGmailService()
    set_provider_override("gmail", gmail)
    clear_executors()
    register_google_executors()
    yield gmail
    reset_providers()
    clear_executors()


class TestGmailReplyDraft:
    MESSAGE_ID = "msg-003"
    ORIGINAL_THREAD = "thread-003"
    REPLY_BODY = "Thursday at 8:30 works for me."

    async def _create_reply_draft(self, client: AsyncClient) -> dict:
        response = await client.post(
            f"/api/v1/gmail/messages/{self.MESSAGE_ID}/reply-draft",
            json={"body": self.REPLY_BODY},
        )
        assert response.status_code == 201
        return response.json()["draft"]

    async def test_reply_draft_stays_in_original_thread(
        self, client: AsyncClient, gmail_provider: MockGmailService
    ) -> None:
        draft = await self._create_reply_draft(client)
        assert draft["thread_id"] == self.ORIGINAL_THREAD
        stored = gmail_provider._drafts[draft["id"]]
        assert stored.thread_id == self.ORIGINAL_THREAD

    async def test_reply_subject_is_correct(self, client: AsyncClient) -> None:
        draft = await self._create_reply_draft(client)
        assert draft["subject"] == "Re: Coffee before the lab meeting?"

    async def test_reply_headers_are_correct(self, client: AsyncClient) -> None:
        draft = await self._create_reply_draft(client)
        assert draft["to"] == ["nathan.chen@example.com"]
        assert draft["in_reply_to_message_id"] == self.MESSAGE_ID
        assert draft["in_reply_to"] == "<msg-003@example.com>"
        assert draft["references"] == "<msg-003@example.com>"

    async def test_creating_reply_draft_does_not_send(
        self, client: AsyncClient, gmail_provider: MockGmailService
    ) -> None:
        draft = await self._create_reply_draft(client)
        assert draft["id"]
        assert gmail_provider.sent == []

    async def test_approved_send_replies_in_same_thread(
        self,
        client: AsyncClient,
        session: AsyncSession,
        settings: Settings,
        gmail_provider: MockGmailService,
    ) -> None:
        draft = await self._create_reply_draft(client)
        ticket = (
            await client.post(
                "/api/v1/gmail/actions/send-draft",
                json={"draft_id": draft["id"], "content_hash": draft["content_hash"]},
            )
        ).json()

        approval = await session.get(Approval, ticket["approval_id"])
        assert approval is not None
        await resolve_by_token(
            session, approval.callback_token, approve=True, chat_id=None, settings=settings
        )

        assert len(gmail_provider.sent) == 1
        assert gmail_provider.sent[0]["thread_id"] == self.ORIGINAL_THREAD
        assert gmail_provider.sent[0]["draft_id"] == draft["id"]
        assert gmail_provider.sent[0]["recipients"] == ["nathan.chen@example.com"]
