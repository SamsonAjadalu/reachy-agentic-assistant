"""Approval/action status lookup for Reachy polling."""

from __future__ import annotations

from datetime import timedelta

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings
from approvals.state_machine import clear_executors, resolve_by_token
from database.models import Approval, PendingAction
from integrations.google.executors import register_google_executors
from integrations.google.mock import MockGmailService
from integrations.registry import reset_providers, set_provider_override
from shared.enums import ApprovalStatus, PendingActionStatus
from shared.timeutils import utcnow


@pytest.fixture(autouse=True)
def providers():
    gmail = MockGmailService()
    set_provider_override("gmail", gmail)
    clear_executors()
    register_google_executors()
    yield {"gmail": gmail}
    reset_providers()
    clear_executors()


async def _request_send_draft(client: AsyncClient) -> dict[str, str]:
    draft = (
        await client.post(
            "/api/v1/gmail/drafts",
            json={
                "to": ["nathan.chen@example.com"],
                "subject": "Status check",
                "body": "Checking approval status.",
            },
        )
    ).json()["draft"]
    ticket = (
        await client.post(
            "/api/v1/gmail/actions/send-draft",
            json={"draft_id": draft["id"], "content_hash": draft["content_hash"]},
        )
    ).json()
    return ticket


class TestActionStatusLookup:
    async def test_pending_send_draft_reports_awaiting_approval(self, client: AsyncClient) -> None:
        ticket = await _request_send_draft(client)

        by_action = (await client.get(f"/api/v1/actions/{ticket['action_id']}/status")).json()
        by_approval = (await client.get(f"/api/v1/approvals/{ticket['approval_id']}/status")).json()

        assert by_action == by_approval
        assert by_action["status"] == "awaiting_approval"
        assert by_action["action_type"] == "google.gmail.send_draft"
        assert by_action["result_summary"] == "Waiting for your approval in Telegram."
        assert "preview" not in by_action
        assert "payload_hash" not in by_action
        assert "payload" not in by_action

    async def test_rejected_send_draft_reports_rejected(
        self,
        client: AsyncClient,
        session: AsyncSession,
        settings: Settings,
    ) -> None:
        ticket = await _request_send_draft(client)
        approval = await session.get(Approval, ticket["approval_id"])
        assert approval is not None
        await resolve_by_token(
            session, approval.callback_token, approve=False, chat_id=None, settings=settings
        )
        await session.commit()

        body = (await client.get(f"/api/v1/actions/{ticket['action_id']}/status")).json()
        assert body["status"] == "rejected"
        assert body["rejected_at"] is not None
        assert body["executed_at"] is None
        assert "Nothing was sent" in body["result_summary"]

    async def test_approved_send_draft_reports_sent(
        self,
        client: AsyncClient,
        session: AsyncSession,
        settings: Settings,
        providers: dict,
    ) -> None:
        ticket = await _request_send_draft(client)
        approval = await session.get(Approval, ticket["approval_id"])
        assert approval is not None
        await resolve_by_token(
            session, approval.callback_token, approve=True, chat_id=None, settings=settings
        )
        await session.commit()

        body = (await client.get(f"/api/v1/actions/{ticket['action_id']}/status")).json()
        assert body["status"] == "sent"
        assert body["approved_at"] is not None
        assert body["executed_at"] is not None
        assert "sent" in body["result_summary"].lower()
        assert len(providers["gmail"].sent) == 1

    async def test_failed_send_draft_reports_failed(
        self,
        client: AsyncClient,
        session: AsyncSession,
        settings: Settings,
        providers: dict,
    ) -> None:
        draft = (
            await client.post(
                "/api/v1/gmail/drafts",
                json={
                    "to": ["nathan.chen@example.com"],
                    "subject": "Status check",
                    "body": "Checking approval status.",
                },
            )
        ).json()["draft"]
        ticket = (
            await client.post(
                "/api/v1/gmail/actions/send-draft",
                json={"draft_id": draft["id"], "content_hash": draft["content_hash"]},
            )
        ).json()
        stored = providers["gmail"]._drafts[draft["id"]]
        stored.body = "changed after approval"
        stored.content_hash = "deadbeef" * 8

        approval = await session.get(Approval, ticket["approval_id"])
        assert approval is not None
        await resolve_by_token(
            session, approval.callback_token, approve=True, chat_id=None, settings=settings
        )
        await session.commit()

        body = (await client.get(f"/api/v1/actions/{ticket['action_id']}/status")).json()
        assert body["status"] == "failed"
        assert body["approved_at"] is not None
        assert body["executed_at"] is None
        assert "could not be completed" in body["result_summary"]

    async def test_expired_pending_reports_expired(
        self, client: AsyncClient, session: AsyncSession
    ) -> None:
        ticket = await _request_send_draft(client)
        approval = await session.get(Approval, ticket["approval_id"])
        action = await session.get(PendingAction, ticket["action_id"])
        assert approval is not None and action is not None
        past = utcnow() - timedelta(minutes=5)
        approval.expires_at = past
        await session.commit()

        body = (await client.get(f"/api/v1/actions/{ticket['action_id']}/status")).json()
        assert body["status"] == "expired"
        assert "expired" in body["result_summary"].lower()

    async def test_executing_reports_executing(
        self, client: AsyncClient, session: AsyncSession
    ) -> None:
        ticket = await _request_send_draft(client)
        approval = await session.get(Approval, ticket["approval_id"])
        action = await session.get(PendingAction, ticket["action_id"])
        assert approval is not None and action is not None
        approval.status = ApprovalStatus.APPROVED.value
        approval.resolved_at = utcnow()
        action.status = PendingActionStatus.EXECUTING.value
        await session.commit()

        body = (await client.get(f"/api/v1/actions/{ticket['action_id']}/status")).json()
        assert body["status"] == "executing"
        assert body["approved_at"] is not None
        assert "in progress" in body["result_summary"].lower()

    async def test_unknown_action_id_is_404(self, client: AsyncClient) -> None:
        response = await client.get("/api/v1/actions/does-not-exist/status")
        assert response.status_code == 404

    async def test_unknown_approval_id_is_404(self, client: AsyncClient) -> None:
        response = await client.get("/api/v1/approvals/does-not-exist/status")
        assert response.status_code == 404
