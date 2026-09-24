"""Notion endpoints against the mock client."""

from __future__ import annotations

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings
from approvals.state_machine import clear_executors, resolve_by_token
from database.models import Approval
from integrations.notion.executors import register_notion_executors
from integrations.notion.mock import (
    PRIVATE_PAGE,
    RESEARCH_PAGE,
    TASKS_DATABASE,
    MockNotionClient,
)
from integrations.registry import reset_providers, set_provider_override


@pytest.fixture(autouse=True)
def notion(settings: Settings):
    client = MockNotionClient(settings)
    set_provider_override("notion", client)
    clear_executors()
    register_notion_executors()
    yield client
    reset_providers()
    clear_executors()


class TestReads:
    async def test_the_allowlist_is_visible(self, client: AsyncClient) -> None:
        body = (await client.get("/api/v1/notion/allowlist")).json()
        assert RESEARCH_PAGE in body["allowed_page_ids"]
        assert PRIVATE_PAGE not in body["allowed_page_ids"]

    async def test_an_allowlisted_page_reads(self, client: AsyncClient) -> None:
        body = (await client.get(f"/api/v1/notion/pages/{RESEARCH_PAGE}")).json()
        assert body["title"] == "Manipulation research log"

    async def test_a_page_outside_the_allowlist_is_refused(self, client: AsyncClient) -> None:
        response = await client.get(f"/api/v1/notion/pages/{PRIVATE_PAGE}")
        assert response.status_code == 403

    async def test_a_malformed_id_is_a_validation_error(self, client: AsyncClient) -> None:
        assert (await client.get("/api/v1/notion/pages/../../secrets")).status_code in {
            403,
            404,
            422,
        }

    async def test_page_content_is_flagged_untrusted(self, client: AsyncClient) -> None:
        body = (await client.get(f"/api/v1/notion/pages/{RESEARCH_PAGE}/content")).json()
        assert body["content_is_untrusted"] is True
        assert "Week 11" in body["text"]

    async def test_a_database_can_be_queried(self, client: AsyncClient) -> None:
        body = (await client.get(f"/api/v1/notion/databases/{TASKS_DATABASE}/rows")).json()
        assert body["total"] == 2
        assert body["items"][0]["properties"]["Status"] == "In progress"

    async def test_search_only_returns_allowlisted_pages(self, client: AsyncClient) -> None:
        body = (await client.get("/api/v1/notion/search?q=finances")).json()
        assert body["total"] == 0


class TestApprovalGatedWrites:
    async def test_appending_creates_a_ticket_not_a_block(
        self, client: AsyncClient, notion: MockNotionClient
    ) -> None:
        response = await client.post(
            f"/api/v1/notion/pages/{RESEARCH_PAGE}/append",
            json={"page_id": RESEARCH_PAGE, "text": "Grasp success improved to 94%."},
        )

        assert response.status_code == 202
        assert notion.appended == []
        assert "Manipulation research log" in response.json()["summary"]

    async def test_approving_appends_once(
        self,
        client: AsyncClient,
        session: AsyncSession,
        settings: Settings,
        notion: MockNotionClient,
    ) -> None:
        ticket = (
            await client.post(
                f"/api/v1/notion/pages/{RESEARCH_PAGE}/append",
                json={"page_id": RESEARCH_PAGE, "text": "Grasp success improved to 94%."},
            )
        ).json()
        approval = await session.get(Approval, ticket["approval_id"])
        assert approval is not None

        await resolve_by_token(
            session, approval.callback_token, approve=True, chat_id=None, settings=settings
        )
        await resolve_by_token(
            session, approval.callback_token, approve=True, chat_id=None, settings=settings
        )

        assert len(notion.appended) == 1

    async def test_appending_to_a_blocked_page_never_reaches_approval(
        self, client: AsyncClient, session: AsyncSession
    ) -> None:
        response = await client.post(
            f"/api/v1/notion/pages/{PRIVATE_PAGE}/append",
            json={"page_id": PRIVATE_PAGE, "text": "note"},
        )
        assert response.status_code == 403
        assert (await session.scalars(Approval.__table__.select())).first() is None

    async def test_empty_text_is_refused(self, client: AsyncClient) -> None:
        response = await client.post(
            f"/api/v1/notion/pages/{RESEARCH_PAGE}/append",
            json={"page_id": RESEARCH_PAGE, "text": "   "},
        )
        assert response.status_code == 422


class TestAuthentication:
    async def test_reads_need_a_token(self, anonymous_client: AsyncClient) -> None:
        assert (await anonymous_client.get("/api/v1/notion/allowlist")).status_code == 401
