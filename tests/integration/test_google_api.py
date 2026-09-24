"""Google endpoints end to end against the mock providers.

Mock mode is the default in tests, which is the point: the whole surface -
reading mail, listing events, resolving contacts, sending an approved email -
works with no Google account, so these paths are exercised on every run.
"""

from __future__ import annotations

from datetime import timedelta

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings
from approvals.state_machine import clear_executors, resolve_by_token
from database.models import Approval, Contact, ContactAlias, PendingAction
from integrations.google.executors import register_google_executors
from integrations.google.mock import (
    ANCHOR,
    MockCalendarService,
    MockContactsService,
    MockDriveService,
    MockGmailService,
)
from integrations.registry import reset_providers, set_provider_override
from shared.enums import PendingActionStatus


@pytest.fixture(autouse=True)
def providers():
    """Uses the configured workflow."""
    gmail = MockGmailService()
    calendar = MockCalendarService()
    contacts = MockContactsService()
    drive = MockDriveService()
    set_provider_override("gmail", gmail)
    set_provider_override("calendar", calendar)
    set_provider_override("contacts", contacts)
    set_provider_override("drive", drive)

    clear_executors()
    register_google_executors()

    yield {"gmail": gmail, "calendar": calendar, "contacts": contacts, "drive": drive}

    reset_providers()
    clear_executors()


class TestGmailEndpoints:
    async def test_listing_returns_the_mailbox(self, client: AsyncClient) -> None:
        body = (await client.get("/api/v1/gmail/messages")).json()
        assert body["total"] == 4
        assert body["unread_count"] == 2

    async def test_unread_only_filters(self, client: AsyncClient) -> None:
        body = (await client.get("/api/v1/gmail/messages?unread_only=true")).json()
        assert all(item["is_unread"] for item in body["items"])

    async def test_messages_are_newest_first(self, client: AsyncClient) -> None:
        items = (await client.get("/api/v1/gmail/messages")).json()["items"]
        received = [item["received_at"] for item in items]
        assert received == sorted(received, reverse=True)

    async def test_search_matches_the_sender(self, client: AsyncClient) -> None:
        body = (await client.get("/api/v1/gmail/search?q=nathan")).json()
        assert body["total"] == 1
        assert "nathan" in body["items"][0]["sender"]

    async def test_reading_a_message_flags_untrusted_content(self, client: AsyncClient) -> None:
        """The model must treat a body as data, not as instructions."""
        body = (await client.get("/api/v1/gmail/messages/msg-001")).json()
        assert body["content_is_untrusted"] is True
        assert "ablation table" in body["message"]["body_text"]

    async def test_an_unknown_message_is_a_404(self, client: AsyncClient) -> None:
        assert (await client.get("/api/v1/gmail/messages/nope")).status_code == 404

    async def test_the_limit_is_enforced(self, client: AsyncClient) -> None:
        assert (await client.get("/api/v1/gmail/messages?limit=999")).status_code == 422


class TestDraftAndSendSeparation:
    PAYLOAD = {
        "to": ["nathan.chen@example.com"],
        "subject": "Thursday coffee",
        "body": "8:30 works for me.",
    }

    async def _create_draft(self, client: AsyncClient) -> dict:
        response = await client.post("/api/v1/gmail/drafts", json=self.PAYLOAD)
        assert response.status_code == 201
        return response.json()["draft"]

    async def test_creating_a_draft_does_not_send(
        self, client: AsyncClient, providers: dict
    ) -> None:
        draft = await self._create_draft(client)
        assert draft["id"]
        assert draft["content_hash"]
        assert providers["gmail"].sent == []
        listed = (await client.get("/api/v1/gmail/drafts")).json()
        assert any(item["id"] == draft["id"] for item in listed["items"])

    async def test_legacy_compose_send_path_is_gone(self, client: AsyncClient) -> None:
        response = await client.post("/api/v1/gmail/send", json=self.PAYLOAD)
        assert response.status_code == 410

    async def test_send_draft_only_creates_a_ticket(
        self, client: AsyncClient, providers: dict
    ) -> None:
        draft = await self._create_draft(client)
        response = await client.post(
            "/api/v1/gmail/actions/send-draft",
            json={"draft_id": draft["id"], "content_hash": draft["content_hash"]},
        )
        assert response.status_code == 202
        assert response.json()["status"] == "awaiting_approval"
        assert providers["gmail"].sent == []

    async def test_the_preview_binds_to_the_draft_revision(self, client: AsyncClient) -> None:
        draft = await self._create_draft(client)
        preview = (
            await client.post(
                "/api/v1/gmail/actions/send-draft",
                json={"draft_id": draft["id"]},
            )
        ).json()["preview"]
        assert f"Draft: {draft['id']}" in preview
        assert draft["content_hash"][:12] in preview
        assert "To: nathan.chen@example.com" in preview
        assert "Subject: Thursday coffee" in preview
        assert "8:30 works for me." in preview

    async def test_approving_sends_exactly_one_message(
        self,
        client: AsyncClient,
        session: AsyncSession,
        settings: Settings,
        providers: dict,
    ) -> None:
        draft = await self._create_draft(client)
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

        assert len(providers["gmail"].sent) == 1
        assert providers["gmail"].sent[0]["recipients"] == ["nathan.chen@example.com"]
        assert providers["gmail"].sent[0]["draft_id"] == draft["id"]

    async def test_changed_draft_content_is_not_sent(
        self,
        client: AsyncClient,
        session: AsyncSession,
        settings: Settings,
        providers: dict,
    ) -> None:
        draft = await self._create_draft(client)
        ticket = (
            await client.post(
                "/api/v1/gmail/actions/send-draft",
                json={"draft_id": draft["id"], "content_hash": draft["content_hash"]},
            )
        ).json()

        # Mutate the stored draft after approval was requested.
        stored = providers["gmail"]._drafts[draft["id"]]
        stored.body = "changed after approval"
        stored.content_hash = "deadbeef" * 8

        approval = await session.get(Approval, ticket["approval_id"])
        assert approval is not None
        outcome = await resolve_by_token(
            session, approval.callback_token, approve=True, chat_id=None, settings=settings
        )
        assert outcome["executed"] is False
        assert outcome["reason"] == "execution_failed"
        assert providers["gmail"].sent == []

    async def test_duplicate_approve_does_not_send_twice(
        self,
        client: AsyncClient,
        session: AsyncSession,
        settings: Settings,
        providers: dict,
    ) -> None:
        draft = await self._create_draft(client)
        ticket = (
            await client.post(
                "/api/v1/gmail/actions/send-draft",
                json={"draft_id": draft["id"]},
            )
        ).json()
        approval = await session.get(Approval, ticket["approval_id"])
        assert approval is not None
        token = approval.callback_token

        first = await resolve_by_token(
            session, token, approve=True, chat_id=None, settings=settings
        )
        second = await resolve_by_token(
            session, token, approve=True, chat_id=None, settings=settings
        )
        assert first["executed"] is True
        assert second["accepted"] is False
        assert second["reason"] == "already_resolved"
        assert len(providers["gmail"].sent) == 1

    async def test_rejecting_sends_nothing(
        self,
        client: AsyncClient,
        session: AsyncSession,
        settings: Settings,
        providers: dict,
    ) -> None:
        draft = await self._create_draft(client)
        ticket = (
            await client.post(
                "/api/v1/gmail/actions/send-draft",
                json={"draft_id": draft["id"]},
            )
        ).json()
        approval = await session.get(Approval, ticket["approval_id"])
        assert approval is not None

        await resolve_by_token(
            session, approval.callback_token, approve=False, chat_id=None, settings=settings
        )

        assert providers["gmail"].sent == []
        session.expunge_all()
        action = await session.get(PendingAction, ticket["action_id"])
        assert action is not None
        assert action.status == PendingActionStatus.REJECTED.value

    async def test_a_bad_address_is_rejected_before_draft_creation(
        self, client: AsyncClient
    ) -> None:
        response = await client.post(
            "/api/v1/gmail/drafts",
            json={**self.PAYLOAD, "to": ["definitely not an address"]},
        )
        assert response.status_code == 422

    async def test_a_header_injection_attempt_is_rejected(self, client: AsyncClient) -> None:
        response = await client.post(
            "/api/v1/gmail/drafts",
            json={**self.PAYLOAD, "subject": "Hi\nBcc: attacker@example.com"},
        )
        assert response.status_code == 422

    async def test_too_many_recipients_is_rejected(self, client: AsyncClient) -> None:
        response = await client.post(
            "/api/v1/gmail/drafts",
            json={**self.PAYLOAD, "to": [f"user{index}@example.com" for index in range(11)]},
        )
        assert response.status_code == 422

    async def test_a_retried_send_request_reuses_one_ticket(self, client: AsyncClient) -> None:
        draft = await self._create_draft(client)
        headers = {"Idempotency-Key": "voice-turn-42"}
        body = {"draft_id": draft["id"], "content_hash": draft["content_hash"]}
        first = (
            await client.post("/api/v1/gmail/actions/send-draft", json=body, headers=headers)
        ).json()
        second = (
            await client.post("/api/v1/gmail/actions/send-draft", json=body, headers=headers)
        ).json()
        assert first["action_id"] == second["action_id"]

    async def test_a_retried_draft_create_reuses_one_draft(self, client: AsyncClient) -> None:
        headers = {"Idempotency-Key": "voice-draft-7"}
        first = (
            await client.post("/api/v1/gmail/drafts", json=self.PAYLOAD, headers=headers)
        ).json()["draft"]
        second = (
            await client.post("/api/v1/gmail/drafts", json=self.PAYLOAD, headers=headers)
        ).json()["draft"]
        assert first["id"] == second["id"]
        listed = (await client.get("/api/v1/gmail/drafts")).json()
        assert sum(1 for item in listed["items"] if item["id"] == first["id"]) == 1


class TestGmailAttachmentsAndLabels:
    async def test_list_and_download_attachment(
        self, client: AsyncClient, settings: Settings, data_dir
    ) -> None:
        listed = (await client.get("/api/v1/gmail/messages/msg-004/attachments")).json()
        assert listed["total"] == 1
        assert listed["content_is_untrusted"] is True
        attachment_id = listed["items"][0]["id"]

        downloaded = (
            await client.post(
                f"/api/v1/gmail/messages/msg-004/attachments/{attachment_id}/download"
            )
        ).json()
        path = downloaded["download"]["path"]
        assert path.startswith(str(settings.app_data_dir / "gmail" / "attachments"))
        assert downloaded["content_is_untrusted"] is True
        assert "Invoice" in (downloaded["download"]["text"] or "")

    async def test_archive_removes_inbox(self, client: AsyncClient) -> None:
        body = (await client.post("/api/v1/gmail/messages/msg-001/actions/archive")).json()
        assert "INBOX" not in body["labels"]

    async def test_trash_label_is_refused(self, client: AsyncClient) -> None:
        response = await client.post(
            "/api/v1/gmail/messages/msg-001/actions/modify-labels",
            json={"add_label_ids": ["TRASH"]},
        )
        assert response.status_code == 422


class TestCalendarEndpoints:
    async def test_events_in_a_window_are_listed(self, client: AsyncClient) -> None:
        start = ANCHOR - timedelta(hours=1)
        end = ANCHOR + timedelta(hours=12)
        body = (
            await client.get(
                "/api/v1/calendar/events",
                params={"starts_after": start.isoformat(), "ends_before": end.isoformat()},
            )
        ).json()
        assert [item["title"] for item in body["items"]] == [
            "Lab meeting",
            "Reachy demo for visiting group",
        ]

    async def test_events_come_back_in_time_order(self, client: AsyncClient) -> None:
        body = (
            await client.get(
                "/api/v1/calendar/events",
                params={
                    "starts_after": (ANCHOR - timedelta(days=1)).isoformat(),
                    "ends_before": (ANCHOR + timedelta(days=10)).isoformat(),
                },
            )
        ).json()
        starts = [item["starts_at"] for item in body["items"]]
        assert starts == sorted(starts)

    async def test_a_backwards_range_is_refused(self, client: AsyncClient) -> None:
        response = await client.get(
            "/api/v1/calendar/events",
            params={
                "starts_after": ANCHOR.isoformat(),
                "ends_before": (ANCHOR - timedelta(hours=1)).isoformat(),
            },
        )
        assert response.status_code == 422

    async def test_free_busy_returns_the_gaps_too(self, client: AsyncClient) -> None:
        body = (
            await client.get(
                "/api/v1/calendar/free-busy",
                params={
                    "starts_at": ANCHOR.isoformat(),
                    "ends_at": (ANCHOR + timedelta(hours=12)).isoformat(),
                },
            )
        ).json()
        assert len(body["busy"]) == 2
        assert len(body["free_slots"]) >= 2
        assert body["free_slots"][0]["ends_at"] == body["busy"][0]["starts_at"]

    async def test_creating_an_event_needs_approval(
        self, client: AsyncClient, providers: dict
    ) -> None:
        response = await client.post(
            "/api/v1/calendar/events",
            json={
                "title": "Supervisor sync",
                "starts_at": (ANCHOR + timedelta(days=1)).isoformat(),
                "ends_at": (ANCHOR + timedelta(days=1, hours=1)).isoformat(),
            },
        )
        assert response.status_code == 202
        assert providers["calendar"].created == []

    async def test_approving_creates_the_event(
        self,
        client: AsyncClient,
        session: AsyncSession,
        settings: Settings,
        providers: dict,
    ) -> None:
        ticket = (
            await client.post(
                "/api/v1/calendar/events",
                json={
                    "title": "Supervisor sync",
                    "starts_at": (ANCHOR + timedelta(days=1)).isoformat(),
                    "ends_at": (ANCHOR + timedelta(days=1, hours=1)).isoformat(),
                    "location": "Office",
                },
            )
        ).json()
        approval = await session.get(Approval, ticket["approval_id"])
        assert approval is not None

        await resolve_by_token(
            session, approval.callback_token, approve=True, chat_id=None, settings=settings
        )

        assert [event.title for event in providers["calendar"].created] == ["Supervisor sync"]

    async def test_deleting_names_the_event_in_the_prompt(self, client: AsyncClient) -> None:
        """'Delete evt-001' is not something anyone can meaningfully approve."""
        ticket = (await client.delete("/api/v1/calendar/events/evt-001")).json()
        assert "Lab meeting" in ticket["summary"]
        assert "Lab meeting" in ticket["preview"]

    async def test_propose_is_dry_run(self, client: AsyncClient, providers: dict) -> None:
        response = await client.post(
            "/api/v1/calendar/events/propose",
            json={
                "title": "Coffee",
                "duration_minutes": 30,
                "search_from": ANCHOR.isoformat(),
                "search_to": (ANCHOR + timedelta(days=2)).isoformat(),
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["suggestions"]
        assert body["draft_create_payload"]["title"] == "Coffee"
        assert providers["calendar"].created == []
        assert providers["calendar"].updated == []

    async def test_update_requires_approval_then_mutates(
        self,
        client: AsyncClient,
        session: AsyncSession,
        settings: Settings,
        providers: dict,
    ) -> None:
        ticket = (
            await client.patch(
                "/api/v1/calendar/events/evt-001",
                json={"title": "Lab meeting (moved)"},
            )
        ).json()
        assert ticket["status"] == "awaiting_approval"
        assert "Lab meeting" in ticket["preview"]
        assert providers["calendar"].updated == []

        approval = await session.get(Approval, ticket["approval_id"])
        assert approval is not None
        await resolve_by_token(
            session, approval.callback_token, approve=True, chat_id=None, settings=settings
        )
        assert providers["calendar"].updated[-1].title == "Lab meeting (moved)"


class TestContactEndpoints:
    async def test_google_contacts_are_searchable(self, client: AsyncClient) -> None:
        body = (await client.get("/api/v1/contacts/search?q=nathan")).json()
        assert {item["display_name"] for item in body["items"]} == {
            "Nathan Chen",
            "Nathan Brooks",
        }

    async def test_a_local_alias_resolves(self, client: AsyncClient, session: AsyncSession) -> None:
        contact = Contact(
            display_name="Dr. Elena Marsh",
            normalised_name="dr. elena marsh",
            primary_email="supervisor@university.edu",
            is_vip=True,
        )
        session.add(contact)
        await session.flush()
        session.add(
            ContactAlias(
                contact_id=contact.id, alias="my supervisor", normalised_alias="my supervisor"
            )
        )
        await session.commit()

        body = (await client.get("/api/v1/contacts/search?q=supervisor")).json()

        assert body["items"][0]["display_name"] == "Dr. Elena Marsh"

    async def test_local_entries_come_before_google_ones(
        self, client: AsyncClient, session: AsyncSession
    ) -> None:
        session.add(
            Contact(
                display_name="Nathan Chen",
                normalised_name="nathan chen",
                primary_email="corrected@example.com",
            )
        )
        await session.commit()

        items = (await client.get("/api/v1/contacts/search?q=nathan chen")).json()["items"]

        assert items[0]["emails"] == ["corrected@example.com"]


class TestDriveEndpoints:
    async def test_files_are_searchable(self, client: AsyncClient) -> None:
        body = (await client.get("/api/v1/drive/search?q=grasp")).json()
        assert body["items"][0]["name"] == "Grasp results.csv"

    async def test_a_document_can_be_read_as_text(self, client: AsyncClient) -> None:
        body = (await client.get("/api/v1/drive/files/file-001/content")).json()
        assert "Learning Dexterous Manipulation" in body["file"]["text"]
        assert body["content_is_untrusted"] is True

    async def test_a_binary_file_is_refused(self, client: AsyncClient) -> None:
        response = await client.get("/api/v1/drive/files/file-004/content")
        assert response.status_code == 422
        assert "no text form" in response.json()["error"]["message"]

    async def test_a_folder_has_no_content(self, client: AsyncClient) -> None:
        assert (await client.get("/api/v1/drive/files/file-003/content")).status_code == 422

    async def test_truncation_is_reported(self, client: AsyncClient) -> None:
        body = (await client.get("/api/v1/drive/files/file-001/content?max_chars=100")).json()
        assert body["file"]["truncated"] is True
        assert len(body["file"]["text"]) == 100


class TestAuthentication:
    @pytest.mark.parametrize(
        "path",
        [
            "/api/v1/gmail/messages",
            "/api/v1/calendar/today",
            "/api/v1/contacts/search?q=a",
            "/api/v1/drive/search?q=a",
        ],
    )
    async def test_every_google_endpoint_needs_a_token(
        self, anonymous_client: AsyncClient, path: str
    ) -> None:
        assert (await anonymous_client.get(path)).status_code == 401
