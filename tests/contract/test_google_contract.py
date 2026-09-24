"""Contract tests for the Google providers.

These pin the request the assistant sends and the parsing of the reply. The
payloads are trimmed copies of real Google responses, so a change in how the
providers build a query or read a field fails here rather than in production
against a live mailbox.
"""

from __future__ import annotations

import base64
from datetime import UTC, datetime
from typing import Any

import httpx
import pytest
import respx

from app.config import Settings
from integrations.google.calendar import CALENDAR_BASE, CalendarService
from integrations.google.client import GoogleClient
from integrations.google.contacts import PEOPLE_BASE, ContactsService
from integrations.google.drive import DRIVE_BASE, DriveService, escape_query_term
from integrations.google.gmail import GMAIL_BASE, GmailService
from integrations.google.oauth import ALL_SCOPES, StoredCredentials, TokenProvider
from shared.errors import IntegrationError, ProviderRateLimitError, ValidationError


class StubTokenProvider(TokenProvider):
    """A token provider that never talks to Google's token endpoint."""

    def __init__(self, settings: Settings, scopes: tuple[str, ...] = ALL_SCOPES) -> None:
        super().__init__(settings)
        self._credentials = StoredCredentials(
            refresh_token="stub-refresh-token",
            scopes=list(scopes),
            account_email="owner@example.com",
        )
        self.tokens_issued = 0

    async def get_access_token(self, client: httpx.AsyncClient | None = None) -> str:
        self.tokens_issued += 1
        return f"stub-access-{self.tokens_issued}"


@pytest.fixture
def tokens(settings: Settings) -> StubTokenProvider:
    return StubTokenProvider(settings)


@pytest.fixture
def client(settings: Settings, tokens: StubTokenProvider) -> GoogleClient:
    return GoogleClient(settings=settings, token_provider=tokens)


def ok(payload: Any) -> httpx.Response:
    return httpx.Response(200, json=payload)


MESSAGE_LIST = {"messages": [{"id": "m1", "threadId": "t1"}], "resultSizeEstimate": 1}
MESSAGE_DETAIL = {
    "id": "m1",
    "threadId": "t1",
    "labelIds": ["INBOX", "UNREAD", "IMPORTANT"],
    "snippet": "The ablation table needs a confidence interval",
    "internalDate": "1772539200000",
    "payload": {
        "mimeType": "multipart/alternative",
        "headers": [
            {"name": "From", "value": "Dr. Elena Marsh <supervisor@university.edu>"},
            {"name": "To", "value": "owner@example.com"},
            {"name": "Subject", "value": "Revised draft"},
            {"name": "Date", "value": "Tue, 3 Mar 2026 12:00:00 -0500"},
        ],
        "parts": [
            {
                "mimeType": "text/plain",
                "body": {
                    "data": base64.urlsafe_b64encode(
                        b"The ablation table needs a confidence interval."
                    )
                    .decode()
                    .rstrip("=")
                },
            },
            {
                "mimeType": "text/html",
                "body": {
                    "data": base64.urlsafe_b64encode(b"<p>HTML version</p>").decode().rstrip("=")
                },
            },
        ],
    },
}


class TestGmailContract:
    @respx.mock
    async def test_listing_asks_for_metadata_only(
        self, client: GoogleClient, settings: Settings
    ) -> None:
        """Listing must not pull whole bodies; that is a different, larger request."""
        respx.get(f"{GMAIL_BASE}/users/me/messages").mock(return_value=ok(MESSAGE_LIST))
        detail = respx.get(f"{GMAIL_BASE}/users/me/messages/m1").mock(
            return_value=ok(MESSAGE_DETAIL)
        )

        await GmailService(client, settings).list_messages(limit=5)

        assert detail.calls.last.request.url.params["format"] == "metadata"

    @respx.mock
    async def test_unread_filter_becomes_a_query_term(
        self, client: GoogleClient, settings: Settings
    ) -> None:
        listing = respx.get(f"{GMAIL_BASE}/users/me/messages").mock(return_value=ok(MESSAGE_LIST))
        respx.get(f"{GMAIL_BASE}/users/me/messages/m1").mock(return_value=ok(MESSAGE_DETAIL))

        await GmailService(client, settings).list_messages(unread_only=True, limit=5)

        assert listing.calls.last.request.url.params["q"] == "is:unread"

    @respx.mock
    async def test_a_message_is_normalised(self, client: GoogleClient, settings: Settings) -> None:
        respx.get(f"{GMAIL_BASE}/users/me/messages/m1").mock(return_value=ok(MESSAGE_DETAIL))

        message = await GmailService(client, settings).get_message("m1")

        assert message.sender == "supervisor@university.edu"
        assert message.sender_name == "Dr. Elena Marsh"
        assert message.subject == "Revised draft"
        assert message.is_unread is True
        assert message.is_important is True
        assert message.received_at == datetime(2026, 3, 3, 12, 0, tzinfo=UTC)

    @respx.mock
    async def test_the_plain_text_part_is_preferred_over_html(
        self, client: GoogleClient, settings: Settings
    ) -> None:
        respx.get(f"{GMAIL_BASE}/users/me/messages/m1").mock(return_value=ok(MESSAGE_DETAIL))
        message = await GmailService(client, settings).get_message("m1")
        assert "HTML version" not in message.body_text
        assert message.body_text.startswith("The ablation table")

    @respx.mock
    async def test_a_long_body_is_truncated_and_says_so(
        self, client: GoogleClient, settings: Settings
    ) -> None:
        payload = dict(MESSAGE_DETAIL)
        payload["payload"] = {
            "mimeType": "text/plain",
            "headers": MESSAGE_DETAIL["payload"]["headers"],
            "body": {"data": base64.urlsafe_b64encode(b"x" * 9000).decode().rstrip("=")},
        }
        respx.get(f"{GMAIL_BASE}/users/me/messages/m1").mock(return_value=ok(payload))

        message = await GmailService(client, settings).get_message("m1", max_chars=500)

        assert len(message.body_text) == 500
        assert message.truncated is True

    @respx.mock
    async def test_creating_a_draft_encodes_mime_and_does_not_send(
        self, client: GoogleClient, settings: Settings
    ) -> None:
        create_route = respx.post(f"{GMAIL_BASE}/users/me/drafts").mock(
            return_value=ok({"id": "d1", "message": {"id": "m9", "threadId": "t-9"}})
        )
        send_route = respx.post(f"{GMAIL_BASE}/users/me/drafts/send").mock(
            return_value=ok({"id": "sent-1", "threadId": "t-9"})
        )

        draft = await GmailService(client, settings).create_draft(
            to=["nathan.chen@example.com"], subject="Coffee", body="Thursday at 8:30 works."
        )

        raw = create_route.calls.last.request.read()
        decoded = base64.urlsafe_b64decode(
            __import__("json").loads(raw)["message"]["raw"] + "=="
        ).decode()
        assert "To: nathan.chen@example.com" in decoded
        assert "Subject: Coffee" in decoded
        assert "Thursday at 8:30 works." in decoded
        assert draft.id == "d1"
        assert draft.content_hash
        assert not send_route.called

    @respx.mock
    async def test_creating_a_reply_draft_sets_thread_and_headers(
        self, client: GoogleClient, settings: Settings
    ) -> None:
        reply_context = {
            "id": "m2",
            "threadId": "t-parent",
            "payload": {
                "headers": [
                    {"name": "From", "value": "Nathan Chen <nathan.chen@example.com>"},
                    {"name": "To", "value": "owner@example.com"},
                    {"name": "Subject", "value": "Coffee before the lab meeting?"},
                    {"name": "Message-ID", "value": "<msg-003@example.com>"},
                    {"name": "References", "value": "<msg-001@example.com>"},
                ]
            },
        }
        respx.get(f"{GMAIL_BASE}/users/me/messages/m2").mock(return_value=ok(reply_context))
        create_route = respx.post(f"{GMAIL_BASE}/users/me/drafts").mock(
            return_value=ok({"id": "d2", "message": {"id": "m10", "threadId": "t-parent"}})
        )
        send_route = respx.post(f"{GMAIL_BASE}/users/me/drafts/send").mock(
            return_value=ok({"id": "sent-2", "threadId": "t-parent"})
        )

        draft = await GmailService(client, settings).create_reply_draft(
            message_id="m2",
            body="Thursday at 8:30 works.",
        )

        request_body = __import__("json").loads(create_route.calls.last.request.read())
        assert request_body["message"]["threadId"] == "t-parent"
        raw = base64.urlsafe_b64decode(request_body["message"]["raw"] + "==").decode()
        assert "To: nathan.chen@example.com" in raw
        assert "Subject: Re: Coffee before the lab meeting?" in raw
        assert "In-Reply-To: <msg-003@example.com>" in raw
        assert "References: <msg-001@example.com> <msg-003@example.com>" in raw
        assert draft.thread_id == "t-parent"
        assert draft.in_reply_to == "<msg-003@example.com>"
        assert draft.references == "<msg-001@example.com> <msg-003@example.com>"
        assert not send_route.called

    @respx.mock
    async def test_send_draft_rechecks_content_hash(
        self, client: GoogleClient, settings: Settings
    ) -> None:
        plain = base64.urlsafe_b64encode(b"Thursday at 8:30 works.").decode().rstrip("=")
        draft_payload = {
            "id": "d1",
            "message": {
                "id": "m9",
                "threadId": "t-9",
                "payload": {
                    "mimeType": "text/plain",
                    "headers": [
                        {"name": "To", "value": "nathan.chen@example.com"},
                        {"name": "Subject", "value": "Coffee"},
                    ],
                    "body": {"data": plain},
                },
            },
        }
        respx.get(f"{GMAIL_BASE}/users/me/drafts/d1").mock(return_value=ok(draft_payload))
        send_route = respx.post(f"{GMAIL_BASE}/users/me/drafts/send").mock(
            return_value=ok({"id": "sent-1", "threadId": "t-9"})
        )

        service = GmailService(client, settings)
        draft = await service.get_draft("d1")
        result = await service.send_draft("d1", expected_content_hash=draft.content_hash)
        assert result["message_id"] == "sent-1"
        assert send_route.called

        with pytest.raises(ValidationError, match="no longer matches"):
            await service.send_draft("d1", expected_content_hash="wrong-hash")

    async def test_a_header_with_a_newline_is_refused(
        self, client: GoogleClient, settings: Settings
    ) -> None:
        """Header injection would let a caller add recipients of their own."""
        with pytest.raises(ValidationError):
            await GmailService(client, settings).create_draft(
                to=["a@example.com"],
                subject="Hello\r\nBcc: attacker@example.com",
                body="text",
            )

    async def test_an_invalid_recipient_is_refused(
        self, client: GoogleClient, settings: Settings
    ) -> None:
        with pytest.raises(ValidationError):
            await GmailService(client, settings).create_draft(
                to=["not-an-address"], subject="Hi", body="text"
            )

    async def test_sending_without_the_send_scope_is_refused(self, settings: Settings) -> None:
        compose_only = StubTokenProvider(
            settings,
            scopes=(
                "https://www.googleapis.com/auth/gmail.readonly",
                "https://www.googleapis.com/auth/gmail.compose",
            ),
        )
        service = GmailService(
            GoogleClient(settings=settings, token_provider=compose_only), settings
        )

        with pytest.raises(IntegrationError, match="gmail.send"):
            await service.send_draft("d1", expected_content_hash="abc")

    @respx.mock
    async def test_modify_labels_posts_add_and_remove(
        self, client: GoogleClient, settings: Settings
    ) -> None:
        route = respx.post(f"{GMAIL_BASE}/users/me/messages/m1/modify").mock(
            return_value=ok({"id": "m1", "labelIds": ["IMPORTANT"]})
        )
        labels = await GmailService(client, settings).modify_labels(
            "m1", add=["IMPORTANT"], remove=["INBOX"]
        )
        assert labels == ["IMPORTANT"]
        body = __import__("json").loads(route.calls.last.request.read())
        assert body["addLabelIds"] == ["IMPORTANT"]
        assert body["removeLabelIds"] == ["INBOX"]


class TestCalendarContract:
    EVENT = {
        "id": "evt-1",
        "summary": "Lab meeting",
        "status": "confirmed",
        "start": {"dateTime": "2026-03-10T14:00:00Z"},
        "end": {"dateTime": "2026-03-10T15:00:00Z"},
        "attendees": [{"email": "nathan.chen@example.com"}],
        "organizer": {"email": "supervisor@university.edu"},
        "htmlLink": "https://calendar.example.com/evt-1",
    }

    @respx.mock
    async def test_listing_requests_expanded_single_events(
        self, client: GoogleClient, settings: Settings
    ) -> None:
        """Without singleEvents a recurring series returns one opaque row."""
        route = respx.get(f"{CALENDAR_BASE}/calendars/primary/events").mock(
            return_value=ok({"items": [self.EVENT]})
        )

        await CalendarService(client, settings).list_events(
            starts_after=datetime(2026, 3, 10, tzinfo=UTC),
            ends_before=datetime(2026, 3, 11, tzinfo=UTC),
        )

        params = route.calls.last.request.url.params
        assert params["singleEvents"] == "true"
        assert params["orderBy"] == "startTime"
        assert params["timeMin"] == "2026-03-10T00:00:00Z"

    @respx.mock
    async def test_a_cancelled_event_is_dropped(
        self, client: GoogleClient, settings: Settings
    ) -> None:
        cancelled = {**self.EVENT, "id": "evt-2", "status": "cancelled"}
        respx.get(f"{CALENDAR_BASE}/calendars/primary/events").mock(
            return_value=ok({"items": [self.EVENT, cancelled]})
        )

        events = await CalendarService(client, settings).list_events(
            starts_after=datetime(2026, 3, 10, tzinfo=UTC),
            ends_before=datetime(2026, 3, 11, tzinfo=UTC),
        )

        assert [event.id for event in events] == ["evt-1"]

    @respx.mock
    async def test_an_all_day_event_keeps_its_local_day(
        self, client: GoogleClient, settings: Settings
    ) -> None:
        all_day = {
            "id": "evt-3",
            "summary": "Conference",
            "status": "confirmed",
            "start": {"date": "2026-03-12", "timeZone": "America/Toronto"},
            "end": {"date": "2026-03-15", "timeZone": "America/Toronto"},
        }
        respx.get(f"{CALENDAR_BASE}/calendars/primary/events").mock(
            return_value=ok({"items": [all_day]})
        )

        events = await CalendarService(client, settings).list_events(
            starts_after=datetime(2026, 3, 1, tzinfo=UTC),
            ends_before=datetime(2026, 3, 20, tzinfo=UTC),
        )

        assert events[0].all_day is True
        # Midnight in Toronto is 05:00 UTC in March.
        assert events[0].starts_at == datetime(2026, 3, 12, 4, 0, tzinfo=UTC)

    @respx.mock
    async def test_creating_an_event_sends_an_explicit_timezone(
        self, client: GoogleClient, settings: Settings
    ) -> None:
        route = respx.post(f"{CALENDAR_BASE}/calendars/primary/events").mock(
            return_value=ok(self.EVENT)
        )

        await CalendarService(client, settings).create_event(
            {
                "title": "Lab meeting",
                "starts_at": "2026-03-10T14:00:00Z",
                "ends_at": "2026-03-10T15:00:00Z",
                "timezone": "America/Toronto",
            }
        )

        body = __import__("json").loads(route.calls.last.request.read())
        assert body["start"]["timeZone"] == "America/Toronto"
        assert body["summary"] == "Lab meeting"

    @respx.mock
    async def test_updating_an_event_sends_a_patch(
        self, client: GoogleClient, settings: Settings
    ) -> None:
        route = respx.patch(f"{CALENDAR_BASE}/calendars/primary/events/evt-1").mock(
            return_value=ok({**self.EVENT, "summary": "Lab meeting (moved)"})
        )
        event = await CalendarService(client, settings).update_event(
            "evt-1", {"title": "Lab meeting (moved)", "timezone": "America/Toronto"}
        )
        assert event.title == "Lab meeting (moved)"
        body = __import__("json").loads(route.calls.last.request.read())
        assert body["summary"] == "Lab meeting (moved)"
        assert "start" not in body

    async def test_an_event_ending_before_it_starts_is_refused(
        self, client: GoogleClient, settings: Settings
    ) -> None:
        with pytest.raises(ValidationError):
            await CalendarService(client, settings).create_event(
                {
                    "title": "Backwards",
                    "starts_at": "2026-03-10T15:00:00Z",
                    "ends_at": "2026-03-10T14:00:00Z",
                }
            )

    @respx.mock
    async def test_free_busy_is_parsed(self, client: GoogleClient, settings: Settings) -> None:
        respx.post(f"{CALENDAR_BASE}/freeBusy").mock(
            return_value=ok(
                {
                    "calendars": {
                        "primary": {
                            "busy": [
                                {
                                    "start": "2026-03-10T14:00:00Z",
                                    "end": "2026-03-10T15:00:00Z",
                                }
                            ]
                        }
                    }
                }
            )
        )

        slots = await CalendarService(client, settings).free_busy(
            starts_at=datetime(2026, 3, 10, tzinfo=UTC),
            ends_at=datetime(2026, 3, 11, tzinfo=UTC),
        )

        assert slots[0].starts_at == datetime(2026, 3, 10, 14, 0, tzinfo=UTC)


class TestContactsContract:
    @respx.mock
    async def test_search_reads_the_person_fields(
        self, client: GoogleClient, settings: Settings
    ) -> None:
        route = respx.get(f"{PEOPLE_BASE}/people:searchContacts").mock(
            return_value=ok(
                {
                    "results": [
                        {
                            "person": {
                                "resourceName": "people/c1",
                                "names": [
                                    {
                                        "displayName": "Nathan Chen",
                                        "givenName": "Nathan",
                                        "familyName": "Chen",
                                    }
                                ],
                                "emailAddresses": [{"value": "nathan.chen@example.com"}],
                                "phoneNumbers": [{"value": "+1-416-555-0142"}],
                            }
                        }
                    ]
                }
            )
        )

        results = await ContactsService(client, settings).search("nathan")

        assert results[0].display_name == "Nathan Chen"
        assert results[0].emails == ["nathan.chen@example.com"]
        assert "emailAddresses" in route.calls.last.request.url.params["readMask"]

    async def test_an_empty_query_does_not_call_google(
        self, client: GoogleClient, settings: Settings
    ) -> None:
        assert await ContactsService(client, settings).search("   ") == []


class TestDriveContract:
    FILE = {
        "id": "file-1",
        "name": "Grasp results.csv",
        "mimeType": "text/csv",
        "size": "8422",
        "modifiedTime": "2026-03-09T10:00:00.000Z",
        "owners": [{"displayName": "Owner"}],
        "webViewLink": "https://drive.example.com/file-1",
    }

    @respx.mock
    async def test_search_excludes_trashed_files(
        self, client: GoogleClient, settings: Settings
    ) -> None:
        route = respx.get(f"{DRIVE_BASE}/files").mock(return_value=ok({"files": [self.FILE]}))

        await DriveService(client, settings).search("grasp")

        query = route.calls.last.request.url.params["q"]
        assert "trashed = false" in query
        assert "name contains 'grasp'" in query

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [("o'brien", r"o\'brien"), ("back\\slash", r"back\\slash")],
    )
    def test_query_terms_are_escaped(self, raw: str, expected: str) -> None:
        """An apostrophe in a filename would otherwise terminate the query string."""
        assert escape_query_term(raw) == expected

    @respx.mock
    async def test_a_google_doc_is_exported_as_text(
        self, client: GoogleClient, settings: Settings
    ) -> None:
        doc = {**self.FILE, "mimeType": "application/vnd.google-apps.document"}
        respx.get(f"{DRIVE_BASE}/files/file-1").mock(return_value=ok(doc))
        export = respx.get(f"{DRIVE_BASE}/files/file-1/export").mock(
            return_value=httpx.Response(200, text="Document body")
        )

        content = await DriveService(client, settings).export_text("file-1")

        assert export.calls.last.request.url.params["mimeType"] == "text/plain"
        assert content.text == "Document body"

    @respx.mock
    async def test_a_binary_file_is_refused_rather_than_downloaded(
        self, client: GoogleClient, settings: Settings
    ) -> None:
        pdf = {**self.FILE, "mimeType": "application/pdf", "name": "Poster.pdf"}
        respx.get(f"{DRIVE_BASE}/files/file-1").mock(return_value=ok(pdf))

        with pytest.raises(ValidationError, match="no text form"):
            await DriveService(client, settings).export_text("file-1")


class TestClientBehaviour:
    @respx.mock
    async def test_a_401_triggers_one_refresh_and_one_retry(
        self, client: GoogleClient, tokens: StubTokenProvider, settings: Settings
    ) -> None:
        route = respx.get(f"{GMAIL_BASE}/users/me/profile").mock(
            side_effect=[
                httpx.Response(401, json={"error": {"message": "Invalid Credentials"}}),
                ok({"emailAddress": "owner@example.com"}),
            ]
        )

        result = await client.request("GET", f"{GMAIL_BASE}/users/me/profile")

        assert result["emailAddress"] == "owner@example.com"
        assert route.call_count == 2
        assert tokens.tokens_issued == 2

    @respx.mock
    async def test_a_repeated_401_is_not_retried_forever(
        self, client: GoogleClient, settings: Settings
    ) -> None:
        route = respx.get(f"{GMAIL_BASE}/users/me/profile").mock(
            return_value=httpx.Response(401, json={"error": {"message": "Invalid Credentials"}})
        )

        with pytest.raises(IntegrationError):
            await client.request("GET", f"{GMAIL_BASE}/users/me/profile")

        assert route.call_count == 2

    @respx.mock
    async def test_rate_limiting_surfaces_as_a_typed_error(
        self, client: GoogleClient, settings: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("integrations.google.client.MAX_ATTEMPTS", 1)
        respx.get(f"{GMAIL_BASE}/users/me/profile").mock(
            return_value=httpx.Response(429, json={"error": {"message": "Rate Limit Exceeded"}})
        )

        with pytest.raises(ProviderRateLimitError):
            await client.request("GET", f"{GMAIL_BASE}/users/me/profile")

    @respx.mock
    async def test_pagination_stops_at_the_requested_limit(
        self, client: GoogleClient, settings: Settings
    ) -> None:
        """A large mailbox must not turn one question into unbounded requests."""
        respx.get(f"{GMAIL_BASE}/users/me/messages").mock(
            return_value=ok(
                {
                    "messages": [{"id": f"m{index}"} for index in range(100)],
                    "nextPageToken": "more",
                }
            )
        )

        collected = [
            item
            async for item in client.paginate(
                f"{GMAIL_BASE}/users/me/messages", items_key="messages", limit=7
            )
        ]

        assert len(collected) == 7

    @respx.mock
    async def test_an_error_body_is_summarised_not_echoed(
        self, client: GoogleClient, settings: Settings
    ) -> None:
        """Google error bodies quote the request, which for Gmail can be content."""
        respx.get(f"{GMAIL_BASE}/users/me/profile").mock(
            return_value=httpx.Response(
                400,
                json={
                    "error": {
                        "message": "Invalid query",
                        "details": [{"echo": "secret-content-from-the-request"}],
                    }
                },
            )
        )

        with pytest.raises(IntegrationError) as caught:
            await client.request("GET", f"{GMAIL_BASE}/users/me/profile")

        assert "secret-content-from-the-request" not in str(caught.value)
