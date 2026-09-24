"""Deterministic in-memory stand-ins for the Google services.

These are not stubs. They hold a small dataset, honour the same filters and
limits as the real providers, and record their writes, so the entire assistant -
briefings, approvals, Reachy tools, the demo script - runs identically with no
Google account.

Timestamps are anchored to a fixed reference date so that a test asserting
"three events tomorrow" keeps passing next month.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from integrations.google.gmail import (
    _write_attachment_bytes,
    build_references_header,
    build_reply_subject,
    draft_content_hash,
    validate_header,
    validate_label_ids,
    validate_recipients,
)
from integrations.google.models import (
    AttachmentDownload,
    CalendarEventModel,
    ContactRecord,
    DriveFileContent,
    DriveFileRecord,
    EmailAttachment,
    EmailBody,
    EmailDraft,
    EmailSummary,
    FreeBusySlot,
)
from security.paths import safe_filename
from shared.errors import NotFoundError, ValidationError
from shared.timeutils import utcnow

ANCHOR = datetime(2026, 3, 10, 9, 0, tzinfo=UTC)


def _hours(offset: float) -> datetime:
    return ANCHOR + timedelta(hours=offset)


class MockGmailService:
    """A small mailbox with a plausible mix of read, unread and important mail."""

    name = "gmail"

    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []
        self._drafts: dict[str, EmailDraft] = {}
        self._draft_seq = 0
        self._attachments: dict[str, list[tuple[EmailAttachment, bytes]]] = {
            "msg-004": [
                (
                    EmailAttachment(
                        id="att-invoice-1",
                        filename="invoice-2026-03.txt",
                        mime_type="text/plain",
                        size_bytes=42,
                    ),
                    b"Invoice 2026-03 for $184.20 is available.\n",
                )
            ]
        }
        self._messages: list[EmailBody] = [
            EmailBody(
                id="msg-001",
                thread_id="thread-001",
                sender="supervisor@university.edu",
                sender_name="Dr. Elena Marsh",
                recipients=["owner@example.com"],
                subject="Revised draft of the manipulation paper",
                snippet="I went through section 4 and left comments on the ablation table.",
                received_at=_hours(-2),
                is_unread=True,
                is_important=True,
                labels=["INBOX", "UNREAD", "IMPORTANT"],
                body_text=(
                    "Hi,\n\nI went through section 4 and left comments on the ablation "
                    "table. The grasp success numbers need a confidence interval before "
                    "we submit.\n\nElena"
                ),
            ),
            EmailBody(
                id="msg-002",
                thread_id="thread-002",
                sender="noreply@lab-cluster.example",
                sender_name="Cluster Notifications",
                recipients=["owner@example.com"],
                subject="Job 884213 completed",
                snippet="Job 884213 (train_policy_v7) finished with exit code 0.",
                received_at=_hours(-5),
                is_unread=True,
                labels=["INBOX", "UNREAD"],
                body_text="Job 884213 (train_policy_v7) finished with exit code 0 after 6h12m.",
            ),
            EmailBody(
                id="msg-003",
                thread_id="thread-003",
                sender="nathan.chen@example.com",
                sender_name="Nathan Chen",
                recipients=["owner@example.com"],
                subject="Coffee before the lab meeting?",
                snippet="Are you free at 8:30 on Thursday?",
                received_at=_hours(-26),
                labels=["INBOX"],
                body_text="Are you free at 8:30 on Thursday? There is a new place by the station.",
                rfc_message_id="<msg-003@example.com>",
            ),
            EmailBody(
                id="msg-004",
                thread_id="thread-004",
                sender="billing@cloudprovider.example",
                sender_name="Cloud Provider Billing",
                recipients=["owner@example.com"],
                subject="Your invoice is ready",
                snippet="Invoice 2026-03 for $184.20 is available.",
                received_at=_hours(-50),
                has_attachments=True,
                labels=["INBOX"],
                body_text="Invoice 2026-03 for $184.20 is available in your account.",
            ),
        ]

    def _summaries(self) -> list[EmailSummary]:
        return [
            EmailSummary(**message.model_dump(exclude={"body_text", "truncated"}))
            for message in self._messages
        ]

    async def list_messages(
        self,
        *,
        query: str | None = None,
        unread_only: bool = False,
        limit: int = 10,
        label: str | None = None,
    ) -> list[EmailSummary]:
        results = self._summaries()
        if unread_only:
            results = [message for message in results if message.is_unread]
        if label:
            results = [message for message in results if label in message.labels]
        if query:
            needle = query.lower()
            results = [
                message
                for message in results
                if needle in message.subject.lower()
                or needle in message.sender.lower()
                or needle in message.snippet.lower()
            ]
        results.sort(key=lambda message: message.received_at, reverse=True)
        return results[:limit]

    async def search(self, query: str, *, limit: int = 10) -> list[EmailSummary]:
        return await self.list_messages(query=query, limit=limit)

    async def unread_count(self) -> int:
        return sum(1 for message in self._messages if message.is_unread)

    async def get_message(self, message_id: str, *, max_chars: int = 4000) -> EmailBody:
        for message in self._messages:
            if message.id == message_id:
                body = message.model_copy()
                if len(body.body_text) > max_chars:
                    body.body_text = body.body_text[:max_chars]
                    body.truncated = True
                return body
        raise NotFoundError(f"No message with id {message_id!r}.")

    async def create_draft(
        self,
        *,
        to: list[str],
        subject: str,
        body: str,
        cc: list[str] | None = None,
    ) -> EmailDraft:
        recipients = validate_recipients(to)
        copies = validate_recipients(cc or [], allow_empty=True)
        clean_subject = validate_header(subject, "subject")
        # Match real Gmail: each create yields a new draft. Voice retries must
        # send Idempotency-Key at the HTTP layer rather than relying on content
        # dedupe that live Gmail does not provide.
        content = draft_content_hash(to=recipients, cc=copies, subject=clean_subject, body=body)
        self._draft_seq += 1
        draft_id = f"draft-{self._draft_seq:03d}"
        draft = EmailDraft(
            id=draft_id,
            message_id=f"draft-msg-{self._draft_seq:03d}",
            thread_id=f"draft-thread-{self._draft_seq:03d}",
            to=recipients,
            cc=copies,
            subject=clean_subject,
            body=body,
            content_hash=content,
            created_at=utcnow(),
        )
        self._drafts[draft_id] = draft
        return draft

    async def create_reply_draft(self, *, message_id: str, body: str) -> EmailDraft:
        original = await self.get_message(message_id)
        if not original.rfc_message_id:
            raise ValidationError("Cannot reply: the original message has no Message-ID header.")
        clean_subject = validate_header(build_reply_subject(original.subject), "subject")
        recipients = validate_recipients([original.sender])
        references = build_references_header(original.references or "", original.rfc_message_id)
        content = draft_content_hash(to=recipients, cc=[], subject=clean_subject, body=body)
        self._draft_seq += 1
        draft_id = f"draft-{self._draft_seq:03d}"
        draft = EmailDraft(
            id=draft_id,
            message_id=f"draft-msg-{self._draft_seq:03d}",
            thread_id=original.thread_id,
            to=recipients,
            cc=[],
            subject=clean_subject,
            body=body,
            content_hash=content,
            created_at=utcnow(),
            in_reply_to_message_id=original.id,
            in_reply_to=original.rfc_message_id,
            references=references,
        )
        self._drafts[draft_id] = draft
        return draft

    async def list_drafts(self, *, limit: int = 20) -> list[EmailDraft]:
        open_drafts = [draft for draft in self._drafts.values() if not draft.sent]
        open_drafts.sort(key=lambda draft: draft.created_at or ANCHOR, reverse=True)
        return open_drafts[: max(1, min(limit, 50))]

    async def get_draft(self, draft_id: str) -> EmailDraft:
        draft = self._drafts.get(draft_id)
        if draft is None:
            raise NotFoundError(f"No draft with id {draft_id!r}.")
        return draft.model_copy()

    async def send_draft(self, draft_id: str, *, expected_content_hash: str) -> dict[str, Any]:
        draft = self._drafts.get(draft_id)
        if draft is None:
            raise NotFoundError(f"No draft with id {draft_id!r}.")
        if draft.sent:
            raise ValidationError("This draft has already been sent.")
        if draft.content_hash != expected_content_hash:
            raise ValidationError(
                "The draft no longer matches the content that was approved, so it was not sent."
            )

        draft.sent = True
        record = {
            "message_id": f"mock-sent-{len(self.sent) + 1}",
            "thread_id": draft.thread_id or f"mock-thread-{len(self.sent) + 1}",
            "draft_id": draft_id,
            "recipients": list(draft.to),
            "cc": list(draft.cc),
            "subject": draft.subject,
            "body": draft.body,
            "sent_at": utcnow().isoformat(),
            "external_id": f"mock-sent-{len(self.sent) + 1}",
        }
        self.sent.append(record)
        return record

    async def list_attachments(self, message_id: str) -> list[EmailAttachment]:
        await self.get_message(message_id)
        return [meta for meta, _ in self._attachments.get(message_id, [])]

    async def download_attachment(
        self,
        message_id: str,
        attachment_id: str,
        *,
        dest_dir: Path,
        max_bytes: int,
    ) -> AttachmentDownload:
        await self.get_message(message_id)
        for meta, raw in self._attachments.get(message_id, []):
            if meta.id == attachment_id:
                if len(raw) > max_bytes:
                    raise ValidationError(
                        f"Attachment is {len(raw)} bytes; the limit is {max_bytes} bytes."
                    )
                filename = safe_filename(meta.filename)
                path = dest_dir / f"{message_id}_{attachment_id}_{filename}"
                await asyncio.to_thread(_write_attachment_bytes, dest_dir, path, raw)
                text = (
                    raw.decode("utf-8", errors="replace")
                    if meta.mime_type.startswith("text/")
                    else None
                )
                return AttachmentDownload(
                    attachment_id=attachment_id,
                    filename=filename,
                    mime_type=meta.mime_type,
                    size_bytes=len(raw),
                    path=str(path),
                    text=text,
                )
        raise NotFoundError(f"No attachment {attachment_id!r} on message {message_id!r}.")

    async def modify_labels(
        self,
        message_id: str,
        *,
        add: list[str] | None = None,
        remove: list[str] | None = None,
    ) -> list[str]:
        message = await self.get_message(message_id)
        labels = set(message.labels)
        for label in validate_label_ids(add or []):
            labels.add(label)
        for label in validate_label_ids(remove or []):
            labels.discard(label)
        message.labels = sorted(labels)
        message.is_unread = "UNREAD" in labels
        for index, existing in enumerate(self._messages):
            if existing.id == message_id:
                self._messages[index] = message
                break
        return list(message.labels)

    async def archive_message(self, message_id: str) -> list[str]:
        return await self.modify_labels(message_id, remove=["INBOX"])

    async def health(self) -> dict[str, Any]:
        return {
            "ok": True,
            "mock": True,
            "messages": len(self._messages),
            "drafts": sum(1 for draft in self._drafts.values() if not draft.sent),
        }


class MockCalendarService:
    name = "calendar"

    def __init__(self) -> None:
        self.created: list[CalendarEventModel] = []
        self.updated: list[CalendarEventModel] = []
        self.deleted: list[str] = []
        self._events: list[CalendarEventModel] = [
            CalendarEventModel(
                id="evt-001",
                title="Lab meeting",
                description="Weekly progress round-table.",
                location="Engineering 4th floor",
                starts_at=_hours(1),
                ends_at=_hours(2),
                attendees=["supervisor@university.edu", "nathan.chen@example.com"],
                organiser="supervisor@university.edu",
                conference_url="https://meet.example.com/lab-weekly",
            ),
            CalendarEventModel(
                id="evt-002",
                title="Reachy demo for visiting group",
                location="Robotics lab",
                starts_at=_hours(6),
                ends_at=_hours(7.5),
                attendees=["visitors@example.org"],
            ),
            CalendarEventModel(
                id="evt-003",
                title="Dentist",
                starts_at=_hours(26),
                ends_at=_hours(27),
            ),
            CalendarEventModel(
                id="evt-004",
                title="Conference travel",
                starts_at=_hours(72),
                ends_at=_hours(96),
                all_day=True,
            ),
        ]

    async def list_events(
        self,
        *,
        starts_after: datetime,
        ends_before: datetime,
        calendar_id: str = "primary",
        limit: int = 25,
    ) -> list[CalendarEventModel]:
        if ends_before <= starts_after:
            raise ValidationError("The end of the range must be after its start.")
        matches = [
            event
            for event in self._events + self.created
            if event.starts_at < ends_before and event.ends_at > starts_after
        ]
        matches.sort(key=lambda event: event.starts_at)
        return matches[:limit]

    async def get_event(self, event_id: str, *, calendar_id: str = "primary") -> CalendarEventModel:
        for event in self._events + self.created:
            if event.id == event_id:
                return event
        raise NotFoundError(f"No event with id {event_id!r}.")

    async def free_busy(
        self, *, starts_at: datetime, ends_at: datetime, calendar_id: str = "primary"
    ) -> list[FreeBusySlot]:
        events = await self.list_events(starts_after=starts_at, ends_before=ends_at, limit=100)
        return [FreeBusySlot(starts_at=event.starts_at, ends_at=event.ends_at) for event in events]

    async def create_event(self, payload: dict[str, Any]) -> CalendarEventModel:
        from integrations.google.calendar import build_event_body

        build_event_body(payload, "UTC")  # same validation as the real provider
        event = CalendarEventModel(
            id=f"mock-evt-{len(self.created) + 1}",
            title=str(payload["title"]),
            description=payload.get("description"),
            location=payload.get("location"),
            starts_at=_as_datetime(payload["starts_at"]),
            ends_at=_as_datetime(payload["ends_at"]),
            attendees=list(payload.get("attendees") or []),
        )
        self.created.append(event)
        return event

    async def update_event(
        self, event_id: str, payload: dict[str, Any], *, calendar_id: str = "primary"
    ) -> CalendarEventModel:
        from integrations.google.calendar import build_event_patch

        build_event_patch(payload, "UTC")
        current = await self.get_event(event_id, calendar_id=calendar_id)
        data = current.model_dump()
        if payload.get("title") is not None:
            data["title"] = str(payload["title"]).strip()
        if payload.get("starts_at") is not None:
            data["starts_at"] = _as_datetime(payload["starts_at"])
        if payload.get("ends_at") is not None:
            data["ends_at"] = _as_datetime(payload["ends_at"])
        if "description" in payload:
            data["description"] = payload["description"]
        if "location" in payload:
            data["location"] = payload["location"]
        if "attendees" in payload and payload["attendees"] is not None:
            data["attendees"] = list(payload["attendees"])
        updated = CalendarEventModel(**data)
        for index, event in enumerate(self._events):
            if event.id == event_id:
                self._events[index] = updated
                break
        else:
            for index, event in enumerate(self.created):
                if event.id == event_id:
                    self.created[index] = updated
                    break
        self.updated.append(updated)
        return updated

    async def delete_event(self, event_id: str, *, calendar_id: str = "primary") -> None:
        self.deleted.append(event_id)

    async def health(self) -> dict[str, Any]:
        return {"ok": True, "mock": True, "events": len(self._events)}


class MockContactsService:
    name = "contacts"

    def __init__(self) -> None:
        self._contacts = [
            ContactRecord(
                id="people/c1",
                display_name="Nathan Chen",
                given_name="Nathan",
                family_name="Chen",
                emails=["nathan.chen@example.com"],
                phones=["+1-416-555-0142"],
                organisation="Example Robotics",
            ),
            ContactRecord(
                id="people/c2",
                display_name="Dr. Elena Marsh",
                given_name="Elena",
                family_name="Marsh",
                emails=["supervisor@university.edu"],
                organisation="University Robotics Lab",
            ),
            ContactRecord(
                id="people/c3",
                display_name="Nathan Brooks",
                given_name="Nathan",
                family_name="Brooks",
                emails=["n.brooks@example.org"],
            ),
        ]

    async def search(self, query: str, *, limit: int = 10) -> list[ContactRecord]:
        needle = query.strip().lower()
        if not needle:
            return []
        matches = [
            contact
            for contact in self._contacts
            if needle in contact.display_name.lower()
            or any(needle in email.lower() for email in contact.emails)
        ]
        return matches[:limit]

    async def list_contacts(self, *, limit: int = 50) -> list[ContactRecord]:
        return self._contacts[:limit]

    async def health(self) -> dict[str, Any]:
        return {"ok": True, "mock": True, "contacts": len(self._contacts)}


class MockDriveService:
    name = "drive"

    def __init__(self) -> None:
        self._files = [
            DriveFileRecord(
                id="file-001",
                name="Manipulation paper draft.gdoc",
                mime_type="application/vnd.google-apps.document",
                modified_at=_hours(-3),
                owner="Owner",
                web_view_link="https://docs.example.com/file-001",
            ),
            DriveFileRecord(
                id="file-002",
                name="Grasp results.csv",
                mime_type="text/csv",
                size_bytes=8422,
                modified_at=_hours(-30),
                owner="Owner",
            ),
            DriveFileRecord(
                id="file-003",
                name="Lab photos",
                mime_type="application/vnd.google-apps.folder",
                modified_at=_hours(-200),
                is_folder=True,
            ),
            DriveFileRecord(
                id="file-004",
                name="Poster.pdf",
                mime_type="application/pdf",
                size_bytes=2_400_000,
                modified_at=_hours(-100),
            ),
        ]
        self._contents = {
            "file-001": (
                "Learning Dexterous Manipulation\n\n"
                "Section 4 reports grasp success across 240 trials spanning three object "
                "classes. Mugs and blocks reach above ninety percent, bottles lag behind "
                "because the gripper loses contact during the lift phase. The ablation in "
                "table 3 removes the tactile channel and success falls by eleven points, "
                "which is the strongest evidence we have that the tactile input matters."
            ),
            "file-002": "object,trials,success\nmug,80,0.91\nblock,80,0.96\nbottle,80,0.84\n",
        }

    async def search(
        self, query: str, *, limit: int = 10, folder_id: str | None = None
    ) -> list[DriveFileRecord]:
        needle = query.strip().lower()
        matches = [record for record in self._files if not needle or needle in record.name.lower()]
        return matches[:limit]

    async def get_metadata(self, file_id: str) -> DriveFileRecord:
        for record in self._files:
            if record.id == file_id:
                return record
        raise NotFoundError(f"No Drive file with id {file_id!r}.")

    async def export_text(self, file_id: str, *, max_chars: int = 20000) -> DriveFileContent:
        record = await self.get_metadata(file_id)
        if record.is_folder:
            raise ValidationError("A folder has no text to read.")
        text = self._contents.get(file_id)
        if text is None:
            raise ValidationError(
                f"{record.name} is a {record.mime_type} file, which has no text form. "
                "Ask for its details instead."
            )
        return DriveFileContent(
            id=record.id,
            name=record.name,
            mime_type=record.mime_type,
            text=text[:max_chars],
            truncated=len(text) > max_chars,
        )

    async def health(self) -> dict[str, Any]:
        return {"ok": True, "mock": True, "files": len(self._files)}


def _as_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
