"""Provider interfaces for the Google services.

Every service has exactly two implementations, real and mock, and the rest of
the application depends only on these protocols. That is what allows the whole
system - API, scheduler, briefings, Reachy tools - to be exercised end to end
with no Google account at all.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

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


@runtime_checkable
class GmailProvider(Protocol):
    async def list_messages(
        self,
        *,
        query: str | None = None,
        unread_only: bool = False,
        limit: int = 10,
        label: str | None = None,
    ) -> list[EmailSummary]: ...

    async def get_message(self, message_id: str, *, max_chars: int = 4000) -> EmailBody: ...

    async def search(self, query: str, *, limit: int = 10) -> list[EmailSummary]: ...

    async def unread_count(self) -> int: ...

    async def create_draft(
        self,
        *,
        to: list[str],
        subject: str,
        body: str,
        cc: list[str] | None = None,
    ) -> EmailDraft: ...

    async def create_reply_draft(self, *, message_id: str, body: str) -> EmailDraft: ...

    async def list_drafts(self, *, limit: int = 20) -> list[EmailDraft]: ...

    async def get_draft(self, draft_id: str) -> EmailDraft: ...

    async def send_draft(self, draft_id: str, *, expected_content_hash: str) -> dict[str, Any]: ...

    async def list_attachments(self, message_id: str) -> list[EmailAttachment]: ...

    async def download_attachment(
        self,
        message_id: str,
        attachment_id: str,
        *,
        dest_dir: Path,
        max_bytes: int,
    ) -> AttachmentDownload: ...

    async def modify_labels(
        self,
        message_id: str,
        *,
        add: list[str] | None = None,
        remove: list[str] | None = None,
    ) -> list[str]: ...

    async def archive_message(self, message_id: str) -> list[str]: ...

    async def health(self) -> dict[str, Any]: ...


@runtime_checkable
class CalendarProvider(Protocol):
    async def list_events(
        self,
        *,
        starts_after: datetime,
        ends_before: datetime,
        calendar_id: str = "primary",
        limit: int = 25,
    ) -> list[CalendarEventModel]: ...

    async def get_event(
        self, event_id: str, *, calendar_id: str = "primary"
    ) -> CalendarEventModel: ...

    async def free_busy(
        self, *, starts_at: datetime, ends_at: datetime, calendar_id: str = "primary"
    ) -> list[FreeBusySlot]: ...

    async def create_event(self, payload: dict[str, Any]) -> CalendarEventModel: ...

    async def update_event(
        self, event_id: str, payload: dict[str, Any], *, calendar_id: str = "primary"
    ) -> CalendarEventModel: ...

    async def delete_event(self, event_id: str, *, calendar_id: str = "primary") -> None: ...

    async def health(self) -> dict[str, Any]: ...


@runtime_checkable
class ContactsProvider(Protocol):
    async def search(self, query: str, *, limit: int = 10) -> list[ContactRecord]: ...

    async def list_contacts(self, *, limit: int = 50) -> list[ContactRecord]: ...

    async def health(self) -> dict[str, Any]: ...


@runtime_checkable
class DriveProvider(Protocol):
    async def search(
        self, query: str, *, limit: int = 10, folder_id: str | None = None
    ) -> list[DriveFileRecord]: ...

    async def get_metadata(self, file_id: str) -> DriveFileRecord: ...

    async def export_text(self, file_id: str, *, max_chars: int = 20000) -> DriveFileContent: ...

    async def health(self) -> dict[str, Any]: ...
