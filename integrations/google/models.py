"""Normalised shapes returned by the Google providers.

Uses the configured workflow.
without notice, and carry fields the assistant has no business handing to a
conversational model. Everything is reduced to a small, stable record here, and
the real and mock providers return the identical type so a test against the mock
proves something about the real path.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class EmailSummary(BaseModel):
    """A message as the assistant describes it out loud."""

    id: str
    thread_id: str
    sender: str
    sender_name: str | None = None
    recipients: list[str] = Field(default_factory=list)
    subject: str
    snippet: str
    received_at: datetime
    is_unread: bool = False
    is_important: bool = False
    has_attachments: bool = False
    labels: list[str] = Field(default_factory=list)
    rfc_message_id: str | None = None
    references: str | None = None


class EmailBody(EmailSummary):
    body_text: str = ""
    truncated: bool = False


class EmailDraft(BaseModel):
    """Uses the configured workflow."""

    id: str
    message_id: str | None = None
    thread_id: str | None = None
    to: list[str] = Field(default_factory=list)
    cc: list[str] = Field(default_factory=list)
    subject: str
    body: str
    content_hash: str
    """SHA-256 of the exact recipients/subject/body that must be re-checked on send."""
    created_at: datetime | None = None
    sent: bool = False
    in_reply_to_message_id: str | None = None
    in_reply_to: str | None = None
    references: str | None = None


class CalendarEventModel(BaseModel):
    id: str
    calendar_id: str = "primary"
    title: str
    description: str | None = None
    location: str | None = None
    starts_at: datetime
    ends_at: datetime
    all_day: bool = False
    attendees: list[str] = Field(default_factory=list)
    organiser: str | None = None
    status: str = "confirmed"
    conference_url: str | None = None
    html_link: str | None = None


class FreeBusySlot(BaseModel):
    starts_at: datetime
    ends_at: datetime


class ProposedSlot(BaseModel):
    starts_at: datetime
    ends_at: datetime
    conflicts: list[str] = Field(default_factory=list)


class EmailAttachment(BaseModel):
    id: str
    filename: str
    mime_type: str
    size_bytes: int | None = None


class AttachmentDownload(BaseModel):
    attachment_id: str
    filename: str
    mime_type: str
    size_bytes: int
    path: str
    text: str | None = None
    truncated: bool = False


class ContactRecord(BaseModel):
    id: str
    display_name: str
    given_name: str | None = None
    family_name: str | None = None
    emails: list[str] = Field(default_factory=list)
    phones: list[str] = Field(default_factory=list)
    organisation: str | None = None
    photo_url: str | None = None


class DriveFileRecord(BaseModel):
    id: str
    name: str
    mime_type: str
    size_bytes: int | None = None
    modified_at: datetime | None = None
    owner: str | None = None
    web_view_link: str | None = None
    is_folder: bool = False


class DriveFileContent(BaseModel):
    id: str
    name: str
    mime_type: str
    text: str
    truncated: bool = False
