"""Gmail provider.

Reads and draft creation are direct. Transmitting mail is not: ``send_draft``
is the only method that leaves the mailbox, and it is only registered as an
approval executor. The executor re-reads the draft and checks its content hash
against the approved revision before calling Gmail's drafts/send.

Message bodies are treated as hostile input. They are decoded, stripped of HTML,
truncated, and marked as untrusted content when they reach the assistant, so a
"forget your instructions" line in an email is data rather than direction.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from email.message import EmailMessage
from email.utils import parseaddr, parsedate_to_datetime
from pathlib import Path
from typing import Any

from app.config import Settings, get_settings
from app.logging_config import get_logger
from integrations.google.client import GMAIL_BASE, GoogleClient
from integrations.google.models import (
    AttachmentDownload,
    EmailAttachment,
    EmailBody,
    EmailDraft,
    EmailSummary,
)
from security.paths import safe_filename
from shared.contracts import payload_hash
from shared.errors import IntegrationError, NotFoundError, ValidationError

logger = get_logger(__name__)

GMAIL_READ_SCOPE = "https://www.googleapis.com/auth/gmail.readonly"
GMAIL_COMPOSE_SCOPE = "https://www.googleapis.com/auth/gmail.compose"
GMAIL_SEND_SCOPE = "https://www.googleapis.com/auth/gmail.send"
GMAIL_MODIFY_SCOPE = "https://www.googleapis.com/auth/gmail.modify"

MAX_RESULTS = 50
MAX_BODY_CHARS = 20000
MAX_RECIPIENTS = 10
# Uses the configured workflow.
FORBIDDEN_LABEL_IDS = frozenset({"TRASH", "SPAM"})
DEFAULT_ATTACHMENT_MAX_BYTES = 15 * 1024 * 1024

_HTML_TAG = re.compile(r"<[^>]+>")
_WHITESPACE = re.compile(r"[ \t]*\n[ \t]*")
_EMAIL_RE = re.compile(r"^[^@\s,;<>\"]+@[^@\s,;<>\"]+\.[a-zA-Z]{2,}$")
_REPLY_PREFIX = re.compile(r"^(re|fwd):\s", re.IGNORECASE)
# A header value containing CR or LF would let a caller append headers of their
# own, which is how a "send to Nathan" request becomes a Bcc to a stranger.
_HEADER_INJECTION = re.compile(r"[\r\n]")


@dataclass(frozen=True)
class ReplyContext:
    message_id: str
    thread_id: str
    subject: str
    reply_to: str
    rfc_message_id: str
    references: str


class GmailService:
    """Real Gmail access over the REST API."""

    name = "gmail"

    def __init__(self, client: GoogleClient | None = None, settings: Settings | None = None):
        self._settings = settings or get_settings()
        self._client = client or GoogleClient(self._settings)

    async def aclose(self) -> None:
        await self._client.aclose()

    # ------------------------------------------------------------------ read
    async def list_messages(
        self,
        *,
        query: str | None = None,
        unread_only: bool = False,
        limit: int = 10,
        label: str | None = None,
    ) -> list[EmailSummary]:
        self._client.tokens.require_scope(GMAIL_READ_SCOPE)
        limit = max(1, min(limit, MAX_RESULTS))

        terms = [query] if query else []
        if unread_only:
            terms.append("is:unread")
        params: dict[str, Any] = {}
        if terms:
            params["q"] = " ".join(terms)
        if label:
            params["labelIds"] = label

        ids: list[str] = []
        async for item in self._client.paginate(
            f"{GMAIL_BASE}/users/me/messages",
            params=params,
            items_key="messages",
            limit=limit,
            page_size_param="maxResults",
        ):
            ids.append(str(item["id"]))

        return [await self._fetch_summary(message_id) for message_id in ids]

    async def search(self, query: str, *, limit: int = 10) -> list[EmailSummary]:
        return await self.list_messages(query=query, limit=limit)

    async def unread_count(self) -> int:
        self._client.tokens.require_scope(GMAIL_READ_SCOPE)
        body = await self._client.request("GET", f"{GMAIL_BASE}/users/me/labels/UNREAD")
        return int(body.get("messagesUnread", 0) or 0)

    async def get_message(self, message_id: str, *, max_chars: int = 4000) -> EmailBody:
        self._client.tokens.require_scope(GMAIL_READ_SCOPE)
        payload = await self._client.request(
            "GET", f"{GMAIL_BASE}/users/me/messages/{message_id}", params={"format": "full"}
        )
        summary = _to_summary(payload)
        text, truncated = _extract_body(
            payload.get("payload") or {}, min(max_chars, MAX_BODY_CHARS)
        )
        return EmailBody(**summary.model_dump(), body_text=text, truncated=truncated)

    async def _fetch_summary(self, message_id: str) -> EmailSummary:
        payload = await self._client.request(
            "GET",
            f"{GMAIL_BASE}/users/me/messages/{message_id}",
            params={
                "format": "metadata",
                "metadataHeaders": ["From", "To", "Subject", "Date"],
            },
        )
        return _to_summary(payload)

    # ----------------------------------------------------------------- drafts
    async def create_draft(
        self,
        *,
        to: list[str],
        subject: str,
        body: str,
        cc: list[str] | None = None,
    ) -> EmailDraft:
        """Uses the configured workflow."""
        self._client.tokens.require_scope(GMAIL_COMPOSE_SCOPE)
        recipients = validate_recipients(to)
        copies = validate_recipients(cc or [], allow_empty=True)
        clean_subject = validate_header(subject, "subject")
        raw = _encode_raw_message(to=recipients, cc=copies, subject=clean_subject, body=body)
        result = await self._client.request(
            "POST",
            f"{GMAIL_BASE}/users/me/drafts",
            json_body={"message": {"raw": raw}},
        )
        draft_id = str(result.get("id", ""))
        message = result.get("message") or {}
        return EmailDraft(
            id=draft_id,
            message_id=str(message.get("id") or "") or None,
            thread_id=str(message.get("threadId") or "") or None,
            to=recipients,
            cc=copies,
            subject=clean_subject,
            body=body,
            content_hash=draft_content_hash(
                to=recipients, cc=copies, subject=clean_subject, body=body
            ),
        )

    async def create_reply_draft(self, *, message_id: str, body: str) -> EmailDraft:
        """Uses the configured workflow."""
        self._client.tokens.require_scope(GMAIL_READ_SCOPE)
        self._client.tokens.require_scope(GMAIL_COMPOSE_SCOPE)
        context = await self._fetch_reply_context(message_id)
        clean_subject = validate_header(build_reply_subject(context.subject), "subject")
        recipients = validate_recipients([context.reply_to])
        references = build_references_header(context.references, context.rfc_message_id)
        raw = _encode_raw_message(
            to=recipients,
            cc=[],
            subject=clean_subject,
            body=body,
            in_reply_to=context.rfc_message_id,
            references=references,
        )
        result = await self._client.request(
            "POST",
            f"{GMAIL_BASE}/users/me/drafts",
            json_body={"message": {"raw": raw, "threadId": context.thread_id}},
        )
        draft_id = str(result.get("id", ""))
        message = result.get("message") or {}
        return EmailDraft(
            id=draft_id,
            message_id=str(message.get("id") or "") or None,
            thread_id=str(message.get("threadId") or context.thread_id) or None,
            to=recipients,
            cc=[],
            subject=clean_subject,
            body=body,
            content_hash=draft_content_hash(to=recipients, cc=[], subject=clean_subject, body=body),
            in_reply_to_message_id=context.message_id,
            in_reply_to=context.rfc_message_id,
            references=references,
        )

    async def _fetch_reply_context(self, message_id: str) -> ReplyContext:
        payload = await self._client.request(
            "GET",
            f"{GMAIL_BASE}/users/me/messages/{message_id}",
            params={
                "format": "metadata",
                "metadataHeaders": [
                    "From",
                    "To",
                    "Cc",
                    "Subject",
                    "Message-ID",
                    "References",
                ],
            },
        )
        headers = {
            str(header.get("name", "")).lower(): str(header.get("value", ""))
            for header in (payload.get("payload") or {}).get("headers") or []
        }
        _, from_email = parseaddr(headers.get("from", ""))
        if not from_email:
            raise ValidationError("Cannot reply: the original message has no From address.")
        rfc_message_id = headers.get("message-id", "").strip()
        if not rfc_message_id:
            raise ValidationError("Cannot reply: the original message has no Message-ID header.")
        thread_id = str(payload.get("threadId") or "").strip()
        if not thread_id:
            raise ValidationError("Cannot reply: the original message has no thread id.")
        return ReplyContext(
            message_id=message_id,
            thread_id=thread_id,
            subject=headers.get("subject", "(no subject)"),
            reply_to=from_email,
            rfc_message_id=rfc_message_id,
            references=headers.get("references", "").strip(),
        )

    async def list_drafts(self, *, limit: int = 20) -> list[EmailDraft]:
        self._client.tokens.require_scope(GMAIL_COMPOSE_SCOPE)
        limit = max(1, min(limit, MAX_RESULTS))
        drafts: list[EmailDraft] = []
        async for item in self._client.paginate(
            f"{GMAIL_BASE}/users/me/drafts",
            params={},
            items_key="drafts",
            limit=limit,
            page_size_param="maxResults",
        ):
            drafts.append(await self.get_draft(str(item["id"])))
        return drafts

    async def get_draft(self, draft_id: str) -> EmailDraft:
        self._client.tokens.require_scope(GMAIL_COMPOSE_SCOPE)
        payload = await self._client.request(
            "GET",
            f"{GMAIL_BASE}/users/me/drafts/{draft_id}",
            params={"format": "full"},
        )
        return _draft_from_payload(payload)

    async def send_draft(self, draft_id: str, *, expected_content_hash: str) -> dict[str, Any]:
        """Transmit an existing draft after verifying its content hash.

        Only ever reached through an approved action. Re-reading the draft and
        hashing it again is what stops a draft edited after approval from going out.
        """
        self._client.tokens.require_scope(GMAIL_SEND_SCOPE)
        try:
            draft = await self.get_draft(draft_id)
        except IntegrationError as exc:
            # Gmail deletes a draft after a successful send; a retry that lands
            # Uses the configured workflow.
            raise ValidationError(
                "This draft is gone or already sent, so nothing was transmitted again."
            ) from exc
        if draft.content_hash != expected_content_hash:
            raise ValidationError(
                "The draft no longer matches the content that was approved, so it was not sent."
            )
        if draft.sent:
            raise ValidationError("This draft has already been sent.")

        result = await self._client.request(
            "POST",
            f"{GMAIL_BASE}/users/me/drafts/send",
            json_body={"id": draft_id},
        )
        logger.info(
            "Sent a Gmail draft",
            extra={
                "draft_id": draft_id,
                "recipient_count": len(draft.to) + len(draft.cc),
                "message_id": result.get("id"),
            },
        )
        return {
            "message_id": result.get("id"),
            "thread_id": result.get("threadId"),
            "draft_id": draft_id,
            "recipients": draft.to,
            "cc": draft.cc,
            "subject": draft.subject,
            "external_id": result.get("id"),
        }

    async def list_attachments(self, message_id: str) -> list[EmailAttachment]:
        self._client.tokens.require_scope(GMAIL_READ_SCOPE)
        payload = await self._client.request(
            "GET",
            f"{GMAIL_BASE}/users/me/messages/{message_id}",
            params={"format": "full"},
        )
        return _collect_attachments(payload.get("payload") or {})

    async def download_attachment(
        self,
        message_id: str,
        attachment_id: str,
        *,
        dest_dir: Path,
        max_bytes: int = DEFAULT_ATTACHMENT_MAX_BYTES,
    ) -> AttachmentDownload:
        self._client.tokens.require_scope(GMAIL_READ_SCOPE)
        attachments = await self.list_attachments(message_id)
        meta = next((item for item in attachments if item.id == attachment_id), None)
        if meta is None:
            raise NotFoundError(f"No attachment {attachment_id!r} on message {message_id!r}.")

        body = await self._client.request(
            "GET",
            f"{GMAIL_BASE}/users/me/messages/{message_id}/attachments/{attachment_id}",
        )
        raw = _decode_bytes(str(body.get("data") or ""))
        if len(raw) > max_bytes:
            raise ValidationError(
                f"Attachment is {len(raw)} bytes; the limit is {max_bytes} bytes."
            )

        filename = safe_filename(meta.filename or "attachment")
        path = dest_dir / f"{message_id}_{attachment_id[:12]}_{filename}"
        await asyncio.to_thread(_write_attachment_bytes, dest_dir, path, raw)

        text: str | None = None
        truncated = False
        if (meta.mime_type or "").startswith("text/"):
            decoded = raw.decode("utf-8", errors="replace")
            if len(decoded) > MAX_BODY_CHARS:
                text = decoded[:MAX_BODY_CHARS]
                truncated = True
            else:
                text = decoded

        return AttachmentDownload(
            attachment_id=attachment_id,
            filename=filename,
            mime_type=meta.mime_type or "application/octet-stream",
            size_bytes=len(raw),
            path=str(path),
            text=text,
            truncated=truncated,
        )

    async def modify_labels(
        self,
        message_id: str,
        *,
        add: list[str] | None = None,
        remove: list[str] | None = None,
    ) -> list[str]:
        self._client.tokens.require_scope(GMAIL_MODIFY_SCOPE)
        add_ids = validate_label_ids(add or [])
        remove_ids = validate_label_ids(remove or [])
        if not add_ids and not remove_ids:
            raise ValidationError("Provide at least one label to add or remove.")

        result = await self._client.request(
            "POST",
            f"{GMAIL_BASE}/users/me/messages/{message_id}/modify",
            json_body={"addLabelIds": add_ids, "removeLabelIds": remove_ids},
        )
        return [str(label) for label in result.get("labelIds") or []]

    async def archive_message(self, message_id: str) -> list[str]:
        return await self.modify_labels(message_id, remove=["INBOX"])

    async def health(self) -> dict[str, Any]:
        if not self._client.tokens.is_authorised():
            return {"ok": False, "detail": "not authorised"}
        try:
            profile = await self._client.request("GET", f"{GMAIL_BASE}/users/me/profile")
        except IntegrationError as exc:
            return {"ok": False, "detail": str(exc)}
        return {
            "ok": True,
            "email": profile.get("emailAddress"),
            "total_messages": profile.get("messagesTotal"),
        }


def validate_recipients(values: list[str], *, allow_empty: bool = False) -> list[str]:
    cleaned: list[str] = []
    for value in values:
        address = parseaddr(value)[1].strip()
        if not _EMAIL_RE.match(address):
            raise ValidationError(f"{value!r} is not a valid email address.")
        cleaned.append(address)

    if not cleaned and not allow_empty:
        raise ValidationError("At least one recipient is required.")
    if len(cleaned) > MAX_RECIPIENTS:
        raise ValidationError(
            f"Too many recipients ({len(cleaned)}); the limit is {MAX_RECIPIENTS}."
        )
    return cleaned


def validate_header(value: str, field: str) -> str:
    if _HEADER_INJECTION.search(value):
        raise ValidationError(f"The {field} supports single-line values.")
    return value.strip()[:500]


def validate_label_ids(values: list[str]) -> list[str]:
    cleaned: list[str] = []
    for value in values:
        label = value.strip()
        if not label:
            continue
        if label.upper() in FORBIDDEN_LABEL_IDS:
            raise ValidationError(
                f"Label {label!r} cannot be changed through this path. "
                "Trash and spam require a separate approved action."
            )
        if len(label) > 100 or "/" in label or ".." in label:
            raise ValidationError(f"Label id {label!r} is not allowed.")
        cleaned.append(label)
    return cleaned


def build_reply_subject(subject: str) -> str:
    stripped = subject.strip() or "(no subject)"
    if _REPLY_PREFIX.match(stripped):
        return stripped
    return f"Re: {stripped}"


def build_references_header(existing_references: str, rfc_message_id: str) -> str:
    existing = existing_references.strip()
    message_id = rfc_message_id.strip()
    if existing and message_id:
        return f"{existing} {message_id}"
    return message_id or existing


def draft_content_hash(
    *,
    to: list[str],
    cc: list[str],
    subject: str,
    body: str,
) -> str:
    """Stable hash of the exact revision that approval must bind to."""
    return payload_hash({"to": to, "cc": cc, "subject": subject, "body": body})


def _encode_raw_message(
    *,
    to: list[str],
    cc: list[str],
    subject: str,
    body: str,
    in_reply_to: str | None = None,
    references: str | None = None,
) -> str:
    message = EmailMessage()
    message["To"] = ", ".join(to)
    if cc:
        message["Cc"] = ", ".join(cc)
    message["Subject"] = subject
    if in_reply_to:
        message["In-Reply-To"] = in_reply_to
    if references:
        message["References"] = references
    message.set_content(body)
    return base64.urlsafe_b64encode(message.as_bytes()).decode("ascii")


def _draft_from_payload(payload: dict[str, Any]) -> EmailDraft:
    message = payload.get("message") or {}
    headers = {
        str(header.get("name", "")).lower(): str(header.get("value", ""))
        for header in (message.get("payload") or {}).get("headers") or []
    }
    to = [
        parseaddr(part)[1]
        for part in headers.get("to", "").split(",")
        if part.strip() and parseaddr(part)[1]
    ]
    cc = [
        parseaddr(part)[1]
        for part in headers.get("cc", "").split(",")
        if part.strip() and parseaddr(part)[1]
    ]
    subject = headers.get("subject", "(no subject)")
    body_text, _ = _extract_body(message.get("payload") or {}, MAX_BODY_CHARS)
    return EmailDraft(
        id=str(payload.get("id", "")),
        message_id=str(message.get("id") or "") or None,
        thread_id=str(message.get("threadId") or "") or None,
        to=to,
        cc=cc,
        subject=subject,
        body=body_text,
        content_hash=draft_content_hash(to=to, cc=cc, subject=subject, body=body_text),
    )


def _to_summary(payload: dict[str, Any]) -> EmailSummary:
    headers = {
        str(header.get("name", "")).lower(): str(header.get("value", ""))
        for header in (payload.get("payload") or {}).get("headers") or []
    }
    sender_name, sender_email = parseaddr(headers.get("from", ""))
    label_ids = [str(label) for label in payload.get("labelIds") or []]

    return EmailSummary(
        id=str(payload.get("id", "")),
        thread_id=str(payload.get("threadId", "")),
        sender=sender_email or headers.get("from", "unknown"),
        sender_name=sender_name or None,
        recipients=[
            parseaddr(part)[1] for part in headers.get("to", "").split(",") if part.strip()
        ],
        subject=headers.get("subject", "(no subject)"),
        snippet=_clean(payload.get("snippet", ""))[:500],
        received_at=_received_at(payload, headers),
        is_unread="UNREAD" in label_ids,
        is_important="IMPORTANT" in label_ids,
        has_attachments=_has_attachments(payload.get("payload") or {}),
        labels=label_ids,
        rfc_message_id=headers.get("message-id") or None,
        references=headers.get("references") or None,
    )


def _received_at(payload: dict[str, Any], headers: dict[str, str]) -> datetime:
    """Prefer Gmail's internal timestamp; a forged Date header is common in spam."""
    internal = payload.get("internalDate")
    if internal:
        try:
            return datetime.fromtimestamp(int(internal) / 1000, tz=UTC)
        except (ValueError, OSError, OverflowError):
            pass
    raw_date = headers.get("date")
    if raw_date:
        try:
            parsed = parsedate_to_datetime(raw_date)
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
        except (TypeError, ValueError):
            pass
    return datetime.now(UTC)


def _has_attachments(part: dict[str, Any]) -> bool:
    if part.get("filename"):
        return True
    return any(_has_attachments(child) for child in part.get("parts") or [])


def _extract_body(part: dict[str, Any], max_chars: int) -> tuple[str, bool]:
    """Pull readable text out of a MIME tree, preferring text/plain."""
    plain = _find_part(part, "text/plain")
    if plain is not None:
        text = _clean(plain)
    else:
        html = _find_part(part, "text/html")
        text = _clean(_HTML_TAG.sub(" ", html)) if html else ""
    if len(text) > max_chars:
        return text[:max_chars], True
    return text, False


def _find_part(part: dict[str, Any], mime_type: str) -> str | None:
    if part.get("mimeType") == mime_type:
        data = (part.get("body") or {}).get("data")
        if data:
            return _decode(data)
    for child in part.get("parts") or []:
        found = _find_part(child, mime_type)
        if found:
            return found
    return None


def _decode(data: str) -> str:
    try:
        return base64.urlsafe_b64decode(data + "=" * (-len(data) % 4)).decode(
            "utf-8", errors="replace"
        )
    except (binascii.Error, ValueError):
        return ""


def _clean(text: str) -> str:
    return _WHITESPACE.sub("\n", text.replace("\r\n", "\n").replace("\xa0", " ")).strip()


def _collect_attachments(part: dict[str, Any]) -> list[EmailAttachment]:
    found: list[EmailAttachment] = []
    filename = str(part.get("filename") or "")
    body = part.get("body") or {}
    attachment_id = body.get("attachmentId")
    if filename and attachment_id:
        found.append(
            EmailAttachment(
                id=str(attachment_id),
                filename=filename,
                mime_type=str(part.get("mimeType") or "application/octet-stream"),
                size_bytes=int(body["size"]) if body.get("size") is not None else None,
            )
        )
    for child in part.get("parts") or []:
        found.extend(_collect_attachments(child))
    return found


def _decode_bytes(data: str) -> bytes:
    try:
        return base64.urlsafe_b64decode(data + "=" * (-len(data) % 4))
    except (binascii.Error, ValueError):
        return b""


def _write_attachment_bytes(dest_dir: Path, path: Path, raw: bytes) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
