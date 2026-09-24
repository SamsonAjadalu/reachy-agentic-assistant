"""Gmail endpoints.

Uses the configured workflow.
here: ``POST /actions/send-draft`` only records a pending action. The message
leaves only after the owner approves the exact draft revision in Telegram.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, get_settings
from app.dependencies.auth import Principal, require_token
from app.dependencies.common import get_session, idempotency_key
from app.schemas.google import (
    ApprovalTicket,
    AttachmentDownloadResponse,
    AttachmentListResponse,
    CreateDraftRequest,
    CreateReplyDraftRequest,
    DraftListResponse,
    DraftResponse,
    EmailDetailResponse,
    EmailListResponse,
    MessageLabelsResponse,
    ModifyLabelsRequest,
    SendDraftRequest,
)
from app.services import idempotency as idem
from approvals.state_machine import request_approval
from integrations.google.executors import SEND_DRAFT, draft_send_preview
from integrations.google.gmail import validate_header, validate_recipients
from integrations.registry import get_gmail, require_enabled
from shared.enums import RiskLevel
from shared.errors import ValidationError

router = APIRouter(prefix="/api/v1/gmail", tags=["gmail"])

SessionDep = Annotated[AsyncSession, Depends(get_session)]
SettingsDep = Annotated[Settings, Depends(get_settings)]
PrincipalDep = Annotated[Principal, Depends(require_token)]
IdempotencyDep = Annotated[str | None, Depends(idempotency_key)]

DRAFT_CREATE_SCOPE = "gmail.drafts.create"
REPLY_DRAFT_CREATE_SCOPE = "gmail.reply_drafts.create"


@router.get("/messages", response_model=EmailListResponse, summary="List recent messages")
async def list_messages(
    _: PrincipalDep,
    settings: SettingsDep,
    unread_only: Annotated[bool, Query()] = False,
    limit: Annotated[int, Query(ge=1, le=50)] = 10,
    query: Annotated[str | None, Query(max_length=200, description="Gmail search syntax.")] = None,
) -> EmailListResponse:
    gmail = get_gmail(settings)
    items = await gmail.list_messages(query=query, unread_only=unread_only, limit=limit)
    return EmailListResponse(items=items, total=len(items), unread_count=await gmail.unread_count())


@router.get("/unread-count", summary="Count unread messages")
async def unread_count(_: PrincipalDep, settings: SettingsDep) -> dict[str, int]:
    return {"unread": await get_gmail(settings).unread_count()}


@router.get("/search", response_model=EmailListResponse, summary="Search the mailbox")
async def search_messages(
    _: PrincipalDep,
    settings: SettingsDep,
    q: Annotated[str, Query(min_length=1, max_length=200)],
    limit: Annotated[int, Query(ge=1, le=50)] = 10,
) -> EmailListResponse:
    items = await get_gmail(settings).search(q, limit=limit)
    return EmailListResponse(items=items, total=len(items))


@router.get(
    "/messages/{message_id}",
    response_model=EmailDetailResponse,
    summary="Read one message",
    description=(
        "Returns the plain-text body, truncated to `max_chars`. The body is third-party "
        "content and is flagged as untrusted in the response."
    ),
)
async def get_message(
    message_id: str,
    _: PrincipalDep,
    settings: SettingsDep,
    max_chars: Annotated[int, Query(ge=100, le=20000)] = 4000,
) -> EmailDetailResponse:
    message = await get_gmail(settings).get_message(message_id, max_chars=max_chars)
    return EmailDetailResponse(message=message)


@router.get(
    "/messages/{message_id}/attachments",
    response_model=AttachmentListResponse,
    summary="List attachments on a message",
)
async def list_attachments(
    message_id: str,
    _: PrincipalDep,
    settings: SettingsDep,
) -> AttachmentListResponse:
    require_enabled("gmail", settings)
    items = await get_gmail(settings).list_attachments(message_id)
    return AttachmentListResponse(items=items, total=len(items))


@router.post(
    "/messages/{message_id}/attachments/{attachment_id}/download",
    response_model=AttachmentDownloadResponse,
    summary="Download one attachment under APP_DATA_DIR",
)
async def download_attachment(
    message_id: str,
    attachment_id: str,
    _: PrincipalDep,
    settings: SettingsDep,
) -> AttachmentDownloadResponse:
    require_enabled("gmail", settings)
    dest = settings.app_data_dir / "gmail" / "attachments"
    download = await get_gmail(settings).download_attachment(
        message_id,
        attachment_id,
        dest_dir=dest,
        max_bytes=settings.gmail_attachment_max_bytes,
    )
    return AttachmentDownloadResponse(download=download)


@router.post(
    "/messages/{message_id}/actions/archive",
    response_model=MessageLabelsResponse,
    summary="Archive a message (remove INBOX)",
)
async def archive_message(
    message_id: str,
    _: PrincipalDep,
    settings: SettingsDep,
) -> MessageLabelsResponse:
    require_enabled("gmail", settings)
    labels = await get_gmail(settings).archive_message(message_id)
    return MessageLabelsResponse(message_id=message_id, labels=labels)


@router.post(
    "/messages/{message_id}/actions/modify-labels",
    response_model=MessageLabelsResponse,
    summary="Add or remove labels (sync, reversible)",
    description="Refuse TRASH/SPAM. Outbound send remains a separate approved action.",
)
async def modify_labels(
    message_id: str,
    payload: ModifyLabelsRequest,
    _: PrincipalDep,
    settings: SettingsDep,
) -> MessageLabelsResponse:
    require_enabled("gmail", settings)
    labels = await get_gmail(settings).modify_labels(
        message_id,
        add=payload.add_label_ids,
        remove=payload.remove_label_ids,
    )
    return MessageLabelsResponse(message_id=message_id, labels=labels)


@router.post(
    "/drafts",
    response_model=DraftResponse,
    status_code=201,
    summary="Create a Gmail draft",
    description=(
        "Stores a draft only. Nothing is transmitted. Sending requires a separate "
        "approved call to `/actions/send-draft`."
    ),
)
async def create_draft(
    payload: CreateDraftRequest,
    _: PrincipalDep,
    session: SessionDep,
    settings: SettingsDep,
    key: IdempotencyDep,
) -> DraftResponse:
    require_enabled("gmail", settings)
    replay = await idem.lookup(session, DRAFT_CREATE_SCOPE, key, payload)
    if replay is not None:
        return DraftResponse.model_validate(replay)

    recipients = validate_recipients(payload.to)
    copies = validate_recipients(payload.cc, allow_empty=True)
    subject = validate_header(payload.subject, "subject")
    draft = await get_gmail(settings).create_draft(
        to=recipients,
        subject=subject,
        body=payload.body,
        cc=copies,
    )
    result = DraftResponse(draft=draft)
    await idem.remember(
        session,
        DRAFT_CREATE_SCOPE,
        key,
        payload,
        result.model_dump(mode="json"),
        resource_id=draft.id,
    )
    return result


@router.post(
    "/messages/{message_id}/reply-draft",
    response_model=DraftResponse,
    status_code=201,
    summary="Create a threaded reply draft",
    description=(
        "Fetches the original message, preserves its thread id, and sets reply headers "
        "(In-Reply-To, References). Stores a draft only; sending still requires a separate "
        "approved call to `/actions/send-draft`."
    ),
)
async def create_reply_draft(
    message_id: str,
    payload: CreateReplyDraftRequest,
    _: PrincipalDep,
    session: SessionDep,
    settings: SettingsDep,
    key: IdempotencyDep,
) -> DraftResponse:
    require_enabled("gmail", settings)
    idem_payload = {"message_id": message_id, "body": payload.body}
    replay = await idem.lookup(session, REPLY_DRAFT_CREATE_SCOPE, key, idem_payload)
    if replay is not None:
        return DraftResponse.model_validate(replay)

    draft = await get_gmail(settings).create_reply_draft(message_id=message_id, body=payload.body)
    result = DraftResponse(draft=draft)
    await idem.remember(
        session,
        REPLY_DRAFT_CREATE_SCOPE,
        key,
        idem_payload,
        result.model_dump(mode="json"),
        resource_id=draft.id,
    )
    return result


@router.get("/drafts", response_model=DraftListResponse, summary="List open drafts")
async def list_drafts(
    _: PrincipalDep,
    settings: SettingsDep,
    limit: Annotated[int, Query(ge=1, le=50)] = 20,
) -> DraftListResponse:
    require_enabled("gmail", settings)
    items = await get_gmail(settings).list_drafts(limit=limit)
    return DraftListResponse(items=items, total=len(items))


@router.get("/drafts/{draft_id}", response_model=DraftResponse, summary="Read one draft")
async def get_draft(
    draft_id: str,
    _: PrincipalDep,
    settings: SettingsDep,
) -> DraftResponse:
    require_enabled("gmail", settings)
    return DraftResponse(draft=await get_gmail(settings).get_draft(draft_id))


@router.post(
    "/actions/send-draft",
    response_model=ApprovalTicket,
    status_code=202,
    summary="Request approval to send an existing draft",
    description=(
        "Does not send anything. Creates a pending action bound to the draft id and "
        "its content hash. The draft is sent only after the owner approves that exact "
        "revision."
    ),
)
async def send_existing_draft(
    payload: SendDraftRequest,
    request: Request,
    _: PrincipalDep,
    session: SessionDep,
    settings: SettingsDep,
    key: IdempotencyDep,
) -> ApprovalTicket:
    require_enabled("gmail", settings)
    draft = await get_gmail(settings).get_draft(payload.draft_id)
    if draft.sent:
        raise ValidationError(f"Draft {payload.draft_id!r} has already been sent.")
    if payload.content_hash and payload.content_hash != draft.content_hash:
        raise ValidationError(
            "The draft no longer matches the content hash you supplied; create a new "
            "approval request against the current revision."
        )

    action_payload = {
        "draft_id": draft.id,
        "content_hash": draft.content_hash,
        "to": list(draft.to),
        "cc": list(draft.cc),
        "subject": draft.subject,
        "body": draft.body,
    }
    preview = draft_send_preview(action_payload)

    action, approval = await request_approval(
        session,
        action_type=SEND_DRAFT,
        summary=f"Send draft to {', '.join(draft.to)}",
        payload=action_payload,
        preview_text=preview,
        risk_level=RiskLevel.EXTERNAL_WRITE,
        settings=settings,
        idempotency_key=key,
        correlation_id=getattr(request.state, "correlation_id", None),
    )

    return ApprovalTicket(
        approval_id=approval.id,
        action_id=action.id,
        summary=action.summary,
        preview=action.preview_text,
        expires_at=approval.expires_at,
    )


@router.post(
    "/send",
    status_code=410,
    summary="Removed: compose-and-send is not supported",
    include_in_schema=False,
)
async def send_email_removed() -> None:
    """Previous agents incorrectly mapped draft creation onto this path."""
    raise HTTPException(
        status_code=410,
        detail=(
            "POST /api/v1/gmail/send is removed. Create a draft with POST /api/v1/gmail/drafts, "
            "then request send approval with POST /api/v1/gmail/actions/send-draft."
        ),
    )
