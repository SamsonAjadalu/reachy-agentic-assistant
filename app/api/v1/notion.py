"""Notion endpoints.

Reads are synchronous and confined to the allowlist. Appending to a page is a
write other people can see, so it goes through approval like any other.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query, Request
from pydantic import Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, get_settings
from app.dependencies.auth import Principal, require_token
from app.dependencies.common import get_session, idempotency_key
from app.schemas.common import ApiModel
from app.schemas.google import ApprovalTicket
from approvals.state_machine import request_approval
from integrations.notion.executors import APPEND_BLOCK, append_preview
from integrations.notion.models import (
    NotionDatabaseRow,
    NotionPageContent,
    NotionPageSummary,
)
from integrations.registry import get_notion
from shared.enums import RiskLevel

router = APIRouter(prefix="/api/v1/notion", tags=["notion"])

SessionDep = Annotated[AsyncSession, Depends(get_session)]
SettingsDep = Annotated[Settings, Depends(get_settings)]
PrincipalDep = Annotated[Principal, Depends(require_token)]
IdempotencyDep = Annotated[str | None, Depends(idempotency_key)]


class PageListResponse(ApiModel):
    items: list[NotionPageSummary]
    total: int


class RowListResponse(ApiModel):
    items: list[NotionDatabaseRow]
    total: int


class AppendRequest(ApiModel):
    page_id: str
    text: Annotated[str, Field(min_length=1, max_length=1900)]


@router.get(
    "/allowlist",
    summary="Which Notion pages are reachable",
    description="Returns the configured page ids. Nothing outside this list can be read.",
)
async def allowlist(_: PrincipalDep, settings: SettingsDep) -> dict[str, object]:
    ids = sorted(get_notion(settings).allowed_ids)
    return {"allowed_page_ids": ids, "total": len(ids)}


@router.get("/search", response_model=PageListResponse, summary="Search allowlisted pages")
async def search_pages(
    _: PrincipalDep,
    settings: SettingsDep,
    q: Annotated[str, Query(min_length=1, max_length=100)],
    limit: Annotated[int, Query(ge=1, le=25)] = 10,
) -> PageListResponse:
    items = await get_notion(settings).search(q, limit=limit)
    return PageListResponse(items=items, total=len(items))


@router.get("/pages/{page_id}", response_model=NotionPageSummary, summary="Page details")
async def get_page(page_id: str, _: PrincipalDep, settings: SettingsDep) -> NotionPageSummary:
    page: NotionPageSummary = await get_notion(settings).get_page(page_id)
    return page


@router.get(
    "/pages/{page_id}/content",
    response_model=NotionPageContent,
    summary="Read a page as text",
    description="Flattens the page's blocks. The text is third-party content and is "
    "flagged as untrusted.",
)
async def get_content(
    page_id: str,
    _: PrincipalDep,
    settings: SettingsDep,
    max_chars: Annotated[int, Query(ge=100, le=50000)] = 8000,
) -> NotionPageContent:
    content: NotionPageContent = await get_notion(settings).get_page_content(
        page_id, max_chars=max_chars
    )
    return content


@router.get(
    "/databases/{database_id}/rows", response_model=RowListResponse, summary="Query a database"
)
async def query_database(
    database_id: str,
    _: PrincipalDep,
    settings: SettingsDep,
    limit: Annotated[int, Query(ge=1, le=100)] = 25,
) -> RowListResponse:
    items = await get_notion(settings).query_database(database_id, limit=limit)
    return RowListResponse(items=items, total=len(items))


@router.post(
    "/pages/{page_id}/append",
    response_model=ApprovalTicket,
    status_code=202,
    summary="Request to append to a page",
)
async def append_to_page(
    page_id: str,
    payload: AppendRequest,
    request: Request,
    _: PrincipalDep,
    session: SessionDep,
    settings: SettingsDep,
    key: IdempotencyDep,
) -> ApprovalTicket:
    client = get_notion(settings)
    # Resolving the page first does two things: it enforces the allowlist before
    # an approval is created, and it puts the page's real title in the prompt.
    page = await client.get_page(page_id)

    action_payload = {"page_id": page.id, "text": payload.text}
    action, approval = await request_approval(
        session,
        action_type=APPEND_BLOCK,
        summary=f"Append a note to '{page.title}' in Notion",
        payload=action_payload,
        preview_text=append_preview(page.title, payload.text),
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
