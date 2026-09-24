"""Approval inspection and resolution endpoints.

Telegram is the primary channel; these endpoints exist so the CLI, the test
suite and a future Reachy voice confirmation can drive the same state machine.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query
from pydantic import Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, get_settings
from app.dependencies.auth import Principal, require_token
from app.dependencies.common import Pagination, get_session, pagination
from app.schemas.action_status import ActionStatusOut
from app.schemas.common import ApiModel
from app.services import action_status as status_service
from app.services import approvals as service
from database.models import Approval, PendingAction
from shared.enums import ApprovalStatus
from shared.errors import NotFoundError

router = APIRouter(prefix="/api/v1/approvals", tags=["approvals"])

SessionDep = Annotated[AsyncSession, Depends(get_session)]
SettingsDep = Annotated[Settings, Depends(get_settings)]
PrincipalDep = Annotated[Principal, Depends(require_token)]
PaginationDep = Annotated[Pagination, Depends(pagination)]


class ApprovalOut(ApiModel):
    id: str
    action_id: str
    action_type: str
    risk_level: str
    status: ApprovalStatus
    summary: str
    preview_text: str
    payload_hash: str = Field(description="SHA-256 of the canonical payload that was approved.")
    requested_at: datetime
    expires_at: datetime
    resolved_at: datetime | None = None
    executed_at: datetime | None = None
    execution_error: str | None = None
    external_id: str | None = None


class ApprovalList(ApiModel):
    items: list[ApprovalOut]
    total: int
    limit: int
    offset: int


class ApprovalDecision(ApiModel):
    approve: bool
    note: str | None = Field(default=None, max_length=500)


def _to_out(approval: Approval, action: PendingAction) -> ApprovalOut:
    return ApprovalOut(
        id=approval.id,
        action_id=action.id,
        action_type=action.action_type,
        risk_level=action.risk_level,
        status=ApprovalStatus(approval.status),
        summary=action.summary,
        preview_text=action.preview_text,
        payload_hash=action.payload_hash,
        requested_at=approval.requested_at,
        expires_at=approval.expires_at,
        resolved_at=approval.resolved_at,
        executed_at=action.executed_at,
        execution_error=action.execution_error,
        external_id=action.external_id,
    )


@router.get("", response_model=ApprovalList, summary="List approvals")
async def list_approvals(
    _: PrincipalDep,
    session: SessionDep,
    page: PaginationDep,
    status: Annotated[ApprovalStatus | None, Query()] = None,
) -> ApprovalList:
    conditions = [Approval.status == status.value] if status is not None else []
    total = await session.scalar(select(func.count()).select_from(Approval).where(*conditions)) or 0
    rows = (
        await session.scalars(
            select(Approval)
            .where(*conditions)
            .order_by(Approval.requested_at.desc())
            .limit(page.limit)
            .offset(page.offset)
        )
    ).all()

    items = []
    for approval in rows:
        action = await session.get(PendingAction, approval.pending_action_id)
        if action is not None:
            items.append(_to_out(approval, action))
    return ApprovalList(items=items, total=int(total), limit=page.limit, offset=page.offset)


@router.get(
    "/pending",
    response_model=ApprovalList,
    summary="Approvals still awaiting a decision",
    description="What Reachy should mention if the owner asks whether anything needs confirming.",
)
async def list_pending(_: PrincipalDep, session: SessionDep, page: PaginationDep) -> ApprovalList:
    return await list_approvals(_, session, page, status=ApprovalStatus.PENDING)


@router.get(
    "/{approval_id}/status",
    response_model=ActionStatusOut,
    summary="Get safe status for one approval",
)
async def get_approval_status(
    approval_id: str, _: PrincipalDep, session: SessionDep
) -> ActionStatusOut:
    return await status_service.get_status_by_approval_id(session, approval_id)


@router.get("/{approval_id}", response_model=ApprovalOut, summary="Get one approval")
async def get_approval(approval_id: str, _: PrincipalDep, session: SessionDep) -> ApprovalOut:
    approval = await session.get(Approval, approval_id)
    if approval is None:
        raise NotFoundError(f"No approval with id {approval_id}.")
    action = await session.get(PendingAction, approval.pending_action_id)
    if action is None:
        raise NotFoundError("The action behind this approval no longer exists.")
    return _to_out(approval, action)


@router.post(
    "/{approval_id}/decide",
    summary="Approve or reject",
    description=(
        "Applies a decision to a pending approval. Single-use: a second decision on the same "
        "approval is refused and performs no side effect."
    ),
)
async def decide(
    approval_id: str,
    payload: ApprovalDecision,
    _: PrincipalDep,
    session: SessionDep,
    settings: SettingsDep,
) -> dict[str, Any]:
    approval = await session.get(Approval, approval_id)
    if approval is None:
        raise NotFoundError(f"No approval with id {approval_id}.")
    return await service.resolve_by_token(
        session,
        approval.callback_token,
        approve=payload.approve,
        chat_id=None,
        settings=settings,
        note=payload.note,
    )
