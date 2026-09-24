"""Read-only approval/action status for Reachy polling."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.schemas.action_status import ActionStatusOut
from database.models import Approval, PendingAction
from integrations.google.executors import SEND_DRAFT
from shared.enums import ApprovalStatus, PendingActionStatus
from shared.errors import NotFoundError
from shared.timeutils import utcnow

_GMAIL_SEND_TYPES = frozenset({SEND_DRAFT, "google.gmail.send"})


async def get_status_by_approval_id(session: AsyncSession, approval_id: str) -> ActionStatusOut:
    approval = await session.get(Approval, approval_id)
    if approval is None:
        raise NotFoundError(f"No approval with id {approval_id}.")
    action = await session.get(PendingAction, approval.pending_action_id)
    if action is None:
        raise NotFoundError("The action behind this approval no longer exists.")
    return build_action_status(approval, action)


async def get_status_by_action_id(session: AsyncSession, action_id: str) -> ActionStatusOut:
    action = await session.get(PendingAction, action_id)
    if action is None:
        raise NotFoundError(f"No action with id {action_id}.")
    approval = await session.scalar(select(Approval).where(Approval.pending_action_id == action.id))
    if approval is None:
        raise NotFoundError("No approval is linked to this action.")
    return build_action_status(approval, action)


def build_action_status(approval: Approval, action: PendingAction) -> ActionStatusOut:
    status = resolve_display_status(approval, action)
    approved_at, rejected_at = _resolution_timestamps(approval, action, status)
    return ActionStatusOut(
        approval_id=approval.id,
        action_id=action.id,
        action_type=action.action_type,
        status=status,
        created_at=action.created_at,
        approved_at=approved_at,
        rejected_at=rejected_at,
        executed_at=action.executed_at,
        result_summary=summarize_result(action.action_type, action, status),
    )


def resolve_display_status(approval: Approval, action: PendingAction) -> str:
    action_status = PendingActionStatus(action.status)
    approval_status = ApprovalStatus(approval.status)
    now = utcnow()

    if action_status == PendingActionStatus.FAILED:
        return "failed"
    if action_status == PendingActionStatus.REJECTED or approval_status == ApprovalStatus.REJECTED:
        return "rejected"
    if action_status == PendingActionStatus.EXPIRED or approval_status == ApprovalStatus.EXPIRED:
        return "expired"
    if (
        action_status == PendingActionStatus.PENDING
        and approval_status == ApprovalStatus.PENDING
        and approval.expires_at <= now
    ):
        return "expired"
    if action_status == PendingActionStatus.EXECUTED or approval_status == ApprovalStatus.EXECUTED:
        if action.action_type in _GMAIL_SEND_TYPES:
            return "sent"
        return "completed"
    if action_status == PendingActionStatus.EXECUTING:
        return "executing"
    if approval_status == ApprovalStatus.APPROVED:
        return "approved"
    if action_status == PendingActionStatus.PENDING and approval_status == ApprovalStatus.PENDING:
        return "awaiting_approval"
    return action_status.value


def _resolution_timestamps(
    approval: Approval, action: PendingAction, status: str
) -> tuple[datetime | None, datetime | None]:
    if status == "rejected":
        return None, approval.resolved_at
    if status in {"approved", "executing", "sent", "completed", "failed"}:
        return approval.resolved_at, None
    return None, None


def summarize_result(action_type: str, action: PendingAction, status: str) -> str | None:
    if status == "awaiting_approval":
        return "Waiting for your approval in Telegram."
    if status == "approved":
        return "Approved and waiting to run."
    if status == "executing":
        return "The action is in progress."
    if status == "rejected":
        return "The action was rejected. Nothing was sent."
    if status == "expired":
        return "The approval expired before a decision was made."
    if status == "failed":
        return "The action was approved but could not be completed."
    if status == "sent":
        return _sent_summary(action)
    if status == "completed":
        return _completed_summary(action_type, action)
    return None


def _sent_summary(action: PendingAction) -> str:
    details = _safe_execution_details(action)
    message_id = details.get("message_id") or action.external_id
    if message_id:
        return f"The draft was sent. Message id {message_id}."
    return "The draft was sent."


def _completed_summary(action_type: str, action: PendingAction) -> str:
    details = _safe_execution_details(action)
    if action_type.endswith(".create_event") and details.get("title"):
        return f"Created calendar event {details['title']}."
    if action_type.endswith(".update_event") and details.get("title"):
        return f"Updated calendar event {details['title']}."
    if action_type.endswith(".delete_event"):
        return "Deleted the calendar event."
    if action_type.endswith(".create_page") and details.get("title"):
        return f"Created Notion page {details['title']}."
    return "The action completed successfully."


def _safe_execution_details(action: PendingAction) -> dict[str, Any]:
    if not action.execution_result:
        return {}
    try:
        payload = json.loads(action.execution_result)
    except json.JSONDecodeError:
        return {}
    if not isinstance(payload, dict):
        return {}

    allowed = (
        "message_id",
        "draft_id",
        "external_id",
        "event_id",
        "title",
        "page_id",
        "job_id",
    )
    return {key: payload[key] for key in allowed if key in payload and payload[key] is not None}
