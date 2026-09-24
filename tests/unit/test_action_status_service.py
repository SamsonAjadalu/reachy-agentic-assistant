"""Unit tests for approval/action status mapping."""

from __future__ import annotations

from datetime import timedelta

from app.services.action_status import build_action_status, resolve_display_status, summarize_result
from database.models import Approval, PendingAction
from integrations.google.executors import SEND_DRAFT
from shared.enums import ApprovalStatus, PendingActionStatus
from shared.timeutils import utcnow


def _pair(
    *,
    approval_status: ApprovalStatus,
    action_status: PendingActionStatus,
    action_type: str = SEND_DRAFT,
    expires_in_minutes: int = 30,
) -> tuple[Approval, PendingAction]:
    now = utcnow()
    action = PendingAction(
        action_type=action_type,
        risk_level="external_write",
        summary="Send draft",
        payload_json="{}",
        payload_hash="abc",
        preview_text="secret preview",
        status=action_status.value,
        expires_at=now + timedelta(minutes=expires_in_minutes),
        created_at=now,
    )
    action.id = "action-1"
    approval = Approval(
        pending_action_id=action.id,
        callback_token="token",
        status=approval_status.value,
        requested_at=now,
        expires_at=now + timedelta(minutes=expires_in_minutes),
        resolved_at=now if approval_status != ApprovalStatus.PENDING else None,
    )
    approval.id = "approval-1"
    return approval, action


def test_pending_maps_to_awaiting_approval() -> None:
    approval, action = _pair(
        approval_status=ApprovalStatus.PENDING,
        action_status=PendingActionStatus.PENDING,
    )
    assert resolve_display_status(approval, action) == "awaiting_approval"


def test_safe_response_omits_sensitive_fields() -> None:
    approval, action = _pair(
        approval_status=ApprovalStatus.PENDING,
        action_status=PendingActionStatus.PENDING,
    )
    body = build_action_status(approval, action)
    dumped = body.model_dump()
    assert "preview_text" not in dumped
    assert "payload_hash" not in dumped
    assert "payload_json" not in dumped


def test_sent_summary_uses_message_id_only() -> None:
    _approval, action = _pair(
        approval_status=ApprovalStatus.EXECUTED,
        action_status=PendingActionStatus.EXECUTED,
    )
    action.executed_at = utcnow()
    action.execution_result = '{"message_id":"msg-123","subject":"secret subject"}'
    action.external_id = "msg-123"
    summary = summarize_result(SEND_DRAFT, action, "sent")
    assert summary is not None
    assert "msg-123" in summary
    assert "secret subject" not in summary
