"""Approval state machine.

The guarantee this module provides: an externally visible action executes at most
once, only after the owner approved the exact payload they were shown, and only
from an allowlisted chat.

That rests on four mechanics:

* The payload is canonicalised and hashed when the approval is requested, and the
  hash is re-verified immediately before execution. Content that changed in the
  meantime cannot be substituted.
* The callback carries an opaque token unrelated to any database id, so a
  captured or guessed value reveals nothing.
* Execution is claimed with a conditional UPDATE from ``pending`` to ``executing``.
  A second approval finds zero rows updated and is refused.
* Expiry is enforced on read as well as by the sweeper, so a stale prompt cannot
  be answered even if the sweeper is behind.
"""

from __future__ import annotations

import json
import secrets
from collections.abc import Awaitable, Callable
from datetime import timedelta
from typing import Any

from sqlalchemy import and_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, get_settings
from app.logging_config import get_logger
from database.models import ActionAuditLog, Approval, PendingAction
from notifications.base import ApprovalRequest
from notifications.dispatcher import get_channel
from shared.contracts import canonical_json, payload_hash
from shared.enums import ApprovalStatus, PendingActionStatus, RiskLevel
from shared.errors import ApprovalError, NotFoundError
from shared.timeutils import utcnow

logger = get_logger(__name__)

CALLBACK_TOKEN_BYTES = 12
MAX_PREVIEW_CHARS = 2500

# Handlers that perform the real side effect once approval is granted. Keyed by
# action type so a pending row written before a restart still resolves. Settings
# are handed in rather than looked up globally, so an executor reads the same
# configuration as the request that raised the approval.
ActionExecutor = Callable[[AsyncSession, dict[str, Any], Settings], Awaitable[dict[str, Any]]]
_EXECUTORS: dict[str, ActionExecutor] = {}


def register_executor(action_type: str) -> Callable[[ActionExecutor], ActionExecutor]:
    def decorator(func: ActionExecutor) -> ActionExecutor:
        if action_type in _EXECUTORS:
            raise RuntimeError(f"An executor for {action_type!r} is already registered.")
        _EXECUTORS[action_type] = func
        return func

    return decorator


def has_executor(action_type: str) -> bool:
    return action_type in _EXECUTORS


def get_executor(action_type: str) -> ActionExecutor:
    executor = _EXECUTORS.get(action_type)
    if executor is None:
        raise ApprovalError(
            f"No executor is registered for action type {action_type!r}. "
            "The action was approved but cannot be performed by this build."
        )
    return executor


def clear_executors() -> None:
    """Test helper."""
    _EXECUTORS.clear()


async def request_approval(
    session: AsyncSession,
    *,
    action_type: str,
    summary: str,
    payload: dict[str, Any],
    preview_text: str,
    risk_level: RiskLevel = RiskLevel.EXTERNAL_WRITE,
    settings: Settings | None = None,
    ttl_minutes: int | None = None,
    idempotency_key: str | None = None,
    correlation_id: str | None = None,
    background_task_id: str | None = None,
) -> tuple[PendingAction, Approval]:
    """Record a pending action and ask the owner to approve it.

    ``preview_text`` is what the owner actually reads, so it must contain every
    consequential detail: recipients, subject, body, destination. A vague preview
    turns an approval into a rubber stamp.
    """
    settings = settings or get_settings()
    now = utcnow()
    ttl = timedelta(minutes=ttl_minutes or settings.telegram_approval_ttl_minutes)

    if idempotency_key:
        existing = await session.scalar(
            select(PendingAction).where(PendingAction.idempotency_key == idempotency_key)
        )
        if existing is not None:
            approval = await session.scalar(
                select(Approval).where(Approval.pending_action_id == existing.id)
            )
            if approval is not None:
                return existing, approval

    canonical = canonical_json(payload)
    action = PendingAction(
        action_type=action_type,
        risk_level=risk_level.value,
        summary=summary[:1000],
        payload_json=canonical,
        payload_hash=payload_hash(payload),
        preview_text=preview_text[:MAX_PREVIEW_CHARS],
        status=PendingActionStatus.PENDING.value,
        expires_at=now + ttl,
        idempotency_key=idempotency_key,
        correlation_id=correlation_id,
        background_task_id=background_task_id,
    )
    session.add(action)
    await session.flush()

    approval = Approval(
        pending_action_id=action.id,
        callback_token=secrets.token_urlsafe(CALLBACK_TOKEN_BYTES)[:32],
        status=ApprovalStatus.PENDING.value,
        requested_at=now,
        expires_at=now + ttl,
        channel="telegram",
    )
    session.add(approval)
    await session.flush()

    channel = get_channel(settings)
    result = await channel.request_approval(
        ApprovalRequest(
            callback_token=approval.callback_token,
            title=summary,
            preview=action.preview_text,
            expires_in_seconds=int(ttl.total_seconds()),
        )
    )
    approval.channel_message_id = result.provider_message_id
    if not result.delivered:
        logger.error(
            "Could not deliver the approval prompt",
            extra={"action_type": action_type, "approval_id": approval.id},
        )

    await _audit(
        session,
        action_type=action_type,
        risk_level=risk_level,
        outcome="approval_requested",
        approval_id=approval.id,
        payload_hash_value=action.payload_hash,
        preview=action.preview_text,
        correlation_id=correlation_id,
    )
    await session.flush()
    return action, approval


async def resolve_by_token(
    session: AsyncSession,
    callback_token: str,
    *,
    approve: bool,
    chat_id: int | None,
    settings: Settings | None = None,
    note: str | None = None,
) -> dict[str, Any]:
    """Apply a decision arriving from the notification channel.

    Every rejection path returns the same shape and performs no side effect, so a
    caller cannot distinguish "unknown token" from "already used" and use that to
    probe.
    """
    settings = settings or get_settings()

    if chat_id is not None and settings.telegram_allowed_chat_ids:
        if chat_id not in settings.telegram_allowed_chat_ids:
            logger.warning(
                "Approval attempt from a chat that is not allowlisted",
                extra={"chat_id": chat_id},
            )
            await _audit(
                session,
                action_type="approval.callback",
                risk_level=RiskLevel.EXTERNAL_WRITE,
                outcome="rejected_unauthorised_chat",
                detail=f"chat_id={chat_id}",
            )
            await session.flush()
            return {"accepted": False, "reason": "not_authorised"}

    approval = await session.scalar(
        select(Approval).where(Approval.callback_token == callback_token)
    )
    if approval is None:
        return {"accepted": False, "reason": "unknown_or_expired"}

    action = await session.get(PendingAction, approval.pending_action_id)
    if action is None:
        return {"accepted": False, "reason": "unknown_or_expired"}

    now = utcnow()
    if approval.status != ApprovalStatus.PENDING.value:
        return {"accepted": False, "reason": "already_resolved", "status": approval.status}
    if approval.expires_at <= now:
        approval.status = ApprovalStatus.EXPIRED.value
        action.status = PendingActionStatus.EXPIRED.value
        await session.flush()
        return {"accepted": False, "reason": "expired"}

    approval.resolved_at = now
    approval.resolved_by_chat_id = chat_id
    approval.decision_note = note

    if not approve:
        approval.status = ApprovalStatus.REJECTED.value
        action.status = PendingActionStatus.REJECTED.value
        await _audit(
            session,
            action_type=action.action_type,
            risk_level=RiskLevel(action.risk_level),
            outcome="rejected",
            approval_id=approval.id,
            payload_hash_value=action.payload_hash,
            correlation_id=action.correlation_id,
        )
        await session.flush()
        await _update_channel_message(approval, "Rejected", "No action was taken.", settings)
        return {"accepted": True, "decision": "rejected", "action_id": action.id}

    approval.status = ApprovalStatus.APPROVED.value
    await session.flush()

    outcome = await execute_approved_action(session, action.id, settings=settings)
    await _update_channel_message(
        approval,
        "Approved" if outcome["executed"] else "Failed",
        outcome.get("message", ""),
        settings,
    )
    return {"accepted": True, "decision": "approved", "action_id": action.id, **outcome}


async def execute_approved_action(
    session: AsyncSession, action_id: str, *, settings: Settings | None = None
) -> dict[str, Any]:
    """Perform the side effect exactly once.

    The claim is a conditional UPDATE: only the caller that moves the row out of
    ``pending`` proceeds. Two approvals racing produce one execution and one
    ``already_executing``.
    """
    settings = settings or get_settings()
    now = utcnow()

    claim = await session.execute(
        update(PendingAction)
        .where(
            and_(
                PendingAction.id == action_id,
                PendingAction.status == PendingActionStatus.PENDING.value,
                PendingAction.expires_at > now,
            )
        )
        .values(status=PendingActionStatus.EXECUTING.value, updated_at=now)
    )
    if claim.rowcount != 1:  # type: ignore[attr-defined]
        current = await session.get(PendingAction, action_id)
        return {
            "executed": False,
            "reason": "already_claimed",
            "status": current.status if current else "unknown",
            "message": "This action was already handled.",
        }
    await session.commit()

    action = await session.get(PendingAction, action_id)
    if action is None:
        raise NotFoundError(f"No pending action with id {action_id}.")

    payload = json.loads(action.payload_json)

    # Re-verify: the approval was granted for these exact bytes.
    if payload_hash(payload) != action.payload_hash:
        action.status = PendingActionStatus.FAILED.value
        action.execution_error = "The stored payload no longer matches the approved hash."
        await _audit(
            session,
            action_type=action.action_type,
            risk_level=RiskLevel(action.risk_level),
            outcome="payload_hash_mismatch",
            payload_hash_value=action.payload_hash,
            correlation_id=action.correlation_id,
        )
        await session.commit()
        return {
            "executed": False,
            "reason": "payload_mismatch",
            "message": "The action content changed after approval, so it was not performed.",
        }

    try:
        executor = get_executor(action.action_type)
        result = await executor(session, payload, settings)
    except Exception as exc:
        action.status = PendingActionStatus.FAILED.value
        action.execution_error = f"{type(exc).__name__}: {exc}"
        await _audit(
            session,
            action_type=action.action_type,
            risk_level=RiskLevel(action.risk_level),
            outcome="execution_failed",
            payload_hash_value=action.payload_hash,
            error_code=type(exc).__name__,
            correlation_id=action.correlation_id,
        )
        await session.commit()
        logger.exception("Approved action failed", extra={"action_id": action_id})
        return {
            "executed": False,
            "reason": "execution_failed",
            "message": "The action was approved but could not be completed.",
        }

    action.status = PendingActionStatus.EXECUTED.value
    action.executed_at = utcnow()
    action.execution_result = json.dumps(result, default=str)[:8000]
    action.external_id = str(result.get("external_id")) if result.get("external_id") else None

    approval = await session.scalar(select(Approval).where(Approval.pending_action_id == action_id))
    if approval is not None:
        approval.status = ApprovalStatus.EXECUTED.value

    await _audit(
        session,
        action_type=action.action_type,
        risk_level=RiskLevel(action.risk_level),
        outcome="executed",
        approval_id=approval.id if approval else None,
        payload_hash_value=action.payload_hash,
        resource_id=action.external_id,
        correlation_id=action.correlation_id,
    )
    await session.commit()
    return {"executed": True, "result": result, "message": "Done."}


async def expire_stale_approvals(session: AsyncSession) -> int:
    """Sweep prompts nobody answered.

    Read paths also check expiry, so this is housekeeping rather than the
    security boundary.
    """
    now = utcnow()
    stale = (
        await session.scalars(
            select(Approval).where(
                Approval.status == ApprovalStatus.PENDING.value, Approval.expires_at <= now
            )
        )
    ).all()

    for approval in stale:
        approval.status = ApprovalStatus.EXPIRED.value
        approval.resolved_at = now
        action = await session.get(PendingAction, approval.pending_action_id)
        if action is not None and action.status == PendingActionStatus.PENDING.value:
            action.status = PendingActionStatus.EXPIRED.value

    if stale:
        await session.flush()
        logger.info("Expired unanswered approvals", extra={"count": len(stale)})
    return len(stale)


async def _update_channel_message(
    approval: Approval, outcome: str, detail: str, settings: Settings
) -> None:
    """Rewrite the prompt so its buttons cannot be pressed again from a stale view."""
    if not approval.channel_message_id:
        return
    try:
        channel = get_channel(settings)
        await channel.resolve_approval_message(approval.channel_message_id, outcome, detail)
    except Exception:
        logger.warning(
            "Could not update the approval message after the decision",
            extra={"approval_id": approval.id},
        )


async def _audit(
    session: AsyncSession,
    *,
    action_type: str,
    risk_level: RiskLevel,
    outcome: str,
    approval_id: str | None = None,
    payload_hash_value: str | None = None,
    preview: str | None = None,
    resource_id: str | None = None,
    error_code: str | None = None,
    detail: str | None = None,
    correlation_id: str | None = None,
) -> None:
    session.add(
        ActionAuditLog(
            occurred_at=utcnow(),
            action_type=action_type,
            risk_level=risk_level.value,
            actor="reachy",
            outcome=outcome,
            resource_id=resource_id,
            approval_id=approval_id,
            correlation_id=correlation_id,
            payload_hash=payload_hash_value,
            payload_preview=preview[:2000] if preview else None,
            error_code=error_code,
            detail=detail,
        )
    )
