"""Approval state machine guarantees.

These tests exist to prove four claims: an action runs at most once, only the
approved payload runs, only an allowlisted chat can decide, and an expired
prompt is inert.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings
from approvals.state_machine import (
    clear_executors,
    execute_approved_action,
    expire_stale_approvals,
    register_executor,
    request_approval,
    resolve_by_token,
)
from database.models import ActionAuditLog, Approval, PendingAction
from notifications.dispatcher import set_channel_override
from notifications.mock import MockNotificationChannel
from shared.contracts import payload_hash
from shared.enums import ApprovalStatus, PendingActionStatus, RiskLevel
from shared.timeutils import utcnow


@pytest.fixture
def channel() -> MockNotificationChannel:
    mock = MockNotificationChannel()
    set_channel_override(mock)
    yield mock
    set_channel_override(None)


@pytest.fixture(autouse=True)
def _clean_executors():
    clear_executors()
    yield
    clear_executors()


@pytest.fixture
def side_effects() -> list[dict[str, Any]]:
    """Records every real execution so double-execution is directly observable."""
    performed: list[dict[str, Any]] = []

    @register_executor("test.send_email")
    async def send_email(
        session: AsyncSession, payload: dict[str, Any], settings: Settings
    ) -> dict[str, Any]:
        performed.append(payload)
        return {"external_id": f"msg-{len(performed)}", "to": payload["to"]}

    return performed


async def _request(
    session: AsyncSession, settings: Settings, **overrides: Any
) -> tuple[PendingAction, Approval]:
    payload = overrides.pop("payload", {"to": "nathan@example.com", "subject": "Hello"})
    return await request_approval(
        session,
        action_type=overrides.pop("action_type", "test.send_email"),
        summary=overrides.pop("summary", "Send an email to Nathan"),
        payload=payload,
        preview_text=overrides.pop(
            "preview_text", "To: nathan@example.com\nSubject: Hello\n\nBody text."
        ),
        risk_level=RiskLevel.EXTERNAL_WRITE,
        settings=settings,
        **overrides,
    )


class TestRequesting:
    async def test_the_prompt_shows_the_exact_payload(
        self, session: AsyncSession, settings: Settings, channel: MockNotificationChannel
    ) -> None:
        await _request(session, settings)
        prompt = channel.last_approval()
        assert prompt is not None
        assert "nathan@example.com" in prompt.preview
        assert "Subject: Hello" in prompt.preview

    async def test_nothing_happens_before_approval(
        self,
        session: AsyncSession,
        settings: Settings,
        channel: MockNotificationChannel,
        side_effects: list,
    ) -> None:
        await _request(session, settings)
        assert side_effects == []

    async def test_the_payload_hash_is_stored(
        self, session: AsyncSession, settings: Settings, channel: MockNotificationChannel
    ) -> None:
        payload = {"to": "a@example.com", "subject": "Hi"}
        action, _ = await _request(session, settings, payload=payload)
        assert action.payload_hash == payload_hash(payload)

    async def test_the_callback_token_is_unrelated_to_any_database_id(
        self, session: AsyncSession, settings: Settings, channel: MockNotificationChannel
    ) -> None:
        action, approval = await _request(session, settings)
        assert approval.callback_token not in (action.id, approval.id)
        assert len(approval.callback_token) >= 12

    async def test_requesting_is_audited(
        self, session: AsyncSession, settings: Settings, channel: MockNotificationChannel
    ) -> None:
        await _request(session, settings)
        outcomes = (await session.scalars(select(ActionAuditLog.outcome))).all()
        assert "approval_requested" in outcomes

    async def test_an_idempotency_key_reuses_the_same_prompt(
        self, session: AsyncSession, settings: Settings, channel: MockNotificationChannel
    ) -> None:
        first, _ = await _request(session, settings, idempotency_key="turn-1")
        second, _ = await _request(session, settings, idempotency_key="turn-1")
        assert first.id == second.id
        assert len(channel.approval_requests) == 1


class TestSingleExecution:
    async def test_approving_performs_the_action_once(
        self,
        session: AsyncSession,
        settings: Settings,
        channel: MockNotificationChannel,
        side_effects: list,
    ) -> None:
        _, approval = await _request(session, settings)
        result = await resolve_by_token(
            session, approval.callback_token, approve=True, chat_id=None, settings=settings
        )
        assert result["decision"] == "approved"
        assert result["executed"] is True
        assert len(side_effects) == 1

    async def test_a_second_approval_of_the_same_prompt_is_refused(
        self,
        session: AsyncSession,
        settings: Settings,
        channel: MockNotificationChannel,
        side_effects: list,
    ) -> None:
        """Uses the configured workflow."""
        _, approval = await _request(session, settings)
        token = approval.callback_token

        first = await resolve_by_token(
            session, token, approve=True, chat_id=None, settings=settings
        )
        second = await resolve_by_token(
            session, token, approve=True, chat_id=None, settings=settings
        )

        assert first["executed"] is True
        assert second["accepted"] is False
        assert second["reason"] == "already_resolved"
        assert len(side_effects) == 1

    async def test_concurrent_approvals_produce_exactly_one_execution(
        self,
        session: AsyncSession,
        settings: Settings,
        channel: MockNotificationChannel,
        side_effects: list,
    ) -> None:
        action, _ = await _request(session, settings)

        results = await asyncio.gather(
            execute_approved_action(session, action.id, settings=settings),
            execute_approved_action(session, action.id, settings=settings),
        )

        assert sum(1 for result in results if result["executed"]) == 1
        assert len(side_effects) == 1

    async def test_rejecting_performs_nothing(
        self,
        session: AsyncSession,
        settings: Settings,
        channel: MockNotificationChannel,
        side_effects: list,
    ) -> None:
        _, approval = await _request(session, settings)
        result = await resolve_by_token(
            session, approval.callback_token, approve=False, chat_id=None, settings=settings
        )
        assert result["decision"] == "rejected"
        assert side_effects == []

    async def test_approving_after_rejecting_is_refused(
        self,
        session: AsyncSession,
        settings: Settings,
        channel: MockNotificationChannel,
        side_effects: list,
    ) -> None:
        _, approval = await _request(session, settings)
        token = approval.callback_token
        await resolve_by_token(session, token, approve=False, chat_id=None, settings=settings)
        second = await resolve_by_token(
            session, token, approve=True, chat_id=None, settings=settings
        )
        assert second["accepted"] is False
        assert side_effects == []


class TestPayloadIntegrity:
    async def test_a_payload_altered_after_approval_is_not_executed(
        self,
        session: AsyncSession,
        settings: Settings,
        channel: MockNotificationChannel,
        side_effects: list,
    ) -> None:
        """Uses the configured workflow."""
        action, _approval = await _request(session, settings)

        tampered = json.loads(action.payload_json)
        tampered["to"] = "attacker@example.com"
        action.payload_json = json.dumps(tampered, separators=(",", ":"), sort_keys=True)
        await session.commit()

        result = await execute_approved_action(session, action.id, settings=settings)

        assert result["executed"] is False
        assert result["reason"] == "payload_mismatch"
        assert side_effects == []

    async def test_the_mismatch_is_audited(
        self,
        session: AsyncSession,
        settings: Settings,
        channel: MockNotificationChannel,
        side_effects: list,
    ) -> None:
        action, _ = await _request(session, settings)
        action.payload_json = json.dumps({"to": "elsewhere@example.com"})
        await session.commit()
        await execute_approved_action(session, action.id, settings=settings)

        outcomes = (await session.scalars(select(ActionAuditLog.outcome))).all()
        assert "payload_hash_mismatch" in outcomes


class TestChatAllowlist:
    async def test_a_chat_outside_the_allowlist_cannot_approve(
        self,
        session: AsyncSession,
        settings: Settings,
        channel: MockNotificationChannel,
        side_effects: list,
    ) -> None:
        settings.telegram_allowed_chat_ids = [111]
        _, approval = await _request(session, settings)

        result = await resolve_by_token(
            session, approval.callback_token, approve=True, chat_id=999, settings=settings
        )

        assert result["accepted"] is False
        assert result["reason"] == "not_authorised"
        assert side_effects == []

    async def test_the_allowlisted_chat_can_approve(
        self,
        session: AsyncSession,
        settings: Settings,
        channel: MockNotificationChannel,
        side_effects: list,
    ) -> None:
        settings.telegram_allowed_chat_ids = [111]
        _, approval = await _request(session, settings)
        result = await resolve_by_token(
            session, approval.callback_token, approve=True, chat_id=111, settings=settings
        )
        assert result["executed"] is True

    async def test_an_unauthorised_attempt_is_audited(
        self, session: AsyncSession, settings: Settings, channel: MockNotificationChannel
    ) -> None:
        settings.telegram_allowed_chat_ids = [111]
        _, approval = await _request(session, settings)
        await resolve_by_token(
            session, approval.callback_token, approve=True, chat_id=999, settings=settings
        )
        outcomes = (await session.scalars(select(ActionAuditLog.outcome))).all()
        assert "rejected_unauthorised_chat" in outcomes


class TestTokenGuessing:
    async def test_an_unknown_token_is_refused_without_detail(
        self, session: AsyncSession, settings: Settings, channel: MockNotificationChannel
    ) -> None:
        result = await resolve_by_token(
            session, "totally-made-up", approve=True, chat_id=None, settings=settings
        )
        assert result == {"accepted": False, "reason": "unknown_or_expired"}

    async def test_the_action_id_is_not_a_usable_token(
        self,
        session: AsyncSession,
        settings: Settings,
        channel: MockNotificationChannel,
        side_effects: list,
    ) -> None:
        action, _ = await _request(session, settings)
        result = await resolve_by_token(
            session, action.id, approve=True, chat_id=None, settings=settings
        )
        assert result["accepted"] is False
        assert side_effects == []


class TestExpiry:
    async def test_an_expired_prompt_cannot_be_approved(
        self,
        session: AsyncSession,
        settings: Settings,
        channel: MockNotificationChannel,
        side_effects: list,
    ) -> None:
        _, approval = await _request(session, settings, ttl_minutes=1)
        approval.expires_at = utcnow()
        await session.commit()

        result = await resolve_by_token(
            session, approval.callback_token, approve=True, chat_id=None, settings=settings
        )
        assert result["reason"] == "expired"
        assert side_effects == []

    async def test_the_sweeper_marks_stale_prompts_expired(
        self, session: AsyncSession, settings: Settings, channel: MockNotificationChannel
    ) -> None:
        action, approval = await _request(session, settings)
        approval.expires_at = utcnow()
        action.expires_at = utcnow()
        await session.commit()

        assert await expire_stale_approvals(session) == 1
        session.expunge_all()

        refreshed = await session.get(Approval, approval.id)
        refreshed_action = await session.get(PendingAction, action.id)
        assert refreshed is not None and refreshed.status == ApprovalStatus.EXPIRED.value
        assert refreshed_action is not None
        assert refreshed_action.status == PendingActionStatus.EXPIRED.value

    async def test_the_sweeper_leaves_live_prompts_alone(
        self, session: AsyncSession, settings: Settings, channel: MockNotificationChannel
    ) -> None:
        await _request(session, settings)
        assert await expire_stale_approvals(session) == 0


class TestExecutionFailure:
    async def test_a_failing_executor_marks_the_action_failed(
        self, session: AsyncSession, settings: Settings, channel: MockNotificationChannel
    ) -> None:
        @register_executor("test.explodes")
        async def explodes(
            db: AsyncSession, payload: dict[str, Any], settings: Settings
        ) -> dict[str, Any]:
            raise RuntimeError("provider rejected it")

        action, approval = await _request(
            session, settings, action_type="test.explodes", payload={"x": 1}
        )
        result = await resolve_by_token(
            session, approval.callback_token, approve=True, chat_id=None, settings=settings
        )

        assert result["executed"] is False
        session.expunge_all()
        refreshed = await session.get(PendingAction, action.id)
        assert refreshed is not None
        assert refreshed.status == PendingActionStatus.FAILED.value

    async def test_an_unregistered_action_type_does_not_execute(
        self, session: AsyncSession, settings: Settings, channel: MockNotificationChannel
    ) -> None:
        _action, approval = await _request(
            session, settings, action_type="test.no_executor", payload={"x": 1}
        )
        result = await resolve_by_token(
            session, approval.callback_token, approve=True, chat_id=None, settings=settings
        )
        assert result["executed"] is False


class TestApi:
    async def test_pending_approvals_are_listed(
        self, client, session: AsyncSession, settings: Settings, channel: MockNotificationChannel
    ) -> None:
        await _request(session, settings)
        await session.commit()

        body = (await client.get("/api/v1/approvals/pending")).json()
        assert body["total"] == 1
        assert body["items"][0]["action_type"] == "test.send_email"
        assert "nathan@example.com" in body["items"][0]["preview_text"]

    async def test_deciding_through_the_api_executes_once(
        self,
        client,
        session: AsyncSession,
        settings: Settings,
        channel: MockNotificationChannel,
        side_effects: list,
    ) -> None:
        _, approval = await _request(session, settings)
        await session.commit()

        first = await client.post(f"/api/v1/approvals/{approval.id}/decide", json={"approve": True})
        second = await client.post(
            f"/api/v1/approvals/{approval.id}/decide", json={"approve": True}
        )

        assert first.json()["executed"] is True
        assert second.json()["accepted"] is False
        assert len(side_effects) == 1
