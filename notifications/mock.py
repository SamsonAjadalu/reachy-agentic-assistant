"""In-memory notification channel.

Used whenever ``MOCK_MODE`` is on or Telegram is disabled. It records everything
it was asked to send so tests can assert on the exact text the owner would have
seen, which is the property that matters for approval previews.
"""

from __future__ import annotations

import itertools
from typing import Any

from app.logging_config import get_logger
from notifications.base import ApprovalRequest, DeliveryResult, OutboundMessage

logger = get_logger(__name__)


class MockNotificationChannel:
    name = "mock"

    def __init__(self, *, fail_next: int = 0) -> None:
        self.messages: list[OutboundMessage] = []
        self.approval_requests: list[ApprovalRequest] = []
        self.resolutions: list[tuple[str, str, str]] = []
        self._ids = itertools.count(1)
        # Lets a test drive the retry path deterministically.
        self.fail_next = fail_next

    async def send(self, message: OutboundMessage) -> DeliveryResult:
        if self.fail_next > 0:
            self.fail_next -= 1
            return DeliveryResult(delivered=False, error="simulated delivery failure")
        self.messages.append(message)
        logger.debug("Mock notification recorded", extra={"title": message.title})
        return DeliveryResult(delivered=True, provider_message_id=f"mock-{next(self._ids)}")

    async def request_approval(self, request: ApprovalRequest) -> DeliveryResult:
        if self.fail_next > 0:
            self.fail_next -= 1
            return DeliveryResult(delivered=False, error="simulated delivery failure")
        self.approval_requests.append(request)
        return DeliveryResult(
            delivered=True, provider_message_id=f"mock-approval-{request.callback_token}"
        )

    async def resolve_approval_message(
        self, provider_message_id: str, outcome: str, detail: str
    ) -> None:
        self.resolutions.append((provider_message_id, outcome, detail))

    async def health(self) -> dict[str, Any]:
        return {
            "ok": True,
            "provider": "mock",
            "messages_sent": len(self.messages),
            "approvals_requested": len(self.approval_requests),
        }

    def last_message(self) -> OutboundMessage | None:
        return self.messages[-1] if self.messages else None

    def last_approval(self) -> ApprovalRequest | None:
        return self.approval_requests[-1] if self.approval_requests else None

    def reset(self) -> None:
        self.messages.clear()
        self.approval_requests.clear()
        self.resolutions.clear()
