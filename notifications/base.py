"""Notification channel interface.

Every channel implements the same protocol so the dispatcher, the approval flow
and the briefing generator never import a specific provider. The mock channel is
a full implementation rather than a stub: the entire notification and approval
path is exercised in tests without a network.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class OutboundMessage:
    title: str
    body: str
    severity: str = "info"
    dedup_key: str | None = None
    attachment_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ApprovalRequest:
    """An approval prompt with its two possible answers.

    ``callback_token`` is an opaque value with no relationship to the action id,
    so a captured callback payload reveals nothing and cannot be crafted.
    """

    callback_token: str
    title: str
    preview: str
    approve_label: str = "Approve"
    reject_label: str = "Reject"
    expires_in_seconds: int = 3600


@dataclass(frozen=True)
class DeliveryResult:
    delivered: bool
    provider_message_id: str | None = None
    error: str | None = None
    suppressed_reason: str | None = None


@runtime_checkable
class NotificationChannel(Protocol):
    name: str

    async def send(self, message: OutboundMessage) -> DeliveryResult: ...

    async def request_approval(self, request: ApprovalRequest) -> DeliveryResult: ...

    async def resolve_approval_message(
        self, provider_message_id: str, outcome: str, detail: str
    ) -> None:
        """Replace the prompt's buttons with the outcome.

        Called after a decision so the same message cannot be answered twice from
        a stale view of the chat.
        """
        ...

    async def health(self) -> dict[str, Any]: ...
