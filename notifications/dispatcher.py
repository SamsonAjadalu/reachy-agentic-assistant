"""Notification dispatch.

Every outbound message is persisted before it is sent and each attempt is
recorded, so a delivery failure is visible after the fact rather than lost in a
log line. Retries are bounded and explicit; a channel that keeps failing marks
the notification failed rather than blocking the caller.
"""

from __future__ import annotations

import asyncio
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, get_settings
from app.logging_config import get_logger
from database.models import (
    AlertDeduplication,
    Notification,
    NotificationDeliveryAttempt,
    PendingReachyNotification,
)
from notifications.base import DeliveryResult, OutboundMessage
from notifications.mock import MockNotificationChannel
from shared.enums import AlertSeverity, DeliveryStatus
from shared.enums import NotificationChannel as ChannelName
from shared.timeutils import utcnow

logger = get_logger(__name__)

MAX_DELIVERY_ATTEMPTS = 3
RETRY_BASE_SECONDS = 1.0

_channel_override: Any | None = None


def set_channel_override(channel: Any | None) -> None:
    """Install a specific channel instance. Used by tests and the mock CLI."""
    global _channel_override
    _channel_override = channel


def get_channel(settings: Settings | None = None) -> Any:
    """Resolve the active channel.

    Falls back to the mock whenever Telegram is off or mock mode is on, so no
    code path has to branch on configuration.
    """
    if _channel_override is not None:
        return _channel_override

    settings = settings or get_settings()
    if settings.mock_mode or not settings.telegram_enabled:
        return MockNotificationChannel()

    from integrations.telegram.real import TelegramChannel

    return TelegramChannel(settings)


async def dispatch(
    session: AsyncSession,
    *,
    kind: str,
    title: str,
    body: str,
    severity: AlertSeverity = AlertSeverity.INFO,
    channel: ChannelName = ChannelName.TELEGRAM,
    dedup_key: str | None = None,
    resource_type: str | None = None,
    resource_id: str | None = None,
    attachment_path: str | None = None,
    settings: Settings | None = None,
    spoken_text: str | None = None,
) -> Notification:
    """Persist and deliver a notification.

    ``ChannelName.REACHY`` does not send anything: the workstation cannot make the
    robot speak. It queues the message for the Pi to collect on its next turn.
    """
    settings = settings or get_settings()

    notification = Notification(
        kind=kind,
        severity=severity.value,
        title=title,
        body=body,
        channel=channel.value,
        dedup_key=dedup_key,
        resource_type=resource_type,
        resource_id=resource_id,
        attachment_path=attachment_path,
        status=DeliveryStatus.PENDING.value,
    )
    session.add(notification)
    await session.flush()

    if channel is ChannelName.NONE:
        notification.status = DeliveryStatus.SUPPRESSED.value
        await session.flush()
        return notification

    if channel is ChannelName.REACHY:
        session.add(
            PendingReachyNotification(
                kind=kind,
                severity=severity.value,
                spoken_text=spoken_text or f"{title}. {body}",
                detail=body,
                resource_type=resource_type,
                resource_id=resource_id,
            )
        )
        notification.status = DeliveryStatus.SENT.value
        notification.sent_at = utcnow()
        await session.flush()
        return notification

    result = await _deliver_with_retries(
        session,
        notification,
        OutboundMessage(
            title=title,
            body=body,
            severity=severity.value,
            dedup_key=dedup_key,
            attachment_path=attachment_path,
        ),
        settings=settings,
    )

    notification.status = (
        DeliveryStatus.SENT.value if result.delivered else DeliveryStatus.FAILED.value
    )
    if result.delivered:
        notification.sent_at = utcnow()
    await session.flush()
    return notification


async def _deliver_with_retries(
    session: AsyncSession,
    notification: Notification,
    message: OutboundMessage,
    *,
    settings: Settings,
) -> DeliveryResult:
    active = get_channel(settings)
    result = DeliveryResult(delivered=False, error="not attempted")

    for attempt in range(1, MAX_DELIVERY_ATTEMPTS + 1):
        try:
            result = await active.send(message)
        except Exception as exc:
            result = DeliveryResult(delivered=False, error=f"{type(exc).__name__}: {exc}")

        session.add(
            NotificationDeliveryAttempt(
                notification_id=notification.id,
                attempt_number=attempt,
                attempted_at=utcnow(),
                succeeded=result.delivered,
                channel=getattr(active, "name", "unknown"),
                provider_message_id=result.provider_message_id,
                error=result.error,
            )
        )
        await session.flush()

        if result.delivered:
            return result
        if attempt < MAX_DELIVERY_ATTEMPTS:
            await asyncio.sleep(RETRY_BASE_SECONDS * (2 ** (attempt - 1)))

    logger.warning(
        "Notification delivery exhausted its retries",
        extra={"notification_id": notification.id, "kind": notification.kind},
    )
    return result


async def should_notify(
    session: AsyncSession,
    dedup_key: str,
    alert_kind: str,
    *,
    cooldown_seconds: int,
    payload_hash: str | None = None,
    record: bool = True,
) -> bool:
    """Cooldown check for repeating conditions.

    A disk that stays above its threshold is still one problem, not one per
    evaluation cycle. Re-notification happens when the cooldown lapses, or
    immediately if the underlying payload changed.

    Uses the configured workflow.
    """
    now = utcnow()
    existing = await session.scalar(
        select(AlertDeduplication).where(AlertDeduplication.dedup_key == dedup_key)
    )

    if existing is None:
        if record:
            session.add(
                AlertDeduplication(
                    dedup_key=dedup_key,
                    alert_kind=alert_kind,
                    first_seen_at=now,
                    last_notified_at=now,
                    cooldown_seconds=cooldown_seconds,
                    payload_hash=payload_hash,
                )
            )
            await session.flush()
        return True

    changed = payload_hash is not None and payload_hash != existing.payload_hash
    elapsed = (now - existing.last_notified_at).total_seconds()
    if not changed and elapsed < existing.cooldown_seconds:
        return False

    if record:
        existing.last_notified_at = now
        existing.notify_count += 1
        existing.cooldown_seconds = cooldown_seconds
        existing.payload_hash = payload_hash
        await session.flush()
    return True


async def clear_dedup(session: AsyncSession, dedup_key: str) -> None:
    """Forget a condition once it resolves, so its next occurrence notifies at once."""
    record = await session.scalar(
        select(AlertDeduplication).where(AlertDeduplication.dedup_key == dedup_key)
    )
    if record is not None:
        await session.delete(record)
        await session.flush()
