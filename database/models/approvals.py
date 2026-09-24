"""Approvals, pending actions, notifications and background tasks."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database.base import Base, TimestampMixin, UtcDateTime, UUIDPrimaryKeyMixin
from shared.enums import (
    AlertSeverity,
    ApprovalStatus,
    BackgroundTaskStatus,
    DeliveryStatus,
    PendingActionStatus,
)


class PendingAction(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """An externally visible action prepared but not yet performed.

    The payload and its hash are captured before the owner is asked. Execution
    re-verifies the hash, so an action whose underlying content changed after the
    Uses the configured workflow.
    """

    __tablename__ = "pending_actions"
    __table_args__ = (
        Index("ix_pending_actions_status_expires_at", "status", "expires_at"),
        UniqueConstraint("idempotency_key", name="uq_pending_actions_idempotency_key"),
    )

    action_type: Mapped[str] = mapped_column(String(80), nullable=False, index=True)
    risk_level: Mapped[str] = mapped_column(String(30), nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    payload_json: Mapped[str] = mapped_column(Text, nullable=False, comment="Canonical JSON")
    payload_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    preview_text: Mapped[str] = mapped_column(Text, nullable=False)

    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default=PendingActionStatus.PENDING.value, index=True
    )
    expires_at: Mapped[datetime] = mapped_column(UtcDateTime, nullable=False)
    executed_at: Mapped[datetime | None] = mapped_column(UtcDateTime)
    execution_result: Mapped[str | None] = mapped_column(Text)
    execution_error: Mapped[str | None] = mapped_column(Text)
    external_id: Mapped[str | None] = mapped_column(
        String(200), comment="Provider-side id created by the execution"
    )
    idempotency_key: Mapped[str | None] = mapped_column(String(200))
    correlation_id: Mapped[str | None] = mapped_column(String(64))
    background_task_id: Mapped[str | None] = mapped_column(String(36), index=True)

    approval: Mapped[Approval | None] = relationship(
        back_populates="pending_action", uselist=False, cascade="all, delete-orphan"
    )


class Approval(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """The owner-facing decision attached to a pending action."""

    __tablename__ = "approvals"
    __table_args__ = (
        UniqueConstraint("pending_action_id", name="uq_approvals_pending_action_id"),
        UniqueConstraint("callback_token", name="uq_approvals_callback_token"),
        Index("ix_approvals_status_expires_at", "status", "expires_at"),
    )

    pending_action_id: Mapped[str] = mapped_column(
        ForeignKey("pending_actions.id", ondelete="CASCADE"), nullable=False
    )
    callback_token: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        comment="Opaque id carried in the Telegram callback. Not the action id.",
    )
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default=ApprovalStatus.PENDING.value, index=True
    )
    requested_at: Mapped[datetime] = mapped_column(UtcDateTime, nullable=False)
    expires_at: Mapped[datetime] = mapped_column(UtcDateTime, nullable=False)
    resolved_at: Mapped[datetime | None] = mapped_column(UtcDateTime)
    resolved_by_chat_id: Mapped[int | None] = mapped_column(Integer)
    channel: Mapped[str] = mapped_column(String(20), nullable=False, default="telegram")
    channel_message_id: Mapped[str | None] = mapped_column(String(60))
    decision_note: Mapped[str | None] = mapped_column(Text)

    pending_action: Mapped[PendingAction] = relationship(back_populates="approval")


class Notification(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "notifications"
    __table_args__ = (Index("ix_notifications_created_at_severity", "created_at", "severity"),)

    kind: Mapped[str] = mapped_column(String(60), nullable=False, index=True)
    severity: Mapped[str] = mapped_column(
        String(20), nullable=False, default=AlertSeverity.INFO.value
    )
    title: Mapped[str] = mapped_column(String(300), nullable=False)
    body: Mapped[str] = mapped_column(Text, nullable=False)
    channel: Mapped[str] = mapped_column(String(20), nullable=False, default="telegram")
    dedup_key: Mapped[str | None] = mapped_column(String(200), index=True)
    resource_type: Mapped[str | None] = mapped_column(String(60))
    resource_id: Mapped[str | None] = mapped_column(String(200))
    attachment_path: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default=DeliveryStatus.PENDING.value, index=True
    )
    sent_at: Mapped[datetime | None] = mapped_column(UtcDateTime)

    attempts: Mapped[list[NotificationDeliveryAttempt]] = relationship(
        back_populates="notification", cascade="all, delete-orphan"
    )


class NotificationDeliveryAttempt(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "notification_delivery_attempts"

    notification_id: Mapped[str] = mapped_column(
        ForeignKey("notifications.id", ondelete="CASCADE"), nullable=False, index=True
    )
    attempt_number: Mapped[int] = mapped_column(Integer, nullable=False)
    attempted_at: Mapped[datetime] = mapped_column(UtcDateTime, nullable=False)
    succeeded: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    channel: Mapped[str] = mapped_column(String(20), nullable=False)
    provider_message_id: Mapped[str | None] = mapped_column(String(60))
    error: Mapped[str | None] = mapped_column(Text)

    notification: Mapped[Notification] = relationship(back_populates="attempts")


class AlertDeduplication(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Cooldown ledger keyed by alert identity.

    Without this, a condition that stays true - a disk still above 90% - would
    re-notify on every evaluation cycle.
    """

    __tablename__ = "alert_deduplications"
    __table_args__ = (UniqueConstraint("dedup_key", name="uq_alert_deduplications_dedup_key"),)

    dedup_key: Mapped[str] = mapped_column(String(200), nullable=False)
    alert_kind: Mapped[str] = mapped_column(String(60), nullable=False, index=True)
    first_seen_at: Mapped[datetime] = mapped_column(UtcDateTime, nullable=False)
    last_notified_at: Mapped[datetime] = mapped_column(UtcDateTime, nullable=False)
    cooldown_seconds: Mapped[int] = mapped_column(Integer, nullable=False, default=3600)
    notify_count: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    payload_hash: Mapped[str | None] = mapped_column(String(64))


class PendingReachyNotification(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Messages queued for Reachy to collect on its next interaction.

    The workstation cannot make the robot speak; it can only expose what is waiting.
    The Pi-side agent polls and acknowledges.
    """

    __tablename__ = "pending_reachy_notifications"
    __table_args__ = (Index("ix_pending_reachy_notifications_ack", "acknowledged", "created_at"),)

    kind: Mapped[str] = mapped_column(String(60), nullable=False)
    severity: Mapped[str] = mapped_column(
        String(20), nullable=False, default=AlertSeverity.INFO.value
    )
    spoken_text: Mapped[str] = mapped_column(
        Text, nullable=False, comment="Short phrasing suitable for a voice turn"
    )
    detail: Mapped[str | None] = mapped_column(Text)
    acknowledged: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    acknowledged_at: Mapped[datetime | None] = mapped_column(UtcDateTime)
    expires_at: Mapped[datetime | None] = mapped_column(UtcDateTime)
    resource_type: Mapped[str | None] = mapped_column(String(60))
    resource_id: Mapped[str | None] = mapped_column(String(200))


class BackgroundTask(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Durable unit of deferred work.

    Only genuinely long-running work belongs here. Fast reads answer inline; see
    Uses the configured workflow.
    """

    __tablename__ = "background_tasks"
    __table_args__ = (
        Index("ix_background_tasks_status_available_at", "status", "available_at"),
        UniqueConstraint("idempotency_key", name="uq_background_tasks_idempotency_key"),
    )

    task_type: Mapped[str] = mapped_column(String(60), nullable=False, index=True)
    payload_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    status: Mapped[str] = mapped_column(
        String(30), nullable=False, default=BackgroundTaskStatus.QUEUED.value, index=True
    )
    priority: Mapped[int] = mapped_column(Integer, nullable=False, default=100)
    progress: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    progress_message: Mapped[str | None] = mapped_column(Text)

    available_at: Mapped[datetime] = mapped_column(UtcDateTime, nullable=False, index=True)
    started_at: Mapped[datetime | None] = mapped_column(UtcDateTime)
    finished_at: Mapped[datetime | None] = mapped_column(UtcDateTime)

    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    max_attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=3)
    timeout_seconds: Mapped[int] = mapped_column(Integer, nullable=False, default=900)

    lease_owner: Mapped[str | None] = mapped_column(String(80), index=True)
    lease_expires_at: Mapped[datetime | None] = mapped_column(UtcDateTime, index=True)

    cancel_requested: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    result_json: Mapped[str | None] = mapped_column(Text)
    result_path: Mapped[str | None] = mapped_column(
        Text, comment="Large results live under APP_DATA_DIR/task_outputs"
    )
    error_code: Mapped[str | None] = mapped_column(String(60))
    error_message: Mapped[str | None] = mapped_column(Text)
    idempotency_key: Mapped[str | None] = mapped_column(String(200))
    correlation_id: Mapped[str | None] = mapped_column(String(64))

    events: Mapped[list[BackgroundTaskEvent]] = relationship(
        back_populates="task", cascade="all, delete-orphan"
    )


class BackgroundTaskEvent(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "background_task_events"

    task_id: Mapped[str] = mapped_column(
        ForeignKey("background_tasks.id", ondelete="CASCADE"), nullable=False, index=True
    )
    occurred_at: Mapped[datetime] = mapped_column(UtcDateTime, nullable=False)
    event_type: Mapped[str] = mapped_column(String(40), nullable=False)
    message: Mapped[str | None] = mapped_column(Text)
    worker_id: Mapped[str | None] = mapped_column(String(80))
    detail_json: Mapped[str | None] = mapped_column(Text)

    task: Mapped[BackgroundTask] = relationship(back_populates="events")
