"""Reminders, tasks, recurrence and scheduled job metadata."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database.base import Base, TimestampMixin, UtcDateTime, UUIDPrimaryKeyMixin
from shared.enums import ReminderStatus, TaskPriority, TaskStatus


class Reminder(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "reminders"
    __table_args__ = (
        Index("ix_reminders_status_trigger_at", "status", "trigger_at"),
        UniqueConstraint("idempotency_key", name="uq_reminders_idempotency_key"),
    )

    text: Mapped[str] = mapped_column(Text, nullable=False)
    trigger_at: Mapped[datetime] = mapped_column(UtcDateTime, nullable=False, index=True)
    original_trigger_at: Mapped[datetime] = mapped_column(UtcDateTime, nullable=False)
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default=ReminderStatus.ACTIVE.value
    )
    channel: Mapped[str] = mapped_column(String(20), nullable=False, default="telegram")
    timezone: Mapped[str] = mapped_column(String(64), nullable=False)

    task_id: Mapped[str | None] = mapped_column(
        ForeignKey("tasks.id", ondelete="SET NULL"), index=True
    )
    recurrence_id: Mapped[str | None] = mapped_column(
        ForeignKey("task_recurrences.id", ondelete="SET NULL")
    )

    fired_at: Mapped[datetime | None] = mapped_column(UtcDateTime)
    completed_at: Mapped[datetime | None] = mapped_column(UtcDateTime)
    cancelled_at: Mapped[datetime | None] = mapped_column(UtcDateTime)
    snooze_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    misfire_grace_seconds: Mapped[int] = mapped_column(Integer, nullable=False, default=300)
    job_id: Mapped[str | None] = mapped_column(String(120), index=True)
    idempotency_key: Mapped[str | None] = mapped_column(String(200))
    notes: Mapped[str | None] = mapped_column(Text)

    recurrence: Mapped[TaskRecurrence | None] = relationship(lazy="selectin")
    task: Mapped[Task | None] = relationship(back_populates="reminders")


class TaskRecurrence(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """A structured recurrence rule.

    Stored as explicit fields rather than an RRULE string so the API contract is
    validated by Pydantic and the next occurrence is computable without parsing.
    """

    __tablename__ = "task_recurrences"

    frequency: Mapped[str] = mapped_column(String(20), nullable=False)
    interval: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    by_weekday: Mapped[str | None] = mapped_column(
        String(40), comment="Comma-separated 0-6, Monday=0"
    )
    by_monthday: Mapped[int | None] = mapped_column(Integer)
    at_hour: Mapped[int] = mapped_column(Integer, nullable=False, default=9)
    at_minute: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    timezone: Mapped[str] = mapped_column(String(64), nullable=False)
    starts_at: Mapped[datetime | None] = mapped_column(UtcDateTime)
    ends_at: Mapped[datetime | None] = mapped_column(UtcDateTime)
    max_occurrences: Mapped[int | None] = mapped_column(Integer)
    occurrences_fired: Mapped[int] = mapped_column(Integer, nullable=False, default=0)


class Task(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "tasks"
    __table_args__ = (
        Index("ix_tasks_status_due_at", "status", "due_at"),
        UniqueConstraint("idempotency_key", name="uq_tasks_idempotency_key"),
    )

    description: Mapped[str] = mapped_column(Text, nullable=False)
    detail: Mapped[str | None] = mapped_column(Text)
    due_at: Mapped[datetime | None] = mapped_column(UtcDateTime, index=True)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default=TaskStatus.OPEN.value)
    priority: Mapped[str] = mapped_column(
        String(20), nullable=False, default=TaskPriority.NORMAL.value
    )
    tags: Mapped[str | None] = mapped_column(Text, comment="JSON array of tag strings")
    recurrence_id: Mapped[str | None] = mapped_column(
        ForeignKey("task_recurrences.id", ondelete="SET NULL")
    )
    completed_at: Mapped[datetime | None] = mapped_column(UtcDateTime)
    cancelled_at: Mapped[datetime | None] = mapped_column(UtcDateTime)
    idempotency_key: Mapped[str | None] = mapped_column(String(200))

    recurrence: Mapped[TaskRecurrence | None] = relationship(lazy="selectin")
    reminders: Mapped[list[Reminder]] = relationship(back_populates="task")


class ScheduledJob(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Metadata mirroring APScheduler's own job store.

    APScheduler owns job execution state; this table records what the
    application believes should exist, which is what makes orphan detection and
    restart reconciliation possible.
    """

    __tablename__ = "scheduled_jobs"
    __table_args__ = (UniqueConstraint("job_id", name="uq_scheduled_jobs_job_id"),)

    job_id: Mapped[str] = mapped_column(String(120), nullable=False)
    job_type: Mapped[str] = mapped_column(String(60), nullable=False, index=True)
    description: Mapped[str | None] = mapped_column(Text)
    trigger_kind: Mapped[str] = mapped_column(String(20), nullable=False)
    trigger_config: Mapped[str | None] = mapped_column(Text, comment="JSON")
    resource_id: Mapped[str | None] = mapped_column(String(64), index=True)
    next_run_at: Mapped[datetime | None] = mapped_column(UtcDateTime)
    last_run_at: Mapped[datetime | None] = mapped_column(UtcDateTime)
    last_status: Mapped[str | None] = mapped_column(String(30))
    last_error: Mapped[str | None] = mapped_column(Text)
    run_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    missed_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
