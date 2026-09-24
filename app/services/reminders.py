"""Reminder lifecycle.

The database is the source of truth; APScheduler is a cache of "what should fire
next". Any divergence is repaired by ``reconcile_reminder_jobs`` rather than
assumed away, which is what lets the service survive a deleted job store, a
crash between the row write and the job add, or a reminder created by a process
that does not own the scheduler.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, get_settings
from app.logging_config import get_logger
from app.schemas.reminders import (
    RecurrenceOut,
    ReminderCreate,
    ReminderOut,
    ReminderUpdate,
)
from database.models import Reminder, ScheduledJob, Task, TaskRecurrence
from notifications.dispatcher import dispatch
from scheduler import jobs as job_targets
from scheduler.service import get_current_scheduler
from shared.enums import AlertSeverity, NotificationChannel, ReminderStatus
from shared.errors import ConflictError, NotFoundError, ValidationError
from shared.recurrence import RecurrenceRule
from shared.timeutils import isoformat_utc, to_local, utcnow

logger = get_logger(__name__)

MAX_SNOOZES = 20
JOB_PREFIX = "reminder:"


def job_id_for(reminder_id: str) -> str:
    return f"{JOB_PREFIX}{reminder_id}"


# --------------------------------------------------------------------- create
async def create_reminder(
    session: AsyncSession,
    payload: ReminderCreate,
    *,
    settings: Settings | None = None,
    idempotency_key: str | None = None,
    correlation_id: str | None = None,
) -> Reminder:
    settings = settings or get_settings()
    now = utcnow()

    recurrence_row: TaskRecurrence | None = None
    if payload.recurrence is not None:
        rule = payload.recurrence.to_rule(settings.app_timezone)
        first = rule.next_occurrence(now)
        if first is None:
            raise ValidationError("That recurrence has no occurrences in the future.")
        trigger_at = first
        recurrence_row = _recurrence_to_row(payload.recurrence, rule)
        session.add(recurrence_row)
        await session.flush()
    else:
        trigger_at = payload.schedule.resolve(from_time=now)
        if trigger_at <= now:
            raise ValidationError(
                f"The reminder time {isoformat_utc(trigger_at)} is in the past. "
                "Supply a future time or use a relative delay."
            )
        if trigger_at > now + timedelta(days=365 * 5):
            raise ValidationError("Reminders cannot be scheduled more than five years ahead.")

    if payload.task_id is not None and await session.get(Task, payload.task_id) is None:
        raise NotFoundError(f"No task with id {payload.task_id}.")

    reminder = Reminder(
        text=payload.text,
        trigger_at=trigger_at,
        original_trigger_at=trigger_at,
        status=ReminderStatus.ACTIVE.value,
        channel=payload.channel.value,
        timezone=settings.app_timezone,
        task_id=payload.task_id,
        recurrence_id=recurrence_row.id if recurrence_row else None,
        misfire_grace_seconds=payload.misfire_grace_seconds,
        idempotency_key=idempotency_key,
        notes=payload.notes,
    )
    session.add(reminder)
    await session.flush()

    _schedule(reminder, correlation_id=correlation_id)
    await _upsert_job_record(session, reminder)
    await session.flush()

    logger.info(
        "Reminder created",
        extra={"reminder_id": reminder.id, "trigger_at": isoformat_utc(trigger_at)},
    )
    return reminder


def _recurrence_to_row(payload: Any, rule: RecurrenceRule) -> TaskRecurrence:
    return TaskRecurrence(
        frequency=rule.frequency.value,
        interval=rule.interval,
        by_weekday=",".join(str(day) for day in rule.by_weekday) if rule.by_weekday else None,
        by_monthday=rule.by_monthday,
        at_hour=rule.at_hour,
        at_minute=rule.at_minute,
        timezone=rule.timezone,
        starts_at=rule.starts_at,
        ends_at=rule.ends_at,
        max_occurrences=payload.max_occurrences,
    )


def _row_to_rule(row: TaskRecurrence) -> RecurrenceRule:
    return RecurrenceRule(
        frequency=row.frequency,
        interval=row.interval,
        by_weekday=[int(part) for part in row.by_weekday.split(",")] if row.by_weekday else None,
        by_monthday=row.by_monthday,
        at_hour=row.at_hour,
        at_minute=row.at_minute,
        timezone=row.timezone,
        starts_at=row.starts_at,
        ends_at=row.ends_at,
        max_occurrences=row.max_occurrences,
    )


# ------------------------------------------------------------------ scheduling
def _schedule(reminder: Reminder, *, correlation_id: str | None = None) -> str | None:
    """Register the reminder with the scheduler, if this process owns one."""
    scheduler = get_current_scheduler()
    if scheduler is None:
        # Another process owns the scheduler, or scheduling is disabled. The
        # reconcile job will pick this row up.
        return None

    target = (
        job_targets.RECURRING_REMINDER_JOB if reminder.recurrence_id else job_targets.REMINDER_JOB
    )
    job_id = job_id_for(reminder.id)
    scheduler.add_date_job(
        target,
        reminder.trigger_at,
        job_id=job_id,
        kwargs={"reminder_id": reminder.id, "correlation_id": correlation_id},
        misfire_grace_seconds=reminder.misfire_grace_seconds,
    )
    reminder.job_id = job_id
    return job_id


def _unschedule(reminder: Reminder) -> None:
    scheduler = get_current_scheduler()
    if scheduler is not None and reminder.job_id:
        scheduler.remove_job(reminder.job_id)
    reminder.job_id = None


async def _upsert_job_record(session: AsyncSession, reminder: Reminder) -> None:
    """Mirror the intended job into ``scheduled_jobs`` for reconciliation."""
    job_id = job_id_for(reminder.id)
    record = await session.scalar(select(ScheduledJob).where(ScheduledJob.job_id == job_id))
    if record is None:
        record = ScheduledJob(job_id=job_id, job_type="reminder", trigger_kind="date")
        session.add(record)
    record.resource_id = reminder.id
    record.description = reminder.text[:200]
    record.next_run_at = reminder.trigger_at
    record.trigger_config = json.dumps({"run_at": isoformat_utc(reminder.trigger_at)})
    record.enabled = reminder.status in {
        ReminderStatus.ACTIVE.value,
        ReminderStatus.SNOOZED.value,
    }


async def reconcile_reminder_jobs(session: AsyncSession) -> dict[str, int]:
    """Bring APScheduler in line with the reminders table.

    Three repairs: schedule active reminders with no live job, drop jobs whose
    reminder is gone or finished, and mark long-overdue reminders missed instead
    of firing them at a time that would confuse the owner.
    """
    scheduler = get_current_scheduler()
    if scheduler is None:
        return {"scheduled": 0, "removed": 0, "missed": 0}

    now = utcnow()
    summary = {"scheduled": 0, "removed": 0, "missed": 0}

    active = (
        await session.scalars(
            select(Reminder).where(
                Reminder.status.in_([ReminderStatus.ACTIVE.value, ReminderStatus.SNOOZED.value])
            )
        )
    ).all()
    live_ids = {job["id"] for job in scheduler.list_jobs()}

    for reminder in active:
        job_id = job_id_for(reminder.id)
        overdue_by = (now - reminder.trigger_at).total_seconds()

        if overdue_by > reminder.misfire_grace_seconds:
            if reminder.recurrence_id:
                # A recurring reminder skips the window it missed and continues.
                recurrence = await session.get(TaskRecurrence, reminder.recurrence_id)
                if recurrence is not None:
                    nxt = _row_to_rule(recurrence).next_occurrence(now)
                    if nxt is not None:
                        reminder.trigger_at = nxt
                        _schedule(reminder)
                        await _upsert_job_record(session, reminder)
                        summary["scheduled"] += 1
                        continue
            reminder.status = ReminderStatus.MISSED.value
            summary["missed"] += 1
            if job_id in live_ids:
                scheduler.remove_job(job_id)
            continue

        if job_id not in live_ids:
            _schedule(reminder)
            await _upsert_job_record(session, reminder)
            summary["scheduled"] += 1

    known = {job_id_for(reminder.id) for reminder in active}
    for job_id in live_ids:
        if job_id.startswith(JOB_PREFIX) and job_id not in known:
            scheduler.remove_job(job_id)
            summary["removed"] += 1

    await session.flush()
    return summary


# ------------------------------------------------------------------- delivery
async def deliver_reminder(session: AsyncSession, reminder_id: str) -> None:
    reminder = await session.get(Reminder, reminder_id)
    if reminder is None:
        logger.warning("Reminder fired but its row is gone", extra={"reminder_id": reminder_id})
        return
    if reminder.status not in {ReminderStatus.ACTIVE.value, ReminderStatus.SNOOZED.value}:
        logger.info(
            "Skipping a reminder that is no longer active",
            extra={"reminder_id": reminder_id, "status": reminder.status},
        )
        return

    await _send(session, reminder)
    reminder.fired_at = utcnow()
    reminder.status = ReminderStatus.COMPLETED.value
    reminder.completed_at = reminder.fired_at
    reminder.job_id = None
    await session.flush()


async def deliver_recurring_reminder(session: AsyncSession, reminder_id: str) -> None:
    reminder = await session.get(Reminder, reminder_id)
    if reminder is None or reminder.status not in {
        ReminderStatus.ACTIVE.value,
        ReminderStatus.SNOOZED.value,
    }:
        return

    await _send(session, reminder)
    reminder.fired_at = utcnow()

    recurrence = (
        await session.get(TaskRecurrence, reminder.recurrence_id)
        if reminder.recurrence_id
        else None
    )
    if recurrence is None:
        reminder.status = ReminderStatus.COMPLETED.value
        reminder.completed_at = utcnow()
        await session.flush()
        return

    recurrence.occurrences_fired += 1
    exhausted = (
        recurrence.max_occurrences is not None
        and recurrence.occurrences_fired >= recurrence.max_occurrences
    )
    # Anchor on the occurrence just delivered so an early or late run does not
    # shift the whole series.
    after = max(utcnow(), reminder.trigger_at)
    nxt = None if exhausted else _row_to_rule(recurrence).next_occurrence(after)

    if nxt is None:
        reminder.status = ReminderStatus.COMPLETED.value
        reminder.completed_at = utcnow()
        reminder.job_id = None
    else:
        reminder.trigger_at = nxt
        reminder.status = ReminderStatus.ACTIVE.value
        _schedule(reminder)
        await _upsert_job_record(session, reminder)
    await session.flush()


async def _send(session: AsyncSession, reminder: Reminder) -> None:
    local = to_local(reminder.trigger_at, reminder.timezone)
    await dispatch(
        session,
        kind="reminder",
        title="Reminder",
        body=f"{reminder.text}\n\nScheduled for {local:%A %d %B, %H:%M} ({reminder.timezone}).",
        severity=AlertSeverity.INFO,
        channel=NotificationChannel(reminder.channel),
        resource_type="reminder",
        resource_id=reminder.id,
        spoken_text=f"Reminder: {reminder.text}",
    )


# --------------------------------------------------------------------- queries
async def get_reminder(session: AsyncSession, reminder_id: str) -> Reminder:
    reminder = await session.get(Reminder, reminder_id)
    if reminder is None:
        raise NotFoundError(f"No reminder with id {reminder_id}.")
    return reminder


async def list_reminders(
    session: AsyncSession,
    *,
    status: ReminderStatus | None = None,
    upcoming_only: bool = False,
    before: datetime | None = None,
    limit: int = 50,
    offset: int = 0,
) -> tuple[list[Reminder], int]:
    conditions = []
    if status is not None:
        conditions.append(Reminder.status == status.value)
    if upcoming_only:
        conditions.append(Reminder.trigger_at >= utcnow())
        conditions.append(
            Reminder.status.in_([ReminderStatus.ACTIVE.value, ReminderStatus.SNOOZED.value])
        )
    if before is not None:
        conditions.append(Reminder.trigger_at <= before)

    total = await session.scalar(select(func.count()).select_from(Reminder).where(*conditions)) or 0
    rows = (
        await session.scalars(
            select(Reminder)
            .where(*conditions)
            .order_by(Reminder.trigger_at.asc())
            .limit(limit)
            .offset(offset)
        )
    ).all()
    return list(rows), int(total)


async def due_reminders(session: AsyncSession, within: timedelta) -> list[Reminder]:
    """Active reminders due inside a window. Used by the briefing."""
    now = utcnow()
    rows = await session.scalars(
        select(Reminder)
        .where(
            Reminder.status.in_([ReminderStatus.ACTIVE.value, ReminderStatus.SNOOZED.value]),
            Reminder.trigger_at >= now,
            Reminder.trigger_at <= now + within,
        )
        .order_by(Reminder.trigger_at.asc())
    )
    return list(rows.all())


# --------------------------------------------------------------------- updates
async def update_reminder(
    session: AsyncSession, reminder_id: str, payload: ReminderUpdate
) -> Reminder:
    reminder = await get_reminder(session, reminder_id)
    if reminder.status in {ReminderStatus.COMPLETED.value, ReminderStatus.CANCELLED.value}:
        raise ConflictError(f"This reminder is {reminder.status} and can no longer be edited.")

    if payload.text is not None:
        reminder.text = payload.text
    if payload.notes is not None:
        reminder.notes = payload.notes
    if payload.channel is not None:
        reminder.channel = payload.channel.value

    if payload.schedule is not None:
        new_time = payload.schedule.resolve()
        if new_time <= utcnow():
            raise ValidationError("The new reminder time is in the past.")
        reminder.trigger_at = new_time
        reminder.status = ReminderStatus.ACTIVE.value
        _schedule(reminder)
        await _upsert_job_record(session, reminder)

    await session.flush()
    return reminder


async def snooze_reminder(session: AsyncSession, reminder_id: str, delay: timedelta) -> Reminder:
    reminder = await get_reminder(session, reminder_id)
    if reminder.status in {ReminderStatus.CANCELLED.value}:
        raise ConflictError("A cancelled reminder cannot be snoozed.")
    if reminder.snooze_count >= MAX_SNOOZES:
        raise ConflictError(
            f"This reminder has been snoozed {MAX_SNOOZES} times. "
            "Reschedule or complete it instead."
        )

    reminder.trigger_at = utcnow() + delay
    reminder.status = ReminderStatus.SNOOZED.value
    reminder.snooze_count += 1
    reminder.completed_at = None
    _schedule(reminder)
    await _upsert_job_record(session, reminder)
    await session.flush()
    return reminder


async def complete_reminder(session: AsyncSession, reminder_id: str) -> Reminder:
    reminder = await get_reminder(session, reminder_id)
    _unschedule(reminder)
    reminder.status = ReminderStatus.COMPLETED.value
    reminder.completed_at = utcnow()
    await _upsert_job_record(session, reminder)
    await session.flush()
    return reminder


async def cancel_reminder(session: AsyncSession, reminder_id: str) -> Reminder:
    reminder = await get_reminder(session, reminder_id)
    _unschedule(reminder)
    reminder.status = ReminderStatus.CANCELLED.value
    reminder.cancelled_at = utcnow()
    await _upsert_job_record(session, reminder)
    await session.flush()
    return reminder


# ------------------------------------------------------------- serialisation
async def to_out(
    session: AsyncSession, reminder: Reminder, *, next_run_at: datetime | None = None
) -> ReminderOut:
    # Loaded explicitly rather than through the relationship: a freshly
    # flushed instance would try to lazy-load outside the async context.
    recurrence_row = (
        await session.get(TaskRecurrence, reminder.recurrence_id)
        if reminder.recurrence_id
        else None
    )
    recurrence = None
    if recurrence_row is not None:
        rule = _row_to_rule(recurrence_row)
        recurrence = RecurrenceOut(
            frequency=rule.frequency.value,
            interval=rule.interval,
            by_weekday=rule.by_weekday or None,
            by_monthday=rule.by_monthday,
            at_hour=rule.at_hour,
            at_minute=rule.at_minute,
            timezone=rule.timezone,
            description=rule.describe(),
            ends_at=rule.ends_at,
            max_occurrences=recurrence_row.max_occurrences,
            occurrences_fired=recurrence_row.occurrences_fired,
        )

    if next_run_at is None:
        scheduler = get_current_scheduler()
        if scheduler is not None and reminder.job_id:
            next_run_at = scheduler.get_job_next_run(reminder.job_id)

    return ReminderOut(
        id=reminder.id,
        text=reminder.text,
        status=ReminderStatus(reminder.status),
        channel=NotificationChannel(reminder.channel),
        timezone=reminder.timezone,
        trigger_at=reminder.trigger_at,
        trigger_at_local=to_local(reminder.trigger_at, reminder.timezone).isoformat(),
        original_trigger_at=reminder.original_trigger_at,
        created_at=reminder.created_at,
        fired_at=reminder.fired_at,
        completed_at=reminder.completed_at,
        cancelled_at=reminder.cancelled_at,
        snooze_count=reminder.snooze_count,
        task_id=reminder.task_id,
        notes=reminder.notes,
        recurrence=recurrence,
        next_run_at=next_run_at,
    )


async def upcoming_summary(session: AsyncSession, within: timedelta) -> list[str]:
    """One-line renderings for the briefing."""
    return [
        f"{to_local(reminder.trigger_at, reminder.timezone):%H:%M} - {reminder.text}"
        for reminder in await due_reminders(session, within)
    ]


async def count_by_status(session: AsyncSession) -> dict[str, int]:
    rows = await session.execute(select(Reminder.status, func.count()).group_by(Reminder.status))
    return dict(rows.all())  # type: ignore[arg-type]


async def overdue_reminders(session: AsyncSession) -> list[Reminder]:
    now = utcnow()
    rows = await session.scalars(
        select(Reminder).where(
            or_(
                Reminder.status == ReminderStatus.ACTIVE.value,
                Reminder.status == ReminderStatus.SNOOZED.value,
            ),
            Reminder.trigger_at < now,
        )
    )
    return list(rows.all())
