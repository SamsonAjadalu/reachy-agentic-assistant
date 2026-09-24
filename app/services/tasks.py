"""To-do task lifecycle.

Tasks are local records with no external side effect, so every operation here is
a reversible write that needs no approval and answers synchronously.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, get_settings
from app.logging_config import get_logger
from app.schemas.reminders import RecurrenceOut
from app.schemas.tasks import TaskCreate, TaskOut, TaskUpdate
from app.services.reminders import _recurrence_to_row, _row_to_rule
from database.models import Reminder, Task, TaskRecurrence
from shared.enums import ReminderStatus, TaskPriority, TaskStatus
from shared.errors import ConflictError, NotFoundError, ValidationError
from shared.timeutils import ensure_utc, to_local, utcnow

logger = get_logger(__name__)

PRIORITY_ORDER = {
    TaskPriority.URGENT.value: 0,
    TaskPriority.HIGH.value: 1,
    TaskPriority.NORMAL.value: 2,
    TaskPriority.LOW.value: 3,
}


async def create_task(
    session: AsyncSession,
    payload: TaskCreate,
    *,
    settings: Settings | None = None,
    idempotency_key: str | None = None,
) -> Task:
    settings = settings or get_settings()

    due_at = ensure_utc(payload.due_at) if payload.due_at else None
    if due_at is not None and due_at > utcnow() + timedelta(days=365 * 10):
        raise ValidationError("A due date more than ten years out is almost certainly a mistake.")

    recurrence_row: TaskRecurrence | None = None
    if payload.recurrence is not None:
        rule = payload.recurrence.to_rule(settings.app_timezone)
        recurrence_row = _recurrence_to_row(payload.recurrence, rule)
        session.add(recurrence_row)
        await session.flush()
        if due_at is None:
            due_at = rule.next_occurrence(utcnow())

    task = Task(
        description=payload.description,
        detail=payload.detail,
        due_at=due_at,
        status=TaskStatus.OPEN.value,
        priority=payload.priority.value,
        tags=json.dumps(sorted({tag.strip().lower() for tag in payload.tags if tag.strip()})),
        recurrence_id=recurrence_row.id if recurrence_row else None,
        idempotency_key=idempotency_key,
    )
    session.add(task)
    await session.flush()
    logger.info("Task created", extra={"task_id": task.id})
    return task


async def get_task(session: AsyncSession, task_id: str) -> Task:
    task = await session.get(Task, task_id)
    if task is None:
        raise NotFoundError(f"No task with id {task_id}.")
    return task


async def list_tasks(
    session: AsyncSession,
    *,
    status: TaskStatus | None = None,
    priority: TaskPriority | None = None,
    tag: str | None = None,
    due_before: datetime | None = None,
    overdue_only: bool = False,
    limit: int = 50,
    offset: int = 0,
) -> tuple[list[Task], int]:
    conditions = []
    if status is not None:
        conditions.append(Task.status == status.value)
    if priority is not None:
        conditions.append(Task.priority == priority.value)
    if tag is not None:
        # Tags are a JSON array of lowercase strings; the quoted match avoids
        # "work" also matching "homework".
        conditions.append(Task.tags.like(f'%"{tag.strip().lower()}"%'))
    if due_before is not None:
        conditions.append(Task.due_at <= ensure_utc(due_before))
    if overdue_only:
        conditions.append(Task.due_at < utcnow())
        conditions.append(Task.status.in_([TaskStatus.OPEN.value, TaskStatus.IN_PROGRESS.value]))

    total = await session.scalar(select(func.count()).select_from(Task).where(*conditions)) or 0
    rows = (
        await session.scalars(
            select(Task)
            .where(*conditions)
            .order_by(
                # Uses the configured workflow.
                Task.due_at.is_(None).asc(),
                Task.due_at.asc(),
                Task.created_at.desc(),
            )
            .limit(limit)
            .offset(offset)
        )
    ).all()
    return list(rows), int(total)


async def update_task(session: AsyncSession, task_id: str, payload: TaskUpdate) -> Task:
    task = await get_task(session, task_id)
    if task.status in {TaskStatus.COMPLETED.value, TaskStatus.CANCELLED.value} and (
        payload.status is None
    ):
        raise ConflictError(
            f"This task is {task.status}. Reopen it by setting status before editing."
        )

    if payload.description is not None:
        task.description = payload.description
    if payload.detail is not None:
        task.detail = payload.detail
    if payload.priority is not None:
        task.priority = payload.priority.value
    if payload.tags is not None:
        task.tags = json.dumps(sorted({tag.strip().lower() for tag in payload.tags if tag.strip()}))
    if payload.clear_due_at:
        task.due_at = None
    elif payload.due_at is not None:
        task.due_at = ensure_utc(payload.due_at)
    if payload.status is not None:
        _apply_status(task, payload.status)

    await session.flush()
    return task


def _apply_status(task: Task, status: TaskStatus) -> None:
    task.status = status.value
    now = utcnow()
    if status is TaskStatus.COMPLETED:
        task.completed_at = now
        task.cancelled_at = None
    elif status is TaskStatus.CANCELLED:
        task.cancelled_at = now
        task.completed_at = None
    else:
        task.completed_at = None
        task.cancelled_at = None


async def complete_task(session: AsyncSession, task_id: str) -> Task:
    """Mark done, rolling a recurring task to its next occurrence.

    A recurring task stays open with a new due date rather than spawning a fresh
    row, so its history and its identity in conversation stay stable.
    """
    task = await get_task(session, task_id)

    if task.recurrence_id:
        recurrence = await session.get(TaskRecurrence, task.recurrence_id)
        if recurrence is not None:
            recurrence.occurrences_fired += 1
            exhausted = (
                recurrence.max_occurrences is not None
                and recurrence.occurrences_fired >= recurrence.max_occurrences
            )
            # Advance past the occurrence being closed, not merely past now:
            # finishing a weekly chore early should schedule next week's, not
            # leave today's still outstanding.
            after = max(utcnow(), task.due_at) if task.due_at else utcnow()
            nxt = None if exhausted else _row_to_rule(recurrence).next_occurrence(after)
            if nxt is not None:
                task.due_at = nxt
                task.status = TaskStatus.OPEN.value
                await session.flush()
                return task

    _apply_status(task, TaskStatus.COMPLETED)
    await session.flush()
    return task


async def cancel_task(session: AsyncSession, task_id: str) -> Task:
    task = await get_task(session, task_id)
    _apply_status(task, TaskStatus.CANCELLED)
    await session.flush()
    return task


async def delete_task(session: AsyncSession, task_id: str) -> None:
    """Remove a task, detaching any reminders that referenced it."""
    task = await get_task(session, task_id)
    linked = await session.scalars(select(Reminder).where(Reminder.task_id == task_id))
    for reminder in linked.all():
        reminder.task_id = None
    await session.delete(task)
    await session.flush()


async def overdue_tasks(session: AsyncSession, limit: int = 20) -> list[Task]:
    rows = await session.scalars(
        select(Task)
        .where(
            Task.due_at < utcnow(),
            Task.status.in_([TaskStatus.OPEN.value, TaskStatus.IN_PROGRESS.value]),
        )
        .order_by(Task.due_at.asc())
        .limit(limit)
    )
    return list(rows.all())


async def tasks_due_within(session: AsyncSession, within: timedelta, limit: int = 20) -> list[Task]:
    now = utcnow()
    rows = await session.scalars(
        select(Task)
        .where(
            Task.due_at.is_not(None),
            Task.due_at <= now + within,
            Task.status.in_([TaskStatus.OPEN.value, TaskStatus.IN_PROGRESS.value]),
        )
        .order_by(Task.due_at.asc())
        .limit(limit)
    )
    return list(rows.all())


async def count_by_status(session: AsyncSession) -> dict[str, int]:
    rows = await session.execute(select(Task.status, func.count()).group_by(Task.status))
    return dict(rows.all())  # type: ignore[arg-type]


async def to_out(session: AsyncSession, task: Task, timezone: str) -> TaskOut:
    # Loaded explicitly rather than through the relationship: a freshly flushed
    # instance would try to lazy-load outside the async context.
    recurrence_row = (
        await session.get(TaskRecurrence, task.recurrence_id) if task.recurrence_id else None
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

    reminder_ids = list(
        (
            await session.scalars(
                select(Reminder.id).where(
                    Reminder.task_id == task.id,
                    Reminder.status.in_(
                        [ReminderStatus.ACTIVE.value, ReminderStatus.SNOOZED.value]
                    ),
                )
            )
        ).all()
    )

    return TaskOut(
        id=task.id,
        description=task.description,
        detail=task.detail,
        status=TaskStatus(task.status),
        priority=TaskPriority(task.priority),
        due_at=task.due_at,
        due_at_local=to_local(task.due_at, timezone).isoformat() if task.due_at else None,
        is_overdue=bool(
            task.due_at
            and task.due_at < utcnow()
            and task.status in {TaskStatus.OPEN.value, TaskStatus.IN_PROGRESS.value}
        ),
        tags=json.loads(task.tags) if task.tags else [],
        created_at=task.created_at,
        completed_at=task.completed_at,
        cancelled_at=task.cancelled_at,
        recurrence=recurrence,
        reminder_ids=reminder_ids,
    )
