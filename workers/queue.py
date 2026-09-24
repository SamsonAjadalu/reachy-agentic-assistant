"""Durable task queue operations.

Claiming is the only subtle part. A worker selects a candidate and then issues a
conditional UPDATE that also matches on the status and lease it expected; if the
row changed underneath it, ``rowcount`` is zero and the worker moves on. That is
what makes a duplicate claim impossible without a broker.
"""

from __future__ import annotations

import json
from datetime import timedelta
from typing import Any

from sqlalchemy import and_, func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.logging_config import get_logger
from database.models import BackgroundTask, BackgroundTaskEvent
from shared.enums import BackgroundTaskStatus
from shared.errors import NotFoundError
from shared.timeutils import utcnow

logger = get_logger(__name__)


async def record_event(
    session: AsyncSession,
    task_id: str,
    event_type: str,
    *,
    message: str | None = None,
    worker_id: str | None = None,
    detail: dict[str, Any] | None = None,
) -> None:
    session.add(
        BackgroundTaskEvent(
            task_id=task_id,
            occurred_at=utcnow(),
            event_type=event_type,
            message=message,
            worker_id=worker_id,
            detail_json=json.dumps(detail) if detail else None,
        )
    )


async def enqueue(
    session: AsyncSession,
    task_type: str,
    payload: dict[str, Any],
    *,
    priority: int = 100,
    max_attempts: int = 3,
    timeout_seconds: int = 900,
    delay_seconds: int = 0,
    idempotency_key: str | None = None,
    correlation_id: str | None = None,
) -> BackgroundTask:
    """Queue work, returning the existing row if the idempotency key repeats."""
    if idempotency_key:
        existing = await session.scalar(
            select(BackgroundTask).where(BackgroundTask.idempotency_key == idempotency_key)
        )
        if existing is not None:
            return existing

    task = BackgroundTask(
        task_type=task_type,
        payload_json=json.dumps(payload),
        status=BackgroundTaskStatus.QUEUED.value,
        priority=priority,
        max_attempts=max_attempts,
        timeout_seconds=timeout_seconds,
        available_at=utcnow() + timedelta(seconds=delay_seconds),
        idempotency_key=idempotency_key,
        correlation_id=correlation_id,
    )
    session.add(task)
    await session.flush()
    await record_event(session, task.id, "queued", message=f"Queued {task_type}")
    return task


async def claim_next(
    session: AsyncSession, worker_id: str, lease_seconds: int
) -> BackgroundTask | None:
    """Atomically take ownership of one runnable task."""
    now = utcnow()
    candidates = (
        await session.scalars(
            select(BackgroundTask)
            .where(
                BackgroundTask.status == BackgroundTaskStatus.QUEUED.value,
                BackgroundTask.available_at <= now,
                BackgroundTask.cancel_requested.is_(False),
                or_(
                    BackgroundTask.lease_owner.is_(None),
                    BackgroundTask.lease_expires_at < now,
                ),
            )
            .order_by(BackgroundTask.priority.asc(), BackgroundTask.available_at.asc())
            .limit(5)
        )
    ).all()

    for candidate in candidates:
        result = await session.execute(
            update(BackgroundTask)
            .where(
                and_(
                    BackgroundTask.id == candidate.id,
                    BackgroundTask.status == BackgroundTaskStatus.QUEUED.value,
                    or_(
                        BackgroundTask.lease_owner.is_(None),
                        BackgroundTask.lease_expires_at < now,
                    ),
                )
            )
            .values(
                status=BackgroundTaskStatus.RUNNING.value,
                lease_owner=worker_id,
                lease_expires_at=now + timedelta(seconds=lease_seconds),
                started_at=candidate.started_at or now,
                attempts=BackgroundTask.attempts + 1,
                updated_at=now,
            )
        )
        if result.rowcount == 1:  # type: ignore[attr-defined]
            await session.commit()
            claimed = await session.get(BackgroundTask, candidate.id)
            if claimed is not None:
                await record_event(session, claimed.id, "claimed", worker_id=worker_id)
                await session.commit()
            return claimed

    return None


async def renew_lease(session: AsyncSession, task_id: str, worker_id: str, seconds: int) -> bool:
    result = await session.execute(
        update(BackgroundTask)
        .where(BackgroundTask.id == task_id, BackgroundTask.lease_owner == worker_id)
        .values(lease_expires_at=utcnow() + timedelta(seconds=seconds))
    )
    await session.commit()
    return bool(result.rowcount == 1)  # type: ignore[attr-defined]


async def set_progress(
    session: AsyncSession, task_id: str, progress: int, message: str | None
) -> None:
    await session.execute(
        update(BackgroundTask)
        .where(BackgroundTask.id == task_id)
        .values(progress=max(0, min(100, progress)), progress_message=message, updated_at=utcnow())
    )
    await session.commit()


async def complete(
    session: AsyncSession,
    task_id: str,
    worker_id: str,
    result: dict[str, Any] | None,
    *,
    result_path: str | None = None,
) -> None:
    now = utcnow()
    await session.execute(
        update(BackgroundTask)
        .where(BackgroundTask.id == task_id)
        .values(
            status=BackgroundTaskStatus.SUCCEEDED.value,
            finished_at=now,
            progress=100,
            result_json=json.dumps(result) if result is not None else None,
            result_path=result_path,
            lease_owner=None,
            lease_expires_at=None,
            updated_at=now,
        )
    )
    await record_event(session, task_id, "succeeded", worker_id=worker_id)
    await session.commit()


async def fail(
    session: AsyncSession,
    task_id: str,
    worker_id: str,
    error_code: str,
    error_message: str,
    *,
    retryable: bool,
    backoff_seconds: int,
) -> bool:
    """Record a failure. Returns ``True`` when the task was re-queued."""
    task = await session.get(BackgroundTask, task_id)
    if task is None:
        return False

    now = utcnow()
    can_retry = retryable and task.attempts < task.max_attempts and not task.cancel_requested
    if can_retry:
        await session.execute(
            update(BackgroundTask)
            .where(BackgroundTask.id == task_id)
            .values(
                status=BackgroundTaskStatus.QUEUED.value,
                available_at=now + timedelta(seconds=backoff_seconds),
                lease_owner=None,
                lease_expires_at=None,
                error_code=error_code,
                error_message=error_message,
                updated_at=now,
            )
        )
        await record_event(
            session,
            task_id,
            "retry_scheduled",
            message=f"Attempt {task.attempts} failed: {error_code}",
            worker_id=worker_id,
            detail={"backoff_seconds": backoff_seconds},
        )
    else:
        await session.execute(
            update(BackgroundTask)
            .where(BackgroundTask.id == task_id)
            .values(
                status=BackgroundTaskStatus.FAILED.value,
                finished_at=now,
                lease_owner=None,
                lease_expires_at=None,
                error_code=error_code,
                error_message=error_message,
                updated_at=now,
            )
        )
        await record_event(session, task_id, "failed", message=error_message, worker_id=worker_id)
    await session.commit()
    return can_retry


async def request_cancel(session: AsyncSession, task_id: str) -> BackgroundTask:
    task = await session.get(BackgroundTask, task_id)
    if task is None:
        raise NotFoundError(f"No background task with id {task_id}.")

    if task.status == BackgroundTaskStatus.QUEUED.value:
        task.cancel_requested = True
        task.status = BackgroundTaskStatus.CANCELLED.value
        task.finished_at = utcnow()
        await record_event(session, task_id, "cancelled", message="Cancelled before it started")
    elif task.status == BackgroundTaskStatus.RUNNING.value:
        # The runner checks this flag at its next progress point.
        task.cancel_requested = True
        await record_event(session, task_id, "cancel_requested")
    await session.commit()
    return task


async def reclaim_stale(session: AsyncSession, *, older_than_seconds: int = 0) -> int:
    """Return tasks whose worker died back to the queue.

    Called at startup and periodically. A worker that was SIGKILLed leaves its
    row in ``running`` with a lease that stops being renewed; once that lease
    expires the work is provably not in flight anywhere.
    """
    cutoff = utcnow() - timedelta(seconds=older_than_seconds)
    stale = (
        await session.scalars(
            select(BackgroundTask).where(
                BackgroundTask.status == BackgroundTaskStatus.RUNNING.value,
                or_(
                    BackgroundTask.lease_expires_at.is_(None),
                    BackgroundTask.lease_expires_at < cutoff,
                ),
            )
        )
    ).all()

    reclaimed = 0
    for task in stale:
        if task.attempts >= task.max_attempts:
            task.status = BackgroundTaskStatus.FAILED.value
            task.finished_at = utcnow()
            task.error_code = "worker_lost"
            task.error_message = "The worker holding this task stopped and no attempts remain."
            await record_event(session, task.id, "failed", message="Worker lost, no retries left")
        else:
            task.status = BackgroundTaskStatus.QUEUED.value
            task.available_at = utcnow()
            await record_event(
                session, task.id, "reclaimed", message="Lease expired; returned to the queue"
            )
        task.lease_owner = None
        task.lease_expires_at = None
        reclaimed += 1

    if reclaimed:
        await session.commit()
        logger.warning("Reclaimed stale background tasks", extra={"count": reclaimed})
    return reclaimed


async def count_by_status(session: AsyncSession) -> dict[str, int]:
    rows = await session.execute(
        select(BackgroundTask.status, func.count()).group_by(BackgroundTask.status)
    )
    return dict(rows.all())  # type: ignore[arg-type]
