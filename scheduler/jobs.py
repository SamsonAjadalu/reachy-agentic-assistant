"""Scheduled job targets.

APScheduler's persistent job store serialises a job as a textual reference to an
importable callable, so every target here must be a module-level coroutine
taking only JSON-serialisable arguments. A closure or bound method would
serialise, then fail to resolve after a restart - exactly when reliability
matters most.

Jobs open their own database session because they run outside any request.
"""

from __future__ import annotations

from typing import Any

from app.logging_config import correlation_id_var, get_logger
from database.session import session_scope

logger = get_logger(__name__)

REMINDER_JOB = "scheduler.jobs:fire_reminder"
RECURRING_REMINDER_JOB = "scheduler.jobs:fire_recurring_reminder"
MAINTENANCE_JOB = "scheduler.jobs:run_maintenance"
RECONCILE_JOB = "scheduler.jobs:reconcile_schedules"
DOCUMENT_INDEX_JOB = "scheduler.jobs:refresh_document_index"
BRIEFING_JOB = "scheduler.jobs:send_daily_briefing"
ALERT_SWEEP_JOB = "scheduler.jobs:evaluate_alerts"
VISUAL_WATCH_JOB = "scheduler.jobs:poll_visual_watches"
VISUAL_ORPHAN_JOB = "scheduler.jobs:sweep_visual_orphans"


async def refresh_document_index() -> dict[str, Any]:
    """Queue an incremental re-index of the document roots.

    The scheduler queues rather than scans: a scan can take minutes, and holding
    a scheduler thread that long delays every reminder behind it.
    """
    from app.config import get_settings
    from documents.tasks import REINDEX_TASK
    from workers.queue import enqueue

    settings = get_settings()
    if not settings.document_index_roots:
        return {"queued": 0}

    async with session_scope() as session:
        task = await enqueue(
            session,
            task_type=REINDEX_TASK,
            payload={"full": False},
            priority=200,  # behind anything the owner asked for directly
        )
    return {"queued": 1, "task_id": task.id}


async def reconcile_schedules() -> dict[str, int]:
    """Make the scheduler's jobs match what the database says should exist.

    Runs at startup and on an interval. This is what makes the system tolerant
    of a scheduler job store that was deleted, a reminder created by a process
    that does not own the scheduler, and a job left behind by a deleted row.
    """
    from app.services.reminders import reconcile_reminder_jobs

    async with session_scope() as session:
        summary = await reconcile_reminder_jobs(session)
    if any(summary.values()):
        logger.info("Reconciled scheduled jobs", extra=summary)
    return summary


async def fire_reminder(reminder_id: str, correlation_id: str | None = None) -> None:
    """Deliver a one-off reminder."""
    token = correlation_id_var.set(correlation_id)
    try:
        from app.services.reminders import deliver_reminder

        async with session_scope() as session:
            await deliver_reminder(session, reminder_id)
    except Exception:
        logger.exception("Reminder delivery failed", extra={"reminder_id": reminder_id})
        raise
    finally:
        correlation_id_var.reset(token)


async def fire_recurring_reminder(reminder_id: str, correlation_id: str | None = None) -> None:
    """Deliver one occurrence of a recurring reminder and schedule the next."""
    token = correlation_id_var.set(correlation_id)
    try:
        from app.services.reminders import deliver_recurring_reminder

        async with session_scope() as session:
            await deliver_recurring_reminder(session, reminder_id)
    except Exception:
        logger.exception("Recurring reminder failed", extra={"reminder_id": reminder_id})
        raise
    finally:
        correlation_id_var.reset(token)


async def run_maintenance() -> dict[str, Any]:
    """Housekeeping: expired approvals, stale idempotency keys, orphaned leases."""
    from app.config import get_settings
    from app.services.approvals import expire_stale_approvals
    from app.services.idempotency import purge_expired
    from vision.retention import mark_expired, refresh_ref_count_hints
    from vision.tasks import RETENTION_UNLINK_TASK
    from workers.queue import enqueue, reclaim_stale
    from workstation.runner import reconcile_orphans

    settings = get_settings()
    async with session_scope() as session:
        summary: dict[str, Any] = {
            "expired_approvals": await expire_stale_approvals(session),
            "purged_idempotency_keys": await purge_expired(session),
            "reclaimed_tasks": await reclaim_stale(session),
            "closed_script_runs": await reconcile_orphans(session),
        }
        if settings.visual_enabled or settings.mock_mode:
            report = await mark_expired(session, settings)
            await refresh_ref_count_hints(session)
            summary["visual_marked_expired"] = report.marked_expired
            summary["visual_referenced_skipped"] = report.referenced_skipped
            if report.marked_expired >= settings.visual_retention_unlink_threshold:
                task = await enqueue(
                    session,
                    task_type=RETENTION_UNLINK_TASK,
                    payload={"limit": 200},
                    priority=180,
                )
                summary["visual_unlink_task_id"] = task.id
    logger.info("Maintenance completed", extra=summary)
    return summary


async def poll_visual_watches() -> dict[str, Any]:
    """Evaluate due visual watches. Scans are not approval-gated; notifies Telegram."""
    from app.config import get_settings
    from vision.watches import evaluate_due_watches

    settings = get_settings()
    if not settings.visual_enabled and not settings.mock_mode:
        return {"due": 0, "fired": 0}
    async with session_scope() as session:
        return await evaluate_due_watches(session, settings)


async def sweep_visual_orphans() -> dict[str, Any]:
    """Weekly phase C: quarantine orphan files and mark missing rows."""
    from app.config import get_settings
    from vision.retention import sweep_orphans

    settings = get_settings()
    async with session_scope() as session:
        report = await sweep_orphans(session, settings)
    return {
        "orphans_quarantined": report.orphans_quarantined,
        "missing_marked": report.missing_marked,
    }


async def send_daily_briefing() -> dict[str, Any]:
    """Assemble and deliver the morning briefing on every configured channel."""
    from app.config import get_settings
    from proactive.briefing import send

    settings = get_settings()
    async with session_scope() as session:
        return await send(session, settings, force=False)


async def evaluate_alerts() -> dict[str, Any]:
    """Run every condition rule once and notify on anything past its cooldown."""
    from app.config import get_settings
    from proactive.alerts import evaluate

    settings = get_settings()
    async with session_scope() as session:
        return await evaluate(session, settings)
