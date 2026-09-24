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
