"""Background handlers for visual memory."""

from __future__ import annotations

from typing import Any

from database.session import session_scope
from vision.perception import SCAN_TASK, execute_scan
from vision.retention import sweep_orphans, unlink_expired
from workers.registry import TaskContext, has_handler, register_task

WATCH_EVAL_TASK = "vision.evaluate_watches"
RETENTION_UNLINK_TASK = "vision.retention_unlink"


async def run_scan(context: TaskContext) -> dict[str, Any]:
    payload = context.payload
    async with session_scope() as session:
        result = await execute_scan(
            session,
            context.settings,
            query=str(payload.get("query") or "object"),
            preset_id=str(payload.get("preset_id") or "CLOSE_LOOK"),
            planned_poses=int(payload.get("planned_poses") or 3),
            correlation_id=context.correlation_id,
        )
    return result.as_dict()


async def run_retention_unlink(context: TaskContext) -> dict[str, Any]:
    limit = int(context.payload.get("limit") or 200)
    async with session_scope() as session:
        unlinked = await unlink_expired(session, context.settings, limit=limit)
        orphans = await sweep_orphans(session, context.settings)
    return {
        "unlinked": unlinked,
        "orphans_quarantined": orphans.orphans_quarantined,
        "missing_marked": orphans.missing_marked,
    }


def register_vision_tasks() -> None:
    if not has_handler(SCAN_TASK):
        register_task(
            SCAN_TASK,
            timeout_seconds=120,
            max_attempts=1,
            retryable=False,
            description="Execute a preset-only active perception scan.",
        )(run_scan)
    if not has_handler(RETENTION_UNLINK_TASK):
        register_task(
            RETENTION_UNLINK_TASK,
            timeout_seconds=300,
            max_attempts=2,
            description="Unlink expired visual evidence and quarantine orphans.",
        )(run_retention_unlink)
