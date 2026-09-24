"""Workstation operations.

Holds the sequencing that the router and the approval executor both need:
validate against the registry, create a run record, queue the execution. Keeping
it here means an approved run and an unapproved one take exactly the same path
once the decision has been made.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings
from app.logging_config import get_logger
from database.models import RegisteredScript, ScriptRun
from shared.enums import ScriptRunStatus
from shared.errors import NotFoundError
from shared.timeutils import utcnow
from workers.queue import enqueue
from workstation.registry import LoadResult, ScriptDefinition, load_registry
from workstation.runner import start_run
from workstation.tasks import RUN_SCRIPT_TASK

logger = get_logger(__name__)


@lru_cache(maxsize=4)
def _cached_registry(path: Path, mtime: float, size: int) -> LoadResult:
    """Keyed on the file's identity so an edit is picked up without a restart."""
    return load_registry(path)


def get_registry(settings: Settings) -> LoadResult:
    path = settings.workstation_script_registry
    try:
        stat = path.stat()
    except OSError:
        return load_registry(path)
    return _cached_registry(path, stat.st_mtime, stat.st_size)


async def sync_registry(session: AsyncSession, settings: Settings) -> dict[str, int]:
    """Mirror the YAML registry into the database.

    The file stays authoritative. The table exists so runs can reference a
    stable id and so entries that failed to load are visible in the API rather
    than only in the logs.
    """
    registry = get_registry(settings)
    existing = {row.name: row for row in await session.scalars(select(RegisteredScript))}
    seen: set[str] = set()

    for script in registry.scripts.values():
        seen.add(script.name)
        row = existing.get(script.name)
        if row is None:
            row = RegisteredScript(name=script.name)
            session.add(row)
        row.description = script.description
        row.executable = script.executable
        row.working_directory = script.working_directory
        row.parameters_json = json.dumps(
            [
                {
                    "name": parameter.name,
                    "type": parameter.type,
                    "required": parameter.required,
                    "choices": list(parameter.choices),
                    "flag": parameter.flag,
                    "description": parameter.description,
                }
                for parameter in script.parameters
            ]
        )
        row.risk_level = script.risk_level.value
        row.requires_approval = script.requires_approval
        row.timeout_seconds = script.timeout_seconds
        row.enabled = script.enabled
        row.checksum = registry.checksum
        row.load_error = None

    for name, message in registry.errors.items():
        seen.add(name)
        row = existing.get(name)
        if row is None:
            row = RegisteredScript(name=name, description="", executable="")
            session.add(row)
        row.enabled = False
        row.load_error = message

    removed = 0
    for name, row in existing.items():
        if name not in seen and row.enabled:
            # The entry is gone from the file, so the capability is gone. The
            # row stays for the sake of historical runs that point at it.
            row.enabled = False
            row.load_error = "No longer present in the registry file."
            removed += 1

    await session.flush()
    return {
        "loaded": len(registry.scripts),
        "failed": len(registry.errors),
        "retired": removed,
    }


async def queue_run(
    session: AsyncSession,
    script: ScriptDefinition,
    arguments: dict[str, Any],
    *,
    settings: Settings,
    approval_id: str | None = None,
    correlation_id: str | None = None,
) -> ScriptRun:
    """Create the run record and queue its execution."""
    run = await start_run(session, script, arguments, settings=settings, approval_id=approval_id)
    await enqueue(
        session,
        RUN_SCRIPT_TASK,
        {"run_id": run.id, "script": script.name},
        max_attempts=1,
        timeout_seconds=script.timeout_seconds + 60,
        correlation_id=correlation_id,
    )
    logger.info("Queued a script run", extra={"script": script.name, "run_id": run.id})
    return run


async def get_run(session: AsyncSession, run_id: str) -> ScriptRun:
    run = await session.get(ScriptRun, run_id)
    if run is None:
        raise NotFoundError(f"No script run {run_id}.")
    return run


async def list_runs(
    session: AsyncSession,
    *,
    script_name: str | None = None,
    status: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> tuple[list[ScriptRun], int]:
    query = select(ScriptRun)
    if script_name:
        query = query.where(ScriptRun.script_name == script_name)
    if status:
        query = query.where(ScriptRun.status == status)

    total = await session.scalar(select(func.count()).select_from(query.subquery())) or 0
    rows = await session.scalars(
        query.order_by(ScriptRun.created_at.desc()).limit(limit).offset(offset)
    )
    return list(rows), total


async def mark_stopped(session: AsyncSession, run: ScriptRun) -> ScriptRun:
    run.status = ScriptRunStatus.STOPPED.value
    run.finished_at = utcnow()
    run.error = "Stopped on request."
    await session.flush()
    return run
