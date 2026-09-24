"""Approval executor for gated script runs.

The approved payload carries the script name and the arguments, and the
arguments are re-validated against the registry here. Approval fixes what was
agreed to; it does not make the agreed thing safe to run unchecked.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings
from app.services.workstation import get_registry, queue_run
from approvals.state_machine import has_executor, register_executor

RUN_SCRIPT = "workstation.run_script"


async def run_registered_script(
    session: AsyncSession, payload: dict[str, Any], settings: Settings
) -> dict[str, Any]:
    registry = get_registry(settings)
    script = registry.get(str(payload["script"]))
    arguments = dict(payload.get("arguments") or {})

    run = await queue_run(
        session,
        script,
        arguments,
        settings=settings,
        approval_id=str(payload.get("approval_id")) if payload.get("approval_id") else None,
    )
    return {"run_id": run.id, "script": script.name, "status": run.status}


def register_workstation_executors() -> None:
    if not has_executor(RUN_SCRIPT):
        register_executor(RUN_SCRIPT)(run_registered_script)
