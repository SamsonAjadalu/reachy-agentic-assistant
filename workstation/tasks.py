"""Background execution of registered scripts.

A script may run for up to an hour, so the API hands back a run id immediately
and the work continues here. The run record is the durable part: it is updated
as the process starts, finishes, times out or is stopped, and it is what the
owner sees whether or not the service stayed up.
"""

from __future__ import annotations

import json
from typing import Any

from app.logging_config import get_logger
from database.models import ScriptRun
from database.session import session_scope
from shared.enums import ScriptRunStatus
from shared.errors import ValidationError
from shared.timeutils import utcnow
from workers.registry import TaskContext, has_handler, register_task
from workstation.registry import load_registry
from workstation.runner import execute

logger = get_logger(__name__)

RUN_SCRIPT_TASK = "workstation.run_script"


async def run_script(context: TaskContext) -> dict[str, Any]:
    """Execute a registered script and record what happened."""
    run_id = str(context.payload["run_id"])
    settings = context.settings

    async with session_scope() as session:
        run = await session.get(ScriptRun, run_id)
        if run is None:
            raise ValidationError(f"No script run {run_id}.")
        script_name = run.script_name
        argv = json.loads(run.resolved_argv_json or "[]")
        run.status = ScriptRunStatus.RUNNING.value
        run.started_at = utcnow()

    registry = load_registry(settings.workstation_script_registry)
    script = registry.get(script_name)
    await context.report_progress(10, f"Running {script_name}")

    outcome = await execute(run_id, script, argv, settings=settings)

    async with session_scope() as session:
        run = await session.get(ScriptRun, run_id)
        if run is not None:
            run.status = outcome.status.value
            run.exit_code = outcome.exit_code
            run.stdout_tail = outcome.stdout_tail or None
            run.stderr_tail = outcome.stderr_tail or None
            run.log_path = outcome.log_path
            run.error = outcome.error
            run.finished_at = utcnow()

    logger.info(
        "A registered script finished",
        extra={"script": script_name, "status": outcome.status.value},
    )
    return {
        "run_id": run_id,
        "script": script_name,
        "status": outcome.status.value,
        "exit_code": outcome.exit_code,
    }


def register_workstation_tasks() -> None:
    if not has_handler(RUN_SCRIPT_TASK):
        register_task(
            RUN_SCRIPT_TASK,
            timeout_seconds=3900,
            max_attempts=1,
            # A half-finished script that is retried could apply its effect
            # twice; the owner decides whether to run it again.
            retryable=False,
            description="Execute a script from the workstation registry.",
        )(run_script)
