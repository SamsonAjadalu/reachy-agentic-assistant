"""Running registered scripts.

Every execution goes through ``create_subprocess_exec`` with an argv list. There
is no shell anywhere in this module, so quoting, globbing, redirection and
command chaining are not features that exist to be escaped: a semicolon in an
argument is a semicolon in an argument.

Output is captured to a file under APP_DATA_DIR and only a bounded tail is kept
in the database, so a script that prints in a loop fills a disk quota rather
than the reminder table.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import signal
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings
from app.logging_config import get_logger
from database.models import RegisteredScript, ScriptRun
from shared.enums import ScriptRunStatus
from shared.errors import ValidationError
from shared.timeutils import utcnow
from workstation.registry import ScriptDefinition

logger = get_logger(__name__)

TAIL_BYTES = 4000
GRACE_PERIOD_SECONDS = 5

# Processes are tracked here rather than in the database because a pid is only
# meaningful within the life of this process; after a restart the run is
# reconciled from its record instead.
_running: dict[str, asyncio.subprocess.Process] = {}
_slots: dict[int, asyncio.Semaphore] = {}


def _semaphore(limit: int) -> asyncio.Semaphore:
    if limit not in _slots:
        _slots[limit] = asyncio.Semaphore(limit)
    return _slots[limit]


@dataclass
class RunOutcome:
    status: ScriptRunStatus
    exit_code: int | None
    stdout_tail: str
    stderr_tail: str
    log_path: str | None
    error: str | None = None


def build_environment(script: ScriptDefinition) -> dict[str, str]:
    """A deliberately small environment.

    Inheriting the service environment would hand every script the API token,
    the encryption key and the Telegram credentials.
    """
    base = {
        "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "HOME": os.environ.get("HOME", tempfile.gettempdir()),
        "LANG": os.environ.get("LANG", "C.UTF-8"),
        "TERM": "dumb",
    }
    base.update(script.environment)
    return base


async def start_run(
    session: AsyncSession,
    script: ScriptDefinition,
    arguments: dict[str, object],
    *,
    settings: Settings,
    approval_id: str | None = None,
) -> ScriptRun:
    """Create the run record and validate argv before anything is executed."""
    argv = script.build_argv(dict(arguments))
    _assert_executable(script)

    record = ScriptRun(
        script_name=script.name,
        arguments_json=json.dumps(arguments, sort_keys=True, default=str),
        resolved_argv_json=json.dumps(argv),
        status=ScriptRunStatus.STARTING.value,
        approval_id=approval_id,
    )
    stored = await session.scalar(
        select(RegisteredScript).where(RegisteredScript.name == script.name)
    )
    if stored is not None:
        record.script_id = stored.id
    session.add(record)
    await session.flush()
    return record


def _assert_executable(script: ScriptDefinition) -> None:
    path = Path(script.executable)
    if not path.exists():
        raise ValidationError(f"Script '{script.name}' points at a file that does not exist.")
    if not os.access(path, os.X_OK):
        raise ValidationError(f"Script '{script.name}' is registered but is not executable.")


async def execute(
    run_id: str,
    script: ScriptDefinition,
    argv: Sequence[str],
    *,
    settings: Settings,
) -> RunOutcome:
    """Run the process to completion, a timeout, or a stop request."""
    log_dir = settings.script_log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{run_id}.log"
    limit = settings.workstation_max_output_bytes

    async with _semaphore(settings.workstation_max_concurrent_runs):
        try:
            process = await asyncio.create_subprocess_exec(
                *argv,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                stdin=asyncio.subprocess.DEVNULL,
                cwd=script.working_directory or None,
                env=build_environment(script),
                # A new session means a runaway child that forks is still
                # reachable as a process group when the run has to be stopped.
                start_new_session=True,
            )
        except (OSError, ValueError) as exc:
            return RunOutcome(
                status=ScriptRunStatus.FAILED,
                exit_code=None,
                stdout_tail="",
                stderr_tail="",
                log_path=None,
                error=f"Could not start the script: {exc}",
            )

        _running[run_id] = process
        try:
            return await _supervise(run_id, process, script, log_path, limit)
        finally:
            _running.pop(run_id, None)


async def _supervise(
    run_id: str,
    process: asyncio.subprocess.Process,
    script: ScriptDefinition,
    log_path: Path,
    limit: int,
) -> RunOutcome:
    stdout_chunks: list[bytes] = []
    stderr_chunks: list[bytes] = []
    truncated = False

    with log_path.open("wb") as handle:

        async def drain(
            stream: asyncio.StreamReader | None, sink: list[bytes], label: bytes
        ) -> None:
            nonlocal truncated
            if stream is None:
                return
            written = 0
            while True:
                chunk = await stream.read(8192)
                if not chunk:
                    break
                if written < limit:
                    allowed = chunk[: limit - written]
                    handle.write(label + allowed)
                    sink.append(allowed)
                    written += len(allowed)
                    if written >= limit:
                        truncated = True
                        handle.write(b"\n--- output truncated ---\n")
                # Uses the configured workflow.
                # blocks the child; the excess is simply discarded.

        readers = asyncio.gather(
            drain(process.stdout, stdout_chunks, b""),
            drain(process.stderr, stderr_chunks, b"[stderr] "),
        )

        timed_out = False
        try:
            await asyncio.wait_for(
                asyncio.gather(readers, process.wait()), timeout=script.timeout_seconds
            )
        except TimeoutError:
            timed_out = True
            await _terminate(process)
            readers.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await readers

    stdout_text = b"".join(stdout_chunks).decode("utf-8", "replace")
    stderr_text = b"".join(stderr_chunks).decode("utf-8", "replace")

    if timed_out:
        return RunOutcome(
            status=ScriptRunStatus.TIMED_OUT,
            exit_code=process.returncode,
            stdout_tail=_tail(stdout_text),
            stderr_tail=_tail(stderr_text),
            log_path=str(log_path),
            error=f"The script exceeded its {script.timeout_seconds} second limit and was stopped.",
        )

    if run_id in _stopped:
        _stopped.discard(run_id)
        return RunOutcome(
            status=ScriptRunStatus.STOPPED,
            exit_code=process.returncode,
            stdout_tail=_tail(stdout_text),
            stderr_tail=_tail(stderr_text),
            log_path=str(log_path),
            error="Stopped on request.",
        )

    code = process.returncode
    note = "Output was truncated." if truncated else None
    return RunOutcome(
        status=ScriptRunStatus.SUCCEEDED if code == 0 else ScriptRunStatus.FAILED,
        exit_code=code,
        stdout_tail=_tail(stdout_text),
        stderr_tail=_tail(stderr_text),
        log_path=str(log_path),
        error=note if code == 0 else f"Exited with code {code}.",
    )


_stopped: set[str] = set()


async def _terminate(process: asyncio.subprocess.Process) -> None:
    """SIGTERM the group, then SIGKILL if it is still there."""
    if process.returncode is not None:
        return
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)

    try:
        await asyncio.wait_for(process.wait(), timeout=GRACE_PERIOD_SECONDS)
        return
    except TimeoutError:
        pass

    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    with contextlib.suppress(TimeoutError):
        await asyncio.wait_for(process.wait(), timeout=GRACE_PERIOD_SECONDS)


async def stop_run(run_id: str) -> bool:
    """Ask a running script to stop. Returns whether one was found."""
    process = _running.get(run_id)
    if process is None:
        return False
    _stopped.add(run_id)
    await _terminate(process)
    logger.info("Stopped a script run on request", extra={"run_id": run_id})
    return True


def is_running(run_id: str) -> bool:
    return run_id in _running


def running_ids() -> list[str]:
    return list(_running)


def _tail(text: str) -> str:
    if len(text) <= TAIL_BYTES:
        return text
    return "...\n" + text[-TAIL_BYTES:]


def read_log(run: ScriptRun, *, max_lines: int = 200) -> list[str]:
    """The last lines of a run's captured output."""
    if not run.log_path:
        return []
    path = Path(run.log_path)
    if not path.exists():
        return []
    # Runs are bounded, so reading the file and slicing is cheaper than seeking.
    content = path.read_text(encoding="utf-8", errors="replace")
    return content.splitlines()[-max_lines:]


async def reconcile_orphans(session: AsyncSession) -> int:
    """Close out runs left mid-flight by a restart.

    Their processes died with the service, so a record still claiming to be
    running is a lie that would otherwise persist forever.
    """
    stale = await session.scalars(
        select(ScriptRun).where(
            ScriptRun.status.in_([ScriptRunStatus.STARTING.value, ScriptRunStatus.RUNNING.value])
        )
    )
    count = 0
    for run in stale:
        if run.id in _running:
            continue
        run.status = ScriptRunStatus.FAILED.value
        run.finished_at = utcnow()
        run.error = "The assistant restarted while this script was running."
        count += 1
    if count:
        await session.flush()
        logger.info("Closed out orphaned script runs", extra={"count": count})
    return count
