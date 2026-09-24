"""In-process background worker.

Runs as asyncio tasks inside the API process by default, and as a standalone
process via ``python -m workers`` when the deployment separates them. Either way
the leasing in ``workers.queue`` means only one runner can hold a given task.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import socket
import uuid
from typing import Any

from app.config import Settings
from app.logging_config import correlation_id_var, get_logger
from database.session import session_scope
from shared.errors import AssistantError
from workers import queue as task_queue
from workers.registry import TaskContext, get_handler, has_handler

logger = get_logger(__name__)

POLL_INTERVAL_SECONDS = 2.0
REAPER_INTERVAL_SECONDS = 60.0
MAX_BACKOFF_SECONDS = 600


def _backoff_for(attempt: int) -> int:
    """Exponential backoff, capped, starting at 5 seconds."""
    return min(MAX_BACKOFF_SECONDS, 5 * (2 ** max(0, attempt - 1)))


class WorkerRunner:
    def __init__(self, settings: Settings, *, worker_id: str | None = None) -> None:
        self._settings = settings
        self._worker_id = (
            worker_id or f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:6]}"
        )
        self._tasks: list[asyncio.Task[None]] = []
        self._stopping = asyncio.Event()

    @property
    def worker_id(self) -> str:
        return self._worker_id

    @property
    def running(self) -> bool:
        return bool(self._tasks) and not self._stopping.is_set()

    async def start(self) -> None:
        # Anything left running belongs to a process that is no longer alive.
        async with session_scope() as session:
            await task_queue.reclaim_stale(session)

        concurrency = max(1, self._settings.worker_concurrency)
        for index in range(concurrency):
            self._tasks.append(asyncio.create_task(self._loop(index), name=f"pa-worker-{index}"))
        self._tasks.append(asyncio.create_task(self._reaper(), name="pa-worker-reaper"))
        logger.info(
            "Background worker started",
            extra={"worker_id": self._worker_id, "concurrency": concurrency},
        )

    async def stop(self) -> None:
        self._stopping.set()
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._tasks.clear()
        logger.info("Background worker stopped", extra={"worker_id": self._worker_id})

    async def _loop(self, slot: int) -> None:
        slot_id = f"{self._worker_id}#{slot}"
        while not self._stopping.is_set():
            try:
                claimed = await self._claim_and_run(slot_id)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Worker loop error", extra={"worker_id": slot_id})
                claimed = False
            if not claimed:
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(self._stopping.wait(), timeout=POLL_INTERVAL_SECONDS)

    async def _reaper(self) -> None:
        while not self._stopping.is_set():
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._stopping.wait(), timeout=REAPER_INTERVAL_SECONDS)
            if self._stopping.is_set():
                return
            try:
                async with session_scope() as session:
                    await task_queue.reclaim_stale(session)
            except Exception:
                logger.exception("Stale task reaper failed")

    async def _claim_and_run(self, slot_id: str) -> bool:
        async with session_scope() as session:
            task = await task_queue.claim_next(
                session, slot_id, self._settings.worker_lease_seconds
            )
            if task is None:
                return False
            task_id = task.id
            task_type = task.task_type
            payload = json.loads(task.payload_json or "{}")
            attempt = task.attempts
            timeout_seconds = task.timeout_seconds
            correlation_id = task.correlation_id

        token = correlation_id_var.set(correlation_id)
        try:
            await self._execute(
                task_id=task_id,
                task_type=task_type,
                payload=payload,
                attempt=attempt,
                timeout_seconds=timeout_seconds,
                slot_id=slot_id,
                correlation_id=correlation_id,
            )
        finally:
            correlation_id_var.reset(token)
        return True

    async def _execute(
        self,
        *,
        task_id: str,
        task_type: str,
        payload: dict[str, Any],
        attempt: int,
        timeout_seconds: int,
        slot_id: str,
        correlation_id: str | None,
    ) -> None:
        if not has_handler(task_type):
            async with session_scope() as session:
                await task_queue.fail(
                    session,
                    task_id,
                    slot_id,
                    "unknown_task_type",
                    f"No handler is registered for task type {task_type!r}.",
                    retryable=False,
                    backoff_seconds=0,
                )
            return

        spec = get_handler(task_type)

        async def report_progress(percent: int, message: str | None = None) -> None:
            async with session_scope() as session:
                await task_queue.set_progress(session, task_id, percent, message)
                await task_queue.renew_lease(
                    session, task_id, slot_id, self._settings.worker_lease_seconds
                )

        async def is_cancelled() -> bool:
            async with session_scope() as session:
                from database.models import BackgroundTask

                row = await session.get(BackgroundTask, task_id)
                return bool(row and row.cancel_requested)

        context = TaskContext(
            task_id=task_id,
            task_type=task_type,
            payload=payload,
            attempt=attempt,
            worker_id=slot_id,
            report_progress=report_progress,
            is_cancelled=is_cancelled,
            settings=self._settings,
            correlation_id=correlation_id,
        )

        heartbeat = asyncio.create_task(self._heartbeat(task_id, slot_id))
        try:
            result = await asyncio.wait_for(spec.handler(context), timeout=timeout_seconds)
        except TimeoutError:
            await self._record_failure(
                task_id,
                slot_id,
                "timeout",
                f"Exceeded {timeout_seconds}s.",
                spec.retryable,
                attempt,
            )
        except asyncio.CancelledError:
            await self._record_failure(
                task_id, slot_id, "cancelled", "The worker was shut down.", True, attempt
            )
            raise
        except AssistantError as exc:
            await self._record_failure(
                task_id, slot_id, exc.code, exc.message, spec.retryable, attempt
            )
        except Exception as exc:
            logger.exception("Background task raised", extra={"task_id": task_id})
            await self._record_failure(
                task_id,
                slot_id,
                "handler_error",
                f"{type(exc).__name__}: {exc}",
                spec.retryable,
                attempt,
            )
        else:
            async with session_scope() as session:
                await task_queue.complete(session, task_id, slot_id, result)
        finally:
            heartbeat.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await heartbeat

    async def _record_failure(
        self, task_id: str, slot_id: str, code: str, message: str, retryable: bool, attempt: int
    ) -> None:
        async with session_scope() as session:
            await task_queue.fail(
                session,
                task_id,
                slot_id,
                code,
                message,
                retryable=retryable,
                backoff_seconds=_backoff_for(attempt),
            )

    async def _heartbeat(self, task_id: str, slot_id: str) -> None:
        """Renew the lease while a long handler runs, so the reaper leaves it alone."""
        interval = max(5, self._settings.worker_lease_seconds // 3)
        while True:
            await asyncio.sleep(interval)
            try:
                async with session_scope() as session:
                    await task_queue.renew_lease(
                        session, task_id, slot_id, self._settings.worker_lease_seconds
                    )
            except Exception:
                logger.warning("Failed to renew task lease", extra={"task_id": task_id})
