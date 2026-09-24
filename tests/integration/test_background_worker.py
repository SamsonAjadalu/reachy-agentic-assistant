"""Durable background task behaviour, including recovery from a lost worker."""

from __future__ import annotations

import asyncio
import json
from datetime import timedelta

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from app.config import Settings
from database.models import BackgroundTask, BackgroundTaskEvent
from shared.enums import BackgroundTaskStatus
from shared.timeutils import utcnow
from workers import queue as task_queue
from workers.registry import TaskContext, clear_registry, register_task
from workers.runner import WorkerRunner, _backoff_for


@pytest.fixture(autouse=True)
def _clean_registry():
    clear_registry()
    yield
    clear_registry()


async def _wait_for(
    session: AsyncSession, task_id: str, statuses: set[str], wait_seconds: float = 10.0
):
    """Poll until the task reaches one of ``statuses``."""
    deadline = asyncio.get_running_loop().time() + wait_seconds
    while asyncio.get_running_loop().time() < deadline:
        session.expunge_all()
        task = await session.get(BackgroundTask, task_id)
        if task is not None and task.status in statuses:
            return task
        await asyncio.sleep(0.05)
    session.expunge_all()
    task = await session.get(BackgroundTask, task_id)
    pytest.fail(f"Task stayed at {task.status if task else 'missing'}, expected one of {statuses}")


class TestEnqueue:
    async def test_enqueued_task_starts_queued(self, session: AsyncSession) -> None:
        task = await task_queue.enqueue(session, "demo", {"n": 1})
        await session.commit()
        assert task.status == BackgroundTaskStatus.QUEUED.value
        assert json.loads(task.payload_json) == {"n": 1}

    async def test_an_idempotency_key_returns_the_existing_task(
        self, session: AsyncSession
    ) -> None:
        first = await task_queue.enqueue(session, "demo", {"n": 1}, idempotency_key="k1")
        await session.commit()
        second = await task_queue.enqueue(session, "demo", {"n": 2}, idempotency_key="k1")
        await session.commit()
        assert first.id == second.id

    async def test_a_delay_defers_availability(self, session: AsyncSession) -> None:
        task = await task_queue.enqueue(session, "demo", {}, delay_seconds=600)
        await session.commit()
        assert task.available_at > utcnow() + timedelta(seconds=500)


class TestClaiming:
    async def test_a_task_can_only_be_claimed_once(self, session: AsyncSession) -> None:
        await task_queue.enqueue(session, "demo", {})
        await session.commit()

        first = await task_queue.claim_next(session, "worker-a", 60)
        second = await task_queue.claim_next(session, "worker-b", 60)

        assert first is not None
        assert second is None

    async def test_claiming_records_the_lease_owner_and_expiry(self, session: AsyncSession) -> None:
        await task_queue.enqueue(session, "demo", {})
        await session.commit()
        task = await task_queue.claim_next(session, "worker-a", 60)
        assert task is not None
        assert task.lease_owner == "worker-a"
        assert task.lease_expires_at is not None
        assert task.attempts == 1

    async def test_a_future_task_is_not_claimed(self, session: AsyncSession) -> None:
        await task_queue.enqueue(session, "demo", {}, delay_seconds=300)
        await session.commit()
        assert await task_queue.claim_next(session, "worker-a", 60) is None

    async def test_higher_priority_is_claimed_first(self, session: AsyncSession) -> None:
        await task_queue.enqueue(session, "low", {}, priority=200)
        await task_queue.enqueue(session, "high", {}, priority=1)
        await session.commit()
        task = await task_queue.claim_next(session, "worker-a", 60)
        assert task is not None
        assert task.task_type == "high"


class TestStaleRecovery:
    async def test_an_expired_lease_returns_the_task_to_the_queue(
        self, session: AsyncSession
    ) -> None:
        """A SIGKILLed worker leaves a lease that simply stops being renewed."""
        await task_queue.enqueue(session, "demo", {})
        await session.commit()
        claimed = await task_queue.claim_next(session, "dead-worker", 60)
        assert claimed is not None

        claimed.lease_expires_at = utcnow() - timedelta(seconds=1)
        await session.commit()

        assert await task_queue.reclaim_stale(session) == 1
        session.expunge_all()
        recovered = await session.get(BackgroundTask, claimed.id)
        assert recovered is not None
        assert recovered.status == BackgroundTaskStatus.QUEUED.value
        assert recovered.lease_owner is None

    async def test_a_live_lease_is_left_alone(self, session: AsyncSession) -> None:
        await task_queue.enqueue(session, "demo", {})
        await session.commit()
        await task_queue.claim_next(session, "live-worker", 600)
        assert await task_queue.reclaim_stale(session) == 0

    async def test_a_task_out_of_attempts_fails_rather_than_looping(
        self, session: AsyncSession
    ) -> None:
        await task_queue.enqueue(session, "demo", {}, max_attempts=1)
        await session.commit()
        claimed = await task_queue.claim_next(session, "dead-worker", 60)
        assert claimed is not None
        claimed.lease_expires_at = utcnow() - timedelta(seconds=1)
        await session.commit()

        await task_queue.reclaim_stale(session)
        session.expunge_all()
        recovered = await session.get(BackgroundTask, claimed.id)
        assert recovered is not None
        assert recovered.status == BackgroundTaskStatus.FAILED.value
        assert recovered.error_code == "worker_lost"

    async def test_recovery_is_recorded_as_an_event(self, session: AsyncSession) -> None:
        await task_queue.enqueue(session, "demo", {})
        await session.commit()
        claimed = await task_queue.claim_next(session, "dead", 60)
        assert claimed is not None
        claimed.lease_expires_at = utcnow() - timedelta(seconds=1)
        await session.commit()
        await task_queue.reclaim_stale(session)

        from sqlalchemy import select

        events = (
            await session.scalars(
                select(BackgroundTaskEvent).where(BackgroundTaskEvent.task_id == claimed.id)
            )
        ).all()
        assert "reclaimed" in {event.event_type for event in events}


class TestExecution:
    async def test_a_handler_runs_and_its_result_is_stored(
        self, settings: Settings, engine: AsyncEngine, session: AsyncSession
    ) -> None:
        @register_task("echo")
        async def echo(context: TaskContext) -> dict:
            return {"echoed": context.payload.get("value")}

        task = await task_queue.enqueue(session, "echo", {"value": 42})
        await session.commit()

        runner = WorkerRunner(settings, worker_id="test-runner")
        await runner.start()
        try:
            finished = await _wait_for(session, task.id, {BackgroundTaskStatus.SUCCEEDED.value})
        finally:
            await runner.stop()

        assert json.loads(finished.result_json) == {"echoed": 42}
        assert finished.progress == 100

    async def test_progress_updates_are_visible_while_running(
        self, settings: Settings, engine: AsyncEngine, session: AsyncSession
    ) -> None:
        seen = asyncio.Event()

        @register_task("slow")
        async def slow(context: TaskContext) -> dict:
            await context.report_progress(50, "halfway")
            seen.set()
            await asyncio.sleep(0.3)
            return {"done": True}

        task = await task_queue.enqueue(session, "slow", {})
        await session.commit()

        runner = WorkerRunner(settings, worker_id="test-runner")
        await runner.start()
        try:
            await asyncio.wait_for(seen.wait(), timeout=10)
            session.expunge_all()
            mid = await session.get(BackgroundTask, task.id)
            assert mid is not None
            assert mid.progress == 50
            assert mid.progress_message == "halfway"
            await _wait_for(session, task.id, {BackgroundTaskStatus.SUCCEEDED.value})
        finally:
            await runner.stop()

    async def test_a_failing_handler_is_retried_then_marked_failed(
        self, settings: Settings, engine: AsyncEngine, session: AsyncSession
    ) -> None:
        attempts = {"count": 0}

        @register_task("always_fails", max_attempts=2)
        async def always_fails(context: TaskContext) -> dict:
            attempts["count"] += 1
            raise RuntimeError("nope")

        task = await task_queue.enqueue(session, "always_fails", {}, max_attempts=2)
        await session.commit()

        runner = WorkerRunner(settings, worker_id="test-runner")
        await runner.start()
        try:
            # The first failure re-queues with a backoff, so the row is briefly
            # 'queued' again; wait for the failure itself to be recorded.
            deadline = asyncio.get_running_loop().time() + 10
            row = None
            while asyncio.get_running_loop().time() < deadline:
                session.expunge_all()
                row = await session.get(BackgroundTask, task.id)
                if row is not None and row.error_code:
                    break
                await asyncio.sleep(0.05)
        finally:
            await runner.stop()

        assert row is not None
        assert row.error_code == "handler_error"
        assert "nope" in (row.error_message or "")
        assert attempts["count"] >= 1

        from sqlalchemy import select

        events = (
            await session.scalars(
                select(BackgroundTaskEvent).where(BackgroundTaskEvent.task_id == task.id)
            )
        ).all()
        assert "retry_scheduled" in {event.event_type for event in events}

    async def test_an_unknown_task_type_fails_without_retrying(
        self, settings: Settings, engine: AsyncEngine, session: AsyncSession
    ) -> None:
        task = await task_queue.enqueue(session, "no_such_handler", {})
        await session.commit()

        runner = WorkerRunner(settings, worker_id="test-runner")
        await runner.start()
        try:
            failed = await _wait_for(session, task.id, {BackgroundTaskStatus.FAILED.value})
        finally:
            await runner.stop()

        assert failed.error_code == "unknown_task_type"

    async def test_a_handler_exceeding_its_timeout_is_stopped(
        self, settings: Settings, engine: AsyncEngine, session: AsyncSession
    ) -> None:
        @register_task("hangs", max_attempts=1)
        async def hangs(context: TaskContext) -> dict:
            await asyncio.sleep(30)
            return {}

        task = await task_queue.enqueue(session, "hangs", {}, max_attempts=1, timeout_seconds=1)
        await session.commit()

        runner = WorkerRunner(settings, worker_id="test-runner")
        await runner.start()
        try:
            failed = await _wait_for(session, task.id, {BackgroundTaskStatus.FAILED.value}, 15)
        finally:
            await runner.stop()

        assert failed.error_code == "timeout"

    async def test_startup_reclaims_work_left_by_a_previous_process(
        self, settings: Settings, engine: AsyncEngine, session: AsyncSession
    ) -> None:
        @register_task("recovered")
        async def recovered(context: TaskContext) -> dict:
            return {"ok": True}

        task = await task_queue.enqueue(session, "recovered", {})
        await session.commit()
        stranded = await task_queue.claim_next(session, "process-that-died", 60)
        assert stranded is not None
        stranded.lease_expires_at = utcnow() - timedelta(seconds=1)
        await session.commit()

        runner = WorkerRunner(settings, worker_id="fresh-process")
        await runner.start()
        try:
            done = await _wait_for(session, task.id, {BackgroundTaskStatus.SUCCEEDED.value})
        finally:
            await runner.stop()

        assert json.loads(done.result_json) == {"ok": True}


class TestCancellation:
    async def test_a_queued_task_cancels_immediately(self, session: AsyncSession) -> None:
        task = await task_queue.enqueue(session, "demo", {})
        await session.commit()
        cancelled = await task_queue.request_cancel(session, task.id)
        assert cancelled.status == BackgroundTaskStatus.CANCELLED.value

    async def test_a_running_task_is_flagged_rather_than_killed(
        self, settings: Settings, engine: AsyncEngine, session: AsyncSession
    ) -> None:
        started = asyncio.Event()
        observed: dict[str, bool] = {}

        @register_task("cancellable")
        async def cancellable(context: TaskContext) -> dict:
            started.set()
            for _ in range(100):
                if await context.is_cancelled():
                    observed["saw_cancel"] = True
                    return {"stopped_early": True}
                await asyncio.sleep(0.05)
            return {"stopped_early": False}

        task = await task_queue.enqueue(session, "cancellable", {})
        await session.commit()

        runner = WorkerRunner(settings, worker_id="test-runner")
        await runner.start()
        try:
            await asyncio.wait_for(started.wait(), timeout=10)
            await task_queue.request_cancel(session, task.id)
            done = await _wait_for(session, task.id, {BackgroundTaskStatus.SUCCEEDED.value}, 15)
        finally:
            await runner.stop()

        assert observed.get("saw_cancel") is True
        assert json.loads(done.result_json) == {"stopped_early": True}

    async def test_cancelling_an_unknown_task_is_a_404(self, session: AsyncSession) -> None:
        from shared.errors import NotFoundError

        with pytest.raises(NotFoundError):
            await task_queue.request_cancel(session, "not-a-task")


class TestBackoff:
    @pytest.mark.parametrize(
        ("attempt", "expected"), [(1, 5), (2, 10), (3, 20), (8, 600), (20, 600)]
    )
    def test_backoff_grows_then_caps(self, attempt: int, expected: int) -> None:
        assert _backoff_for(attempt) == expected


class TestApi:
    async def test_polling_returns_progress_and_events(self, client, session) -> None:
        task = await task_queue.enqueue(session, "demo", {"a": 1})
        await session.commit()

        body = (await client.get(f"/api/v1/background-tasks/{task.id}")).json()
        assert body["status"] == "queued"
        assert body["task_type"] == "demo"
        assert any(event["event_type"] == "queued" for event in body["events"])

    async def test_listing_filters_by_status(self, client, session) -> None:
        await task_queue.enqueue(session, "demo", {})
        await session.commit()
        listed = (await client.get("/api/v1/background-tasks", params={"status": "queued"})).json()
        assert listed["total"] == 1

    async def test_polling_an_unknown_ticket_is_a_404(self, client) -> None:
        assert (await client.get("/api/v1/background-tasks/nope")).status_code == 404

    async def test_task_types_are_discoverable(self, client) -> None:
        @register_task("documented", description="Does a documented thing")
        async def documented(context: TaskContext) -> dict:
            return {}

        types = (await client.get("/api/v1/background-tasks/types")).json()
        assert any(entry["name"] == "documented" for entry in types)
