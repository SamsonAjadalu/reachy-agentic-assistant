"""Queue backpressure, cancel, and idempotency."""

from __future__ import annotations

import pytest

from vision_sidecar.errors import QueueFullError
from vision_sidecar.queue import Job, JobQueue
from vision_sidecar.types import JobKind, JobStatus


def _job(**kwargs: object) -> Job:
    return Job(kind=JobKind.DETECT, payload=object(), **kwargs)  # type: ignore[arg-type]


class TestJobQueue:
    def test_rejects_when_full(self) -> None:
        queue = JobQueue(maxsize=1, max_finished=8, ttl_seconds=60)
        queue.submit(_job())
        with pytest.raises(QueueFullError):
            queue.submit(_job())
        assert queue.rejected_backpressure == 1

    def test_idempotency_returns_the_same_job(self) -> None:
        queue = JobQueue(maxsize=4, max_finished=8, ttl_seconds=60)
        first = queue.submit(_job(idempotency_key="same"))
        second = queue.submit(_job(idempotency_key="same"))
        assert first.id == second.id
        assert queue.accepted == 1

    def test_cancel_queued_job(self) -> None:
        queue = JobQueue(maxsize=4, max_finished=8, ttl_seconds=60)
        job = queue.submit(_job())
        cancelled = queue.cancel(job.id)
        assert cancelled.status is JobStatus.CANCELLED
        assert cancelled.done.is_set()

    async def test_get_next_returns_submitted_job(self) -> None:
        queue = JobQueue(maxsize=4, max_finished=8, ttl_seconds=60)
        job = queue.submit(_job())
        claimed = await queue.get_next()
        assert claimed.id == job.id
