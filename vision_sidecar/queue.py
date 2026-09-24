"""Bounded in-memory job queue with backpressure and cancellation."""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from shared.errors import NotFoundError
from vision_sidecar.errors import JobCancelledError, QueueFullError
from vision_sidecar.types import JobKind, JobPublic, JobStatus


def _iso(ts: float | None) -> str | None:
    if ts is None:
        return None
    return datetime.fromtimestamp(ts, tz=UTC).isoformat()


@dataclass
class Job:
    kind: JobKind
    payload: Any
    correlation_id: str | None = None
    idempotency_key: str | None = None
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    status: JobStatus = JobStatus.QUEUED
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    provider_name: str | None = None
    model_name: str | None = None
    model_version: str | None = None
    latency_ms: float | None = None
    degraded: bool = False
    degraded_reason: str | None = None
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    cancel: asyncio.Event = field(default_factory=asyncio.Event)
    done: asyncio.Event = field(default_factory=asyncio.Event)

    def raise_if_cancelled(self) -> None:
        if self.cancel.is_set():
            raise JobCancelledError("Cancelled.")

    def public(self) -> JobPublic:
        return JobPublic(
            job_id=self.id,
            kind=self.kind,
            status=self.status,
            result=self.result,
            error=self.error,
            provider_name=self.provider_name,
            model_name=self.model_name,
            model_version=self.model_version,
            latency_ms=self.latency_ms,
            degraded=self.degraded,
            degraded_reason=self.degraded_reason,
            correlation_id=self.correlation_id,
            created_at=_iso(self.created_at) or "",
            started_at=_iso(self.started_at),
            finished_at=_iso(self.finished_at),
        )


class JobQueue:
    def __init__(self, *, maxsize: int, max_finished: int, ttl_seconds: int) -> None:
        self.maxsize = maxsize
        self.max_finished = max_finished
        self.ttl_seconds = ttl_seconds
        self._pending: asyncio.Queue[Job] = asyncio.Queue(maxsize=maxsize)
        self._jobs: dict[str, Job] = {}
        self._idempotency: dict[str, str] = {}
        self.rejected_backpressure = 0
        self.accepted = 0

    @property
    def depth(self) -> int:
        return self._pending.qsize()

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def submit(self, job: Job) -> Job:
        self._sweep()
        if job.idempotency_key:
            existing_id = self._idempotency.get(job.idempotency_key)
            if existing_id and existing_id in self._jobs:
                return self._jobs[existing_id]
        try:
            self._pending.put_nowait(job)
        except asyncio.QueueFull as exc:
            self.rejected_backpressure += 1
            raise QueueFullError(
                f"Perception queue is full ({self.maxsize}). Retry shortly."
            ) from exc
        self._jobs[job.id] = job
        if job.idempotency_key:
            self._idempotency[job.idempotency_key] = job.id
        self.accepted += 1
        return job

    async def get_next(self) -> Job:
        return await self._pending.get()

    def task_done(self) -> None:
        self._pending.task_done()

    def cancel(self, job_id: str) -> Job:
        job = self._jobs.get(job_id)
        if job is None:
            raise NotFoundError(f"No vision job {job_id}.")
        if job.status in {JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED}:
            return job
        job.cancel.set()
        if job.status is JobStatus.QUEUED:
            job.status = JobStatus.CANCELLED
            job.error = {"code": "job_cancelled", "message": "Cancelled before execution."}
            job.finished_at = time.time()
            job.done.set()
        return job

    def _sweep(self) -> None:
        cutoff = time.time() - self.ttl_seconds
        finished = [
            job_id
            for job_id, job in self._jobs.items()
            if job.done.is_set() and job.created_at < cutoff
        ]
        overflow = max(0, (len(self._jobs) - len(finished)) - self.max_finished)
        if overflow:
            extra = sorted(
                (j for j in self._jobs.values() if j.done.is_set()),
                key=lambda item: item.created_at,
            )[:overflow]
            finished.extend(item.id for item in extra)
        for job_id in finished:
            job = self._jobs.pop(job_id, None)
            if job and job.idempotency_key:
                self._idempotency.pop(job.idempotency_key, None)
