"""APScheduler integration.

Pinned to the APScheduler 3.11 API. Two details of that version drive the design:

* ``SQLAlchemyJobStore`` is synchronous, so it gets its own ``sqlite://`` URL and
  its own database file rather than sharing the async application engine.
* Persisted jobs are stored as a textual reference to a module-level callable, so
  every job target lives in ``scheduler.jobs`` and takes only JSON-serialisable
  arguments. A lambda or bound method cannot be restored after a restart.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Any

from apscheduler.events import EVENT_JOB_ERROR, EVENT_JOB_EXECUTED, EVENT_JOB_MISSED, JobEvent
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger
from apscheduler.triggers.interval import IntervalTrigger

from app.config import Settings
from app.logging_config import get_logger
from scheduler.lock import SchedulerLock
from shared.timeutils import utcnow

logger = get_logger(__name__)

Trigger = DateTrigger | CronTrigger | IntervalTrigger

DEFAULT_MISFIRE_GRACE_SECONDS = 300


class SchedulerService:
    """Owns the process-wide scheduler, guarded by an exclusive lock."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._lock = SchedulerLock(settings.scheduler_lock_path)
        self._scheduler: AsyncIOScheduler | None = None

    @property
    def running(self) -> bool:
        return self._scheduler is not None and self._scheduler.running

    @property
    def scheduler(self) -> AsyncIOScheduler:
        if self._scheduler is None:
            raise RuntimeError("Scheduler is not running in this process.")
        return self._scheduler

    async def start(self) -> bool:
        """Start scheduling, unless another process already holds the lock."""
        if not self._lock.acquire():
            return False

        jobstore = SQLAlchemyJobStore(
            url=self._settings.scheduler_jobstore_url,
            tablename="apscheduler_jobs",
        )
        scheduler = AsyncIOScheduler(
            jobstores={"default": jobstore},
            job_defaults={
                "coalesce": True,  # one catch-up run, not one per missed interval
                "max_instances": 1,
                "misfire_grace_time": DEFAULT_MISFIRE_GRACE_SECONDS,
            },
            timezone="UTC",
        )
        scheduler.add_listener(
            self._on_job_event, EVENT_JOB_EXECUTED | EVENT_JOB_ERROR | EVENT_JOB_MISSED
        )
        scheduler.start()
        self._scheduler = scheduler

        restored = len(scheduler.get_jobs())
        logger.info(
            "Scheduler started",
            extra={"restored_jobs": restored, "jobstore": self._settings.scheduler_jobstore_url},
        )
        return True

    async def stop(self) -> None:
        if self._scheduler is not None:
            self._scheduler.shutdown(wait=False)
            self._scheduler = None
        self._lock.release()

    # ------------------------------------------------------------------ jobs
    def add_date_job(
        self,
        func: str | Callable[..., Any],
        run_at: datetime,
        *,
        job_id: str,
        kwargs: dict[str, Any] | None = None,
        misfire_grace_seconds: int = DEFAULT_MISFIRE_GRACE_SECONDS,
    ) -> str:
        job = self.scheduler.add_job(
            func,
            trigger=DateTrigger(run_date=run_at, timezone="UTC"),
            id=job_id,
            kwargs=kwargs or {},
            replace_existing=True,
            misfire_grace_time=misfire_grace_seconds,
        )
        return str(job.id)

    def add_cron_job(
        self,
        func: str | Callable[..., Any],
        *,
        job_id: str,
        timezone: str,
        hour: int = 0,
        minute: int = 0,
        day_of_week: str | None = None,
        day: str | None = None,
        kwargs: dict[str, Any] | None = None,
        misfire_grace_seconds: int = DEFAULT_MISFIRE_GRACE_SECONDS,
    ) -> str:
        """Schedule a wall-clock time in a named zone.

        Passing the zone to the trigger rather than converting to UTC ourselves
        is what keeps "07:00 every day" at 07:00 across a DST transition.
        """
        trigger = CronTrigger(
            hour=hour,
            minute=minute,
            day_of_week=day_of_week,
            day=day,
            timezone=timezone,
        )
        job = self.scheduler.add_job(
            func,
            trigger=trigger,
            id=job_id,
            kwargs=kwargs or {},
            replace_existing=True,
            misfire_grace_time=misfire_grace_seconds,
        )
        return str(job.id)

    def add_interval_job(
        self,
        func: str | Callable[..., Any],
        *,
        job_id: str,
        seconds: int,
        kwargs: dict[str, Any] | None = None,
        start_immediately: bool = False,
        misfire_grace_seconds: int = DEFAULT_MISFIRE_GRACE_SECONDS,
    ) -> str:
        job = self.scheduler.add_job(
            func,
            trigger=IntervalTrigger(seconds=seconds, timezone="UTC"),
            id=job_id,
            kwargs=kwargs or {},
            replace_existing=True,
            next_run_time=utcnow() if start_immediately else None,
            misfire_grace_time=misfire_grace_seconds,
        )
        return str(job.id)

    def remove_job(self, job_id: str) -> bool:
        try:
            self.scheduler.remove_job(job_id)
        except Exception:  # APScheduler raises JobLookupError; treat as idempotent
            return False
        return True

    def get_job_next_run(self, job_id: str) -> datetime | None:
        job = self.scheduler.get_job(job_id)
        return job.next_run_time if job else None

    def list_jobs(self) -> list[dict[str, Any]]:
        return [
            {
                "id": job.id,
                "name": job.name,
                "next_run_at": job.next_run_time,
                "trigger": str(job.trigger),
            }
            for job in self.scheduler.get_jobs()
        ]

    def _on_job_event(self, event: JobEvent) -> None:
        if event.code == EVENT_JOB_MISSED:
            logger.warning("Scheduled job missed its window", extra={"job_id": event.job_id})
        elif event.code == EVENT_JOB_ERROR:
            logger.error(
                "Scheduled job raised",
                extra={"job_id": event.job_id, "exception_type": type(event.exception).__name__},
            )
