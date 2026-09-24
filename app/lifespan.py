"""Application startup and shutdown.

Startup order matters: secrets are registered for redaction before anything can
log, then the database, then the scheduler and worker. The scheduler and worker
are optional so a second Uvicorn worker, the CLI and the test suite can run the
API without them.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI

from app.config import Settings, get_settings
from app.logging_config import configure_logging, get_logger
from database.session import create_engine, dispose_engine, set_engine
from security.redaction import register_secrets

if TYPE_CHECKING:
    from scheduler.service import SchedulerService
    from workers.runner import WorkerRunner

logger = get_logger(__name__)


@dataclass
class AppState:
    """Long-lived objects attached to ``app.state``."""

    settings: Settings
    scheduler: SchedulerService | None = None
    worker: WorkerRunner | None = None
    started_at: Any = None
    extras: dict[str, Any] = field(default_factory=dict)


def register_settings_secrets(settings: Settings) -> None:
    """Teach the redaction filter every secret this process holds."""
    register_secrets(
        [
            settings.pa_api_token.get_secret_value(),
            settings.pa_secret_key.get_secret_value(),
            settings.telegram_bot_token.get_secret_value(),
            settings.google_client_secret.get_secret_value(),
            settings.notion_token.get_secret_value(),
            settings.weather_api_key.get_secret_value(),
        ]
    )


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    from shared.timeutils import utcnow

    settings: Settings = getattr(app.state, "settings", None) or get_settings()
    register_settings_secrets(settings)
    configure_logging(settings.app_log_level, settings.app_log_format)

    settings.ensure_directories()
    set_engine(create_engine(settings))

    state = AppState(settings=settings, started_at=utcnow())
    app.state.app_state = state

    logger.info(
        "Starting personal assistant API",
        extra={
            "app_env": settings.app_env.value,
            "mock_mode": settings.mock_mode,
            "data_dir": str(settings.app_data_dir),
        },
    )

    if settings.scheduler_enabled:
        from scheduler.service import SchedulerService

        scheduler = SchedulerService(settings)
        if await scheduler.start():
            state.scheduler = scheduler
        else:
            logger.info("Scheduler not started in this process; another instance holds the lock.")

    if settings.worker_enabled:
        from workers.runner import WorkerRunner

        worker = WorkerRunner(settings)
        await worker.start()
        state.worker = worker

    try:
        yield
    finally:
        if state.worker is not None:
            await state.worker.stop()
        if state.scheduler is not None:
            await state.scheduler.stop()
        await dispose_engine()
        logger.info("Personal assistant API stopped")
