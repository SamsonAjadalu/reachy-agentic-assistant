"""Health, readiness, status and integration inventory."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, get_settings
from app.dependencies.auth import Principal, require_token
from app.dependencies.common import get_session
from database.session import check_integrity, get_engine
from shared.timeutils import isoformat_utc, utcnow
from workers.queue import count_by_status

router = APIRouter(tags=["system"])

SessionDep = Annotated[AsyncSession, Depends(get_session)]
SettingsDep = Annotated[Settings, Depends(get_settings)]
PrincipalDep = Annotated[Principal, Depends(require_token)]


class HealthResponse(BaseModel):
    status: str = Field(description="'ok' when the process is alive.")
    version: str
    timestamp: str


class ReadyResponse(BaseModel):
    ready: bool
    checks: dict[str, Any]


class PingResponse(BaseModel):
    pong: bool = True
    message: str
    server_time_utc: str
    server_time_local: str
    timezone: str
    request_id: str | None = None


class IntegrationStatus(BaseModel):
    name: str
    enabled: bool
    mode: str = Field(description="'real', 'mock' or 'disabled'.")
    configured: bool
    detail: str | None = None


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness probe",
    description="Unauthenticated. Answers only whether the process is running.",
)
async def health() -> HealthResponse:
    return HealthResponse(status="ok", version=_version(), timestamp=isoformat_utc(utcnow()))


@router.get(
    "/ready",
    response_model=ReadyResponse,
    summary="Readiness probe",
    description=(
        "Unauthenticated. Verifies the database answers and reports whether the scheduler "
        "and worker are active in this process."
    ),
)
async def ready(request: Request) -> ReadyResponse:
    checks: dict[str, Any] = {}
    ok = True

    try:
        async with get_engine().connect() as conn:
            await conn.execute(text("SELECT 1"))
        checks["database"] = {"ok": True}
    except Exception as exc:
        ok = False
        checks["database"] = {"ok": False, "error": type(exc).__name__}

    state = getattr(request.app.state, "app_state", None)
    checks["scheduler"] = {
        "running": bool(state and state.scheduler and state.scheduler.running),
        "owned_by_this_process": bool(state and state.scheduler),
    }
    checks["worker"] = {"running": bool(state and state.worker and state.worker.running)}
    return ReadyResponse(ready=ok, checks=checks)


@router.get(
    "/api/v1/ping",
    response_model=PingResponse,
    summary="Authenticated round-trip check",
    description=(
        "The endpoint a Reachy tool calls to prove connectivity and authentication. "
        "Read-only, no side effects, answers synchronously."
    ),
)
async def ping(request: Request, _: PrincipalDep, settings: SettingsDep) -> PingResponse:
    from shared.timeutils import to_local

    now = utcnow()
    return PingResponse(
        message="The personal assistant service is reachable.",
        server_time_utc=isoformat_utc(now),
        server_time_local=to_local(now, settings.app_timezone).isoformat(),
        timezone=settings.app_timezone,
        request_id=getattr(request.state, "request_id", None),
    )


@router.get(
    "/api/v1/status",
    summary="Service status",
    description="Uptime, database integrity, background queue depth and scheduler state.",
)
async def status(
    request: Request, _: PrincipalDep, settings: SettingsDep, session: SessionDep
) -> dict[str, Any]:
    state = getattr(request.app.state, "app_state", None)
    now = utcnow()
    started_at = getattr(state, "started_at", None)

    integrity = await check_integrity()
    queue_depth = await count_by_status(session)

    scheduler_jobs: list[dict[str, Any]] = []
    if state and state.scheduler and state.scheduler.running:
        scheduler_jobs = [
            {
                "id": job["id"],
                "next_run_at": isoformat_utc(job["next_run_at"]) if job["next_run_at"] else None,
            }
            for job in state.scheduler.list_jobs()
        ]

    return {
        "version": _version(),
        "environment": settings.app_env.value,
        "mock_mode": settings.mock_mode,
        "timezone": settings.app_timezone,
        "server_time_utc": isoformat_utc(now),
        "started_at": isoformat_utc(started_at) if started_at else None,
        "uptime_seconds": int((now - started_at).total_seconds()) if started_at else None,
        "database": {
            "ok": integrity["ok"],
            "journal_mode": integrity["journal_mode"],
            "foreign_keys_enabled": integrity["foreign_keys_enabled"],
        },
        "scheduler": {
            "running": bool(state and state.scheduler and state.scheduler.running),
            "job_count": len(scheduler_jobs),
            "jobs": scheduler_jobs[:20],
        },
        "worker": {
            "running": bool(state and state.worker and state.worker.running),
            "queue_depth": queue_depth,
        },
        "visual": {
            "enabled": settings.visual_enabled,
            "keep_full_frames": settings.visual_keep_full_frames,
            "disk": _visual_disk(settings),
        },
    }


@router.get(
    "/api/v1/integrations",
    response_model=list[IntegrationStatus],
    summary="Integration inventory",
    description=(
        "Which providers are enabled and whether each is running against the real service "
        "or its mock. Credentials remain private."
    ),
)
async def integrations(_: PrincipalDep, settings: SettingsDep) -> list[IntegrationStatus]:
    def entry(
        name: str, enabled: bool, configured: bool, detail: str | None = None
    ) -> IntegrationStatus:
        if settings.mock_mode:
            mode = "mock"
        elif not enabled:
            mode = "disabled"
        else:
            mode = "real" if configured else "mock"
        return IntegrationStatus(
            name=name, enabled=enabled, mode=mode, configured=configured, detail=detail
        )

    google_configured = bool(
        settings.google_client_id and settings.google_client_secret.get_secret_value()
    )
    return [
        entry(
            "telegram",
            settings.telegram_enabled,
            bool(
                settings.telegram_bot_token.get_secret_value()
                and settings.telegram_allowed_chat_ids
            ),
            "Notification and approval channel.",
        ),
        entry(
            "google", settings.google_enabled, google_configured, "Shared OAuth for Google APIs."
        ),
        entry("gmail", settings.google_enabled, google_configured),
        entry("calendar", settings.google_enabled, google_configured),
        entry("contacts", settings.google_enabled, google_configured),
        entry("drive", settings.google_enabled, google_configured),
        entry(
            "notion",
            settings.notion_enabled,
            bool(settings.notion_token.get_secret_value()),
            "Only pages shared with the integration are reachable.",
        ),
        entry(
            "weather",
            True,
            settings.weather_provider.value != "mock",
            f"Provider: {settings.weather_provider.value}.",
        ),
        entry("workstation", True, settings.workstation_script_registry.exists()),
        entry("documents", True, bool(settings.document_index_roots)),
        entry("wardrobe", True, True),
        entry(
            "vision",
            settings.visual_enabled,
            settings.reachy_camera_enabled or settings.mock_mode,
            "Keyframe memory, watches, and HF look tools. Sidecar is a separate process.",
        ),
        entry(
            "vision_sidecar",
            settings.visual_sidecar_enabled,
            bool(settings.vision_sidecar_token.get_secret_value() or settings.mock_mode),
            "Perception process on loopback. Mocked unless visual_sidecar_enabled.",
        ),
    ]


@router.get(
    "/api/v1/integrations/google",
    summary="Google authorisation status",
    description=(
        "Whether a Google grant is stored, which account it belongs to and which scopes "
        "it covers. Returns status information."
    ),
)
async def google_status(_: PrincipalDep, settings: SettingsDep) -> dict[str, Any]:
    if settings.mock_mode or not settings.google_enabled:
        return {
            "enabled": settings.google_enabled,
            "mode": "mock",
            "authorised": False,
            "detail": "Running against the mock provider.",
        }

    from integrations.google.oauth import TokenProvider

    try:
        status_payload = TokenProvider(settings).status()
    except Exception as exc:
        return {"enabled": True, "mode": "real", "authorised": False, "detail": str(exc)}

    return {"enabled": True, "mode": "real", **status_payload}


def _visual_disk(settings: Settings) -> dict[str, Any]:
    from vision.metrics import collect_disk_metrics, metrics_as_dict

    return metrics_as_dict(collect_disk_metrics(settings))


def _version() -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("reachy-personal-assistant")
    except PackageNotFoundError:  # pragma: no cover - only when running from a bare checkout
        return "0.0.0-dev"
