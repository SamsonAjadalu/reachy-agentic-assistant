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

def _version() -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("reachy-personal-assistant")
    except PackageNotFoundError:  # pragma: no cover - only when running from a bare checkout
        return "0.0.0-dev"
