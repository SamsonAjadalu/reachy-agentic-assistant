"""Briefing endpoints.

Assembly is a handful of local queries plus whatever providers are enabled, so
it answers synchronously. Delivery is explicit: asking for a briefing returns
one, and does not also send it to Telegram unless that was the request.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query
from pydantic import Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, get_settings
from app.dependencies.auth import Principal, require_token
from app.dependencies.common import get_session
from app.schemas.common import ApiModel
from proactive import briefing as service
from proactive.formatters import formatter_names
from proactive.sections import available_sections
from shared.errors import ValidationError

router = APIRouter(prefix="/api/v1/briefings", tags=["briefings"])

SessionDep = Annotated[AsyncSession, Depends(get_session)]
SettingsDep = Annotated[Settings, Depends(get_settings)]
PrincipalDep = Annotated[Principal, Depends(require_token)]


class PreferencesIn(ApiModel):
    enabled: bool | None = None
    send_hour: Annotated[int | None, Field(ge=0, le=23)] = None
    send_minute: Annotated[int | None, Field(ge=0, le=59)] = None
    timezone: str | None = None
    sections: list[str] | None = None
    channels: list[str] | None = None
    quiet_hours_start: Annotated[int | None, Field(ge=0, le=23)] = None
    quiet_hours_end: Annotated[int | None, Field(ge=0, le=23)] = None
    formatter: str | None = None


class PreferencesOut(ApiModel):
    enabled: bool
    send_hour: int
    send_minute: int
    timezone: str
    sections: list[str]
    channels: list[str]
    quiet_hours_start: int | None
    quiet_hours_end: int | None
    formatter: str
    available_sections: list[str]
    available_formatters: list[str]


@router.get("/today", summary="Today's briefing")
async def today(
    _: PrincipalDep,
    session: SessionDep,
    settings: SettingsDep,
    sections: Annotated[str | None, Query(max_length=200)] = None,
    formatter: Annotated[str | None, Query(max_length=40)] = None,
) -> dict[str, Any]:
    wanted = [part.strip() for part in sections.split(",") if part.strip()] if sections else None
    if wanted:
        unknown = sorted(set(wanted) - set(available_sections()))
        if unknown:
            raise ValidationError(f"Unknown section(s): {', '.join(unknown)}.")

    result = await service.assemble(session, settings, sections=wanted, formatter_name=formatter)
    return result.as_dict()


@router.post(
    "/send",
    summary="Assemble and deliver the briefing now",
    description="Respects quiet hours unless 'force' is set.",
)
@router.post(
    "/generate",
    summary="Assemble and deliver the briefing now",
    description="Alias of /send. Respects quiet hours unless 'force' is set.",
    include_in_schema=False,
)
async def send(
    _: PrincipalDep,
    session: SessionDep,
    settings: SettingsDep,
    force: bool = False,
) -> dict[str, Any]:
    result = await service.send(session, settings, force=force)
    await session.commit()
    return result


@router.get("/preferences", response_model=PreferencesOut, summary="Briefing preferences")
async def get_preferences(
    _: PrincipalDep, session: SessionDep, settings: SettingsDep
) -> PreferencesOut:
    return _out(await service.load_preferences(session, settings))


@router.patch("/preferences", response_model=PreferencesOut, summary="Update preferences")
async def update_preferences(
    payload: PreferencesIn,
    _: PrincipalDep,
    session: SessionDep,
    settings: SettingsDep,
) -> PreferencesOut:
    updated = await service.save_preferences(
        session, payload.model_dump(exclude_unset=True), settings
    )
    await session.commit()
    return _out(updated)


def _out(preferences: service.BriefingSettings) -> PreferencesOut:
    return PreferencesOut(
        enabled=preferences.enabled,
        send_hour=preferences.send_hour,
        send_minute=preferences.send_minute,
        timezone=preferences.timezone,
        sections=preferences.sections,
        channels=preferences.channels,
        quiet_hours_start=preferences.quiet_hours_start,
        quiet_hours_end=preferences.quiet_hours_end,
        formatter=preferences.formatter,
        available_sections=available_sections(),
        available_formatters=formatter_names(),
    )
