"""Weather endpoints.

All reads, all synchronous, all cached. Weather is the most frequently asked
question in a system like this and the least expensive to serve.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query

from app.config import Settings, get_settings
from app.dependencies.auth import Principal, require_token
from app.schemas.common import ApiModel
from integrations.registry import get_weather
from integrations.weather.advice import advise, notable_changes
from integrations.weather.models import DailyForecast, WeatherReport

router = APIRouter(prefix="/api/v1/weather", tags=["weather"])

SettingsDep = Annotated[Settings, Depends(get_settings)]
PrincipalDep = Annotated[Principal, Depends(require_token)]


class AdviceResponse(ApiModel):
    warmth_band: str
    needs_umbrella: bool
    needs_winter_layers: bool
    needs_sun_protection: bool
    windy: bool
    reasons: list[str]
    summary: str


class WeatherResponse(ApiModel):
    report: WeatherReport
    advice: AdviceResponse
    notable: list[str]


class ForecastResponse(ApiModel):
    location: str
    days: list[DailyForecast]


@router.get(
    "/current",
    response_model=WeatherResponse,
    summary="Current conditions with derived guidance",
    description=(
        "Returns the forecast plus the interpretation the briefing and wardrobe use, "
        "so a caller does not have to re-derive 'cold enough for a coat' from raw numbers."
    ),
)
async def current(
    _: PrincipalDep,
    settings: SettingsDep,
    location: Annotated[str | None, Query(max_length=100)] = None,
    latitude: Annotated[float | None, Query(ge=-90, le=90)] = None,
    longitude: Annotated[float | None, Query(ge=-180, le=180)] = None,
) -> WeatherResponse:
    report = await get_weather(settings).get_weather(
        latitude=latitude, longitude=longitude, location=location, days=3
    )
    return _respond(report)


@router.get("/forecast", response_model=ForecastResponse, summary="Daily forecast")
async def forecast(
    _: PrincipalDep,
    settings: SettingsDep,
    days: Annotated[int, Query(ge=1, le=16)] = 5,
    location: Annotated[str | None, Query(max_length=100)] = None,
    latitude: Annotated[float | None, Query(ge=-90, le=90)] = None,
    longitude: Annotated[float | None, Query(ge=-180, le=180)] = None,
) -> ForecastResponse:
    report = await get_weather(settings).get_weather(
        latitude=latitude, longitude=longitude, location=location, days=days
    )
    return ForecastResponse(location=report.location, days=report.daily)


@router.get(
    "/advice",
    response_model=AdviceResponse,
    summary="What the forecast implies for one day",
    description="`day_offset` 0 is today, 1 is tomorrow.",
)
async def advice(
    _: PrincipalDep,
    settings: SettingsDep,
    day_offset: Annotated[int, Query(ge=0, le=6)] = 0,
    location: Annotated[str | None, Query(max_length=100)] = None,
) -> AdviceResponse:
    report = await get_weather(settings).get_weather(location=location, days=day_offset + 1)
    return _advice(report, day_offset)


def _advice(report: WeatherReport, day_index: int) -> AdviceResponse:
    result = advise(report, day_index=day_index)
    return AdviceResponse(
        warmth_band=result.warmth_band,
        needs_umbrella=result.needs_umbrella,
        needs_winter_layers=result.needs_winter_layers,
        needs_sun_protection=result.needs_sun_protection,
        windy=result.windy,
        reasons=result.reasons,
        summary=result.summary(),
    )


def _respond(report: WeatherReport) -> WeatherResponse:
    return WeatherResponse(
        report=report, advice=_advice(report, 0), notable=notable_changes(report)
    )
