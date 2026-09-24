"""Normalised weather shapes.

Weather feeds a lot of downstream logic - the morning briefing, outfit
recommendations, alerts - so the fields here are chosen for what those consumers
actually need, not for what any one provider happens to return.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class Condition(StrEnum):
    """A small vocabulary the rest of the system can branch on.

    Provider condition codes are far more granular than anything the assistant
    can usefully say out loud, and mapping them down here keeps that granularity
    from leaking into the wardrobe rules.
    """

    CLEAR = "clear"
    PARTLY_CLOUDY = "partly_cloudy"
    CLOUDY = "cloudy"
    FOG = "fog"
    DRIZZLE = "drizzle"
    RAIN = "rain"
    FREEZING_RAIN = "freezing_rain"
    SNOW = "snow"
    THUNDERSTORM = "thunderstorm"
    UNKNOWN = "unknown"


class CurrentWeather(BaseModel):
    location: str
    latitude: float
    longitude: float
    observed_at: datetime
    temperature_c: float
    feels_like_c: float
    condition: Condition
    description: str
    humidity_percent: int | None = None
    wind_kph: float | None = None
    precipitation_mm: float = 0.0
    is_day: bool = True


class DailyForecast(BaseModel):
    day: date
    high_c: float
    low_c: float
    condition: Condition
    description: str
    precipitation_mm: float = 0.0
    precipitation_probability: int | None = None
    wind_kph: float | None = None
    sunrise: datetime | None = None
    sunset: datetime | None = None


class HourlyForecast(BaseModel):
    at: datetime
    temperature_c: float
    condition: Condition
    precipitation_mm: float = 0.0
    precipitation_probability: int | None = None


class WeatherReport(BaseModel):
    location: str
    current: CurrentWeather
    daily: list[DailyForecast] = Field(default_factory=list)
    hourly: list[HourlyForecast] = Field(default_factory=list)
    provider: str = "open-meteo"
    retrieved_at: datetime
    from_cache: bool = False
