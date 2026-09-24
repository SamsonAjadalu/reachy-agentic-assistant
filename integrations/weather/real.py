"""Open-Meteo weather provider.

Chosen because it needs no API key: one fewer credential to store, rotate and
leak, for a service whose data is not sensitive. The provider interface stays
generic so swapping in a keyed service later touches only this directory.

Responses are cached in memory for ``WEATHER_CACHE_TTL_SECONDS``. Weather does
not change between two questions asked a minute apart, and the briefing, the
wardrobe recommender and an alert check all want the same forecast.
"""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, date, datetime
from typing import Any

import httpx

from app.config import Settings, get_settings
from app.logging_config import get_logger
from integrations.weather.models import (
    Condition,
    CurrentWeather,
    DailyForecast,
    HourlyForecast,
    WeatherReport,
)
from shared.errors import IntegrationError, ProviderTimeoutError, ValidationError
from shared.timeutils import utcnow

logger = get_logger(__name__)

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"

MAX_ATTEMPTS = 3
MAX_FORECAST_DAYS = 16

# WMO weather interpretation codes, collapsed to the vocabulary the assistant
# speaks. https://open-meteo.com/en/docs documents the full table.
WMO_CONDITIONS: dict[int, tuple[Condition, str]] = {
    0: (Condition.CLEAR, "clear sky"),
    1: (Condition.PARTLY_CLOUDY, "mainly clear"),
    2: (Condition.PARTLY_CLOUDY, "partly cloudy"),
    3: (Condition.CLOUDY, "overcast"),
    45: (Condition.FOG, "fog"),
    48: (Condition.FOG, "freezing fog"),
    51: (Condition.DRIZZLE, "light drizzle"),
    53: (Condition.DRIZZLE, "drizzle"),
    55: (Condition.DRIZZLE, "heavy drizzle"),
    56: (Condition.FREEZING_RAIN, "light freezing drizzle"),
    57: (Condition.FREEZING_RAIN, "freezing drizzle"),
    61: (Condition.RAIN, "light rain"),
    63: (Condition.RAIN, "rain"),
    65: (Condition.RAIN, "heavy rain"),
    66: (Condition.FREEZING_RAIN, "light freezing rain"),
    67: (Condition.FREEZING_RAIN, "freezing rain"),
    71: (Condition.SNOW, "light snow"),
    73: (Condition.SNOW, "snow"),
    75: (Condition.SNOW, "heavy snow"),
    77: (Condition.SNOW, "snow grains"),
    80: (Condition.RAIN, "light showers"),
    81: (Condition.RAIN, "showers"),
    82: (Condition.RAIN, "violent showers"),
    85: (Condition.SNOW, "snow showers"),
    86: (Condition.SNOW, "heavy snow showers"),
    95: (Condition.THUNDERSTORM, "thunderstorm"),
    96: (Condition.THUNDERSTORM, "thunderstorm with hail"),
    99: (Condition.THUNDERSTORM, "thunderstorm with heavy hail"),
}


def interpret(code: int | None) -> tuple[Condition, str]:
    if code is None:
        return Condition.UNKNOWN, "unknown conditions"
    return WMO_CONDITIONS.get(int(code), (Condition.UNKNOWN, f"weather code {code}"))


class OpenMeteoProvider:
    name = "open-meteo"

    def __init__(self, settings: Settings | None = None, client: httpx.AsyncClient | None = None):
        self._settings = settings or get_settings()
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=httpx.Timeout(15.0, connect=5.0))
        self._cache: dict[str, tuple[float, WeatherReport]] = {}

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> OpenMeteoProvider:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    def clear_cache(self) -> None:
        self._cache.clear()

    async def get_weather(
        self,
        *,
        latitude: float | None = None,
        longitude: float | None = None,
        location: str | None = None,
        days: int = 5,
        include_hourly: bool = False,
    ) -> WeatherReport:
        latitude, longitude, label = self._resolve_location(latitude, longitude, location)
        days = max(1, min(days, MAX_FORECAST_DAYS))

        cache_key = f"{latitude:.3f},{longitude:.3f}:{days}:{include_hourly}"
        cached = self._read_cache(cache_key)
        if cached is not None:
            return cached.model_copy(update={"from_cache": True})

        params: dict[str, Any] = {
            "latitude": round(latitude, 4),
            "longitude": round(longitude, 4),
            "current": (
                "temperature_2m,apparent_temperature,relative_humidity_2m,"
                "precipitation,weather_code,wind_speed_10m,is_day"
            ),
            "daily": (
                "weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum,"
                "precipitation_probability_max,wind_speed_10m_max,sunrise,sunset"
            ),
            "timezone": "UTC",
            "forecast_days": days,
            "wind_speed_unit": "kmh",
        }
        if include_hourly:
            params["hourly"] = "temperature_2m,weather_code,precipitation,precipitation_probability"

        payload = await self._get(FORECAST_URL, params)
        report = _to_report(payload, label, days, include_hourly)
        self._write_cache(cache_key, report)
        return report

    async def geocode(self, query: str) -> tuple[float, float, str]:
        """Turn a place name into coordinates."""
        payload = await self._get(GEOCODE_URL, {"name": query[:100], "count": 1})
        results = payload.get("results") or []
        if not results:
            raise ValidationError(f"No place matched {query!r}.")
        first = results[0]
        label = ", ".join(part for part in [first.get("name"), first.get("country_code")] if part)
        return float(first["latitude"]), float(first["longitude"]), label

    def _resolve_location(
        self, latitude: float | None, longitude: float | None, location: str | None
    ) -> tuple[float, float, str]:
        if latitude is not None and longitude is not None:
            if not (-90 <= latitude <= 90) or not (-180 <= longitude <= 180):
                raise ValidationError("Latitude or longitude is out of range.")
            return latitude, longitude, location or f"{latitude:.2f},{longitude:.2f}"
        return (
            self._settings.weather_default_latitude,
            self._settings.weather_default_longitude,
            location or self._settings.weather_default_location,
        )

    async def _get(self, url: str, params: dict[str, Any]) -> dict[str, Any]:
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                response = await self._client.get(url, params=params)
            except httpx.TimeoutException as exc:
                if attempt == MAX_ATTEMPTS:
                    raise ProviderTimeoutError(
                        "The weather service did not respond in time.",
                        details={"integration": "weather"},
                    ) from exc
            except httpx.HTTPError as exc:
                raise IntegrationError(
                    f"Could not reach the weather service ({type(exc).__name__}).",
                    integration="weather",
                ) from exc
            else:
                if response.status_code == httpx.codes.OK:
                    body: dict[str, Any] = response.json()
                    return body
                if response.status_code < httpx.codes.INTERNAL_SERVER_ERROR:
                    raise IntegrationError(
                        f"The weather service rejected the request (HTTP {response.status_code}).",
                        integration="weather",
                    )
            await asyncio.sleep(min(2**attempt, 8))

        raise IntegrationError("The weather service failed after retries.", integration="weather")

    def _read_cache(self, key: str) -> WeatherReport | None:
        entry = self._cache.get(key)
        if entry is None:
            return None
        expires_at, report = entry
        if time.monotonic() > expires_at:
            del self._cache[key]
            return None
        return report

    def _write_cache(self, key: str, report: WeatherReport) -> None:
        ttl = self._settings.weather_cache_ttl_seconds
        if ttl > 0:
            self._cache[key] = (time.monotonic() + ttl, report)

    async def health(self) -> dict[str, Any]:
        try:
            await self.get_weather(days=1)
        except (IntegrationError, ValidationError) as exc:
            return {"ok": False, "detail": str(exc)}
        return {"ok": True, "provider": self.name, "cached_entries": len(self._cache)}


def _to_report(
    payload: dict[str, Any], label: str, days: int, include_hourly: bool
) -> WeatherReport:
    current = payload.get("current") or {}
    condition, description = interpret(current.get("weather_code"))

    report_current = CurrentWeather(
        location=label,
        latitude=float(payload.get("latitude", 0.0)),
        longitude=float(payload.get("longitude", 0.0)),
        observed_at=_parse(current.get("time")) or utcnow(),
        temperature_c=float(current.get("temperature_2m", 0.0)),
        feels_like_c=float(current.get("apparent_temperature", current.get("temperature_2m", 0.0))),
        condition=condition,
        description=description,
        humidity_percent=_optional_int(current.get("relative_humidity_2m")),
        wind_kph=_optional_float(current.get("wind_speed_10m")),
        precipitation_mm=float(current.get("precipitation", 0.0) or 0.0),
        is_day=bool(current.get("is_day", 1)),
    )

    daily_raw = payload.get("daily") or {}
    daily: list[DailyForecast] = []
    for index, day_value in enumerate(daily_raw.get("time") or []):
        day_condition, day_description = interpret(_at(daily_raw, "weather_code", index))
        daily.append(
            DailyForecast(
                day=date.fromisoformat(day_value),
                high_c=float(_at(daily_raw, "temperature_2m_max", index) or 0.0),
                low_c=float(_at(daily_raw, "temperature_2m_min", index) or 0.0),
                condition=day_condition,
                description=day_description,
                precipitation_mm=float(_at(daily_raw, "precipitation_sum", index) or 0.0),
                precipitation_probability=_optional_int(
                    _at(daily_raw, "precipitation_probability_max", index)
                ),
                wind_kph=_optional_float(_at(daily_raw, "wind_speed_10m_max", index)),
                sunrise=_parse(_at(daily_raw, "sunrise", index)),
                sunset=_parse(_at(daily_raw, "sunset", index)),
            )
        )

    hourly: list[HourlyForecast] = []
    if include_hourly:
        hourly_raw = payload.get("hourly") or {}
        for index, at_value in enumerate(hourly_raw.get("time") or []):
            hour_condition, _ = interpret(_at(hourly_raw, "weather_code", index))
            parsed = _parse(at_value)
            if parsed is None:
                continue
            hourly.append(
                HourlyForecast(
                    at=parsed,
                    temperature_c=float(_at(hourly_raw, "temperature_2m", index) or 0.0),
                    condition=hour_condition,
                    precipitation_mm=float(_at(hourly_raw, "precipitation", index) or 0.0),
                    precipitation_probability=_optional_int(
                        _at(hourly_raw, "precipitation_probability", index)
                    ),
                )
            )

    return WeatherReport(
        location=label,
        current=report_current,
        daily=daily[:days],
        hourly=hourly,
        retrieved_at=utcnow(),
    )


def _at(block: dict[str, Any], key: str, index: int) -> Any:
    values = block.get(key) or []
    return values[index] if index < len(values) else None


def _parse(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    # Open-Meteo returns naive local times when a timezone is requested; UTC was
    # requested, so a naive value is UTC.
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _optional_int(value: Any) -> int | None:
    return int(value) if isinstance(value, int | float) else None


def _optional_float(value: Any) -> float | None:
    return float(value) if isinstance(value, int | float) else None
