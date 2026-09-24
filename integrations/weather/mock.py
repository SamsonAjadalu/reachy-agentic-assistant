"""Deterministic weather.

Fixed values rather than random ones: the wardrobe recommender and the briefing
are tested against specific conditions, and "cold with snow tomorrow" has to
mean the same thing on every run.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from integrations.weather.models import (
    Condition,
    CurrentWeather,
    DailyForecast,
    HourlyForecast,
    WeatherReport,
)
from shared.errors import ValidationError

ANCHOR = datetime(2026, 3, 10, 9, 0, tzinfo=UTC)

# A week that exercises every branch a consumer might have: mild, wet, cold,
# freezing and hot.
DAILY_PATTERN: list[tuple[float, float, Condition, str, float, int]] = [
    (8.0, 1.0, Condition.PARTLY_CLOUDY, "partly cloudy", 0.0, 10),
    (11.0, 4.0, Condition.RAIN, "rain", 12.4, 85),
    (2.0, -6.0, Condition.SNOW, "snow", 8.1, 90),
    (-4.0, -12.0, Condition.CLEAR, "clear sky", 0.0, 0),
    (14.0, 6.0, Condition.CLOUDY, "overcast", 0.4, 20),
    (19.0, 9.0, Condition.CLEAR, "clear sky", 0.0, 5),
    (27.0, 17.0, Condition.THUNDERSTORM, "thunderstorm", 16.0, 70),
]


class MockWeatherProvider:
    name = "mock"

    def __init__(self, settings: Any | None = None) -> None:
        self._settings = settings
        self.calls = 0

    def clear_cache(self) -> None:
        return None

    async def geocode(self, query: str) -> tuple[float, float, str]:
        if not query.strip():
            raise ValidationError("A place name is required.")
        return 43.6532, -79.3832, "Toronto, CA"

    async def get_weather(
        self,
        *,
        latitude: float | None = None,
        longitude: float | None = None,
        location: str | None = None,
        days: int = 5,
        include_hourly: bool = False,
    ) -> WeatherReport:
        self.calls += 1
        days = max(1, min(days, len(DAILY_PATTERN)))
        label = location or "Toronto, CA"

        current = CurrentWeather(
            location=label,
            latitude=latitude if latitude is not None else 43.6532,
            longitude=longitude if longitude is not None else -79.3832,
            observed_at=ANCHOR,
            temperature_c=6.0,
            feels_like_c=3.0,
            condition=Condition.PARTLY_CLOUDY,
            description="partly cloudy",
            humidity_percent=62,
            wind_kph=14.0,
            precipitation_mm=0.0,
            is_day=True,
        )

        daily = [
            DailyForecast(
                day=(ANCHOR + timedelta(days=index)).date(),
                high_c=high,
                low_c=low,
                condition=condition,
                description=description,
                precipitation_mm=precipitation,
                precipitation_probability=probability,
                wind_kph=12.0 + index,
                sunrise=(ANCHOR + timedelta(days=index)).replace(hour=11, minute=15),
                sunset=(ANCHOR + timedelta(days=index)).replace(hour=23, minute=20),
            )
            for index, (high, low, condition, description, precipitation, probability) in enumerate(
                DAILY_PATTERN[:days]
            )
        ]

        hourly: list[HourlyForecast] = []
        if include_hourly:
            hourly = [
                HourlyForecast(
                    at=ANCHOR + timedelta(hours=hour),
                    temperature_c=6.0 + (hour % 12) * 0.5,
                    condition=Condition.PARTLY_CLOUDY if hour % 6 else Condition.RAIN,
                    precipitation_mm=0.0 if hour % 6 else 1.2,
                    precipitation_probability=10 if hour % 6 else 80,
                )
                for hour in range(24)
            ]

        return WeatherReport(
            location=label,
            current=current,
            daily=daily,
            hourly=hourly,
            provider="mock",
            retrieved_at=ANCHOR,
        )

    async def health(self) -> dict[str, Any]:
        return {"ok": True, "mock": True, "provider": self.name}
