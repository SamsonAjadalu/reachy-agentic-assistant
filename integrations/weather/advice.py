"""Deterministic interpretation of a forecast.

The rules here are plain thresholds rather than anything learned, for two
reasons: the wardrobe recommender needs to explain itself ("it will be below
freezing and snowing"), and a rule that can be read is a rule the owner can
disagree with and change.

Temperatures are Celsius throughout.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from integrations.weather.models import Condition, DailyForecast, WeatherReport

FREEZING_C = 0.0
COLD_C = 8.0
MILD_C = 18.0
WARM_C = 25.0
HOT_C = 30.0

WET_CONDITIONS = frozenset(
    {Condition.RAIN, Condition.DRIZZLE, Condition.FREEZING_RAIN, Condition.THUNDERSTORM}
)
SNOW_CONDITIONS = frozenset({Condition.SNOW, Condition.FREEZING_RAIN})

SIGNIFICANT_RAIN_MM = 2.0
HIGH_PROBABILITY = 60
STRONG_WIND_KPH = 35.0


@dataclass
class WeatherAdvice:
    """What the forecast implies, with the reasons attached."""

    warmth_band: str
    needs_umbrella: bool
    needs_winter_layers: bool
    needs_sun_protection: bool
    windy: bool
    reasons: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return "; ".join(self.reasons) if self.reasons else "Nothing notable in the forecast."


def warmth_band(temperature_c: float) -> str:
    if temperature_c < FREEZING_C:
        return "freezing"
    if temperature_c < COLD_C:
        return "cold"
    if temperature_c < MILD_C:
        return "cool"
    if temperature_c < WARM_C:
        return "mild"
    if temperature_c < HOT_C:
        return "warm"
    return "hot"


def advise(report: WeatherReport, *, day_index: int = 0) -> WeatherAdvice:
    """Interpret one day of the forecast, falling back to current conditions."""
    day: DailyForecast | None = report.daily[day_index] if day_index < len(report.daily) else None

    if day is not None:
        # The low matters as much as the high: a 14 degree afternoon after a
        # 2 degree morning still calls for a coat on the way out.
        representative = (day.high_c + day.low_c) / 2
        condition = day.condition
        precipitation = day.precipitation_mm
        probability = day.precipitation_probability or 0
        wind = day.wind_kph or 0.0
        low = day.low_c
    else:
        representative = report.current.feels_like_c
        condition = report.current.condition
        precipitation = report.current.precipitation_mm
        probability = 0
        wind = report.current.wind_kph or 0.0
        low = report.current.feels_like_c

    reasons: list[str] = []
    band = warmth_band(representative)

    wet = (
        condition in WET_CONDITIONS
        or precipitation >= SIGNIFICANT_RAIN_MM
        or probability >= HIGH_PROBABILITY
    )
    if wet:
        reasons.append(
            f"{condition.value.replace('_', ' ')} expected"
            + (f" ({probability}% chance)" if probability else "")
        )

    winter = low < FREEZING_C or condition in SNOW_CONDITIONS
    if winter:
        reasons.append(f"low of {low:.0f}C" if low < FREEZING_C else "snow expected")

    sunny = condition is Condition.CLEAR and representative >= WARM_C
    if sunny:
        reasons.append("clear and warm")

    windy = wind >= STRONG_WIND_KPH
    if windy:
        reasons.append(f"wind around {wind:.0f} km/h")

    if not reasons:
        reasons.append(f"{band} and unremarkable")

    return WeatherAdvice(
        warmth_band=band,
        needs_umbrella=wet,
        needs_winter_layers=winter,
        needs_sun_protection=sunny,
        windy=windy,
        reasons=reasons,
    )


def notable_changes(report: WeatherReport) -> list[str]:
    """Things worth mentioning unprompted in a briefing."""
    notes: list[str] = []
    if not report.daily:
        return notes

    today = report.daily[0]
    if today.precipitation_probability and today.precipitation_probability >= HIGH_PROBABILITY:
        notes.append(
            f"{today.precipitation_probability}% chance of "
            f"{today.condition.value.replace('_', ' ')} today"
        )
    if today.low_c < FREEZING_C:
        notes.append(f"below freezing today, low of {today.low_c:.0f}C")

    if len(report.daily) > 1:
        tomorrow = report.daily[1]
        swing = tomorrow.high_c - today.high_c
        if abs(swing) >= 8:
            direction = "warmer" if swing > 0 else "colder"
            notes.append(f"{abs(swing):.0f} degrees {direction} tomorrow")
        if tomorrow.condition in SNOW_CONDITIONS and today.condition not in SNOW_CONDITIONS:
            notes.append("snow starting tomorrow")

    return notes
