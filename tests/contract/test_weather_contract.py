"""Open-Meteo parsing, caching and the derived advice.

The advice rules get the most attention here because the wardrobe recommender
and the briefing both depend on them, and a silent change to a threshold would
show up as odd clothing suggestions rather than as a failure.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import httpx
import pytest
import respx

from app.config import Settings
from integrations.weather.advice import advise, notable_changes, warmth_band
from integrations.weather.mock import MockWeatherProvider
from integrations.weather.models import Condition, WeatherReport
from integrations.weather.real import FORECAST_URL, GEOCODE_URL, OpenMeteoProvider, interpret
from shared.errors import IntegrationError, ValidationError

FORECAST_PAYLOAD: dict[str, Any] = {
    "latitude": 43.65,
    "longitude": -79.38,
    "current": {
        "time": "2026-03-10T14:00",
        "temperature_2m": 6.4,
        "apparent_temperature": 2.1,
        "relative_humidity_2m": 71,
        "precipitation": 0.0,
        "weather_code": 3,
        "wind_speed_10m": 18.5,
        "is_day": 1,
    },
    "daily": {
        "time": ["2026-03-10", "2026-03-11"],
        "weather_code": [3, 71],
        "temperature_2m_max": [8.2, 1.0],
        "temperature_2m_min": [1.4, -7.5],
        "precipitation_sum": [0.0, 9.3],
        "precipitation_probability_max": [10, 90],
        "wind_speed_10m_max": [22.0, 41.0],
        "sunrise": ["2026-03-10T11:38", "2026-03-11T11:36"],
        "sunset": ["2026-03-10T23:16", "2026-03-11T23:17"],
    },
}


@pytest.fixture
def provider(settings: Settings) -> OpenMeteoProvider:
    settings.weather_cache_ttl_seconds = 900
    return OpenMeteoProvider(settings)


class TestParsing:
    @respx.mock
    async def test_current_conditions_are_normalised(self, provider: OpenMeteoProvider) -> None:
        respx.get(FORECAST_URL).mock(return_value=httpx.Response(200, json=FORECAST_PAYLOAD))

        report = await provider.get_weather(days=2)

        assert report.current.temperature_c == 6.4
        assert report.current.feels_like_c == 2.1
        assert report.current.condition is Condition.CLOUDY
        assert report.current.observed_at == datetime(2026, 3, 10, 14, 0, tzinfo=UTC)

    @respx.mock
    async def test_the_daily_forecast_is_parsed(self, provider: OpenMeteoProvider) -> None:
        respx.get(FORECAST_URL).mock(return_value=httpx.Response(200, json=FORECAST_PAYLOAD))

        report = await provider.get_weather(days=2)

        assert len(report.daily) == 2
        assert report.daily[1].condition is Condition.SNOW
        assert report.daily[1].low_c == -7.5
        assert report.daily[1].precipitation_probability == 90

    @respx.mock
    async def test_utc_is_requested_explicitly(self, provider: OpenMeteoProvider) -> None:
        """Local times without an offset would be ambiguous across a DST change."""
        route = respx.get(FORECAST_URL).mock(
            return_value=httpx.Response(200, json=FORECAST_PAYLOAD)
        )
        await provider.get_weather(days=2)
        assert route.calls.last.request.url.params["timezone"] == "UTC"

    @respx.mock
    async def test_hourly_data_is_optional(self, provider: OpenMeteoProvider) -> None:
        route = respx.get(FORECAST_URL).mock(
            return_value=httpx.Response(200, json=FORECAST_PAYLOAD)
        )
        await provider.get_weather(days=2)
        assert "hourly" not in route.calls.last.request.url.params

    @pytest.mark.parametrize(
        ("code", "expected"),
        [
            (0, Condition.CLEAR),
            (3, Condition.CLOUDY),
            (63, Condition.RAIN),
            (75, Condition.SNOW),
            (95, Condition.THUNDERSTORM),
            (None, Condition.UNKNOWN),
            (12345, Condition.UNKNOWN),
        ],
    )
    def test_wmo_codes_map_to_the_shared_vocabulary(
        self, code: int | None, expected: Condition
    ) -> None:
        assert interpret(code)[0] is expected


class TestCaching:
    @respx.mock
    async def test_a_repeated_question_does_not_refetch(self, provider: OpenMeteoProvider) -> None:
        route = respx.get(FORECAST_URL).mock(
            return_value=httpx.Response(200, json=FORECAST_PAYLOAD)
        )

        first = await provider.get_weather(days=2)
        second = await provider.get_weather(days=2)

        assert route.call_count == 1
        assert first.from_cache is False
        assert second.from_cache is True

    @respx.mock
    async def test_a_different_location_is_a_different_entry(
        self, provider: OpenMeteoProvider
    ) -> None:
        route = respx.get(FORECAST_URL).mock(
            return_value=httpx.Response(200, json=FORECAST_PAYLOAD)
        )

        await provider.get_weather(latitude=43.65, longitude=-79.38, days=2)
        await provider.get_weather(latitude=45.42, longitude=-75.69, days=2)

        assert route.call_count == 2

    @respx.mock
    async def test_caching_can_be_disabled(self, settings: Settings) -> None:
        settings.weather_cache_ttl_seconds = 0
        provider = OpenMeteoProvider(settings)
        route = respx.get(FORECAST_URL).mock(
            return_value=httpx.Response(200, json=FORECAST_PAYLOAD)
        )

        await provider.get_weather(days=2)
        await provider.get_weather(days=2)

        assert route.call_count == 2


class TestFailures:
    @pytest.fixture(autouse=True)
    def _no_backoff(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def instant(_seconds: float) -> None:
            return None

        monkeypatch.setattr("integrations.weather.real.asyncio.sleep", instant)

    @respx.mock
    async def test_a_client_error_is_not_retried(self, provider: OpenMeteoProvider) -> None:
        route = respx.get(FORECAST_URL).mock(return_value=httpx.Response(400))
        with pytest.raises(IntegrationError):
            await provider.get_weather()
        assert route.call_count == 1

    @respx.mock
    async def test_a_server_error_is_retried(self, provider: OpenMeteoProvider) -> None:
        route = respx.get(FORECAST_URL).mock(
            side_effect=[
                httpx.Response(503),
                httpx.Response(200, json=FORECAST_PAYLOAD),
            ]
        )
        report = await provider.get_weather(days=2)
        assert route.call_count == 2
        assert report.current.temperature_c == 6.4

    async def test_impossible_coordinates_are_refused(self, provider: OpenMeteoProvider) -> None:
        with pytest.raises(ValidationError):
            await provider.get_weather(latitude=200.0, longitude=0.0)

    @respx.mock
    async def test_an_unknown_place_is_reported(self, provider: OpenMeteoProvider) -> None:
        respx.get(GEOCODE_URL).mock(return_value=httpx.Response(200, json={"results": []}))
        with pytest.raises(ValidationError, match="No place matched"):
            await provider.geocode("Atlantis")


class TestAdvice:
    async def _report(self) -> WeatherReport:
        return await MockWeatherProvider().get_weather(days=7)

    @pytest.mark.parametrize(
        ("temperature", "band"),
        [
            (-5.0, "freezing"),
            (3.0, "cold"),
            (12.0, "cool"),
            (20.0, "mild"),
            (27.0, "warm"),
            (34.0, "hot"),
        ],
    )
    def test_the_warmth_bands_are_thresholds_not_guesses(
        self, temperature: float, band: str
    ) -> None:
        assert warmth_band(temperature) == band

    async def test_rain_calls_for_an_umbrella(self) -> None:
        result = advise(await self._report(), day_index=1)
        assert result.needs_umbrella is True
        assert any("rain" in reason for reason in result.reasons)

    async def test_snow_calls_for_winter_layers(self) -> None:
        result = advise(await self._report(), day_index=2)
        assert result.needs_winter_layers is True

    async def test_a_freezing_low_calls_for_winter_layers_without_snow(self) -> None:
        result = advise(await self._report(), day_index=3)
        assert result.needs_winter_layers is True
        assert result.needs_umbrella is False

    async def test_a_clear_warm_day_calls_for_sun_protection(self) -> None:
        result = advise(await self._report(), day_index=5)
        assert result.needs_sun_protection is False  # 19/9 averages to mild
        result = advise(await self._report(), day_index=6)
        assert result.needs_umbrella is True  # thunderstorm

    async def test_advice_always_explains_itself(self) -> None:
        report = await self._report()
        for index in range(len(report.daily)):
            assert advise(report, day_index=index).reasons

    async def test_a_large_swing_is_worth_mentioning(self) -> None:
        """Day two of the mock week is nine degrees colder than day one."""
        report = await self._report()
        shifted = report.model_copy(update={"daily": report.daily[1:]})

        notes = notable_changes(shifted)

        assert any("colder tomorrow" in note for note in notes)
        assert any("snow starting tomorrow" in note for note in notes)

    async def test_an_unremarkable_day_produces_no_notes(self) -> None:
        """A briefing should stay quiet when there is nothing to say."""
        report = await self._report()
        assert notable_changes(report.model_copy(update={"daily": report.daily[:1]})) == []

    async def test_advice_falls_back_to_current_conditions(self) -> None:
        report = await self._report()
        result = advise(report, day_index=99)
        assert result.warmth_band == warmth_band(report.current.feels_like_c)
