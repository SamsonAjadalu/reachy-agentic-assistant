"""Weather endpoints against the deterministic provider."""

from __future__ import annotations

import pytest
from httpx import AsyncClient

from integrations.registry import reset_providers, set_provider_override
from integrations.weather.mock import MockWeatherProvider


@pytest.fixture(autouse=True)
def weather():
    provider = MockWeatherProvider()
    set_provider_override("weather", provider)
    yield provider
    reset_providers()


class TestCurrent:
    async def test_current_conditions_include_advice(self, client: AsyncClient) -> None:
        body = (await client.get("/api/v1/weather/current")).json()
        assert body["report"]["current"]["temperature_c"] == 6.0
        assert body["advice"]["warmth_band"] in {"cold", "cool"}
        assert body["advice"]["summary"]

    async def test_a_named_location_is_echoed(self, client: AsyncClient) -> None:
        body = (await client.get("/api/v1/weather/current?location=Ottawa,CA")).json()
        assert body["report"]["location"] == "Ottawa,CA"

    async def test_impossible_coordinates_are_refused(self, client: AsyncClient) -> None:
        assert (await client.get("/api/v1/weather/current?latitude=120")).status_code == 422


class TestForecast:
    async def test_the_requested_number_of_days_comes_back(self, client: AsyncClient) -> None:
        body = (await client.get("/api/v1/weather/forecast?days=4")).json()
        assert len(body["days"]) == 4

    async def test_days_are_in_order(self, client: AsyncClient) -> None:
        days = (await client.get("/api/v1/weather/forecast?days=5")).json()["days"]
        assert [day["day"] for day in days] == sorted(day["day"] for day in days)

    async def test_an_absurd_range_is_refused(self, client: AsyncClient) -> None:
        assert (await client.get("/api/v1/weather/forecast?days=400")).status_code == 422


class TestAdviceEndpoint:
    async def test_tomorrows_rain_is_reported(self, client: AsyncClient) -> None:
        body = (await client.get("/api/v1/weather/advice?day_offset=1")).json()
        assert body["needs_umbrella"] is True

    async def test_the_snow_day_needs_layers(self, client: AsyncClient) -> None:
        body = (await client.get("/api/v1/weather/advice?day_offset=2")).json()
        assert body["needs_winter_layers"] is True

    async def test_advice_always_carries_reasons(self, client: AsyncClient) -> None:
        for offset in range(5):
            body = (await client.get(f"/api/v1/weather/advice?day_offset={offset}")).json()
            assert body["reasons"]


class TestAuthentication:
    async def test_weather_needs_a_token(self, anonymous_client: AsyncClient) -> None:
        assert (await anonymous_client.get("/api/v1/weather/current")).status_code == 401
