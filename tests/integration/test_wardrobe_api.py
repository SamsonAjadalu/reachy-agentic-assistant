"""Wardrobe endpoints, exercised as a wardrobe would actually be used."""

from __future__ import annotations

import io
from datetime import date, timedelta
from typing import Any

import pytest
from httpx import AsyncClient
from PIL import Image

from app.config import Settings
from integrations.registry import reset_providers, set_provider_override
from integrations.weather.mock import MockWeatherProvider


@pytest.fixture(autouse=True)
def weather():
    set_provider_override("weather", MockWeatherProvider())
    yield
    reset_providers()


@pytest.fixture(autouse=True)
def image_root(settings: Settings, tmp_path):
    settings.wardrobe_image_root = tmp_path / "wardrobe"
    return settings.wardrobe_image_root


def photo(colour: tuple[int, int, int] = (20, 40, 130)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (400, 400), colour).save(buffer, format="JPEG")
    return buffer.getvalue()


async def add_item(client: AsyncClient, **overrides: Any) -> dict[str, Any]:
    payload = {
        "name": "Navy jumper",
        "category": "top",
        "primary_color": "navy",
        "warmth": 4,
        "formality": 3,
    } | overrides
    response = await client.post("/api/v1/wardrobe/items", json=payload)
    assert response.status_code == 201, response.text
    return response.json()


async def add_outfit(client: AsyncClient, item_ids: list[str], **overrides: Any) -> dict[str, Any]:
    payload = {
        "name": "Everyday",
        "items": [{"item_id": item_id} for item_id in item_ids],
    } | overrides
    response = await client.post("/api/v1/wardrobe/outfits", json=payload)
    assert response.status_code == 201, response.text
    return response.json()


class TestItems:
    async def test_a_garment_can_be_added_and_read_back(self, client: AsyncClient) -> None:
        created = await add_item(client)
        fetched = (await client.get(f"/api/v1/wardrobe/items/{created['id']}")).json()
        assert fetched["name"] == "Navy jumper"
        assert fetched["laundry_status"] == "clean"

    async def test_categories_and_colours_are_normalised(self, client: AsyncClient) -> None:
        """Spoken input arrives capitalised as often as not."""
        created = await add_item(client, category="TOP", primary_color="Navy")
        assert created["category"] == "top"
        assert created["primary_color"] == "navy"

    async def test_items_can_be_filtered(self, client: AsyncClient) -> None:
        await add_item(client, name="Jeans", category="bottom", primary_color="blue")
        await add_item(client, name="Tee", category="top", primary_color="white")

        body = (await client.get("/api/v1/wardrobe/items?category=top")).json()

        assert body["total"] == 1
        assert body["items"][0]["name"] == "Tee"

    async def test_an_invalid_formality_is_refused(self, client: AsyncClient) -> None:
        response = await client.post(
            "/api/v1/wardrobe/items",
            json={"name": "X", "category": "top", "primary_color": "red", "formality": 9},
        )
        assert response.status_code == 422

    async def test_a_garment_can_be_deleted(self, client: AsyncClient) -> None:
        created = await add_item(client)
        assert (await client.delete(f"/api/v1/wardrobe/items/{created['id']}")).status_code == 200
        assert (await client.get(f"/api/v1/wardrobe/items/{created['id']}")).status_code == 404


class TestImages:
    async def test_a_photo_proposes_a_colour(self, client: AsyncClient) -> None:
        item = await add_item(client)

        response = await client.post(
            f"/api/v1/wardrobe/items/{item['id']}/images",
            files={"file": ("jumper.jpg", photo(), "image/jpeg")},
        )

        assert response.status_code == 201
        body = response.json()
        assert body["proposed_primary_color"] in {"navy", "blue"}
        assert "Correct them if they are wrong" in body["note"]

    async def test_a_non_image_upload_is_refused(self, client: AsyncClient) -> None:
        item = await add_item(client)
        response = await client.post(
            f"/api/v1/wardrobe/items/{item['id']}/images",
            files={"file": ("payload.jpg", b"not an image", "image/jpeg")},
        )
        assert response.status_code == 422

    async def test_a_declared_content_type_is_not_trusted(self, client: AsyncClient) -> None:
        """Uses the configured workflow."""
        item = await add_item(client)
        buffer = io.BytesIO()
        Image.new("RGB", (10, 10), (0, 0, 0)).save(buffer, format="BMP")

        response = await client.post(
            f"/api/v1/wardrobe/items/{item['id']}/images",
            files={"file": ("real.jpg", buffer.getvalue(), "image/jpeg")},
        )

        assert response.status_code == 422

    async def test_the_same_photo_twice_is_refused(self, client: AsyncClient) -> None:
        item = await add_item(client)
        data = photo()
        await client.post(
            f"/api/v1/wardrobe/items/{item['id']}/images",
            files={"file": ("a.jpg", data, "image/jpeg")},
        )
        second = await client.post(
            f"/api/v1/wardrobe/items/{item['id']}/images",
            files={"file": ("b.jpg", data, "image/jpeg")},
        )
        assert second.status_code == 422

    async def test_a_stored_image_can_be_fetched(self, client: AsyncClient) -> None:
        item = await add_item(client)
        uploaded = (
            await client.post(
                f"/api/v1/wardrobe/items/{item['id']}/images",
                files={"file": ("jumper.jpg", photo(), "image/jpeg")},
            )
        ).json()

        response = await client.get(f"/api/v1/wardrobe/images/{uploaded['image']['id']}")

        assert response.status_code == 200
        assert response.headers["content-type"] == "image/jpeg"


class TestOutfits:
    async def test_an_outfit_groups_garments(self, client: AsyncClient) -> None:
        top = await add_item(client, name="Shirt", category="top")
        bottom = await add_item(client, name="Chinos", category="bottom")

        outfit = await add_outfit(client, [top["id"], bottom["id"]])

        assert {item["name"] for item in outfit["items"]} == {"Shirt", "Chinos"}

    async def test_the_same_garment_twice_is_refused(self, client: AsyncClient) -> None:
        item = await add_item(client)
        response = await client.post(
            "/api/v1/wardrobe/outfits",
            json={
                "name": "Doubled",
                "items": [{"item_id": item["id"]}, {"item_id": item["id"]}],
            },
        )
        assert response.status_code == 422

    async def test_an_unknown_garment_is_refused(self, client: AsyncClient) -> None:
        response = await client.post(
            "/api/v1/wardrobe/outfits",
            json={"name": "Ghost", "items": [{"item_id": "does-not-exist"}]},
        )
        assert response.status_code == 422

    async def test_wearing_an_outfit_ages_its_garments(self, client: AsyncClient) -> None:
        item = await add_item(client)
        outfit = await add_outfit(client, [item["id"]])

        await client.post(
            f"/api/v1/wardrobe/outfits/{outfit['id']}/wear",
            json={"occasion": "work", "rating": 4},
        )

        refreshed = (await client.get(f"/api/v1/wardrobe/items/{item['id']}")).json()
        assert refreshed["times_worn"] == 1
        assert refreshed["wears_since_wash"] == 1
        assert refreshed["last_worn_on"] == date.today().isoformat()

    async def test_hitting_the_wear_limit_moves_a_garment_to_worn(
        self, client: AsyncClient
    ) -> None:
        item = await add_item(client, max_wears_before_wash=2)
        outfit = await add_outfit(client, [item["id"]])

        for _ in range(2):
            await client.post(f"/api/v1/wardrobe/outfits/{outfit['id']}/wear", json={})

        refreshed = (await client.get(f"/api/v1/wardrobe/items/{item['id']}")).json()
        assert refreshed["laundry_status"] == "worn"


class TestLaundry:
    async def test_a_worn_out_garment_appears_on_the_laundry_list(
        self, client: AsyncClient
    ) -> None:
        item = await add_item(client, max_wears_before_wash=1)
        outfit = await add_outfit(client, [item["id"]])
        await client.post(f"/api/v1/wardrobe/outfits/{outfit['id']}/wear", json={})

        body = (await client.get("/api/v1/wardrobe/laundry")).json()

        assert body["total"] == 1
        assert "worn 1 times" in body["items"][0]["reason"]

    async def test_washing_resets_the_counter(self, client: AsyncClient) -> None:
        item = await add_item(client, max_wears_before_wash=1)
        outfit = await add_outfit(client, [item["id"]])
        await client.post(f"/api/v1/wardrobe/outfits/{outfit['id']}/wear", json={})

        await client.post("/api/v1/wardrobe/laundry/done", json={"item_ids": [item["id"]]})

        refreshed = (await client.get(f"/api/v1/wardrobe/items/{item['id']}")).json()
        assert refreshed["wears_since_wash"] == 0
        assert refreshed["laundry_status"] == "clean"

    async def test_a_garment_in_the_wash_is_excluded_from_availability(
        self, client: AsyncClient
    ) -> None:
        item = await add_item(client)
        await client.patch(
            f"/api/v1/wardrobe/items/{item['id']}", json={"laundry_status": "in_laundry"}
        )

        body = (await client.get("/api/v1/wardrobe/items?available_only=true")).json()

        assert body["total"] == 0


class TestRecommendations:
    async def test_the_recommendation_explains_itself(self, client: AsyncClient) -> None:
        item = await add_item(client, warmth=4)
        await add_outfit(client, [item["id"]])

        body = (await client.get("/api/v1/wardrobe/recommend")).json()

        assert body["recommendations"]
        top = body["recommendations"][0]
        assert top["explanation"]
        assert sum(component["weight"] for component in top["components"]) == 100

    async def test_an_unavailable_outfit_is_reported_as_excluded(self, client: AsyncClient) -> None:
        """Silence about a missing option is worse than a reason."""
        item = await add_item(client)
        await add_outfit(client, [item["id"]], name="In the wash")
        await client.patch(
            f"/api/v1/wardrobe/items/{item['id']}", json={"laundry_status": "in_laundry"}
        )

        body = (await client.get("/api/v1/wardrobe/recommend")).json()

        assert body["recommendations"] == []
        assert body["excluded"][0]["outfit_name"] == "In the wash"
        assert "laundry" in body["excluded"][0]["reason"]

    async def test_the_warmer_outfit_wins_on_the_snow_day(self, client: AsyncClient) -> None:
        """Day two of the mock forecast is snow with a low of minus six."""
        warm = await add_item(client, name="Parka", category="outerwear", warmth=5)
        light = await add_item(client, name="Linen shirt", category="top", warmth=1)
        await add_outfit(client, [warm["id"]], name="Winter")
        await add_outfit(client, [light["id"]], name="Summer")

        body = (await client.get("/api/v1/wardrobe/recommend?day_offset=2")).json()

        assert body["recommendations"][0]["outfit_name"] == "Winter"

    async def test_an_empty_wardrobe_answers_cleanly(self, client: AsyncClient) -> None:
        body = (await client.get("/api/v1/wardrobe/recommend")).json()
        assert body["recommendations"] == []
        assert body["considered"] == 0


class TestPackingList:
    async def test_a_trip_produces_quantities_with_reasons(self, client: AsyncClient) -> None:
        await add_item(client, name="Shirt", category="top")
        await add_item(client, name="Jeans", category="bottom")

        start = date(2026, 3, 10)
        body = (
            await client.post(
                "/api/v1/wardrobe/packing-list",
                json={
                    "destination": "Ottawa",
                    "start_on": start.isoformat(),
                    "end_on": (start + timedelta(days=3)).isoformat(),
                },
            )
        ).json()

        tops = next(line for line in body["lines"] if line["category"] == "top")
        assert tops["quantity"] == 4
        assert "per day" in tops["reason"]

    async def test_a_cold_trip_adds_winter_essentials(self, client: AsyncClient) -> None:
        start = date(2026, 3, 10)
        body = (
            await client.post(
                "/api/v1/wardrobe/packing-list",
                json={
                    "destination": "Ottawa",
                    "start_on": start.isoformat(),
                    "end_on": (start + timedelta(days=4)).isoformat(),
                },
            )
        ).json()

        assert "gloves" in body["essentials"]
        assert any(line["category"] == "rain layer" for line in body["lines"])

    async def test_a_backwards_trip_is_refused(self, client: AsyncClient) -> None:
        response = await client.post(
            "/api/v1/wardrobe/packing-list",
            json={"destination": "X", "start_on": "2026-03-10", "end_on": "2026-03-01"},
        )
        assert response.status_code == 422

    async def test_an_absurdly_long_trip_is_refused(self, client: AsyncClient) -> None:
        response = await client.post(
            "/api/v1/wardrobe/packing-list",
            json={"destination": "X", "start_on": "2026-03-10", "end_on": "2027-03-10"},
        )
        assert response.status_code == 422


class TestStatsAndGaps:
    async def test_stats_summarise_the_wardrobe(self, client: AsyncClient) -> None:
        await add_item(client, name="Shirt", category="top")
        await add_item(client, name="Jeans", category="bottom")

        body = (await client.get("/api/v1/wardrobe/stats")).json()

        assert body["items"] == 2
        assert body["never_worn"] == 2
        assert body["by_category"]["top"] == 1

    async def test_never_worn_garments_show_up_as_gaps(self, client: AsyncClient) -> None:
        await add_item(client, name="Forgotten blazer", category="top")
        body = (await client.get("/api/v1/wardrobe/gaps")).json()
        assert body["items"][0]["reason"] == "unworn"


class TestAuthentication:
    async def test_the_wardrobe_needs_a_token(self, anonymous_client: AsyncClient) -> None:
        assert (await anonymous_client.get("/api/v1/wardrobe/items")).status_code == 401
