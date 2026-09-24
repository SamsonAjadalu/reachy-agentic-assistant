"""Image handling and the recommendation rules.

The scoring tests are written as scenarios rather than as assertions about
individual numbers, because the numbers are allowed to be tuned and the
behaviour is not: a coat should win in the cold whatever the weights say.
"""

from __future__ import annotations

import io
from datetime import date, timedelta

import pytest
from PIL import Image

from integrations.weather.advice import WeatherAdvice
from shared.enums import LaundryStatus
from shared.errors import ValidationError
from wardrobe.images import (
    MAX_IMAGE_BYTES,
    dominant_colours,
    load_image,
    name_colour,
    store_image,
)
from wardrobe.recommend import (
    ItemView,
    OutfitView,
    rank,
    rotation_gaps,
    score_outfit,
    suggest_laundry,
    target_warmth,
)

TODAY = date(2026, 3, 10)

MILD_DRY = WeatherAdvice(
    warmth_band="cool",
    needs_umbrella=False,
    needs_winter_layers=False,
    needs_sun_protection=False,
    windy=False,
    reasons=["cool and unremarkable"],
)
COLD_SNOW = WeatherAdvice(
    warmth_band="freezing",
    needs_umbrella=False,
    needs_winter_layers=True,
    needs_sun_protection=False,
    windy=True,
    reasons=["snow expected"],
)
WET = WeatherAdvice(
    warmth_band="cool",
    needs_umbrella=True,
    needs_winter_layers=False,
    needs_sun_protection=False,
    windy=False,
    reasons=["rain expected"],
)


def make_image(colour: tuple[int, int, int], size: tuple[int, int] = (200, 200)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, colour).save(buffer, format="PNG")
    return buffer.getvalue()


def item(
    name: str,
    category: str,
    *,
    colour: str = "black",
    warmth: int = 3,
    formality: int = 3,
    water_resistant: bool = False,
    available: bool = True,
    laundry: str = LaundryStatus.CLEAN.value,
    wears: int = 0,
    limit: int = 3,
    rating: int | None = None,
    last_worn: date | None = None,
) -> ItemView:
    return ItemView(
        id=f"item-{name.lower().replace(' ', '-')}",
        name=name,
        category=category,
        primary_color=colour,
        formality=formality,
        warmth=warmth,
        water_resistant=water_resistant,
        seasons=["all"],
        laundry_status=laundry,
        available=available,
        wears_since_wash=wears,
        max_wears_before_wash=limit,
        favourite_rating=rating,
        last_worn_on=last_worn,
    )


def outfit(name: str, items: list[ItemView], **kwargs: object) -> OutfitView:
    return OutfitView(
        id=f"outfit-{name.lower().replace(' ', '-')}",
        name=name,
        formality=int(kwargs.pop("formality", 3)),  # type: ignore[call-overload]
        items=items,
        **kwargs,  # type: ignore[arg-type]
    )


class TestImageValidation:
    def test_a_valid_png_decodes(self) -> None:
        assert load_image(make_image((10, 40, 120))).size == (200, 200)

    def test_a_text_file_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="could not be decoded"):
            load_image(b"this is not an image")

    def test_an_empty_upload_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="empty"):
            load_image(b"")

    def test_an_oversized_upload_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="over the"):
            load_image(b"x" * (MAX_IMAGE_BYTES + 1))

    def test_a_truncated_image_is_refused(self) -> None:
        data = make_image((200, 30, 30))
        with pytest.raises(ValidationError):
            load_image(data[: len(data) // 3])

    def test_a_disallowed_format_is_refused(self) -> None:
        buffer = io.BytesIO()
        Image.new("RGB", (10, 10), (0, 0, 0)).save(buffer, format="BMP")
        with pytest.raises(ValidationError, match="not accepted"):
            load_image(buffer.getvalue())


class TestImageStorage:
    def test_storage_strips_metadata(self, tmp_path):
        """Phone photos carry GPS coordinates in EXIF."""
        from PIL import ExifTags

        buffer = io.BytesIO()
        image = Image.new("RGB", (300, 300), (30, 90, 160))
        exif = image.getexif()
        exif[ExifTags.Base.Make] = "TestCamera"
        image.save(buffer, format="JPEG", exif=exif)

        stored = store_image(buffer.getvalue(), tmp_path, item_id="item-1")

        reopened = Image.open(tmp_path / stored.relative_path)
        assert not dict(reopened.getexif())

    def test_a_large_image_is_downscaled(self, tmp_path):
        stored = store_image(make_image((60, 60, 60), (4000, 3000)), tmp_path, item_id="i")
        assert max(stored.width, stored.height) <= 1600

    def test_a_thumbnail_is_written(self, tmp_path):
        stored = store_image(make_image((60, 60, 60)), tmp_path, item_id="i")
        assert (tmp_path / stored.thumbnail_path).is_file()

    def test_the_stored_name_is_a_content_hash(self, tmp_path):
        """Uses the configured workflow."""
        stored = store_image(make_image((60, 60, 60)), tmp_path, item_id="i")
        assert stored.content_hash[:16] in stored.relative_path

    def test_identical_uploads_land_on_the_same_path(self, tmp_path):
        data = make_image((10, 200, 90))
        first = store_image(data, tmp_path, item_id="i")
        second = store_image(data, tmp_path, item_id="i")
        assert first.relative_path == second.relative_path


class TestColourNaming:
    @pytest.mark.parametrize(
        ("rgb", "expected"),
        [
            ((5, 5, 5), "black"),
            ((250, 250, 250), "white"),
            ((128, 128, 128), "grey"),
            ((200, 20, 20), "red"),
            ((20, 40, 120), "navy"),
            ((30, 130, 60), "green"),
        ],
    )
    def test_colours_get_names_people_use(self, rgb: tuple[int, int, int], expected: str) -> None:
        assert name_colour(*rgb) == expected

    def test_a_garment_photo_proposes_its_colour(self) -> None:
        colours = dominant_colours(load_image(make_image((20, 40, 130))))
        assert colours[0]["name"] in {"navy", "blue"}
        assert colours[0]["hex"].startswith("#")

    def test_the_proposal_is_deterministic(self) -> None:
        image = load_image(make_image((150, 40, 40)))
        assert dominant_colours(image) == dominant_colours(image)


class TestWarmthTargets:
    @pytest.mark.parametrize(
        ("temperature", "warmth"),
        [(-15.0, 5), (-2.0, 5), (5.0, 4), (12.0, 3), (20.0, 2), (30.0, 1)],
    )
    def test_colder_weather_wants_warmer_clothes(self, temperature: float, warmth: int) -> None:
        assert target_warmth(temperature) == warmth


class TestScoring:
    def test_an_unavailable_garment_disqualifies_the_outfit(self) -> None:
        score = score_outfit(
            outfit("Loaned out", [item("Lent jacket", "outerwear", available=False)]),
            temperature_c=10,
            advice=MILD_DRY,
            today=TODAY,
        )
        assert score.disqualified
        assert "Lent jacket" in (score.disqualified_reason or "")

    def test_laundry_disqualifies_the_outfit(self) -> None:
        score = score_outfit(
            outfit("Dirty", [item("Shirt", "top", laundry=LaundryStatus.IN_LAUNDRY.value)]),
            temperature_c=10,
            advice=MILD_DRY,
            today=TODAY,
        )
        assert score.disqualified
        assert "laundry" in (score.disqualified_reason or "")

    def test_a_warm_outfit_wins_in_the_cold(self) -> None:
        coat = outfit("Winter", [item("Parka", "outerwear", warmth=5)])
        linen = outfit("Summer", [item("Linen shirt", "top", warmth=1)])

        ranked = rank([linen, coat], temperature_c=-5, advice=COLD_SNOW, today=TODAY)

        assert ranked[0].outfit_name == "Winter"

    def test_a_light_outfit_wins_in_the_heat(self) -> None:
        coat = outfit("Winter", [item("Parka", "outerwear", warmth=5)])
        linen = outfit("Summer", [item("Linen shirt", "top", warmth=1)])

        ranked = rank([coat, linen], temperature_c=29, advice=MILD_DRY, today=TODAY)

        assert ranked[0].outfit_name == "Summer"

    def test_being_underdressed_costs_more_than_being_overdressed(self) -> None:
        """Cold is a worse outcome than warm."""
        too_light = score_outfit(
            outfit("Light", [item("Tee", "top", warmth=2)]),
            temperature_c=0,
            advice=COLD_SNOW,
            today=TODAY,
        )
        too_warm = score_outfit(
            outfit("Heavy", [item("Parka", "outerwear", warmth=5)]),
            temperature_c=18,
            advice=MILD_DRY,
            today=TODAY,
        )
        light_weather = next(c for c in too_light.components if c.name == "weather")
        warm_weather = next(c for c in too_warm.components if c.name == "weather")
        assert light_weather.score < warm_weather.score

    def test_rain_favours_a_water_resistant_layer(self) -> None:
        dry = outfit("Wool", [item("Wool coat", "outerwear", warmth=4)])
        proofed = outfit("Shell", [item("Rain shell", "outerwear", warmth=4, water_resistant=True)])

        ranked = rank([dry, proofed], temperature_c=8, advice=WET, today=TODAY)

        assert ranked[0].outfit_name == "Shell"

    def test_an_outfit_with_no_rain_layer_is_warned_about(self) -> None:
        score = score_outfit(
            outfit("Wool", [item("Wool coat", "outerwear")]),
            temperature_c=8,
            advice=WET,
            today=TODAY,
        )
        assert any("water resistant" in warning for warning in score.warnings)

    def test_formality_is_matched_when_it_is_asked_for(self) -> None:
        smart = outfit("Suit", [item("Blazer", "top", formality=5)], formality=5)
        casual = outfit("Jeans", [item("Tee", "top", formality=1)], formality=1)

        ranked = rank(
            [casual, smart],
            temperature_c=15,
            advice=MILD_DRY,
            required_formality=5,
            today=TODAY,
        )

        assert ranked[0].outfit_name == "Suit"

    def test_mixed_formality_is_penalised_without_an_occasion(self) -> None:
        mixed = score_outfit(
            outfit(
                "Odd", [item("Blazer", "top", formality=5), item("Shorts", "bottom", formality=1)]
            ),
            temperature_c=15,
            advice=MILD_DRY,
            today=TODAY,
        )
        component = next(c for c in mixed.components if c.name == "formality")
        assert component.score < 0.5
        assert "dressy" in component.reason

    def test_clashing_colours_are_penalised(self) -> None:
        clash = score_outfit(
            outfit(
                "Clash",
                [item("Top", "top", colour="red"), item("Trousers", "bottom", colour="pink")],
            ),
            temperature_c=15,
            advice=MILD_DRY,
            today=TODAY,
        )
        component = next(c for c in clash.components if c.name == "colour")
        assert component.score < 0.5
        assert "red" in component.reason

    def test_neutrals_are_never_treated_as_a_clash(self) -> None:
        neutral = score_outfit(
            outfit(
                "Neutral",
                [item("Top", "top", colour="white"), item("Trousers", "bottom", colour="grey")],
            ),
            temperature_c=15,
            advice=MILD_DRY,
            today=TODAY,
        )
        assert next(c for c in neutral.components if c.name == "colour").score >= 0.8

    def test_something_worn_today_is_pushed_down(self) -> None:
        fresh = outfit("Fresh", [item("Shirt A", "top")])
        repeat = outfit("Repeat", [item("Shirt B", "top")], last_worn_on=TODAY)

        ranked = rank([repeat, fresh], temperature_c=15, advice=MILD_DRY, today=TODAY)

        assert ranked[0].outfit_name == "Fresh"

    def test_an_old_favourite_is_not_penalised_forever(self) -> None:
        score = score_outfit(
            outfit("Old", [item("Shirt", "top")], last_worn_on=TODAY - timedelta(days=30)),
            temperature_c=15,
            advice=MILD_DRY,
            today=TODAY,
        )
        assert next(c for c in score.components if c.name == "freshness").score == 1.0

    def test_a_rated_outfit_beats_an_unrated_twin(self) -> None:
        loved = outfit("Loved", [item("Shirt A", "top")], user_rating=5)
        plain = outfit("Plain", [item("Shirt B", "top")])

        ranked = rank([plain, loved], temperature_c=15, advice=MILD_DRY, today=TODAY)

        assert ranked[0].outfit_name == "Loved"


class TestExplanations:
    def test_every_component_states_a_reason(self) -> None:
        score = score_outfit(
            outfit("Anything", [item("Shirt", "top")]),
            temperature_c=15,
            advice=MILD_DRY,
            today=TODAY,
        )
        assert all(component.reason for component in score.components)

    def test_the_weights_are_reported_with_the_scores(self) -> None:
        """The owner can only argue with the weighting if they can see it."""
        payload = score_outfit(
            outfit("Anything", [item("Shirt", "top")]),
            temperature_c=15,
            advice=MILD_DRY,
            today=TODAY,
        ).as_dict()
        assert sum(component["weight"] for component in payload["components"]) == 100

    def test_the_same_wardrobe_ranks_identically_every_time(self) -> None:
        """A recommendation that changes between identical calls cannot be reported."""
        outfits = [
            outfit("A", [item("Shirt A", "top")]),
            outfit("B", [item("Shirt B", "top")]),
            outfit("C", [item("Shirt C", "top")]),
        ]
        first = [
            score.outfit_name
            for score in rank(outfits, temperature_c=15, advice=MILD_DRY, today=TODAY)
        ]
        second = [
            score.outfit_name
            for score in rank(outfits, temperature_c=15, advice=MILD_DRY, today=TODAY)
        ]
        assert first == second


class TestLaundryAndRotation:
    def test_an_item_at_its_wear_limit_is_flagged(self) -> None:
        due = suggest_laundry([item("Jeans", "bottom", wears=3, limit=3)], today=TODAY)
        assert due[0]["name"] == "Jeans"

    def test_an_item_already_in_the_wash_is_not_flagged_again(self) -> None:
        already = item("Jeans", "bottom", wears=5, limit=3, laundry=LaundryStatus.IN_LAUNDRY.value)
        assert suggest_laundry([already], today=TODAY) == []

    def test_a_forgotten_garment_is_surfaced(self) -> None:
        old = item("Green coat", "outerwear", last_worn=TODAY - timedelta(days=200))
        recent = item("Daily jacket", "outerwear", last_worn=TODAY - timedelta(days=2))

        gaps = rotation_gaps([old, recent], today=TODAY, days=90)

        assert [entry["name"] for entry in gaps] == ["Green coat"]

    def test_a_never_worn_garment_is_surfaced(self) -> None:
        gaps = rotation_gaps([item("New shirt", "top")], today=TODAY)
        assert gaps[0]["reason"] == "unworn"
