"""Time conversions at the boundaries.

Everything inside the service is UTC. These tests cover the two places that is
not enough: the owner's calendar day, and DST transitions.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from shared.errors import ValidationError
from shared.timeutils import (
    duration_from_parts,
    end_of_local_day,
    ensure_utc,
    from_local,
    isoformat_utc,
    local_today,
    parse_iso8601,
    start_of_local_day,
    to_local,
)

TORONTO = "America/Toronto"


class TestEnsureUtc:
    def test_a_naive_time_is_rejected_rather_than_guessed(self) -> None:
        """Assuming local here is how a reminder fires five hours early."""
        with pytest.raises(ValidationError, match="Naive datetime"):
            ensure_utc(datetime(2026, 3, 10, 9, 0))

    def test_another_zone_is_converted_not_relabelled(self) -> None:
        local = to_local(datetime(2026, 3, 10, 9, 0, tzinfo=UTC), TORONTO)
        assert ensure_utc(local) == datetime(2026, 3, 10, 9, 0, tzinfo=UTC)


class TestLocalDay:
    def test_late_evening_is_still_today_locally(self) -> None:
        """23:30 in Toronto is 03:30 UTC tomorrow; the owner would say today."""
        evening = datetime(2026, 8, 3, 3, 30, tzinfo=UTC)
        assert to_local(evening, TORONTO).date().isoformat() == "2026-08-02"

    def test_the_local_day_starts_at_local_midnight(self) -> None:
        moment = datetime(2026, 8, 2, 18, 0, tzinfo=UTC)
        assert start_of_local_day(moment, TORONTO) == datetime(2026, 8, 2, 4, 0, tzinfo=UTC)

    def test_the_local_day_is_twenty_four_hours_long(self) -> None:
        moment = datetime(2026, 8, 2, 18, 0, tzinfo=UTC)
        span = end_of_local_day(moment, TORONTO) - start_of_local_day(moment, TORONTO)
        assert span == timedelta(days=1)

    def test_today_follows_the_configured_zone(self) -> None:
        assert local_today(TORONTO) == to_local(datetime.now(UTC), TORONTO).date()

    def test_today_can_differ_between_zones(self) -> None:
        """The reason wardrobe and briefing dates are zone-aware at all."""
        assert local_today("Pacific/Kiritimati") >= local_today("Pacific/Midway")


class TestDaylightSaving:
    def test_a_time_that_does_not_exist_is_still_resolved(self) -> None:
        """02:30 on the spring-forward morning never happens in Toronto."""
        resolved = from_local(datetime(2026, 3, 8, 2, 30), TORONTO)
        assert resolved.tzinfo is UTC

    def test_the_local_hour_is_preserved_across_a_transition(self) -> None:
        """A 09:00 reminder is 09:00 in March and 09:00 in December."""
        march = to_local(from_local(datetime(2026, 3, 10, 9, 0), TORONTO), TORONTO)
        december = to_local(from_local(datetime(2026, 12, 10, 9, 0), TORONTO), TORONTO)
        assert march.hour == december.hour == 9
        # The UTC offset differs, which is exactly what storing UTC absorbs.
        assert march.utcoffset() != december.utcoffset()

    def test_an_unknown_zone_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            to_local(datetime.now(UTC), "Mars/Olympus_Mons")


class TestParsing:
    @pytest.mark.parametrize(
        "raw",
        ["2026-03-10T09:00:00Z", "2026-03-10T09:00:00+00:00", "2026-03-10T04:00:00-05:00"],
    )
    def test_offsets_normalise_to_one_instant(self, raw: str) -> None:
        assert parse_iso8601(raw) == datetime(2026, 3, 10, 9, 0, tzinfo=UTC)

    def test_a_string_without_an_offset_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="Naive datetime"):
            parse_iso8601("2026-03-10T09:00:00")

    @pytest.mark.parametrize("raw", ["tomorrow", "10/03/2026", "", "2026-13-45T00:00:00Z"])
    def test_unparseable_input_is_rejected(self, raw: str) -> None:
        with pytest.raises(ValidationError):
            parse_iso8601(raw)

    def test_output_always_carries_a_z(self) -> None:
        assert isoformat_utc(datetime(2026, 3, 10, 9, 0, tzinfo=UTC)).endswith("Z")


class TestDurations:
    def test_parts_combine(self) -> None:
        assert duration_from_parts(hours=1, minutes=30) == timedelta(minutes=90)

    def test_a_zero_duration_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            duration_from_parts()

    def test_a_negative_duration_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            duration_from_parts(minutes=-5)

    def test_an_absurd_duration_is_rejected(self) -> None:
        """Guards against a misheard number becoming a reminder in 3025."""
        with pytest.raises(ValidationError):
            duration_from_parts(days=4000)
