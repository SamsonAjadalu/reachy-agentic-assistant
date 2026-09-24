"""Recurrence arithmetic, including the DST cases that make it worth testing."""

from __future__ import annotations

from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import pytest

from shared.errors import ValidationError
from shared.recurrence import Frequency, RecurrenceRule

TORONTO = ZoneInfo("America/Toronto")


def local(year: int, month: int, day: int, hour: int = 0, minute: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=TORONTO)


class TestDaily:
    def test_next_occurrence_is_today_when_the_time_has_not_passed(self) -> None:
        rule = RecurrenceRule(Frequency.DAILY, at_hour=9, timezone="America/Toronto")
        nxt = rule.next_occurrence(local(2026, 6, 15, 7, 0))
        assert nxt.astimezone(TORONTO) == local(2026, 6, 15, 9, 0)

    def test_next_occurrence_rolls_to_tomorrow_once_the_time_has_passed(self) -> None:
        rule = RecurrenceRule(Frequency.DAILY, at_hour=9, timezone="America/Toronto")
        nxt = rule.next_occurrence(local(2026, 6, 15, 10, 0))
        assert nxt.astimezone(TORONTO) == local(2026, 6, 16, 9, 0)

    def test_interval_skips_days(self) -> None:
        rule = RecurrenceRule(Frequency.DAILY, interval=3, at_hour=8, timezone="America/Toronto")
        occurrences = rule.occurrences(3, local(2026, 6, 15, 9, 0))
        days = [o.astimezone(TORONTO).day for o in occurrences]
        assert days == [18, 21, 24]


class TestWeekly:
    def test_selected_weekdays_are_visited_in_order(self) -> None:
        # Monday=0, Wednesday=2, Friday=4
        rule = RecurrenceRule(
            Frequency.WEEKLY, by_weekday=[0, 2, 4], at_hour=8, timezone="America/Toronto"
        )
        occurrences = rule.occurrences(4, local(2026, 6, 15, 9, 0))  # Monday 15 June 2026
        weekdays = [o.astimezone(TORONTO).weekday() for o in occurrences]
        assert weekdays == [2, 4, 0, 2]

    def test_weekday_selection_with_an_interval_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="interval=1"):
            RecurrenceRule(Frequency.WEEKLY, interval=2, by_weekday=[1])

    def test_plain_weekly_steps_seven_days(self) -> None:
        rule = RecurrenceRule(Frequency.WEEKLY, at_hour=8, timezone="America/Toronto")
        occurrences = rule.occurrences(2, local(2026, 6, 15, 9, 0))
        assert occurrences[0].astimezone(TORONTO) == local(2026, 6, 22, 8, 0)
        assert occurrences[1].astimezone(TORONTO) == local(2026, 6, 29, 8, 0)


class TestMonthly:
    def test_lands_on_the_requested_day(self) -> None:
        rule = RecurrenceRule(
            Frequency.MONTHLY, by_monthday=15, at_hour=10, timezone="America/Toronto"
        )
        occurrences = rule.occurrences(3, local(2026, 1, 20, 9, 0))
        assert [o.astimezone(TORONTO).day for o in occurrences] == [15, 15, 15]
        assert [o.astimezone(TORONTO).month for o in occurrences] == [2, 3, 4]

    def test_day_31_clamps_to_the_end_of_a_short_month(self) -> None:
        rule = RecurrenceRule(
            Frequency.MONTHLY, by_monthday=31, at_hour=10, timezone="America/Toronto"
        )
        occurrences = rule.occurrences(3, local(2026, 1, 31, 11, 0))
        days = [(o.astimezone(TORONTO).month, o.astimezone(TORONTO).day) for o in occurrences]
        assert days == [(2, 28), (3, 31), (4, 30)]

    def test_interval_skips_months(self) -> None:
        rule = RecurrenceRule(
            Frequency.MONTHLY, interval=3, by_monthday=1, at_hour=9, timezone="America/Toronto"
        )
        occurrences = rule.occurrences(3, local(2026, 1, 15))
        assert [o.astimezone(TORONTO).month for o in occurrences] == [4, 7, 10]


class TestDaylightSaving:
    def test_wall_clock_time_survives_the_spring_forward(self) -> None:
        """Toronto springs forward on 8 March 2026. 07:00 must stay 07:00."""
        rule = RecurrenceRule(Frequency.DAILY, at_hour=7, timezone="America/Toronto")
        occurrences = rule.occurrences(3, local(2026, 3, 6, 8, 0))

        for occurrence in occurrences:
            rendered = occurrence.astimezone(TORONTO)
            assert (rendered.hour, rendered.minute) == (7, 0)

        # The offset genuinely changes across the boundary, which is the point:
        # the UTC instants differ by 23 hours, not 24.
        offsets = {o.astimezone(TORONTO).utcoffset() for o in occurrences}
        assert len(offsets) == 2

    def test_wall_clock_time_survives_the_fall_back(self) -> None:
        """Toronto falls back on 1 November 2026."""
        rule = RecurrenceRule(Frequency.DAILY, at_hour=7, timezone="America/Toronto")
        occurrences = rule.occurrences(3, local(2026, 10, 30, 8, 0))
        for occurrence in occurrences:
            assert occurrence.astimezone(TORONTO).hour == 7

    def test_a_utc_rule_is_unaffected_by_local_dst(self) -> None:
        rule = RecurrenceRule(Frequency.DAILY, at_hour=12, timezone="UTC")
        occurrences = rule.occurrences(3, datetime(2026, 3, 6, 13, 0, tzinfo=UTC))
        assert {o.hour for o in occurrences} == {12}


class TestBoundaries:
    def test_ends_at_terminates_the_series(self) -> None:
        rule = RecurrenceRule(
            Frequency.DAILY,
            at_hour=9,
            timezone="America/Toronto",
            ends_at=datetime(2026, 6, 18, 0, 0, tzinfo=UTC),
        )
        occurrences = rule.occurrences(10, local(2026, 6, 15, 10, 0))
        assert len(occurrences) == 2

    def test_starts_at_defers_the_first_occurrence(self) -> None:
        rule = RecurrenceRule(
            Frequency.DAILY,
            at_hour=9,
            timezone="America/Toronto",
            starts_at=datetime(2026, 7, 1, 0, 0, tzinfo=UTC),
        )
        nxt = rule.next_occurrence(local(2026, 6, 15, 10, 0))
        assert nxt.astimezone(TORONTO).month == 7

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"interval": 0}, "interval"),
            ({"interval": 500}, "interval"),
            ({"at_hour": 24}, "valid hour"),
            ({"at_minute": 60}, "valid hour"),
            ({"frequency": Frequency.DAILY, "by_weekday": [1]}, "weekly"),
            ({"frequency": Frequency.DAILY, "by_monthday": 5}, "monthly"),
            ({"frequency": Frequency.WEEKLY, "by_weekday": [9]}, "0 \\(Monday\\)"),
        ],
    )
    def test_invalid_rules_are_rejected(self, kwargs: dict, message: str) -> None:
        params = {"frequency": Frequency.WEEKLY, **kwargs}
        with pytest.raises(ValidationError, match=message):
            RecurrenceRule(**params)

    def test_unknown_timezone_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="Unknown timezone"):
            RecurrenceRule(Frequency.DAILY, timezone="Nowhere/Fictional")


class TestDescription:
    @pytest.mark.parametrize(
        ("rule", "expected"),
        [
            (RecurrenceRule(Frequency.DAILY, at_hour=7, timezone="UTC"), "every day at 07:00 UTC"),
            (
                RecurrenceRule(Frequency.DAILY, interval=2, at_hour=7, timezone="UTC"),
                "every 2 day at 07:00 UTC",
            ),
            (
                RecurrenceRule(Frequency.WEEKLY, by_weekday=[0, 4], at_hour=8, timezone="UTC"),
                "every week on Monday, Friday at 08:00 UTC",
            ),
            (
                RecurrenceRule(Frequency.MONTHLY, by_monthday=1, at_hour=9, timezone="UTC"),
                "every month on day 1 at 09:00 UTC",
            ),
        ],
    )
    def test_describe_reads_naturally(self, rule: RecurrenceRule, expected: str) -> None:
        assert rule.describe() == expected
