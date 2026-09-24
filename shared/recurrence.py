"""Recurrence computation.

Deliberately narrow: daily, weekly, monthly and yearly with an interval, plus
weekday selection. Anything richer would need a full RRULE engine, and the
conversational model on the Reachy side is a better place to turn "every other
Tuesday" into these fields than a regex in this backend.

Occurrences are computed in the rule's own timezone and converted to UTC at the
end. That is what keeps a 07:00 reminder at 07:00 across a DST change instead of
drifting to 06:00 or 08:00.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from enum import StrEnum

from dateutil.relativedelta import relativedelta

from shared.errors import ValidationError
from shared.timeutils import ensure_utc, get_zone, utcnow

MAX_SEARCH_ITERATIONS = 400


def _with_monthday(moment: datetime, monthday: int) -> datetime:
    """Set the day of month, clamping to the last valid day.

    "The 31st of every month" has to mean something in February; clamping is the
    behaviour people expect from calendar apps.
    """
    last_day = (moment.replace(day=1) + relativedelta(months=1, days=-1)).day
    return moment.replace(day=min(monthday, last_day))


class Frequency(StrEnum):
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    YEARLY = "yearly"


class RecurrenceRule:
    """A validated recurrence definition."""

    def __init__(
        self,
        frequency: Frequency | str,
        *,
        interval: int = 1,
        by_weekday: list[int] | None = None,
        by_monthday: int | None = None,
        at_hour: int = 9,
        at_minute: int = 0,
        timezone: str = "UTC",
        starts_at: datetime | None = None,
        ends_at: datetime | None = None,
        max_occurrences: int | None = None,
    ) -> None:
        self.frequency = Frequency(frequency)
        if interval < 1 or interval > 366:
            raise ValidationError("Recurrence interval must be between 1 and 366.")
        if not 0 <= at_hour <= 23 or not 0 <= at_minute <= 59:
            raise ValidationError("Recurrence time must be a valid hour and minute.")
        if by_weekday:
            if any(day < 0 or day > 6 for day in by_weekday):
                raise ValidationError("Weekdays must be 0 (Monday) through 6 (Sunday).")
            if self.frequency is not Frequency.WEEKLY:
                raise ValidationError("by_weekday only applies to a weekly recurrence.")
            if interval != 1:
                # "Every other Tuesday" needs a week-of-year anchor to be
                # unambiguous. Rejecting it is better than guessing wrong.
                raise ValidationError(
                    "Selecting weekdays requires interval=1. For a longer cycle, use a "
                    "weekly recurrence without by_weekday."
                )
        if by_monthday is not None:
            if not 1 <= by_monthday <= 31:
                raise ValidationError("by_monthday must be between 1 and 31.")
            if self.frequency is not Frequency.MONTHLY:
                raise ValidationError("by_monthday only applies to a monthly recurrence.")
        if ends_at and starts_at and ends_at <= starts_at:
            raise ValidationError("Recurrence ends_at must be after starts_at.")

        self.interval = interval
        self.by_weekday = sorted(set(by_weekday)) if by_weekday else []
        self.by_monthday = by_monthday
        self.at_hour = at_hour
        self.at_minute = at_minute
        self.timezone = timezone
        self.starts_at = ensure_utc(starts_at) if starts_at else None
        self.ends_at = ensure_utc(ends_at) if ends_at else None
        self.max_occurrences = max_occurrences
        get_zone(timezone)  # fail fast on an unknown zone

    def next_occurrence(self, after: datetime | None = None) -> datetime | None:
        """First occurrence strictly after ``after``, or ``None`` if exhausted."""
        reference = ensure_utc(after) if after else utcnow()
        if self.starts_at and reference < self.starts_at:
            reference = self.starts_at - timedelta(seconds=1)

        zone = get_zone(self.timezone)
        local = reference.astimezone(zone)
        candidate = self._first_candidate(local)

        for _ in range(MAX_SEARCH_ITERATIONS):
            if candidate > local and self._matches(candidate):
                # Re-localise from the naive wall-clock value: replace(tzinfo=...)
                # on an already-aware datetime keeps the old UTC offset, which is
                # wrong on the day a DST transition happens.
                as_utc = ensure_utc(candidate.replace(tzinfo=None).replace(tzinfo=zone))
                if self.ends_at and as_utc > self.ends_at:
                    return None
                return as_utc
            candidate = self._advance(candidate)

        return None

    def _first_candidate(self, local: datetime) -> datetime:
        at_time = local.replace(hour=self.at_hour, minute=self.at_minute, second=0, microsecond=0)
        if self.frequency is Frequency.MONTHLY and self.by_monthday:
            return _with_monthday(at_time, self.by_monthday)
        return at_time

    def occurrences(self, count: int, after: datetime | None = None) -> list[datetime]:
        """The next ``count`` occurrences, for previews and tests."""
        results: list[datetime] = []
        cursor = after
        for _ in range(count):
            nxt = self.next_occurrence(cursor)
            if nxt is None:
                break
            results.append(nxt)
            cursor = nxt
        return results

    def _matches(self, candidate: datetime) -> bool:
        if self.frequency is Frequency.WEEKLY and self.by_weekday:
            return candidate.weekday() in self.by_weekday
        if self.frequency is Frequency.MONTHLY and self.by_monthday:
            return candidate.day == _with_monthday(candidate, self.by_monthday).day
        return True

    def _advance(self, candidate: datetime) -> datetime:
        """Step to the next candidate slot.

        Weekly-with-weekdays steps a day at a time so every selected weekday is
        visited (which is why that mode is restricted to interval=1); the other
        modes step by their own unit times the interval.
        """
        if self.frequency is Frequency.DAILY:
            return candidate + relativedelta(days=self.interval)
        if self.frequency is Frequency.WEEKLY:
            return candidate + relativedelta(days=1 if self.by_weekday else 7 * self.interval)
        if self.frequency is Frequency.MONTHLY:
            stepped = candidate.replace(day=1) + relativedelta(months=self.interval)
            if self.by_monthday:
                return _with_monthday(stepped, self.by_monthday)
            return candidate + relativedelta(months=self.interval)
        return candidate + relativedelta(years=self.interval)

    def describe(self) -> str:
        """Short human phrasing, used in approval previews and briefings."""
        time_part = f"{self.at_hour:02d}:{self.at_minute:02d}"
        every = "" if self.interval == 1 else f" {self.interval}"
        if self.frequency is Frequency.WEEKLY and self.by_weekday:
            names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            days = ", ".join(names[day] for day in self.by_weekday)
            return f"every{every} week on {days} at {time_part} {self.timezone}"
        if self.frequency is Frequency.MONTHLY and self.by_monthday:
            return f"every{every} month on day {self.by_monthday} at {time_part} {self.timezone}"
        unit = {
            Frequency.DAILY: "day",
            Frequency.WEEKLY: "week",
            Frequency.MONTHLY: "month",
            Frequency.YEARLY: "year",
        }[self.frequency]
        return f"every{every} {unit} at {time_part} {self.timezone}"
