"""Time handling.

The whole system stores and computes in UTC. Conversion to the owner's local
zone happens only at the input and output boundaries, which is what makes DST
transitions predictable rather than a source of silent off-by-one-hour bugs.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from shared.errors import ValidationError


def utcnow() -> datetime:
    """Timezone-aware current time in UTC."""
    return datetime.now(UTC)


def ensure_utc(value: datetime) -> datetime:
    """Normalise any datetime to an aware UTC datetime.

    Naive datetimes are rejected rather than assumed to be local: guessing here
    is how reminders end up firing hours early.
    """
    if value.tzinfo is None:
        raise ValidationError(
            "Naive datetime received. Supply an ISO-8601 timestamp with an explicit "
            "offset or a 'Z' suffix."
        )
    return value.astimezone(UTC)


def get_zone(name: str) -> ZoneInfo:
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise ValidationError(f"Unknown timezone: {name!r}") from exc


def to_local(value: datetime, timezone_name: str) -> datetime:
    """Render a UTC instant in the configured local zone for display."""
    return ensure_utc(value).astimezone(get_zone(timezone_name))


def from_local(value: datetime, timezone_name: str) -> datetime:
    """Interpret a naive wall-clock time as local, returning UTC.

    Used for schedule definitions such as "07:30 every weekday", where the
    intent is a wall-clock time that should survive a DST shift.
    """
    zone = get_zone(timezone_name)
    localised = value.replace(tzinfo=zone) if value.tzinfo is None else value.astimezone(zone)
    return localised.astimezone(UTC)


def parse_iso8601(raw: str) -> datetime:
    """Parse a strict ISO-8601 timestamp that carries an offset."""
    text = raw.strip()
    if text.endswith(("z", "Z")):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValidationError(f"Invalid ISO-8601 timestamp: {raw!r}") from exc
    return ensure_utc(parsed)


def isoformat_utc(value: datetime) -> str:
    """Serialise as ``...Z``, which every JSON client parses consistently."""
    return ensure_utc(value).isoformat().replace("+00:00", "Z")


def duration_from_parts(
    *,
    days: int = 0,
    hours: int = 0,
    minutes: int = 0,
    seconds: int = 0,
) -> timedelta:
    """Build a delay from structured fields.

    Relative reminders arrive as structured parts, never as free text: date
    parsing belongs to the conversational model on the Reachy side, not to this
    backend.
    """
    total = timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds)
    if total <= timedelta(0):
        raise ValidationError("Duration must be positive.")
    if total > timedelta(days=365 * 5):
        raise ValidationError("Duration must be under five years.")
    return total


def start_of_local_day(moment: datetime, timezone_name: str) -> datetime:
    local = to_local(moment, timezone_name)
    return local.replace(hour=0, minute=0, second=0, microsecond=0).astimezone(UTC)


def end_of_local_day(moment: datetime, timezone_name: str) -> datetime:
    return start_of_local_day(moment, timezone_name) + timedelta(days=1)


def local_today(timezone_name: str) -> date:
    """The owner's current calendar date.

    Anything a person would call "today" - what was worn today, what is due
    today - has to be resolved in their timezone. Using the UTC date would file
    an evening event under tomorrow for anyone west of Greenwich.
    """
    return to_local(utcnow(), timezone_name).date()
