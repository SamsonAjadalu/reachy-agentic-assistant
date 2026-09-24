"""Shared request and response schema pieces."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from shared.enums import NotificationChannel
from shared.errors import ValidationError
from shared.recurrence import Frequency, RecurrenceRule
from shared.timeutils import duration_from_parts, ensure_utc, utcnow


class ApiModel(BaseModel):
    """Base for every request and response body."""

    model_config = ConfigDict(
        extra="forbid",  # a typo'd field is a bug worth surfacing, not ignoring
        str_strip_whitespace=True,
        validate_assignment=True,
    )


class Delay(ApiModel):
    """A relative offset expressed as structured fields.

    Free-text date parsing is deliberately absent from this backend. Turning
    "tomorrow morning" into an instant is the conversational model's job; the
    API takes either an absolute timestamp or these explicit parts, so a
    misheard phrase cannot silently become the wrong time.
    """

    days: Annotated[int, Field(ge=0, le=3650)] = 0
    hours: Annotated[int, Field(ge=0, le=87600)] = 0
    minutes: Annotated[int, Field(ge=0, le=525600)] = 0
    seconds: Annotated[int, Field(ge=0, le=31536000)] = 0

    def resolve(self, *, from_time: datetime | None = None) -> datetime:
        delta = duration_from_parts(
            days=self.days, hours=self.hours, minutes=self.minutes, seconds=self.seconds
        )
        return (from_time or utcnow()) + delta


class RecurrenceInput(ApiModel):
    frequency: Frequency
    interval: Annotated[int, Field(ge=1, le=366)] = 1
    by_weekday: list[Annotated[int, Field(ge=0, le=6)]] | None = Field(
        default=None, description="0 is Monday. Weekly recurrences only."
    )
    by_monthday: Annotated[int, Field(ge=1, le=31)] | None = Field(
        default=None, description="Monthly recurrences only. Clamped to the last valid day."
    )
    at_hour: Annotated[int, Field(ge=0, le=23)] = 9
    at_minute: Annotated[int, Field(ge=0, le=59)] = 0
    timezone: str | None = Field(
        default=None, description="Defaults to APP_TIMEZONE. The wall-clock time is preserved."
    )
    ends_at: datetime | None = None
    max_occurrences: Annotated[int, Field(ge=1, le=10000)] | None = None

    def to_rule(self, default_timezone: str, starts_at: datetime | None = None) -> RecurrenceRule:
        return RecurrenceRule(
            frequency=self.frequency,
            interval=self.interval,
            by_weekday=self.by_weekday,
            by_monthday=self.by_monthday,
            at_hour=self.at_hour,
            at_minute=self.at_minute,
            timezone=self.timezone or default_timezone,
            starts_at=starts_at,
            ends_at=ensure_utc(self.ends_at) if self.ends_at else None,
            max_occurrences=self.max_occurrences,
        )


class ScheduleInput(ApiModel):
    """Exactly one of an absolute time or a relative delay."""

    at: datetime | None = Field(
        default=None, description="ISO-8601 with an explicit offset or a 'Z' suffix."
    )
    delay: Delay | None = None

    @model_validator(mode="after")
    def _exactly_one(self) -> Self:
        if (self.at is None) == (self.delay is None):
            raise ValueError("Provide exactly one of 'at' or 'delay'.")
        return self

    def resolve(self, *, from_time: datetime | None = None) -> datetime:
        if self.at is not None:
            if self.at.tzinfo is None:
                raise ValidationError(
                    "'at' must carry a timezone offset, for example 2026-08-03T09:00:00-04:00."
                )
            return ensure_utc(self.at)
        assert self.delay is not None
        return self.delay.resolve(from_time=from_time)


class ChannelPreference(ApiModel):
    channel: NotificationChannel = NotificationChannel.TELEGRAM


class DeleteResponse(ApiModel):
    id: str
    deleted: bool = True


class AcceptedTask(ApiModel):
    """Returned by endpoints that genuinely defer work.

    Fast reads answer inline; this shape appears only where the operation cannot
    complete inside a voice turn.
    """

    task_id: str
    status: str = "queued"
    message: str = "The task was started and the result will be delivered later."
    poll_url: str | None = None
