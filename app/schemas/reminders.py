"""Reminder request and response bodies."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Self

from pydantic import Field, model_validator

from app.schemas.common import ApiModel, Delay, RecurrenceInput, ScheduleInput
from shared.enums import NotificationChannel, ReminderStatus


class ReminderCreate(ApiModel):
    text: Annotated[str, Field(min_length=1, max_length=2000)]
    schedule: ScheduleInput
    channel: NotificationChannel = NotificationChannel.TELEGRAM
    recurrence: RecurrenceInput | None = None
    task_id: str | None = Field(default=None, description="Optional link to an existing task.")
    notes: str | None = Field(default=None, max_length=4000)
    misfire_grace_seconds: Annotated[int, Field(ge=0, le=86400)] = Field(
        default=300,
        description=(
            "How late a missed reminder may still fire. Past that, it is marked missed "
            "rather than delivered at a confusing time."
        ),
    )


class ReminderUpdate(ApiModel):
    text: Annotated[str, Field(min_length=1, max_length=2000)] | None = None
    schedule: ScheduleInput | None = None
    channel: NotificationChannel | None = None
    notes: str | None = Field(default=None, max_length=4000)

    @model_validator(mode="after")
    def _at_least_one_field(self) -> Self:
        if not any(
            value is not None for value in (self.text, self.schedule, self.channel, self.notes)
        ):
            raise ValueError("Supply at least one field to update.")
        return self


class ReminderSnooze(ApiModel):
    delay: Delay = Field(default_factory=lambda: Delay(minutes=10))


class RecurrenceOut(ApiModel):
    frequency: str
    interval: int
    by_weekday: list[int] | None = None
    by_monthday: int | None = None
    at_hour: int
    at_minute: int
    timezone: str
    description: str
    ends_at: datetime | None = None
    max_occurrences: int | None = None
    occurrences_fired: int = 0


class ReminderOut(ApiModel):
    id: str
    text: str
    status: ReminderStatus
    channel: NotificationChannel
    timezone: str
    trigger_at: datetime
    trigger_at_local: str
    original_trigger_at: datetime
    created_at: datetime
    fired_at: datetime | None = None
    completed_at: datetime | None = None
    cancelled_at: datetime | None = None
    snooze_count: int = 0
    task_id: str | None = None
    notes: str | None = None
    recurrence: RecurrenceOut | None = None
    next_run_at: datetime | None = Field(
        default=None, description="What the scheduler currently has queued, if anything."
    )


class ReminderList(ApiModel):
    items: list[ReminderOut]
    total: int
    limit: int
    offset: int
