"""Task and background-task request and response bodies."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Self

from pydantic import Field, model_validator

from app.schemas.common import ApiModel, RecurrenceInput
from app.schemas.reminders import RecurrenceOut
from shared.enums import BackgroundTaskStatus, TaskPriority, TaskStatus


class TaskCreate(ApiModel):
    description: Annotated[str, Field(min_length=1, max_length=2000)]
    detail: str | None = Field(default=None, max_length=8000)
    due_at: datetime | None = Field(
        default=None, description="ISO-8601 with an explicit offset or a 'Z' suffix."
    )
    priority: TaskPriority = TaskPriority.NORMAL
    tags: list[Annotated[str, Field(min_length=1, max_length=40)]] = Field(default_factory=list)
    recurrence: RecurrenceInput | None = None

    @model_validator(mode="after")
    def _limit_tags(self) -> Self:
        if len(self.tags) > 20:
            raise ValueError("A task may carry at most 20 tags.")
        return self


class TaskUpdate(ApiModel):
    description: Annotated[str, Field(min_length=1, max_length=2000)] | None = None
    detail: str | None = Field(default=None, max_length=8000)
    due_at: datetime | None = None
    clear_due_at: bool = False
    priority: TaskPriority | None = None
    status: TaskStatus | None = None
    tags: list[str] | None = None

    @model_validator(mode="after")
    def _at_least_one_field(self) -> Self:
        provided = [
            self.description,
            self.detail,
            self.due_at,
            self.priority,
            self.status,
            self.tags,
        ]
        if not any(value is not None for value in provided) and not self.clear_due_at:
            raise ValueError("Supply at least one field to update.")
        if self.due_at is not None and self.clear_due_at:
            raise ValueError("Set either due_at or clear_due_at, not both.")
        return self


class TaskOut(ApiModel):
    id: str
    description: str
    detail: str | None = None
    status: TaskStatus
    priority: TaskPriority
    due_at: datetime | None = None
    due_at_local: str | None = None
    is_overdue: bool = False
    tags: list[str] = Field(default_factory=list)
    created_at: datetime
    completed_at: datetime | None = None
    cancelled_at: datetime | None = None
    recurrence: RecurrenceOut | None = None
    reminder_ids: list[str] = Field(default_factory=list)


class TaskList(ApiModel):
    items: list[TaskOut]
    total: int
    limit: int
    offset: int


class BackgroundTaskEventOut(ApiModel):
    occurred_at: datetime
    event_type: str
    message: str | None = None


class BackgroundTaskOut(ApiModel):
    id: str
    task_type: str
    status: BackgroundTaskStatus
    progress: int
    progress_message: str | None = None
    created_at: datetime
    available_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    attempts: int
    max_attempts: int
    cancel_requested: bool
    result: dict[str, Any] | None = None
    result_path: str | None = None
    error_code: str | None = None
    error_message: str | None = None
    events: list[BackgroundTaskEventOut] = Field(default_factory=list)


class BackgroundTaskList(ApiModel):
    items: list[BackgroundTaskOut]
    total: int
    limit: int
    offset: int
