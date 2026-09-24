"""Briefing section builders.

Each builder answers one question ("what is on today?", "what needs washing?")
and returns a structured section. Nothing here formats text: a section carries
facts and a one-line spoken summary, and the formatter decides how to present
them. That separation is what lets the same briefing go to Telegram as Markdown
and to Reachy as something a person would say out loud.

A section that cannot be built - an integration that is off, a provider that is
down - degrades to a note rather than failing the briefing. A morning summary
that is missing the weather is still useful; one that fails entirely is not.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings
from app.logging_config import get_logger
from database.models import Reminder, Task, WardrobeItem
from shared.enums import LaundryStatus, ReminderStatus, TaskStatus
from shared.timeutils import end_of_local_day, local_today, start_of_local_day, to_local, utcnow

logger = get_logger(__name__)

MAX_ITEMS_PER_SECTION = 8


@dataclass
class Section:
    """One part of a briefing."""

    name: str
    title: str
    items: list[dict[str, Any]] = field(default_factory=list)
    summary: str = ""
    """A single sentence suitable for speech."""
    available: bool = True
    note: str | None = None
    """Why the section is empty or unavailable, when that is worth saying."""

    @property
    def is_empty(self) -> bool:
        return not self.items


@dataclass
class BriefingContext:
    session: AsyncSession
    settings: Settings
    timezone: str
    today: date
    now: Any


SectionBuilder = Callable[[BriefingContext], Awaitable[Section]]
_BUILDERS: dict[str, SectionBuilder] = {}


def section(name: str) -> Callable[[SectionBuilder], SectionBuilder]:
    def decorator(func: SectionBuilder) -> SectionBuilder:
        _BUILDERS[name] = func
        return func

    return decorator


def available_sections() -> list[str]:
    return sorted(_BUILDERS)


async def build_section(name: str, context: BriefingContext) -> Section:
    """Build one section, converting any provider failure into a note."""
    builder = _BUILDERS.get(name)
    if builder is None:
        return Section(
            name=name,
            title=name.title(),
            available=False,
            note=f"No builder is registered for the '{name}' section.",
        )

    try:
        return await builder(context)
    except Exception as exc:
        logger.warning(
            "A briefing section could not be built",
            extra={"section": name, "reason": f"{type(exc).__name__}: {exc}"},
        )
        return Section(
            name=name,
            title=name.title(),
            available=False,
            note="This part could not be retrieved just now.",
        )


# ------------------------------------------------------------------ builders
@section("calendar")
async def _calendar(context: BriefingContext) -> Section:
    from integrations.registry import get_calendar

    if not (context.settings.google_enabled or context.settings.mock_mode):
        return Section(
            name="calendar",
            title="Calendar",
            available=False,
            note="Google Calendar is not connected.",
        )

    start = start_of_local_day(context.now, context.timezone)
    end = end_of_local_day(context.now, context.timezone)
    events = await get_calendar(context.settings).list_events(
        starts_after=start, ends_before=end, limit=MAX_ITEMS_PER_SECTION
    )

    items = [
        {
            "id": event.id,
            "title": event.title,
            "starts_at": event.starts_at.isoformat(),
            "local_time": (
                "All day"
                if event.all_day
                else to_local(event.starts_at, context.timezone).strftime("%H:%M")
            ),
            "location": event.location,
            "all_day": event.all_day,
        }
        for event in events
    ]

    if not items:
        return Section(name="calendar", title="Calendar", summary="Nothing in the calendar today.")

    first = items[0]
    summary = (
        f"{len(items)} event{'s' if len(items) != 1 else ''} today, "
        f"starting with {first['title']} at {first['local_time']}."
    )
    return Section(name="calendar", title="Calendar", items=items, summary=summary)


@section("tasks")
async def _tasks(context: BriefingContext) -> Section:
    end = end_of_local_day(context.now, context.timezone)
    rows = list(
        await context.session.scalars(
            select(Task)
            .where(
                Task.status.in_([TaskStatus.OPEN.value, TaskStatus.IN_PROGRESS.value]),
                Task.due_at.is_not(None),
                Task.due_at <= end,
            )
            .order_by(Task.due_at)
            .limit(MAX_ITEMS_PER_SECTION)
        )
    )

    overdue = [task for task in rows if task.due_at and task.due_at < context.now]
    items = [
        {
            "id": task.id,
            "title": task.description,
            "due_at": task.due_at.isoformat() if task.due_at else None,
            "priority": task.priority,
            "overdue": bool(task.due_at and task.due_at < context.now),
        }
        for task in rows
    ]

    if not items:
        return Section(name="tasks", title="Tasks", summary="No tasks are due today.")

    summary = f"{len(items)} task{'s' if len(items) != 1 else ''} due today"
    if overdue:
        summary += f", {len(overdue)} already overdue"
    return Section(name="tasks", title="Tasks", items=items, summary=summary + ".")


@section("reminders")
async def _reminders(context: BriefingContext) -> Section:
    end = end_of_local_day(context.now, context.timezone)
    rows = list(
        await context.session.scalars(
            select(Reminder)
            .where(
                Reminder.status.in_([ReminderStatus.ACTIVE.value, ReminderStatus.SNOOZED.value]),
                Reminder.trigger_at >= context.now,
                Reminder.trigger_at <= end,
            )
            .order_by(Reminder.trigger_at)
            .limit(MAX_ITEMS_PER_SECTION)
        )
    )

    items = [
        {
            "id": reminder.id,
            "title": reminder.text,
            "trigger_at": reminder.trigger_at.isoformat(),
            "local_time": to_local(reminder.trigger_at, context.timezone).strftime("%H:%M"),
        }
        for reminder in rows
    ]

    if not items:
        return Section(name="reminders", title="Reminders", summary="No reminders left today.")
    return Section(
        name="reminders",
        title="Reminders",
        items=items,
        summary=f"{len(items)} reminder{'s' if len(items) != 1 else ''} still to come today.",
    )


@section("weather")
async def _weather(context: BriefingContext) -> Section:
    from integrations.registry import get_weather
    from integrations.weather.advice import advise

    report = await get_weather(context.settings).get_weather(days=2)
    guidance = advise(report)

    today = report.daily[0] if report.daily else None
    items = (
        [
            {
                "date": today.day.isoformat(),
                "high_c": today.high_c,
                "low_c": today.low_c,
                "condition": today.condition.value,
                "description": today.description,
                "precipitation_probability": today.precipitation_probability,
            }
        ]
        if today
        else []
    )

    if today is None:
        return Section(name="weather", title="Weather", summary="No forecast is available.")

    summary = (
        f"{today.description}, {today.high_c:.0f} degrees at the warmest "
        f"and {today.low_c:.0f} at the coldest."
    )
    return Section(
        name="weather",
        title="Weather",
        items=items,
        summary=summary,
        note=guidance.summary(),
    )


@section("outfit")
async def _outfit(context: BriefingContext) -> Section:
    from app.services import wardrobe as wardrobe_service
    from integrations.registry import get_weather
    from integrations.weather.advice import advise
    from wardrobe.recommend import rank

    outfits, _total = await wardrobe_service.list_outfits(context.session, limit=100)
    if not outfits:
        return Section(
            name="outfit",
            title="Outfit",
            summary="No outfits are saved yet, so there is nothing to suggest.",
        )

    report = await get_weather(context.settings).get_weather(days=1)
    guidance = advise(report)
    day = report.daily[0] if report.daily else None
    temperature = (day.high_c + day.low_c) / 2 if day else report.current.feels_like_c

    views = [await wardrobe_service.to_outfit_view(context.session, outfit) for outfit in outfits]
    ranked = rank(views, temperature_c=temperature, advice=guidance, today=context.today)
    if not ranked:
        return Section(
            name="outfit",
            title="Outfit",
            summary="Nothing in the wardrobe is available and clean today.",
        )

    best = ranked[0]
    items = [scored.as_dict() for scored in ranked]
    return Section(
        name="outfit",
        title="Outfit",
        items=items,
        summary=f"Suggested outfit: {best.outfit_name}.",
        note=best.explain(),
    )


@section("laundry")
async def _laundry(context: BriefingContext) -> Section:
    rows = list(
        await context.session.scalars(
            select(WardrobeItem).where(
                WardrobeItem.laundry_status.in_(
                    [LaundryStatus.IN_LAUNDRY.value, LaundryStatus.WORN.value]
                )
            )
        )
    )
    items = [{"id": item.id, "name": item.name} for item in rows[:MAX_ITEMS_PER_SECTION]]
    if not items:
        return Section(name="laundry", title="Laundry", summary="Nothing is in the wash.")
    return Section(
        name="laundry",
        title="Laundry",
        items=items,
        summary=(
            f"{len(rows)} garment{'s' if len(rows) != 1 else ''} "
            f"{'are' if len(rows) != 1 else 'is'} worn or in the laundry."
        ),
    )


@section("email")
async def _email(context: BriefingContext) -> Section:
    from integrations.registry import get_gmail

    if not (context.settings.google_enabled or context.settings.mock_mode):
        return Section(name="email", title="Email", available=False, note="Gmail is not connected.")

    messages = await get_gmail(context.settings).list_messages(
        unread_only=True, limit=MAX_ITEMS_PER_SECTION
    )
    items = [
        {
            "id": message.id,
            "from": message.sender_name or message.sender,
            "subject": message.subject,
            "received_at": message.received_at.isoformat(),
            "important": message.is_important,
        }
        for message in messages
    ]

    if not items:
        return Section(name="email", title="Email", summary="No unread email.")
    return Section(
        name="email",
        title="Email",
        items=items,
        summary=f"{len(items)} unread message{'s' if len(items) != 1 else ''}.",
        # Sender names and subjects are attacker-controlled text.
        note="Sender and subject text comes from external senders.",
    )


@section("workstation")
async def _workstation(context: BriefingContext) -> Section:
    from workstation.status import collect_status, summarise

    status = await collect_status(context.settings)
    items = [
        {
            "cpu_percent": status.cpu_percent,
            "memory_percent": status.memory_percent,
            "disks": [disk.model_dump(mode="json") for disk in status.disks],
        }
    ]
    return Section(
        name="workstation",
        title="Workstation",
        items=items,
        summary=summarise(status),
        note="; ".join(status.warnings) if status.warnings else None,
    )


def default_sections() -> list[str]:
    return ["calendar", "tasks", "reminders", "weather", "outfit"]


def yesterdays_window(today: date) -> tuple[date, date]:
    return today - timedelta(days=1), today


def now_context(session: AsyncSession, settings: Settings, timezone: str) -> BriefingContext:
    return BriefingContext(
        session=session,
        settings=settings,
        timezone=timezone,
        today=local_today(timezone),
        now=utcnow(),
    )
