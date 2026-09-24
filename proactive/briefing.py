"""Assembling and sending the daily briefing.

Assembly is deterministic: the same database and the same fixtures produce the
same briefing, which is what makes it testable at all. Section order follows the
owner's configured list, and a section that cannot be built becomes a note
rather than an exception.

Quiet hours suppress delivery, not assembly. The briefing is still recorded, so
"what did I miss?" has an answer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings
from app.logging_config import get_logger
from database.models import BriefingPreference, OwnerProfile
from notifications.dispatcher import dispatch
from proactive.formatters import get_formatter
from proactive.sections import (
    BriefingContext,
    Section,
    available_sections,
    build_section,
    default_sections,
)
from shared.enums import AlertSeverity, NotificationChannel
from shared.timeutils import local_today, to_local, utcnow

logger = get_logger(__name__)


@dataclass
class BriefingSettings:
    """The owner's briefing preferences, with defaults for a fresh install."""

    enabled: bool = True
    send_hour: int = 7
    send_minute: int = 0
    timezone: str = "America/Toronto"
    sections: list[str] = field(default_factory=default_sections)
    channels: list[str] = field(default_factory=lambda: [NotificationChannel.TELEGRAM.value])
    quiet_hours_start: int | None = None
    quiet_hours_end: int | None = None
    formatter: str = "template"

    def in_quiet_hours(self, moment: datetime) -> bool:
        """Whether local ``moment`` falls inside the do-not-disturb window.

        The window is allowed to wrap midnight, which is the common case: 22:00
        to 07:00 is one interval, not two.
        """
        if self.quiet_hours_start is None or self.quiet_hours_end is None:
            return False
        hour = to_local(moment, self.timezone).hour
        start, end = self.quiet_hours_start, self.quiet_hours_end
        if start == end:
            return False
        if start < end:
            return start <= hour < end
        return hour >= start or hour < end


@dataclass
class Briefing:
    generated_at: datetime
    local_date: str
    greeting: str
    sections: list[Section]
    text: str
    formatter: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at.isoformat(),
            "local_date": self.local_date,
            "greeting": self.greeting,
            "formatter": self.formatter,
            "text": self.text,
            "sections": [
                {
                    "name": section.name,
                    "title": section.title,
                    "summary": section.summary,
                    "items": section.items,
                    "available": section.available,
                    "note": section.note,
                }
                for section in self.sections
            ],
        }


async def load_preferences(session: AsyncSession, settings: Settings) -> BriefingSettings:
    """Read the owner's preferences, falling back to sane defaults."""
    row = (await session.scalars(select(BriefingPreference).limit(1))).first()
    if row is None:
        return BriefingSettings(timezone=settings.app_timezone)

    return BriefingSettings(
        enabled=row.enabled,
        send_hour=row.send_hour,
        send_minute=row.send_minute,
        timezone=row.timezone or settings.app_timezone,
        sections=_json_list(row.sections_json) or default_sections(),
        channels=_json_list(row.channels_json) or [NotificationChannel.TELEGRAM.value],
        quiet_hours_start=row.quiet_hours_start,
        quiet_hours_end=row.quiet_hours_end,
        formatter=row.formatter or "template",
    )


async def save_preferences(
    session: AsyncSession, updates: dict[str, Any], settings: Settings
) -> BriefingSettings:
    """Update the stored preferences, creating the row and owner if needed."""
    unknown = sorted(set(updates.get("sections") or []) - set(available_sections()))
    if unknown:
        from shared.errors import ValidationError

        raise ValidationError(f"Unknown briefing section(s): {', '.join(unknown)}.")

    row = (await session.scalars(select(BriefingPreference).limit(1))).first()
    if row is None:
        owner = (await session.scalars(select(OwnerProfile).limit(1))).first()
        if owner is None:
            owner = OwnerProfile(display_name="Owner", timezone=settings.app_timezone)
            session.add(owner)
            await session.flush()
        row = BriefingPreference(owner_id=owner.id, timezone=settings.app_timezone)
        session.add(row)

    for key in ("enabled", "send_hour", "send_minute", "timezone", "formatter"):
        if key in updates and updates[key] is not None:
            setattr(row, key, updates[key])
    if "sections" in updates and updates["sections"] is not None:
        row.sections_json = json.dumps(list(updates["sections"]))
    if "channels" in updates and updates["channels"] is not None:
        row.channels_json = json.dumps(list(updates["channels"]))
    for key in ("quiet_hours_start", "quiet_hours_end"):
        if key in updates:
            setattr(row, key, updates[key])

    await session.flush()
    return await load_preferences(session, settings)


async def assemble(
    session: AsyncSession,
    settings: Settings,
    *,
    preferences: BriefingSettings | None = None,
    sections: list[str] | None = None,
    formatter_name: str | None = None,
) -> Briefing:
    """Build the briefing. Never raises for a section that cannot be produced."""
    preferences = preferences or await load_preferences(session, settings)
    wanted = sections if sections is not None else preferences.sections
    now = utcnow()
    today = local_today(preferences.timezone)

    context = BriefingContext(
        session=session,
        settings=settings,
        timezone=preferences.timezone,
        today=today,
        now=now,
    )

    built = [await build_section(name, context) for name in wanted]
    greeting = _greeting(to_local(now, preferences.timezone))
    formatter = get_formatter(formatter_name or preferences.formatter)

    return Briefing(
        generated_at=now,
        local_date=today.isoformat(),
        greeting=greeting,
        sections=built,
        text=formatter.render(greeting, built),
        formatter=formatter.name,
    )


async def send(
    session: AsyncSession,
    settings: Settings,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Assemble and deliver the briefing on every configured channel."""
    preferences = await load_preferences(session, settings)
    if not preferences.enabled and not force:
        return {"sent": False, "reason": "briefings_disabled"}

    now = utcnow()
    if preferences.in_quiet_hours(now) and not force:
        # Assembled anyway so the record exists; simply not delivered.
        briefing = await assemble(session, settings, preferences=preferences)
        logger.info("Briefing suppressed during quiet hours")
        return {"sent": False, "reason": "quiet_hours", "briefing": briefing.as_dict()}

    briefing = await assemble(session, settings, preferences=preferences)
    spoken = await assemble(session, settings, preferences=preferences, formatter_name="spoken")

    delivered: list[str] = []
    for channel_name in preferences.channels:
        try:
            channel = NotificationChannel(channel_name)
        except ValueError:
            logger.warning("Unknown briefing channel", extra={"channel": channel_name})
            continue

        await dispatch(
            session,
            kind="briefing.daily",
            title=f"Daily briefing for {briefing.local_date}",
            body=briefing.text,
            severity=AlertSeverity.INFO,
            channel=channel,
            dedup_key=f"briefing:{briefing.local_date}:{channel.value}",
            settings=settings,
            spoken_text=spoken.text,
        )
        delivered.append(channel.value)

    return {"sent": bool(delivered), "channels": delivered, "briefing": briefing.as_dict()}


def _greeting(local_now: datetime) -> str:
    hour = local_now.hour
    if hour < 12:
        part = "Good morning"
    elif hour < 18:
        part = "Good afternoon"
    else:
        part = "Good evening"
    return f"{part}. Here is your briefing for {local_now.strftime('%A %d %B')}."


def _json_list(raw: str | None) -> list[str]:
    if not raw:
        return []
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return []
    return [str(entry) for entry in value] if isinstance(value, list) else []
