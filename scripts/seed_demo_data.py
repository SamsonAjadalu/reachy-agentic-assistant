#!/usr/bin/env python3
"""Seed demonstration data for local walkthroughs and mock-mode testing.

    python scripts/seed_demo_data.py
    python scripts/seed_demo_data.py --reset-first

Creates an owner profile, briefing preferences, sample tasks and reminders, and
a few wardrobe items. Idempotent: existing demo rows (tagged in notes) are
skipped unless --reset-first is passed.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from sqlalchemy import delete, select  # noqa: E402

import database.models  # noqa: E402, F401
from app.config import Settings  # noqa: E402
from app.schemas.common import ScheduleInput  # noqa: E402
from app.schemas.reminders import ReminderCreate  # noqa: E402
from app.schemas.tasks import TaskCreate  # noqa: E402
from app.schemas.wardrobe import ItemCreate  # noqa: E402
from app.services import reminders as reminder_service  # noqa: E402
from app.services import tasks as task_service  # noqa: E402
from app.services import wardrobe as wardrobe_service  # noqa: E402
from database.models import OwnerProfile, Reminder, Task, WardrobeItem  # noqa: E402
from database.session import (  # noqa: E402
    create_all,
    create_engine,
    dispose_engine,
    session_scope,
    set_engine,
)
from proactive.briefing import save_preferences  # noqa: E402
from shared.enums import TaskPriority  # noqa: E402
from shared.timeutils import utcnow  # noqa: E402

DEMO_TAG = "demo-seed"


async def _clear_demo_rows(settings: Settings) -> None:
    engine = create_engine(settings)
    set_engine(engine)
    try:
        async with session_scope() as session:
            for model, column in (
                (Task, Task.detail),
                (Reminder, Reminder.notes),
                (WardrobeItem, WardrobeItem.notes),
            ):
                await session.execute(delete(model).where(column == DEMO_TAG))
            await session.commit()
    finally:
        await dispose_engine()


def _load_briefing_yaml(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    import yaml

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return raw if isinstance(raw, dict) else None


async def seed(settings: Settings, *, reset_first: bool) -> dict[str, int]:
    if reset_first:
        await _clear_demo_rows(settings)

    counts = {"tasks": 0, "reminders": 0, "wardrobe_items": 0}

    async with session_scope() as session:
        owner = (await session.scalars(select(OwnerProfile).limit(1))).first()
        if owner is None:
            owner = OwnerProfile(display_name="Demo Owner", timezone=settings.app_timezone)
            session.add(owner)
            await session.flush()

        briefing_yaml = _load_briefing_yaml(settings.briefing_config_path)
        briefing_updates: dict[str, object] = {
            "enabled": True,
            "send_hour": 7,
            "send_minute": 30,
            "sections": ["calendar", "tasks", "reminders", "weather", "outfit"],
            "channels": ["telegram"],
        }
        if briefing_yaml:
            briefing_updates.update({k: v for k, v in briefing_yaml.items() if k != "notes"})
        await save_preferences(session, briefing_updates, settings)

        existing_tasks = (await session.scalars(select(Task).where(Task.detail == DEMO_TAG))).all()
        if not existing_tasks:
            samples = [
                TaskCreate(
                    description="Review lab notes", detail=DEMO_TAG, priority=TaskPriority.NORMAL
                ),
                TaskCreate(
                    description="Prepare Monday stand-up talking points",
                    detail=DEMO_TAG,
                    priority=TaskPriority.HIGH,
                ),
            ]
            for payload in samples:
                await task_service.create_task(session, payload, settings=settings)
                counts["tasks"] += 1

        existing_reminders = (
            await session.scalars(select(Reminder).where(Reminder.notes == DEMO_TAG))
        ).all()
        if not existing_reminders:
            trigger = utcnow() + timedelta(hours=2)
            payload = ReminderCreate(
                text="Demo reminder: check the weather before leaving",
                schedule=ScheduleInput(at=trigger),
                notes=DEMO_TAG,
            )
            await reminder_service.create_reminder(session, payload, settings=settings)
            counts["reminders"] += 1

        existing_items = (
            await session.scalars(select(WardrobeItem).where(WardrobeItem.notes == DEMO_TAG))
        ).all()
        if not existing_items:
            items = [
                ItemCreate(
                    name="Blue Oxford shirt",
                    category="top",
                    primary_color="blue",
                    seasons=["all"],
                    notes=DEMO_TAG,
                ),
                ItemCreate(
                    name="Grey chinos",
                    category="bottom",
                    primary_color="grey",
                    seasons=["spring", "fall"],
                    notes=DEMO_TAG,
                ),
            ]
            for item in items:
                await wardrobe_service.create_item(session, item)
                counts["wardrobe_items"] += 1

        await session.commit()

    return counts


async def _run(settings: Settings, reset_first: bool) -> int:
    settings.ensure_directories()
    engine = create_engine(settings)
    set_engine(engine)
    await create_all(engine)
    try:
        counts = await seed(settings, reset_first=reset_first)
    finally:
        await dispose_engine()

    print(f"Demo data ready under {settings.app_data_dir}")
    for key, value in counts.items():
        print(f"  {key}: {value} created")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--reset-first",
        action="store_true",
        help="Clear demo-tagged rows before seeding.",
    )
    args = parser.parse_args()
    return asyncio.run(_run(Settings(), reset_first=args.reset_first))


if __name__ == "__main__":
    raise SystemExit(main())
