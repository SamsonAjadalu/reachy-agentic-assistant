#!/usr/bin/env python3
"""Remove demonstration data or reset the local database entirely.

    python scripts/reset_demo_data.py              # delete demo-tagged rows only
    python scripts/reset_demo_data.py --full       # delete all SQLite files and re-init

Stop the API before --full. Demo-tagged rows are those with detail/notes equal
to 'demo-seed' (see seed_demo_data.py).
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from sqlalchemy import delete, select  # noqa: E402

import database.models  # noqa: E402, F401
from app.config import Settings  # noqa: E402
from database.models import BriefingPreference, Reminder, Task, WardrobeItem  # noqa: E402
from database.session import create_engine, dispose_engine, session_scope, set_engine  # noqa: E402

DEMO_TAG = "demo-seed"


async def reset_demo(settings: Settings, *, full: bool) -> None:
    if full:
        await dispose_engine()
        paths = [
            settings.database_path,
            settings.app_data_dir / "scheduler.db",
            settings.document_index_path,
        ]
        for path in paths:
            if path is not None and path.exists():
                path.unlink()
                print(f"Removed {path}")
        from importlib.util import module_from_spec, spec_from_file_location

        init_path = REPO_ROOT / "scripts" / "init_database.py"
        spec = spec_from_file_location("init_database_script", init_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot load {init_path}")
        module = module_from_spec(spec)
        spec.loader.exec_module(module)
        module.main()
        return

    engine = create_engine(settings)
    set_engine(engine)
    try:
        async with session_scope() as session:
            for model, column in (
                (Task, Task.detail),
                (Reminder, Reminder.notes),
                (WardrobeItem, WardrobeItem.notes),
            ):
                result = await session.execute(delete(model).where(column == DEMO_TAG))
                print(f"Deleted {result.rowcount} row(s) from {model.__tablename__}")

            pref = (await session.scalars(select(BriefingPreference).limit(1))).first()
            if pref is not None:
                pref.sections_json = '["calendar","tasks","reminders","weather","outfit"]'
            await session.commit()
    finally:
        await dispose_engine()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Delete all SQLite databases and run init_database.py.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip confirmation for --full.",
    )
    args = parser.parse_args()

    settings = Settings()

    if args.full and not args.yes:
        print(f"This will delete all data under {settings.app_data_dir}.")
        reply = input("Type 'yes' to continue: ").strip()
        if reply != "yes":
            print("Aborted.")
            return 1

    asyncio.run(reset_demo(settings, full=args.full))
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
