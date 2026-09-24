#!/usr/bin/env python3
"""Create the data directories and bring the database to the latest migration.

python scripts/init_database.py
python scripts/init_database.py --check   # report status without changing anything
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from alembic import command  # noqa: E402
from alembic.config import Config  # noqa: E402
from alembic.runtime.migration import MigrationContext  # noqa: E402
from alembic.script import ScriptDirectory  # noqa: E402
from sqlalchemy import create_engine  # noqa: E402

from app.config import Settings  # noqa: E402
from database.session import check_integrity  # noqa: E402
from database.session import create_engine as create_async_engine  # noqa: E402


def alembic_config(settings: Settings) -> Config:
    config = Config(str(REPO_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(REPO_ROOT / "database" / "migrations"))
    config.set_main_option("sqlalchemy.url", settings.sync_database_url)
    return config


def current_revision(settings: Settings) -> str | None:
    engine = create_engine(settings.sync_database_url)
    try:
        with engine.connect() as connection:
            return MigrationContext.configure(connection).get_current_revision()
    finally:
        engine.dispose()


def head_revision(settings: Settings) -> str | None:
    return ScriptDirectory.from_config(alembic_config(settings)).get_current_head()


async def _integrity(settings: Settings) -> dict[str, object]:
    engine = create_async_engine(settings)
    try:
        return await check_integrity(engine)
    finally:
        await engine.dispose()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--check", action="store_true", help="Report status only.")
    args = parser.parse_args()

    settings = Settings()
    settings.ensure_directories()
    print(f"Data directory: {settings.app_data_dir}")
    print(f"Database:       {settings.sync_database_url}")

    head = head_revision(settings)
    current = current_revision(settings)

    if args.check:
        print(f"Head revision:    {head}")
        print(f"Current revision: {current or '(none - not initialised)'}")
        if current != head:
            print(
                "\nThe database is not at the latest revision. Run: python scripts/init_database.py"
            )
            return 1
        result = asyncio.run(_integrity(settings))
        print(f"Integrity:        {result['integrity_check']}")
        print(f"Journal mode:     {result['journal_mode']}")
        return 0 if result["ok"] else 1

    print(f"Upgrading {current or '(empty)'} -> {head}")
    command.upgrade(alembic_config(settings), "head")

    result = asyncio.run(_integrity(settings))
    print(f"Integrity check:  {result['integrity_check']}")
    print(f"Journal mode:     {result['journal_mode']}")
    print(f"Foreign keys:     {'on' if result['foreign_keys_enabled'] else 'OFF'}")
    print("\nDatabase ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
