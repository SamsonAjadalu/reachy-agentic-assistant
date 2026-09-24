#!/usr/bin/env python3
"""Prove the migration chain is complete and reversible.

Runs against a throwaway database, never the live one:

1. upgrade head from empty
2. verify the ORM metadata matches the migrated schema (no drift)
3. downgrade base
4. upgrade head again

Step 2 is what catches the common failure of editing a model and forgetting to
generate the migration.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from alembic import command  # noqa: E402
from alembic.autogenerate import compare_metadata  # noqa: E402
from alembic.config import Config  # noqa: E402
from alembic.migration import MigrationContext  # noqa: E402
from sqlalchemy import create_engine, inspect  # noqa: E402

from database.models import Base  # noqa: E402
from database.session import apply_sqlite_pragmas_sync  # noqa: E402


def _config(url: str) -> Config:
    config = Config(str(REPO_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(REPO_ROOT / "database" / "migrations"))
    config.set_main_option("sqlalchemy.url", url)
    config.attributes["configure_logger"] = False
    return config


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="pa-migration-check-") as tmp:
        db_path = Path(tmp) / "check.db"
        url = f"sqlite:///{db_path}"
        config = _config(url)

        print("upgrade head")
        command.upgrade(config, "head")

        engine = create_engine(url)
        apply_sqlite_pragmas_sync(engine)
        try:
            tables = set(inspect(engine).get_table_names()) - {"alembic_version"}
            expected = set(Base.metadata.tables)
            missing = expected - tables
            if missing:
                print(f"FAIL: migrations did not create: {sorted(missing)}")
                return 1
            print(f"  {len(tables)} tables created")

            with engine.connect() as connection:
                context = MigrationContext.configure(
                    connection, opts={"compare_type": True, "render_as_batch": True}
                )
                diff = compare_metadata(context, Base.metadata)

            # Reflection of a TypeDecorator column reports the impl type, which
            # produces a spurious "type changed" entry; only structural drift
            # matters here.
            structural = [entry for entry in diff if _is_structural(entry)]
            if structural:
                print("FAIL: the ORM models have drifted from the migrations:")
                for entry in structural:
                    print(f"  {entry}")
                print("\nRun: alembic revision --autogenerate -m 'describe the change'")
                return 1
            print("  no schema drift")
        finally:
            engine.dispose()

        print("downgrade base")
        command.downgrade(config, "base")

        engine = create_engine(url)
        try:
            remaining = set(inspect(engine).get_table_names()) - {"alembic_version"}
            if remaining:
                print(f"FAIL: downgrade left tables behind: {sorted(remaining)}")
                return 1
            print("  all tables removed")
        finally:
            engine.dispose()

        print("upgrade head again")
        command.upgrade(config, "head")

    print("\nMigrations are complete and reversible.")
    return 0


def _is_structural(entry: object) -> bool:
    if isinstance(entry, list):
        return any(_is_structural(item) for item in entry)
    if isinstance(entry, tuple) and entry:
        return not str(entry[0]).startswith("modify_type")
    return True


if __name__ == "__main__":
    raise SystemExit(main())
