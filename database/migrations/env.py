"""Alembic environment.

Runs synchronously against the plain ``sqlite://`` form of the configured URL.
``render_as_batch`` is essential: SQLite cannot ALTER most constraints, so
Alembic has to rebuild the table, and batch mode is what makes downgrades work.
"""

from __future__ import annotations

import sys
from logging.config import fileConfig
from pathlib import Path
from typing import Any, Literal

from alembic import context
from sqlalchemy import create_engine, pool

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.config import get_settings  # noqa: E402
from database.models import Base  # noqa: E402
from database.session import apply_sqlite_pragmas_sync  # noqa: E402

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def render_item(type_: str, obj: Any, autogen_context: Any) -> str | Literal[False]:
    """Render custom column types as their plain SQLAlchemy equivalent.

    ``UtcDateTime`` is a TypeDecorator over ``DateTime``; emitting the decorator
    name would make every migration import application code, coupling the schema
    history to the current shape of the package.
    """
    from database.base import UtcDateTime

    if type_ == "type" and isinstance(obj, UtcDateTime):
        return "sa.DateTime()"
    return False


def _database_url() -> str:
    override = config.get_main_option("sqlalchemy.url")
    if override:
        return override
    return get_settings().sync_database_url


def run_migrations_offline() -> None:
    context.configure(
        url=_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=True,
        compare_type=True,
        compare_server_default=True,
        render_item=render_item,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    url = _database_url()
    if url.startswith("sqlite:///"):
        Path(url[len("sqlite:///") :]).parent.mkdir(parents=True, exist_ok=True)

    engine = create_engine(url, poolclass=pool.NullPool)
    if url.startswith("sqlite"):
        # Without WAL and synchronous=NORMAL, every batch-mode table rebuild
        # fsyncs, which turns a 30-table migration into minutes of disk waits.
        apply_sqlite_pragmas_sync(engine)

    with engine.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            render_as_batch=True,
            compare_type=True,
            compare_server_default=True,
            render_item=render_item,
        )
        with context.begin_transaction():
            context.run_migrations()
    engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
