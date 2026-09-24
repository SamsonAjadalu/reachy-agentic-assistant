"""Engine and session management.

SQLite needs three things configured explicitly or it will quietly misbehave:
WAL mode (so a read during a write does not block), ``foreign_keys=ON`` (off by
default, which would silently accept orphan rows), and a busy timeout (so a
concurrent writer waits instead of raising "database is locked").
"""

from __future__ import annotations

import sqlite3
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.config import Settings, get_settings
from app.logging_config import get_logger
from database.base import Base

logger = get_logger(__name__)

_engine: AsyncEngine | None = None
_sessionmaker: async_sessionmaker[AsyncSession] | None = None

BUSY_TIMEOUT_MS = 10_000


def _apply_sqlite_pragmas(dbapi_connection: Any, _record: Any) -> None:
    """Apply the pragmas SQLite does not enable by default.

    The connection handed to this hook is the driver's own object - a raw
    ``sqlite3.Connection`` under pysqlite, but an adapter wrapper under
    aiosqlite - so it is used duck-typed rather than isinstance-checked.
    """
    cursor = dbapi_connection.cursor()
    try:
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS}")
        # NORMAL is durable under WAL for anything short of an OS crash, and
        # avoids an fsync on every commit for a workload that commits often.
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA temp_store=MEMORY")
    finally:
        cursor.close()


def apply_sqlite_pragmas_sync(engine: Any) -> None:
    """Attach the same pragmas to a synchronous engine (Alembic, backups, FTS)."""
    event.listen(engine, "connect", _apply_sqlite_pragmas)


def create_engine(settings: Settings | None = None, *, echo: bool = False) -> AsyncEngine:
    settings = settings or get_settings()
    url = settings.database_url

    if url.startswith("sqlite"):
        db_path = settings.database_path
        if db_path is not None:
            db_path.parent.mkdir(parents=True, exist_ok=True)

    engine = create_async_engine(
        url,
        echo=echo,
        future=True,
        # NullPool would reopen the file per request; the default pool with
        # pre-ping keeps a warm connection without holding stale handles.
        pool_pre_ping=True,
        connect_args={"timeout": BUSY_TIMEOUT_MS / 1000} if url.startswith("sqlite") else {},
    )
    if url.startswith("sqlite"):
        event.listen(engine.sync_engine, "connect", _apply_sqlite_pragmas)
    return engine


def get_engine() -> AsyncEngine:
    global _engine
    if _engine is None:
        _engine = create_engine()
    return _engine


def get_sessionmaker() -> async_sessionmaker[AsyncSession]:
    global _sessionmaker
    if _sessionmaker is None:
        _sessionmaker = async_sessionmaker(
            get_engine(),
            expire_on_commit=False,
            autoflush=False,
        )
    return _sessionmaker


def set_engine(engine: AsyncEngine) -> None:
    """Install a specific engine. Used by tests and by the worker process."""
    global _engine, _sessionmaker
    _engine = engine
    _sessionmaker = async_sessionmaker(engine, expire_on_commit=False, autoflush=False)


async def dispose_engine() -> None:
    global _engine, _sessionmaker
    if _engine is not None:
        await _engine.dispose()
    _engine = None
    _sessionmaker = None


@asynccontextmanager
async def session_scope() -> AsyncIterator[AsyncSession]:
    """Transactional session for use outside the request lifecycle."""
    factory = get_sessionmaker()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except BaseException:
            await session.rollback()
            raise


async def create_all(engine: AsyncEngine | None = None) -> None:
    """Create the schema directly.

    Used by tests and ``scripts/init_database.py``. Production schema changes go
    through Alembic.
    """
    target = engine or get_engine()
    async with target.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def check_integrity(engine: AsyncEngine | None = None) -> dict[str, Any]:
    """Run SQLite's own consistency checks."""
    target = engine or get_engine()
    async with target.connect() as conn:
        integrity = (await conn.execute(text("PRAGMA integrity_check"))).scalar_one()
        foreign_keys = (await conn.execute(text("PRAGMA foreign_key_check"))).fetchall()
        journal_mode = (await conn.execute(text("PRAGMA journal_mode"))).scalar_one()
        fk_enabled = (await conn.execute(text("PRAGMA foreign_keys"))).scalar_one()
    return {
        "integrity_check": integrity,
        "ok": integrity == "ok" and not foreign_keys,
        "foreign_key_violations": len(foreign_keys),
        "journal_mode": journal_mode,
        "foreign_keys_enabled": bool(fk_enabled),
    }


def sqlite_backup(source: Path, destination: Path) -> None:
    """Consistent online backup using SQLite's own backup API.

    Copying the file with ``cp`` while the service is running can capture a torn
    page or miss the WAL; ``Connection.backup`` handles both correctly.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    src = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
    dst = sqlite3.connect(str(destination))
    try:
        with dst:
            src.backup(dst)
    finally:
        src.close()
        dst.close()
    destination.chmod(0o600)
