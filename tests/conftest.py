"""Shared test fixtures.

Every test runs against a throwaway data directory and its own SQLite file, with
the scheduler and worker off unless a test asks for them. Tests never touch the
developer's real ``.env``: ``PA_ENV_FILE`` is pointed at a path that does not
exist before ``app.config`` is imported.
"""

from __future__ import annotations

import os
import sys
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ["PA_ENV_FILE"] = str(REPO_ROOT / "tests" / ".env.absent")
os.environ.setdefault("APP_ENV", "test")

from cryptography.fernet import Fernet  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from httpx import ASGITransport, AsyncClient  # noqa: E402
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession  # noqa: E402

from app.config import AppEnv, Settings  # noqa: E402
from app.logging_config import configure_logging  # noqa: E402
from database.session import (  # noqa: E402
    create_all,
    create_engine,
    dispose_engine,
    set_engine,
)
from security.redaction import clear_registered_secrets  # noqa: E402

TEST_API_TOKEN = "test-token-0123456789abcdef0123456789abcdef"


@pytest.fixture(scope="session", autouse=True)
def _configure_test_logging() -> None:
    configure_logging("WARNING", "console")


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "pa-data"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


@pytest.fixture
def secret_key_file(tmp_path: Path) -> Path:
    """A key file outside the data directory, mirroring the production layout."""
    key_dir = tmp_path / "secret-store"
    key_dir.mkdir(parents=True, exist_ok=True)
    path = key_dir / "secret.key"
    path.write_text(Fernet.generate_key().decode("ascii"), encoding="utf-8")
    path.chmod(0o600)
    return path


@pytest.fixture
def settings(data_dir: Path, secret_key_file: Path) -> Settings:
    return Settings(
        app_env=AppEnv.TEST,
        app_data_dir=data_dir,
        app_timezone="America/Toronto",
        app_log_level="WARNING",
        app_log_format="console",
        database_url=f"sqlite+aiosqlite:///{data_dir / 'test.db'}",
        pa_api_token=TEST_API_TOKEN,
        pa_secret_key_file=secret_key_file,
        pa_rate_limit_per_minute=0,
        mock_mode=True,
        scheduler_enabled=False,
        worker_enabled=False,
        telegram_enabled=False,
        google_enabled=False,
        notion_enabled=False,
        workstation_script_registry=REPO_ROOT / "tests" / "fixtures" / "registered_scripts.yaml",
    )


@pytest.fixture(autouse=True)
def _reset_redaction() -> Iterator[None]:
    clear_registered_secrets()
    yield
    clear_registered_secrets()


@pytest.fixture
async def engine(settings: Settings) -> AsyncIterator[AsyncEngine]:
    settings.ensure_directories()
    test_engine = create_engine(settings)
    set_engine(test_engine)
    await create_all(test_engine)
    try:
        yield test_engine
    finally:
        await dispose_engine()


@pytest.fixture
async def session(engine: AsyncEngine) -> AsyncIterator[AsyncSession]:
    from database.session import get_sessionmaker

    factory = get_sessionmaker()
    async with factory() as db_session:
        yield db_session
        await db_session.rollback()


@pytest.fixture
async def app(settings: Settings, engine: AsyncEngine) -> AsyncIterator[FastAPI]:
    """The real application, with lifespan startup executed.

    Uses the engine fixture's already-migrated database rather than letting
    lifespan build its own, so a test can seed rows before the first request.
    """
    from app.main import create_app

    application = create_app(settings)

    async with _lifespan_context(application, engine):
        yield application


class _lifespan_context:
    """Run the app's lifespan while keeping the test engine installed."""

    def __init__(self, application: FastAPI, engine: AsyncEngine) -> None:
        self._app = application
        self._engine = engine
        self._manager: object = None

    async def __aenter__(self) -> FastAPI:
        from app.lifespan import lifespan

        self._manager = lifespan(self._app)
        await self._manager.__aenter__()  # type: ignore[attr-defined]
        # lifespan installs a fresh engine; put the test one back so seeded data
        # and request data share a database.
        set_engine(self._engine)
        return self._app

    async def __aexit__(self, *exc_info: object) -> None:
        await self._manager.__aexit__(*exc_info)  # type: ignore[attr-defined,arg-type]


@pytest.fixture
async def client(app: FastAPI) -> AsyncIterator[AsyncClient]:
    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport,
        base_url="http://testserver",
        headers={"Authorization": f"Bearer {TEST_API_TOKEN}"},
    ) as http_client:
        yield http_client


@pytest.fixture
async def anonymous_client(app: FastAPI) -> AsyncIterator[AsyncClient]:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as http_client:
        yield http_client
