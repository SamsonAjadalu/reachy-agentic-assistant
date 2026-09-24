"""Provider selection.

One place decides whether a caller gets the real integration or its mock, based
on configuration rather than on the caller's knowledge. Two consequences that
matter: a service that is switched off fails with a clear, actionable error
instead of a stack trace, and no request handler ever contains
``if settings.mock_mode``.

Instances are cached per process because the real providers hold an HTTP
connection pool and a shared OAuth token, and building one per request would
mean a token refresh per request.
"""

from __future__ import annotations

from typing import Any

from app.config import Settings, get_settings
from app.logging_config import get_logger
from integrations.google.protocols import (
    CalendarProvider,
    ContactsProvider,
    DriveProvider,
    GmailProvider,
)
from shared.errors import IntegrationDisabledError

logger = get_logger(__name__)

_cache: dict[str, Any] = {}
_overrides: dict[str, Any] = {}


def set_provider_override(name: str, provider: Any | None) -> None:
    """Install a specific provider instance. Used by tests and the demo CLI."""
    if provider is None:
        _overrides.pop(name, None)
    else:
        _overrides[name] = provider


def reset_providers() -> None:
    _cache.clear()
    _overrides.clear()


def _resolve(name: str, settings: Settings, build_real: Any, build_mock: Any) -> Any:
    if name in _overrides:
        return _overrides[name]

    use_mock = settings.mock_mode or not _is_enabled(name, settings)
    cache_key = f"{name}:{'mock' if use_mock else 'real'}"
    if cache_key not in _cache:
        _cache[cache_key] = build_mock() if use_mock else build_real()
        logger.debug("Built provider", extra={"provider": name, "mock": use_mock})
    return _cache[cache_key]


def _is_enabled(name: str, settings: Settings) -> bool:
    if name in {"gmail", "calendar", "contacts", "drive"}:
        return settings.google_enabled
    if name == "notion":
        return settings.notion_enabled
    return True


def require_enabled(name: str, settings: Settings | None = None) -> None:
    """Refuse a write that has nowhere to go.

    Mock mode is deliberately exempt: the mock providers perform the write
    against their in-memory state, which is what makes an end-to-end demo and
    the test suite meaningful.
    """
    settings = settings or get_settings()
    if not settings.mock_mode and not _is_enabled(name, settings):
        raise IntegrationDisabledError(
            f"The {name} integration is disabled. Set the matching *_ENABLED flag and "
            "supply credentials to use it."
        )


def get_gmail(settings: Settings | None = None) -> GmailProvider:
    settings = settings or get_settings()

    def build_real() -> Any:
        from integrations.google.gmail import GmailService

        return GmailService(settings=settings)

    def build_mock() -> Any:
        from integrations.google.mock import MockGmailService

        return MockGmailService()

    provider: GmailProvider = _resolve("gmail", settings, build_real, build_mock)
    return provider


def get_calendar(settings: Settings | None = None) -> CalendarProvider:
    settings = settings or get_settings()

    def build_real() -> Any:
        from integrations.google.calendar import CalendarService

        return CalendarService(settings=settings)

    def build_mock() -> Any:
        from integrations.google.mock import MockCalendarService

        return MockCalendarService()

    provider: CalendarProvider = _resolve("calendar", settings, build_real, build_mock)
    return provider


def get_contacts(settings: Settings | None = None) -> ContactsProvider:
    settings = settings or get_settings()

    def build_real() -> Any:
        from integrations.google.contacts import ContactsService

        return ContactsService(settings=settings)

    def build_mock() -> Any:
        from integrations.google.mock import MockContactsService

        return MockContactsService()

    provider: ContactsProvider = _resolve("contacts", settings, build_real, build_mock)
    return provider


def get_weather(settings: Settings | None = None) -> Any:
    """Weather has no enable flag: it always works, on the mock if need be."""
    from app.config import WeatherProvider

    settings = settings or get_settings()

    def build_real() -> Any:
        from integrations.weather.real import OpenMeteoProvider

        return OpenMeteoProvider(settings)

    def build_mock() -> Any:
        from integrations.weather.mock import MockWeatherProvider

        return MockWeatherProvider(settings)

    if settings.weather_provider is WeatherProvider.MOCK:
        if "weather" in _overrides:
            return _overrides["weather"]
        if "weather:mock" not in _cache:
            _cache["weather:mock"] = build_mock()
        return _cache["weather:mock"]

    return _resolve("weather", settings, build_real, build_mock)


def get_notion(settings: Settings | None = None) -> Any:
    settings = settings or get_settings()

    def build_real() -> Any:
        from integrations.notion.real import NotionClient

        return NotionClient(settings)

    def build_mock() -> Any:
        from integrations.notion.mock import MockNotionClient

        return MockNotionClient(settings)

    return _resolve("notion", settings, build_real, build_mock)


def get_drive(settings: Settings | None = None) -> DriveProvider:
    settings = settings or get_settings()

    def build_real() -> Any:
        from integrations.google.drive import DriveService

        return DriveService(settings=settings)

    def build_mock() -> Any:
        from integrations.google.mock import MockDriveService

        return MockDriveService()

    provider: DriveProvider = _resolve("drive", settings, build_real, build_mock)
    return provider
