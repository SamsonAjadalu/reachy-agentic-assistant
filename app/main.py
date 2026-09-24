"""FastAPI application factory."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi

from app.api.v1 import (
    actions,
    alerts,
    approvals,
    background_tasks,
    briefings,
    calendar,
    contacts,
    documents,
    drive,
    gmail,
    notion,
    reachy,
    reminders,
    system,
    tasks,
    vision,
    wardrobe,
    weather,
    workstation,
)
from app.config import Settings, get_settings
from app.error_handlers import register_error_handlers
from app.lifespan import lifespan
from app.middleware.context import (
    BodySizeLimitMiddleware,
    RateLimitMiddleware,
    RequestContextMiddleware,
    SecurityHeadersMiddleware,
)
from app.middleware.network import NetworkAllowlistMiddleware

# Probes must answer even when the caller is off-network and unauthenticated, so
# systemd and a remote monitor can distinguish "down" from "blocked".
PUBLIC_PATHS = frozenset({"/health", "/ready"})

DESCRIPTION = """
Reachy Agentic Assistant is a workstation service that extends Reachy Mini with
Gmail, Calendar, Telegram, Notion, scheduling, memory, and visual-perception
tools.

Every endpoint requires a bearer token except `/health` and `/ready`. Fast read
operations answer synchronously within a bounded timeout; only genuinely
long-running work returns a task ticket and completes in the background.

Any externally visible or destructive action creates a pending action, shows the
exact payload for approval over Telegram, and performs the side effect only
after that approval is granted.
"""


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    app = FastAPI(
        title="Reachy Personal Assistant API",
        version="0.1.0",
        description=DESCRIPTION,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url=None,
        openapi_url="/openapi.json",
    )
    app.state.settings = settings
    # Bind the dependency to this instance's settings rather than the cached
    # global, so a test or an embedded instance can run with its own config.
    app.dependency_overrides[get_settings] = lambda: settings

    # Starlette runs middleware in reverse registration order, so the context
    # middleware is added last to make the request id available to all the others.
    app.add_middleware(
        RateLimitMiddleware,
        requests_per_minute=settings.pa_rate_limit_per_minute,
        exempt_paths=PUBLIC_PATHS,
    )
    app.add_middleware(
        BodySizeLimitMiddleware,
        max_bytes=settings.pa_max_request_bytes,
        exempt_prefixes=("/api/v1/wardrobe/items",),
    )
    app.add_middleware(
        NetworkAllowlistMiddleware,
        allowed_networks=settings.pa_api_allowed_networks,
        exempt_paths=PUBLIC_PATHS,
    )
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(RequestContextMiddleware)

    register_error_handlers(app)
    _register_routers(app)
    _customise_openapi(app)
    return app


def _register_routers(app: FastAPI) -> None:
    app.include_router(system.router)
    app.include_router(reminders.router)
    app.include_router(tasks.router)
    app.include_router(background_tasks.router)
    app.include_router(approvals.router)
    app.include_router(actions.router)
    app.include_router(gmail.router)
    app.include_router(calendar.router)
    app.include_router(contacts.router)
    app.include_router(drive.router)
    app.include_router(notion.router)
    app.include_router(weather.router)
    app.include_router(documents.router)
    app.include_router(wardrobe.router)
    app.include_router(workstation.router)
    app.include_router(briefings.router)
    app.include_router(alerts.router)
    app.include_router(reachy.router)
    app.include_router(vision.router)


def _customise_openapi(app: FastAPI) -> None:
    def openapi() -> dict[str, object]:
        if app.openapi_schema:
            return app.openapi_schema
        schema = get_openapi(
            title=app.title,
            version=app.version,
            description=app.description,
            routes=app.routes,
        )
        schema.setdefault("components", {}).setdefault("securitySchemes", {})["BearerToken"] = {
            "type": "http",
            "scheme": "bearer",
            "description": "Static token from PA_API_TOKEN.",
        }
        app.openapi_schema = schema
        return schema

    app.openapi = openapi  # type: ignore[method-assign]


app = create_app()
