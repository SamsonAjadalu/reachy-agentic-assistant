"""Background task handler registry.

Handlers are registered by name so a queued row that outlives a restart, or a
deploy that renames a function, still resolves to something explicit rather than
executing whatever happens to be at that import path.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from shared.errors import ValidationError

TaskHandler = Callable[["TaskContext"], Awaitable[dict[str, Any]]]


@dataclass
class TaskContext:
    """Everything a handler is given.

    ``report_progress`` and ``is_cancelled`` are supplied by the runner so a
    handler never touches the database or the task row directly.
    """

    task_id: str
    task_type: str
    payload: dict[str, Any]
    attempt: int
    worker_id: str
    report_progress: Callable[[int, str | None], Awaitable[None]]
    is_cancelled: Callable[[], Awaitable[bool]]
    correlation_id: str | None = None


@dataclass(frozen=True)
class HandlerSpec:
    name: str
    handler: TaskHandler
    default_timeout_seconds: int = 900
    max_attempts: int = 3
    retryable: bool = True
    description: str = ""


_REGISTRY: dict[str, HandlerSpec] = {}


def register_task(
    name: str,
    *,
    timeout_seconds: int = 900,
    max_attempts: int = 3,
    retryable: bool = True,
    description: str = "",
) -> Callable[[TaskHandler], TaskHandler]:
    """Register a coroutine as the handler for ``name``.

    ``retryable=False`` is the correct choice for anything with an irreversible
    external side effect, where a blind retry could duplicate the effect.
    """

    def decorator(func: TaskHandler) -> TaskHandler:
        if name in _REGISTRY:
            raise RuntimeError(f"Background task handler {name!r} is already registered.")
        _REGISTRY[name] = HandlerSpec(
            name=name,
            handler=func,
            default_timeout_seconds=timeout_seconds,
            max_attempts=max_attempts,
            retryable=retryable,
            description=description or (func.__doc__ or "").strip().split("\n")[0],
        )
        return func

    return decorator


def get_handler(name: str) -> HandlerSpec:
    spec = _REGISTRY.get(name)
    if spec is None:
        raise ValidationError(f"Unknown background task type: {name!r}")
    return spec


def has_handler(name: str) -> bool:
    return name in _REGISTRY


def registered_tasks() -> list[HandlerSpec]:
    return sorted(_REGISTRY.values(), key=lambda spec: spec.name)


def clear_registry() -> None:
    """Test helper."""
    _REGISTRY.clear()
