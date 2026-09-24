"""Condition alerts.

An alert rule answers one question about the current state of the world and,
when the answer is bad, produces a message with a stable dedup key. The key is
what makes repetition survivable: a disk that stays above ninety percent is one
problem, not one problem per evaluation cycle.

Rules are deliberately dull - thresholds and counts, no inference - so an alert
can always be explained and the owner can change the number that produced it.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings
from app.logging_config import get_logger
from database.models import Reminder, Task
from notifications.dispatcher import clear_dedup, dispatch, should_notify
from shared.contracts import payload_hash
from shared.enums import AlertSeverity, NotificationChannel, ReminderStatus, TaskStatus
from shared.timeutils import utcnow

logger = get_logger(__name__)

DISK_WARNING_PERCENT = 90.0
DISK_CRITICAL_PERCENT = 96.0
MEMORY_WARNING_PERCENT = 92.0
OVERDUE_TASK_THRESHOLD = 3

DEFAULT_COOLDOWNS = {
    AlertSeverity.INFO: 24 * 3600,
    AlertSeverity.WARNING: 6 * 3600,
    AlertSeverity.CRITICAL: 3600,
}


@dataclass
class Alert:
    kind: str
    title: str
    body: str
    severity: AlertSeverity = AlertSeverity.WARNING
    dedup_key: str = ""
    cooldown_seconds: int | None = None
    spoken_text: str | None = None
    resource_type: str | None = None
    resource_id: str | None = None
    facts: dict[str, Any] = field(default_factory=dict)
    """Included in the dedup hash so a worsening condition re-notifies at once."""

    @property
    def cooldown(self) -> int:
        return self.cooldown_seconds or DEFAULT_COOLDOWNS[self.severity]


@dataclass
class AlertContext:
    session: AsyncSession
    settings: Settings


AlertRule = Callable[[AlertContext], Awaitable[list[Alert]]]
_RULES: dict[str, AlertRule] = {}


def rule(name: str) -> Callable[[AlertRule], AlertRule]:
    def decorator(func: AlertRule) -> AlertRule:
        _RULES[name] = func
        return func

    return decorator


def rule_names() -> list[str]:
    return sorted(_RULES)


# -------------------------------------------------------------------- rules
@rule("disk_space")
async def _disk_space(context: AlertContext) -> list[Alert]:
    from workstation.status import collect_status

    status = await collect_status(context.settings)
    alerts: list[Alert] = []

    for disk in status.disks:
        if disk.percent_used < DISK_WARNING_PERCENT:
            # Resolved conditions forget their cooldown, so a recurrence is
            # reported immediately rather than after the window lapses.
            await clear_dedup(context.session, f"disk:{disk.mount}")
            continue

        critical = disk.percent_used >= DISK_CRITICAL_PERCENT
        alerts.append(
            Alert(
                kind="workstation.disk_space",
                title="Disk space is running low",
                body=(
                    f"{disk.mount} is {disk.percent_used:.0f} percent full "
                    f"with {disk.free_gb:.1f} gigabytes free."
                ),
                severity=AlertSeverity.CRITICAL if critical else AlertSeverity.WARNING,
                dedup_key=f"disk:{disk.mount}",
                spoken_text=(f"Heads up, {disk.mount} is {disk.percent_used:.0f} percent full."),
                resource_type="mount",
                resource_id=disk.mount,
                # Bucketed so a percent of drift does not re-notify, but a real
                # deterioration does.
                facts={"bucket": int(disk.percent_used // 2)},
            )
        )
    return alerts


@rule("memory")
async def _memory(context: AlertContext) -> list[Alert]:
    from workstation.status import collect_status

    status = await collect_status(context.settings)
    if status.memory_percent < MEMORY_WARNING_PERCENT:
        await clear_dedup(context.session, "memory:host")
        return []

    return [
        Alert(
            kind="workstation.memory",
            title="Memory is nearly exhausted",
            body=(
                f"Memory is {status.memory_percent:.0f} percent used "
                f"({status.memory_used_gb:.1f} of {status.memory_total_gb:.1f} GB)."
            ),
            severity=AlertSeverity.WARNING,
            dedup_key="memory:host",
            facts={"bucket": int(status.memory_percent // 2)},
        )
    ]


@rule("overdue_tasks")
async def _overdue_tasks(context: AlertContext) -> list[Alert]:
    now = utcnow()
    overdue = list(
        await context.session.scalars(
            select(Task).where(
                Task.status.in_([TaskStatus.OPEN.value, TaskStatus.IN_PROGRESS.value]),
                Task.due_at.is_not(None),
                Task.due_at < now,
            )
        )
    )
    if len(overdue) < OVERDUE_TASK_THRESHOLD:
        await clear_dedup(context.session, "tasks:overdue")
        return []

    names = ", ".join(task.description[:40] for task in overdue[:3])
    return [
        Alert(
            kind="tasks.overdue",
            title="Several tasks are overdue",
            body=f"{len(overdue)} tasks are past their due date, including {names}.",
            severity=AlertSeverity.INFO,
            dedup_key="tasks:overdue",
            spoken_text=f"You have {len(overdue)} overdue tasks.",
            facts={"count": len(overdue)},
        )
    ]


@rule("missed_reminders")
async def _missed_reminders(context: AlertContext) -> list[Alert]:
    """A reminder whose time passed without delivery means something is wrong.

    The scheduler should have fired it; if it did not, silence is the worst
    possible outcome, so the condition itself is reported.
    """
    cutoff = utcnow() - timedelta(minutes=15)
    stale = list(
        await context.session.scalars(
            select(Reminder).where(
                Reminder.status == ReminderStatus.ACTIVE.value,
                Reminder.trigger_at < cutoff,
            )
        )
    )
    if not stale:
        await clear_dedup(context.session, "reminders:missed")
        return []

    return [
        Alert(
            kind="reminders.missed",
            title="Reminders did not fire on time",
            body=(
                f"{len(stale)} reminder(s) are past due but still active, "
                f"starting with '{stale[0].text}'. The scheduler may not be running."
            ),
            severity=AlertSeverity.CRITICAL,
            dedup_key="reminders:missed",
            facts={"count": len(stale)},
        )
    ]


@rule("integration_health")
async def _integration_health(context: AlertContext) -> list[Alert]:
    from database.models import IntegrationHealth

    unhealthy = list(
        await context.session.scalars(
            select(IntegrationHealth).where(IntegrationHealth.consecutive_failures >= 3)
        )
    )
    alerts: list[Alert] = []
    for record in unhealthy:
        alerts.append(
            Alert(
                kind="integration.unhealthy",
                title=f"{record.name} is failing",
                body=(
                    f"{record.name} has failed {record.consecutive_failures} times in a row. "
                    f"Last error: {record.last_error or 'not recorded'}."
                ),
                severity=AlertSeverity.WARNING,
                dedup_key=f"integration:{record.name}",
                resource_type="integration",
                resource_id=record.name,
                facts={"failures": record.consecutive_failures},
            )
        )
    return alerts


# ------------------------------------------------------------------ engine
async def evaluate(
    session: AsyncSession,
    settings: Settings,
    *,
    rules: list[str] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run every rule once and notify on anything past its cooldown."""
    context = AlertContext(session=session, settings=settings)
    wanted = rules or rule_names()

    raised: list[dict[str, Any]] = []
    suppressed = 0
    failed: list[str] = []

    for name in wanted:
        rule_fn = _RULES.get(name)
        if rule_fn is None:
            failed.append(name)
            continue

        try:
            alerts = await rule_fn(context)
        except Exception as exc:
            # One broken rule must not stop the others; a monitoring system that
            # stops monitoring on the first error is worse than none.
            logger.warning(
                "An alert rule failed",
                extra={"rule": name, "reason": f"{type(exc).__name__}: {exc}"},
            )
            failed.append(name)
            continue

        for alert in alerts:
            fingerprint = payload_hash(alert.facts) if alert.facts else None
            # dry_run must not advance the cooldown ledger, or a "what would fire?"
            # check would silently suppress the real notification.
            allowed = await should_notify(
                session,
                alert.dedup_key or f"{alert.kind}:default",
                alert.kind,
                cooldown_seconds=alert.cooldown,
                payload_hash=fingerprint,
                record=not dry_run,
            )
            if not allowed:
                suppressed += 1
                continue

            if not dry_run:
                await dispatch(
                    session,
                    kind=alert.kind,
                    title=alert.title,
                    body=alert.body,
                    severity=alert.severity,
                    channel=NotificationChannel.TELEGRAM,
                    dedup_key=alert.dedup_key,
                    resource_type=alert.resource_type,
                    resource_id=alert.resource_id,
                    settings=settings,
                    spoken_text=alert.spoken_text,
                )
            raised.append(
                {
                    "kind": alert.kind,
                    "severity": alert.severity.value,
                    "title": alert.title,
                    "body": alert.body,
                    "dedup_key": alert.dedup_key,
                }
            )

    return {
        "evaluated": len(wanted),
        "raised": len(raised),
        "suppressed": suppressed,
        "failed_rules": failed,
        "alerts": raised,
    }
