"""Command-line interface for the personal assistant API."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable
from datetime import timedelta

from reachy_client import PersonalAssistantClient
from reachy_client.models import Delay, ReminderCreate, ScheduleInput, TaskCreate
from reachy_tools.proposed import adapters
from shared.timeutils import utcnow


def _env_client() -> PersonalAssistantClient:
    base_url = os.environ.get("PA_API_BASE_URL", "http://127.0.0.1:8000")
    token = os.environ.get("PA_API_TOKEN", "")
    if not token:
        print("PA_API_TOKEN is not set.", file=sys.stderr)
        sys.exit(2)
    return PersonalAssistantClient(base_url, token)


def _print_json(data: object) -> None:
    if isinstance(data, dict):
        print(json.dumps(data, indent=2, default=str))
        return
    dump = getattr(data, "model_dump", None)
    if callable(dump):
        print(json.dumps(dump(mode="json"), indent=2, default=str))
        return
    print(json.dumps(data, indent=2, default=str))


def cmd_ping(args: argparse.Namespace) -> int:
    with _env_client() as client:
        if args.adapter:
            _print_json(adapters.assistant_ping(client))
        else:
            _print_json(client.ping())
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    with _env_client() as client:
        _print_json(client.status())
    return 0


def cmd_reminders_list(args: argparse.Namespace) -> int:
    with _env_client() as client:
        if args.adapter:
            _print_json(adapters.list_reminders(client, upcoming_only=args.upcoming))
        else:
            _print_json(client.list_reminders(upcoming_only=args.upcoming, limit=args.limit))
    return 0


def cmd_reminders_create(args: argparse.Namespace) -> int:
    with _env_client() as client:
        if args.adapter:
            _print_json(
                adapters.create_reminder(
                    client,
                    text=args.text,
                    delay_minutes=args.delay_minutes,
                )
            )
            return 0
        schedule = ScheduleInput(delay=Delay(minutes=args.delay_minutes))
        payload = ReminderCreate(text=args.text, schedule=schedule)
        _print_json(client.create_reminder(payload, idempotency_key=args.idempotency_key))
    return 0


def cmd_mock_reachy(args: argparse.Namespace) -> int:
    """Exercise every major endpoint the way a Reachy tool adapter would."""
    failures: list[str] = []
    with _env_client() as client:
        checks: list[tuple[str, Callable[[], object]]] = [
            ("ping", lambda: client.ping()),
            ("status", lambda: client.status()),
            ("list_reminders", lambda: client.list_reminders(limit=3)),
            ("list_tasks", lambda: client.list_tasks(limit=3)),
            ("search_gmail", lambda: client.search_gmail("in:inbox", limit=3)),
            ("get_calendar_events", lambda: client.get_calendar_events(days=3, limit=5)),
            (
                "check_free_busy",
                lambda: client.check_free_busy(utcnow(), utcnow() + timedelta(hours=4)),
            ),
            ("search_contacts", lambda: client.search_contacts("test", limit=3)),
            ("search_notion", lambda: client.search_notion("notes", limit=3)),
            ("search_documents", lambda: client.search_documents("report", limit=3)),
            ("get_weather", lambda: client.get_weather()),
            ("search_wardrobe", lambda: client.search_wardrobe(limit=5)),
            ("recommend_outfit", lambda: client.recommend_outfit(limit=2)),
            ("get_workstation_status", lambda: client.get_workstation_status()),
            ("list_pending_notifications", lambda: client.list_pending_notifications()),
        ]

        for name, call in checks:
            try:
                result = call()
                label = "ok"
                if hasattr(result, "total"):
                    label = f"ok (total={getattr(result, 'total', '?')})"
                print(f"[pass] {name}: {label}")
            except Exception as exc:
                failures.append(name)
                print(f"[fail] {name}: {type(exc).__name__}: {exc}", file=sys.stderr)

        if args.create_samples:
            try:
                reminder = client.create_reminder(
                    ReminderCreate(
                        text="CLI mock-reachy sample",
                        schedule=ScheduleInput(
                            delay=Delay(minutes=30),
                        ),
                    ),
                    idempotency_key=f"mock-{utcnow().timestamp()}",
                )
                print(f"[pass] create_reminder: ok (id={reminder.id})")
            except Exception as exc:
                failures.append("create_reminder")
                print(f"[fail] create_reminder: {exc}", file=sys.stderr)

            try:
                task = client.create_task(
                    TaskCreate(description="CLI mock-reachy sample task"),
                    idempotency_key=f"mock-task-{utcnow().timestamp()}",
                )
                print(f"[pass] create_task: ok (id={task.id})")
            except Exception as exc:
                failures.append("create_task")
                print(f"[fail] create_task: {exc}", file=sys.stderr)

    if failures:
        print(f"\n{len(failures)} check(s) failed.", file=sys.stderr)
        return 1
    print("\nAll checks passed.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pa-cli", description="Personal assistant CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    ping = sub.add_parser("ping", help="Authenticated round-trip check")
    ping.add_argument("--adapter", action="store_true", help="Run through proposed adapter")
    ping.set_defaults(func=cmd_ping)

    status = sub.add_parser("status", help="Service status")
    status.set_defaults(func=cmd_status)

    reminders = sub.add_parser("reminders", help="Reminder commands")
    rem_sub = reminders.add_subparsers(dest="rem_action", required=True)

    rem_list = rem_sub.add_parser("list", help="List reminders")
    rem_list.add_argument("--upcoming", action="store_true", default=True)
    rem_list.add_argument("--limit", type=int, default=10)
    rem_list.add_argument("--adapter", action="store_true")
    rem_list.set_defaults(func=cmd_reminders_list)

    rem_create = rem_sub.add_parser("create", help="Create a reminder")
    rem_create.add_argument("text", help="Reminder text")
    rem_create.add_argument("--delay-minutes", type=int, default=15)
    rem_create.add_argument("--idempotency-key")
    rem_create.add_argument("--adapter", action="store_true")
    rem_create.set_defaults(func=cmd_reminders_create)

    mock = sub.add_parser("mock-reachy", help="Exercise major API endpoints")
    mock.add_argument(
        "--create-samples",
        action="store_true",
        help="Also create a sample reminder and task",
    )
    mock.set_defaults(func=cmd_mock_reachy)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
