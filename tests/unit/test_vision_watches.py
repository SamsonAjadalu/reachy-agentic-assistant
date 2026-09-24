"""Watch quiet hours without constructing a mapped ORM row."""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from vision.watches import in_quiet_hours


def test_quiet_hours_overnight() -> None:
    watch = SimpleNamespace(
        quiet_hours_json='{"start":"22:00","end":"07:00","timezone":"America/Toronto"}'
    )
    late = datetime(2026, 9, 2, 23, 30, tzinfo=ZoneInfo("America/Toronto"))
    noon = datetime(2026, 9, 2, 12, 0, tzinfo=ZoneInfo("America/Toronto"))
    assert in_quiet_hours(watch, now=late, timezone="America/Toronto")  # type: ignore[arg-type]
    assert not in_quiet_hours(watch, now=noon, timezone="America/Toronto")  # type: ignore[arg-type]
