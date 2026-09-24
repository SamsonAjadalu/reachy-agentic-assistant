"""Prove SQLite online backup and restore round-trip preserves application data."""

from __future__ import annotations

import asyncio
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from database.models import Reminder
from scripts.backup_database import backup_file
from scripts.restore_database import restore_one
from shared.enums import ReminderStatus
from shared.timeutils import utcnow

REPO_ROOT = Path(__file__).resolve().parents[2]


def _assert_reminder_in_db(db_path: Path, reminder_id: str, text: str) -> None:
    conn = sqlite3.connect(db_path)
    try:
        found = conn.execute("SELECT text FROM reminders WHERE id = ?", (reminder_id,)).fetchone()
    finally:
        conn.close()
    assert found is not None
    assert found[0] == text


async def test_backup_and_restore_preserves_rows(
    session: AsyncSession, data_dir: Path, settings
) -> None:
    reminder = Reminder(
        text="Survive backup",
        trigger_at=utcnow(),
        original_trigger_at=utcnow(),
        status=ReminderStatus.ACTIVE.value,
        timezone="America/Toronto",
    )
    session.add(reminder)
    await session.commit()
    reminder_id = reminder.id

    db_path = Path(settings.database_url.split("sqlite+aiosqlite:///")[-1])
    backup_path = data_dir / "backups" / "assistant-test.db"
    restored = data_dir / "restored.db"

    assert backup_file(db_path, backup_path, dry_run=False) is True
    _assert_reminder_in_db(backup_path, reminder_id, "Survive backup")

    restore_one(backup_path, restored, force=True, dry_run=False)
    _assert_reminder_in_db(restored, reminder_id, "Survive backup")


async def test_backup_and_restore_scripts_cli(
    session: AsyncSession, data_dir: Path, settings, secret_key_file: Path
) -> None:
    reminder = Reminder(
        text="CLI round trip",
        trigger_at=utcnow(),
        original_trigger_at=utcnow(),
        status=ReminderStatus.ACTIVE.value,
        timezone="America/Toronto",
    )
    session.add(reminder)
    await session.commit()
    reminder_id = reminder.id
    await session.close()

    env = os.environ.copy()
    env.update(
        {
            "PA_ENV_FILE": str(REPO_ROOT / "tests" / ".env.absent"),
            "APP_ENV": "test",
            "APP_DATA_DIR": str(data_dir),
            "DATABASE_URL": settings.database_url,
            "PA_API_TOKEN": settings.pa_api_token.get_secret_value(),
            "PA_SECRET_KEY_FILE": str(secret_key_file),
            "MOCK_MODE": "true",
            "SCHEDULER_ENABLED": "false",
            "WORKER_ENABLED": "false",
            "TELEGRAM_ENABLED": "false",
            "PYTHONPATH": str(REPO_ROOT),
        }
    )
    backup_dir = data_dir / "cli-backups"
    backup_dir.mkdir(parents=True, exist_ok=True)

    backup = await asyncio.to_thread(
        subprocess.run,
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "backup_database.py"),
            "--output-dir",
            str(backup_dir),
        ],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert backup.returncode == 0, backup.stdout + backup.stderr
    archives = list(backup_dir.glob("assistant-*.db"))
    assert archives, backup.stdout

    # Point restore at a sidecar path under the data dir, not the live test DB.
    restore_target = data_dir / "assistant-from-cli.db"
    restore = await asyncio.to_thread(
        subprocess.run,
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "restore_database.py"),
            "--backup",
            str(archives[0]),
            "--force",
        ],
        cwd=str(REPO_ROOT),
        env={
            **env,
            "DATABASE_URL": f"sqlite+aiosqlite:///{restore_target}",
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert restore.returncode == 0, restore.stdout + restore.stderr
    _assert_reminder_in_db(restore_target, reminder_id, "CLI round trip")
