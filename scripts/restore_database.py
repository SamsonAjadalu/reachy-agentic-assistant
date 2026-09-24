#!/usr/bin/env python3
"""Restore SQLite databases from a backup produced by backup_database.py.

    python scripts/restore_database.py --backup /path/to/assistant-20260101T120000Z.db
    python scripts/restore_database.py --backup-dir $APP_DATA_DIR/backups --latest

Stop the API (and any worker unit) before restoring. The script refuses to
overwrite a live database unless --force is passed.
"""

from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app.config import Settings  # noqa: E402


def _latest_backup(directory: Path, label: str) -> Path | None:
    matches = sorted(directory.glob(f"{label}-*.db"), key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def _verify_sqlite(path: Path) -> None:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        conn.execute("PRAGMA integrity_check").fetchone()
    finally:
        conn.close()


def restore_one(source: Path, destination: Path, *, force: bool, dry_run: bool) -> None:
    if not source.exists():
        raise FileNotFoundError(source)
    _verify_sqlite(source)

    if destination.exists() and not force:
        raise RuntimeError(
            f"{destination} already exists. Stop the service and pass --force to overwrite."
        )

    if dry_run:
        print(f"Would restore {source} -> {destination}")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        sidecar = destination.with_suffix(destination.suffix + ".pre-restore")
        shutil.copy2(destination, sidecar)
        print(f"Previous database preserved at {sidecar}")

    shutil.copy2(source, destination)
    print(f"Restored {destination.name} from {source.name}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--backup", type=Path, help="Path to a single assistant backup file.")
    parser.add_argument(
        "--backup-dir",
        type=Path,
        help="Directory containing timestamped backups (used with --latest).",
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Restore the newest assistant, scheduler and document index backups.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing database files.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    settings = Settings()
    backup_dir = (
        args.backup_dir or settings.backup_root or settings.app_data_dir / "backups"
    ).expanduser()

    if args.latest:
        if settings.database_path is None:
            print("DATABASE_URL is not a local SQLite path.", file=sys.stderr)
            return 1
        pairs = [
            (_latest_backup(backup_dir, "assistant"), settings.database_path),
            (_latest_backup(backup_dir, "scheduler"), settings.app_data_dir / "scheduler.db"),
            (_latest_backup(backup_dir, "documents_index"), settings.document_index_path),
        ]
        restored = 0
        for source, dest in pairs:
            if source is None or dest is None:
                continue
            restore_one(source, dest, force=args.force, dry_run=args.dry_run)
            restored += 1
        if restored == 0:
            print(f"No backups found in {backup_dir}", file=sys.stderr)
            return 1
        return 0

    if args.backup is None:
        parser.error("Supply --backup or --latest.")
        return 2

    if settings.database_path is None:
        print("DATABASE_URL is not a local SQLite path.", file=sys.stderr)
        return 1

    restore_one(args.backup, settings.database_path, force=args.force, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
