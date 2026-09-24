#!/usr/bin/env python3
"""Online SQLite backup using the sqlite3 backup API.

    python scripts/backup_database.py
    python scripts/backup_database.py --output-dir /path/to/backups
    python scripts/backup_database.py --dry-run

Copies assistant.db, scheduler.db and documents/index.db when present. Uses the
SQLite online backup protocol so readers and writers can continue during the copy.
Prunes archives older than BACKUP_RETENTION_DAYS.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app.config import Settings  # noqa: E402


def backup_file(source: Path, destination: Path, *, dry_run: bool) -> bool:
    if not source.exists():
        return False
    if dry_run:
        print(f"Would back up {source} -> {destination}")
        return True

    destination.parent.mkdir(parents=True, exist_ok=True)
    source_conn = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
    dest_conn = sqlite3.connect(destination)
    try:
        source_conn.backup(dest_conn)
    finally:
        dest_conn.close()
        source_conn.close()
    print(f"Backed up {source.name} -> {destination}")
    return True


def prune_old(backups: list[Path], retention_days: int, *, dry_run: bool) -> int:
    if retention_days <= 0:
        return 0
    cutoff = datetime.now(tz=UTC) - timedelta(days=retention_days)
    removed = 0
    for path in sorted(backups):
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
        if mtime < cutoff:
            if dry_run:
                print(f"Would prune {path}")
            else:
                path.unlink()
                print(f"Pruned {path}")
            removed += 1
    return removed


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Override BACKUP_ROOT / APP_DATA_DIR/backups.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Report actions only.")
    args = parser.parse_args()

    settings = Settings()
    stamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir = (
        args.output_dir or settings.backup_root or settings.app_data_dir / "backups"
    ).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    targets: list[tuple[Path, str]] = []
    if settings.database_path is not None:
        targets.append((settings.database_path, "assistant"))
    targets.append((settings.app_data_dir / "scheduler.db", "scheduler"))
    targets.append((settings.document_index_path, "documents_index"))

    backed = 0
    created: list[Path] = []
    for source, label in targets:
        dest = output_dir / f"{label}-{stamp}.db"
        if backup_file(source, dest, dry_run=args.dry_run):
            backed += 1
            if not args.dry_run:
                created.append(dest)

    all_backups = list(output_dir.glob("*-*.db"))
    pruned = prune_old(all_backups, settings.backup_retention_days, dry_run=args.dry_run)

    print(f"Backup directory: {output_dir}")
    print(f"Files copied:     {backed}")
    print(f"Files pruned:     {pruned}")
    return 0 if backed > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
