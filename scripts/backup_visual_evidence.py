#!/usr/bin/env python3
"""Copy PINNED and EVALUATION evidence only.

Thumbnails, crops and full frames are excluded — they are re-derivable and would
multiply 30x under the daily backup retention policy.

    python scripts/backup_visual_evidence.py
    python scripts/backup_visual_evidence.py --dry-run
"""

from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app.config import Settings  # noqa: E402
from vision.enums import RetentionClass  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    settings = Settings()
    if settings.database_path is None or not settings.database_path.exists():
        print("assistant.db is not present; nothing to copy.")
        return 1

    stamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    output = (
        args.output_dir
        or (settings.backup_root or settings.app_data_dir / "backups") / f"visual-pinned-{stamp}"
    )
    if not args.dry_run:
        output.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(f"file:{settings.database_path}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            "SELECT relative_path, retention_class FROM evidence_assets "
            "WHERE retention_class IN (?, ?) AND state = 'active'",
            (RetentionClass.PINNED.value, RetentionClass.EVALUATION.value),
        ).fetchall()
    except sqlite3.OperationalError:
        print("evidence_assets table is not present yet.")
        return 0
    finally:
        connection.close()

    copied = 0
    for relative_path, _class in rows:
        source = settings.visual_evidence_path / relative_path
        if not source.is_file():
            print(f"Missing {relative_path}")
            continue
        destination = output / relative_path
        if args.dry_run:
            print(f"Would copy {source} -> {destination}")
            copied += 1
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied += 1
        print(f"Copied {relative_path}")

    print(f"Pinned/evaluation files: {copied}")
    print(f"Output: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
