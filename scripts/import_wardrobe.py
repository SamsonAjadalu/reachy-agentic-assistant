#!/usr/bin/env python
"""Batch-import wardrobe items from a CSV, optionally with photographs.

Cataloguing a wardrobe one API call at a time is tedious enough that it would
not get done. This reads a spreadsheet the owner can fill in once.

    python scripts/import_wardrobe.py items.csv --images ./photos --dry-run

Required columns: name, category, primary_color.
Optional: subcategory, pattern, material, seasons, formality, warmth,
water_resistant, fit, size, brand, notes, max_wears_before_wash,
favourite_rating, image.

``seasons`` is semicolon-separated. ``image`` is a filename inside --images.
Nothing is written until the run is repeated without --dry-run.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("PYTHONPATH", "")

from app.config import get_settings  # noqa: E402
from app.schemas.wardrobe import ItemCreate  # noqa: E402
from app.services import wardrobe as service  # noqa: E402
from database.models import WardrobeImage  # noqa: E402
from database.session import create_engine, dispose_engine, session_scope, set_engine  # noqa: E402
from shared.errors import ValidationError  # noqa: E402
from wardrobe.images import store_image  # noqa: E402

REQUIRED_COLUMNS = {"name", "category", "primary_color"}
TRUTHY = {"1", "true", "yes", "y"}


def parse_row(row: dict[str, str]) -> ItemCreate:
    def text(key: str, default: str = "") -> str:
        return (row.get(key) or default).strip()

    def number(key: str, default: int) -> int:
        raw = text(key)
        return int(raw) if raw.isdigit() else default

    seasons = [part.strip() for part in text("seasons", "all").split(";") if part.strip()]
    rating = text("favourite_rating")

    return ItemCreate(
        name=text("name"),
        category=text("category"),
        primary_color=text("primary_color"),
        subcategory=text("subcategory") or None,
        pattern=text("pattern", "solid"),
        material=text("material") or None,
        seasons=seasons or ["all"],
        formality=min(5, max(1, number("formality", 3))),
        warmth=min(5, max(1, number("warmth", 3))),
        water_resistant=text("water_resistant").lower() in TRUTHY,
        fit=text("fit") or None,
        size=text("size") or None,
        brand=text("brand") or None,
        notes=text("notes") or None,
        max_wears_before_wash=max(1, number("max_wears_before_wash", 3)),
        favourite_rating=int(rating) if rating.isdigit() else None,
    )


def read_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS - {name.strip() for name in reader.fieldnames or []}
        if missing:
            raise SystemExit(f"The CSV is missing required columns: {', '.join(sorted(missing))}")
        return [row for row in reader if (row.get("name") or "").strip()]


async def import_rows(
    rows: list[dict[str, str]], image_root: Path | None, *, dry_run: bool
) -> dict[str, Any]:
    settings = get_settings()
    summary: dict[str, Any] = {"created": 0, "images": 0, "skipped": 0, "errors": []}

    if dry_run:
        for index, row in enumerate(rows, start=2):
            try:
                item = parse_row(row)
            except (ValidationError, ValueError) as exc:
                summary["errors"].append(f"row {index}: {exc}")
                continue
            print(f"  would add {item.name} ({item.category}, {item.primary_color})")
            summary["created"] += 1
        return summary

    engine = create_engine(settings)
    set_engine(engine)
    try:
        async with session_scope() as session:
            for index, row in enumerate(rows, start=2):
                try:
                    created = await service.create_item(session, parse_row(row))
                except (ValidationError, ValueError) as exc:
                    summary["errors"].append(f"row {index}: {exc}")
                    summary["skipped"] += 1
                    continue

                summary["created"] += 1
                filename = (row.get("image") or "").strip()
                if not filename or image_root is None:
                    continue

                source = image_root / filename
                if not source.is_file():
                    summary["errors"].append(f"row {index}: no image at {filename}")
                    continue
                try:
                    stored = store_image(
                        source.read_bytes(), settings.wardrobe_image_path, item_id=created.id
                    )
                except ValidationError as exc:
                    summary["errors"].append(f"row {index}: {exc}")
                    continue

                session.add(
                    WardrobeImage(
                        item_id=created.id,
                        relative_path=stored.relative_path,
                        thumbnail_path=stored.thumbnail_path,
                        width=stored.width,
                        height=stored.height,
                        bytes=stored.bytes,
                        content_hash=stored.content_hash,
                        dominant_colors=service._dumps(stored.dominant_colours),
                        is_primary=True,
                    )
                )
                summary["images"] += 1
    finally:
        await dispose_engine()

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path, help="CSV describing the garments")
    parser.add_argument("--images", type=Path, default=None, help="Directory holding photos")
    parser.add_argument(
        "--dry-run", action="store_true", help="Report what would be imported and stop"
    )
    args = parser.parse_args()

    if not args.csv_path.is_file():
        print(f"No such file: {args.csv_path}", file=sys.stderr)
        return 1

    rows = read_rows(args.csv_path)
    print(f"Read {len(rows)} row(s) from {args.csv_path}")

    summary = asyncio.run(import_rows(rows, args.images, dry_run=args.dry_run))

    print(f"\nCreated: {summary['created']}  Images: {summary['images']}")
    if summary["errors"]:
        print(f"Problems ({len(summary['errors'])}):")
        for error in summary["errors"][:20]:
            print(f"  - {error}")
    if args.dry_run:
        print("\nDry run: nothing was written. Re-run without --dry-run to import.")
    return 1 if summary["errors"] and not summary["created"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
