"""The batch import path.

Cataloguing a wardrobe by hand is the step most likely to be abandoned, so the
importer has to be forgiving about messy spreadsheets and clear about what it
refused.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
from PIL import Image
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from app.config import Settings
from database.models import WardrobeImage, WardrobeItem
from scripts.import_wardrobe import import_rows, parse_row, read_rows

ROWS = [
    {
        "name": "Navy jumper",
        "category": "top",
        "primary_color": "navy",
        "warmth": "4",
        "seasons": "autumn;winter",
        "image": "jumper.jpg",
    },
    {"name": "Grey chinos", "category": "bottom", "primary_color": "grey", "warmth": "3"},
]


@pytest.fixture
def import_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "import"
    directory.mkdir()
    Image.new("RGB", (300, 300), (20, 40, 130)).save(directory / "jumper.jpg")

    with (directory / "items.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["name", "category", "primary_color", "warmth", "seasons", "image"],
        )
        writer.writeheader()
        writer.writerows(ROWS)
    return directory


class TestParsing:
    def test_a_minimal_row_gets_sensible_defaults(self) -> None:
        item = parse_row({"name": "Tee", "category": "top", "primary_color": "white"})
        assert item.formality == 3
        assert item.seasons == ["all"]
        assert item.max_wears_before_wash == 3

    def test_seasons_are_split_on_semicolons(self) -> None:
        item = parse_row(
            {
                "name": "Coat",
                "category": "outerwear",
                "primary_color": "black",
                "seasons": "autumn; winter",
            }
        )
        assert item.seasons == ["autumn", "winter"]

    def test_out_of_range_values_are_clamped_not_rejected(self) -> None:
        """A typo in one cell should not fail an import of two hundred rows."""
        item = parse_row(
            {
                "name": "Coat",
                "category": "outerwear",
                "primary_color": "black",
                "formality": "9",
                "warmth": "0",
            }
        )
        assert item.formality == 5
        assert item.warmth == 1

    def test_truthy_spellings_are_all_accepted(self) -> None:
        for spelling in ("yes", "TRUE", "1", "y"):
            item = parse_row(
                {
                    "name": "Shell",
                    "category": "outerwear",
                    "primary_color": "green",
                    "water_resistant": spelling,
                }
            )
            assert item.water_resistant is True

    def test_a_csv_missing_a_required_column_stops_the_run(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.csv"
        path.write_text("name,category\nTee,top\n", encoding="utf-8")
        with pytest.raises(SystemExit, match="primary_color"):
            read_rows(path)

    def test_blank_rows_are_ignored(self, tmp_path: Path) -> None:
        path = tmp_path / "gappy.csv"
        path.write_text("name,category,primary_color\nTee,top,white\n,,\n", encoding="utf-8")
        assert len(read_rows(path)) == 1


class TestImport:
    async def test_a_dry_run_writes_nothing(
        self, import_dir: Path, engine: AsyncEngine, session: AsyncSession
    ) -> None:
        summary = await import_rows(read_rows(import_dir / "items.csv"), import_dir, dry_run=True)

        assert summary["created"] == 2
        assert (await session.scalars(select(WardrobeItem))).first() is None

    async def test_a_real_run_creates_items_and_photos(
        self,
        import_dir: Path,
        engine: AsyncEngine,
        session: AsyncSession,
        settings: Settings,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr("scripts.import_wardrobe.get_settings", lambda: settings)
        monkeypatch.setattr("scripts.import_wardrobe.create_engine", lambda _s: engine)
        monkeypatch.setattr("scripts.import_wardrobe.dispose_engine", _noop)

        summary = await import_rows(read_rows(import_dir / "items.csv"), import_dir, dry_run=False)

        assert summary["created"] == 2
        assert summary["images"] == 1
        names = {item.name for item in await session.scalars(select(WardrobeItem))}
        assert names == {"Navy jumper", "Grey chinos"}
        assert (await session.scalars(select(WardrobeImage))).first() is not None

    async def test_a_missing_photo_is_reported_but_the_item_survives(
        self,
        import_dir: Path,
        engine: AsyncEngine,
        session: AsyncSession,
        settings: Settings,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr("scripts.import_wardrobe.get_settings", lambda: settings)
        monkeypatch.setattr("scripts.import_wardrobe.create_engine", lambda _s: engine)
        monkeypatch.setattr("scripts.import_wardrobe.dispose_engine", _noop)
        (import_dir / "jumper.jpg").unlink()

        summary = await import_rows(read_rows(import_dir / "items.csv"), import_dir, dry_run=False)

        assert summary["created"] == 2
        assert summary["images"] == 0
        assert any("jumper.jpg" in error for error in summary["errors"])

    async def test_a_corrupt_photo_does_not_lose_the_row(
        self,
        import_dir: Path,
        engine: AsyncEngine,
        session: AsyncSession,
        settings: Settings,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr("scripts.import_wardrobe.get_settings", lambda: settings)
        monkeypatch.setattr("scripts.import_wardrobe.create_engine", lambda _s: engine)
        monkeypatch.setattr("scripts.import_wardrobe.dispose_engine", _noop)
        (import_dir / "jumper.jpg").write_bytes(b"not an image")

        summary = await import_rows(read_rows(import_dir / "items.csv"), import_dir, dry_run=False)

        assert summary["created"] == 2
        assert summary["images"] == 0
        assert summary["errors"]


async def _noop() -> None:
    """The engine is owned by the test fixture, not by the import."""
    return None
