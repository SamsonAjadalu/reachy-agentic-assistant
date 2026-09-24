"""Wardrobe business logic.

The database stores lists as JSON text because SQLite has no array type; the
conversion lives here so nothing above this layer has to know that.
"""

from __future__ import annotations

import json
from datetime import date
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings
from app.schemas.wardrobe import (
    ImageOut,
    ItemCreate,
    ItemOut,
    ItemUpdate,
    OutfitCreate,
    OutfitOut,
)
from database.models import (
    Outfit,
    OutfitHistory,
    OutfitItem,
    WardrobeAvailabilityEvent,
    WardrobeImage,
    WardrobeItem,
)
from shared.enums import LaundryStatus
from shared.errors import NotFoundError, ValidationError
from shared.timeutils import local_today, utcnow
from wardrobe.recommend import ItemView, OutfitView


def _loads(raw: str | None, default: list[Any] | None = None) -> list[Any]:
    if not raw:
        return list(default or [])
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return list(default or [])
    return value if isinstance(value, list) else list(default or [])


def _dumps(values: list[Any]) -> str:
    return json.dumps(values, ensure_ascii=False)


# ------------------------------------------------------------------- items
async def create_item(session: AsyncSession, payload: ItemCreate) -> WardrobeItem:
    item = WardrobeItem(
        name=payload.name.strip(),
        category=payload.category.strip().lower(),
        subcategory=payload.subcategory,
        primary_color=payload.primary_color.strip().lower(),
        secondary_colors=_dumps([colour.lower() for colour in payload.secondary_colors]),
        pattern=payload.pattern,
        material=payload.material,
        seasons=_dumps(payload.seasons or ["all"]),
        formality=payload.formality,
        warmth=payload.warmth,
        water_resistant=payload.water_resistant,
        fit=payload.fit,
        size=payload.size,
        brand=payload.brand,
        notes=payload.notes,
        max_wears_before_wash=payload.max_wears_before_wash,
        favourite_rating=payload.favourite_rating,
        purchase_date=payload.purchase_date,
        purchase_price=payload.purchase_price,
    )
    session.add(item)
    await session.flush()
    return item


async def get_item(session: AsyncSession, item_id: str) -> WardrobeItem:
    item = await session.get(WardrobeItem, item_id)
    if item is None:
        raise NotFoundError("No wardrobe item with that id.")
    return item


async def list_items(
    session: AsyncSession,
    *,
    category: str | None = None,
    colour: str | None = None,
    available_only: bool = False,
    include_archived: bool = False,
    limit: int = 50,
    offset: int = 0,
) -> tuple[list[WardrobeItem], int]:
    query = select(WardrobeItem)
    if not include_archived:
        query = query.where(WardrobeItem.archived.is_(False))
    if category:
        query = query.where(WardrobeItem.category == category.lower())
    if colour:
        query = query.where(WardrobeItem.primary_color == colour.lower())
    if available_only:
        query = query.where(
            WardrobeItem.available.is_(True),
            WardrobeItem.laundry_status != LaundryStatus.IN_LAUNDRY.value,
        )

    total = await session.scalar(select(func.count()).select_from(query.subquery())) or 0
    rows = await session.scalars(
        query.order_by(WardrobeItem.category, WardrobeItem.name).limit(limit).offset(offset)
    )
    return list(rows), total


async def update_item(session: AsyncSession, item_id: str, payload: ItemUpdate) -> WardrobeItem:
    item = await get_item(session, item_id)
    data = payload.model_dump(exclude_unset=True)

    if "laundry_status" in data and data["laundry_status"] is not None:
        new_status = LaundryStatus(data.pop("laundry_status")).value
        await _record_status_change(session, item, new_status)

    for field, value in data.items():
        if value is not None:
            setattr(item, field, value)
    await session.flush()
    return item


async def _record_status_change(session: AsyncSession, item: WardrobeItem, new_status: str) -> None:
    """Status is kept as history, not just a flag, so laundry can be reasoned about."""
    if item.laundry_status == new_status:
        return
    session.add(
        WardrobeAvailabilityEvent(
            item_id=item.id,
            occurred_at=utcnow(),
            from_status=item.laundry_status,
            to_status=new_status,
        )
    )
    item.laundry_status = new_status
    if new_status == LaundryStatus.CLEAN.value:
        item.wears_since_wash = 0
    item.available = new_status != LaundryStatus.IN_LAUNDRY.value


async def mark_laundered(session: AsyncSession, item_ids: list[str]) -> int:
    for item_id in item_ids:
        item = await get_item(session, item_id)
        await _record_status_change(session, item, LaundryStatus.CLEAN.value)
    await session.flush()
    return len(item_ids)


# ----------------------------------------------------------------- outfits
async def create_outfit(session: AsyncSession, payload: OutfitCreate) -> Outfit:
    item_ids = [ref.item_id for ref in payload.items]
    if len(set(item_ids)) != len(item_ids):
        raise ValidationError("An outfit cannot contain the same garment twice.")

    found = list(await session.scalars(select(WardrobeItem).where(WardrobeItem.id.in_(item_ids))))
    if len(found) != len(item_ids):
        raise ValidationError("One or more of those garments do not exist.")

    outfit = Outfit(
        name=payload.name.strip(),
        occasion=payload.occasion,
        seasons=_dumps(payload.seasons or ["all"]),
        formality=payload.formality,
        min_temperature_c=payload.min_temperature_c,
        max_temperature_c=payload.max_temperature_c,
        notes=payload.notes,
    )
    session.add(outfit)
    await session.flush()

    for ref in payload.items:
        session.add(OutfitItem(outfit_id=outfit.id, item_id=ref.item_id, role=ref.role))
    await session.flush()
    return outfit


async def get_outfit(session: AsyncSession, outfit_id: str) -> Outfit:
    outfit = await session.get(Outfit, outfit_id)
    if outfit is None:
        raise NotFoundError("No outfit with that id.")
    return outfit


async def list_outfits(
    session: AsyncSession,
    *,
    occasion: str | None = None,
    include_archived: bool = False,
    limit: int = 50,
    offset: int = 0,
) -> tuple[list[Outfit], int]:
    query = select(Outfit)
    if not include_archived:
        query = query.where(Outfit.archived.is_(False))
    if occasion:
        query = query.where(Outfit.occasion == occasion)

    total = await session.scalar(select(func.count()).select_from(query.subquery())) or 0
    rows = await session.scalars(query.order_by(Outfit.name).limit(limit).offset(offset))
    return list(rows), total


async def record_wear(
    session: AsyncSession,
    outfit_id: str,
    *,
    settings: Settings,
    worn_on: date | None = None,
    occasion: str | None = None,
    rating: int | None = None,
    notes: str | None = None,
    weather_summary: str | None = None,
    temperature_c: float | None = None,
) -> OutfitHistory:
    """Log that an outfit was worn, ageing every garment in it."""
    outfit = await get_outfit(session, outfit_id)
    when = worn_on or local_today(settings.app_timezone)

    history = OutfitHistory(
        outfit_id=outfit.id,
        worn_on=when,
        occasion=occasion,
        rating=rating,
        notes=notes,
        weather_summary=weather_summary,
        temperature_c=temperature_c,
    )
    session.add(history)

    outfit.last_worn_on = when
    outfit.times_worn += 1
    if rating is not None:
        outfit.user_rating = rating

    links = await session.scalars(select(OutfitItem).where(OutfitItem.outfit_id == outfit.id))
    for link in links:
        item = await session.get(WardrobeItem, link.item_id)
        if item is None:
            continue
        item.last_worn_on = when
        item.times_worn += 1
        item.wears_since_wash += 1
        if (
            item.wears_since_wash >= item.max_wears_before_wash
            and item.laundry_status == LaundryStatus.CLEAN.value
        ):
            item.laundry_status = LaundryStatus.WORN.value

    await session.flush()
    return history


# --------------------------------------------------------------- projections
def to_item_view(item: WardrobeItem) -> ItemView:
    return ItemView(
        id=item.id,
        name=item.name,
        category=item.category,
        primary_color=item.primary_color,
        formality=item.formality,
        warmth=item.warmth,
        water_resistant=item.water_resistant,
        seasons=_loads(item.seasons, ["all"]),
        laundry_status=item.laundry_status,
        available=item.available,
        wears_since_wash=item.wears_since_wash,
        max_wears_before_wash=item.max_wears_before_wash,
        favourite_rating=item.favourite_rating,
        last_worn_on=item.last_worn_on,
    )


async def to_outfit_view(session: AsyncSession, outfit: Outfit) -> OutfitView:
    links = list(await session.scalars(select(OutfitItem).where(OutfitItem.outfit_id == outfit.id)))
    items: list[ItemView] = []
    for link in links:
        item = await session.get(WardrobeItem, link.item_id)
        if item is not None:
            items.append(to_item_view(item))

    return OutfitView(
        id=outfit.id,
        name=outfit.name,
        formality=outfit.formality,
        items=items,
        occasion=outfit.occasion,
        min_temperature_c=outfit.min_temperature_c,
        max_temperature_c=outfit.max_temperature_c,
        user_rating=outfit.user_rating,
        last_worn_on=outfit.last_worn_on,
        times_worn=outfit.times_worn,
    )


async def to_item_out(session: AsyncSession, item: WardrobeItem) -> ItemOut:
    images = list(
        await session.scalars(select(WardrobeImage).where(WardrobeImage.item_id == item.id))
    )
    return ItemOut(
        id=item.id,
        name=item.name,
        category=item.category,
        subcategory=item.subcategory,
        primary_color=item.primary_color,
        secondary_colors=_loads(item.secondary_colors),
        pattern=item.pattern,
        material=item.material,
        seasons=_loads(item.seasons, ["all"]),
        formality=item.formality,
        warmth=item.warmth,
        water_resistant=item.water_resistant,
        available=item.available,
        laundry_status=item.laundry_status,
        wears_since_wash=item.wears_since_wash,
        max_wears_before_wash=item.max_wears_before_wash,
        times_worn=item.times_worn,
        last_worn_on=item.last_worn_on,
        favourite_rating=item.favourite_rating,
        archived=item.archived,
        images=[to_image_out(image) for image in images],
    )


def to_image_out(image: WardrobeImage) -> ImageOut:
    return ImageOut(
        id=image.id,
        relative_path=image.relative_path,
        thumbnail_path=image.thumbnail_path,
        width=image.width,
        height=image.height,
        dominant_colors=_loads(image.dominant_colors),
        is_primary=image.is_primary,
    )


async def to_outfit_out(session: AsyncSession, outfit: Outfit) -> OutfitOut:
    links = list(await session.scalars(select(OutfitItem).where(OutfitItem.outfit_id == outfit.id)))
    items: list[ItemOut] = []
    for link in links:
        item = await session.get(WardrobeItem, link.item_id)
        if item is not None:
            items.append(await to_item_out(session, item))

    return OutfitOut(
        id=outfit.id,
        name=outfit.name,
        occasion=outfit.occasion,
        formality=outfit.formality,
        seasons=_loads(outfit.seasons, ["all"]),
        min_temperature_c=outfit.min_temperature_c,
        max_temperature_c=outfit.max_temperature_c,
        user_rating=outfit.user_rating,
        times_worn=outfit.times_worn,
        last_worn_on=outfit.last_worn_on,
        archived=outfit.archived,
        items=items,
    )
