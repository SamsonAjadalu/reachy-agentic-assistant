"""Wardrobe endpoints.

Everything here is local: the garments, the images and the reasoning. No part of
this leaves the machine, which is why photographs of the owner's clothes are
acceptable to store at all.
"""

from __future__ import annotations

from datetime import date
from typing import Annotated, Any

from fastapi import APIRouter, Depends, File, Query, UploadFile
from fastapi.responses import Response
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, get_settings
from app.dependencies.auth import Principal, require_token
from app.dependencies.common import get_session
from app.schemas.common import ApiModel, DeleteResponse
from app.schemas.wardrobe import (
    ColourProposal,
    ItemCreate,
    ItemList,
    ItemOut,
    ItemUpdate,
    OutfitCreate,
    OutfitList,
    OutfitOut,
    PackingRequest,
    RecommendationOut,
    RecommendationResponse,
    WearRecord,
)
from app.services import wardrobe as service
from database.models import WardrobeImage, WardrobeItem
from integrations.registry import get_weather
from integrations.weather.advice import advise
from shared.errors import ValidationError
from shared.timeutils import local_today
from wardrobe.images import read_stored, store_image
from wardrobe.packing import build_packing_list
from wardrobe.recommend import rank, rotation_gaps, score_outfit, suggest_laundry

router = APIRouter(prefix="/api/v1/wardrobe", tags=["wardrobe"])

SessionDep = Annotated[AsyncSession, Depends(get_session)]
SettingsDep = Annotated[Settings, Depends(get_settings)]
PrincipalDep = Annotated[Principal, Depends(require_token)]


class LaundryRequest(ApiModel):
    item_ids: list[str]


def _image_root(settings: Settings) -> Any:
    root = settings.wardrobe_image_path
    root.mkdir(parents=True, exist_ok=True)
    return root


# ------------------------------------------------------------------- items
@router.post("/items", response_model=ItemOut, status_code=201, summary="Add a garment")
async def create_item(payload: ItemCreate, _: PrincipalDep, session: SessionDep) -> ItemOut:
    item = await service.create_item(session, payload)
    await session.commit()
    return await service.to_item_out(session, item)


@router.get("/items", response_model=ItemList, summary="List garments")
async def list_items(
    _: PrincipalDep,
    session: SessionDep,
    category: Annotated[str | None, Query(max_length=40)] = None,
    color: Annotated[str | None, Query(max_length=40)] = None,
    available_only: bool = False,
    include_archived: bool = False,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> ItemList:
    items, total = await service.list_items(
        session,
        category=category,
        colour=color,
        available_only=available_only,
        include_archived=include_archived,
        limit=limit,
        offset=offset,
    )
    return ItemList(items=[await service.to_item_out(session, item) for item in items], total=total)


@router.get("/items/{item_id}", response_model=ItemOut, summary="One garment")
async def get_item(item_id: str, _: PrincipalDep, session: SessionDep) -> ItemOut:
    return await service.to_item_out(session, await service.get_item(session, item_id))


@router.patch("/items/{item_id}", response_model=ItemOut, summary="Update a garment")
async def update_item(
    item_id: str, payload: ItemUpdate, _: PrincipalDep, session: SessionDep
) -> ItemOut:
    item = await service.update_item(session, item_id, payload)
    await session.commit()
    return await service.to_item_out(session, item)


@router.delete("/items/{item_id}", response_model=DeleteResponse, summary="Delete a garment")
async def delete_item(item_id: str, _: PrincipalDep, session: SessionDep) -> DeleteResponse:
    item = await service.get_item(session, item_id)
    await session.delete(item)
    await session.commit()
    return DeleteResponse(id=item_id, deleted=True)


# ------------------------------------------------------------------ images
@router.post(
    "/items/{item_id}/images",
    response_model=ColourProposal,
    status_code=201,
    summary="Attach a photo and propose its colours",
    description=(
        "The image is re-encoded before it is stored, which strips EXIF - including the "
        "GPS tag most phones attach. Colours are a proposal for you to confirm."
    ),
)
async def upload_image(
    item_id: str,
    _: PrincipalDep,
    session: SessionDep,
    settings: SettingsDep,
    file: Annotated[UploadFile, File()],
    is_primary: bool = False,
) -> ColourProposal:
    item = await service.get_item(session, item_id)
    data = await file.read()
    stored = store_image(data, _image_root(settings), item_id=item.id)

    existing = await session.scalar(
        select(WardrobeImage).where(
            WardrobeImage.item_id == item.id,
            WardrobeImage.content_hash == stored.content_hash,
        )
    )
    if existing is not None:
        raise ValidationError("That photo is already attached to this garment.")

    if is_primary:
        for other in await session.scalars(
            select(WardrobeImage).where(WardrobeImage.item_id == item.id)
        ):
            other.is_primary = False

    image = WardrobeImage(
        item_id=item.id,
        relative_path=stored.relative_path,
        thumbnail_path=stored.thumbnail_path,
        width=stored.width,
        height=stored.height,
        bytes=stored.bytes,
        content_hash=stored.content_hash,
        dominant_colors=service._dumps(stored.dominant_colours),
        is_primary=is_primary,
    )
    session.add(image)
    await session.commit()

    return ColourProposal(
        proposed_primary_color=stored.primary_colour_name,
        colors=stored.dominant_colours,
        image=service.to_image_out(image),
    )


@router.get(
    "/images/{image_id}",
    summary="Fetch a stored image",
    response_class=Response,
    responses={200: {"content": {"image/jpeg": {}}}},
)
async def get_image(
    image_id: str,
    _: PrincipalDep,
    session: SessionDep,
    settings: SettingsDep,
    thumbnail: bool = False,
) -> Response:
    image = await session.get(WardrobeImage, image_id)
    if image is None:
        raise ValidationError("No image with that id.")
    relative = (image.thumbnail_path if thumbnail else image.relative_path) or (image.relative_path)
    return Response(content=read_stored(_image_root(settings), relative), media_type="image/jpeg")


# ----------------------------------------------------------------- outfits
@router.post("/outfits", response_model=OutfitOut, status_code=201, summary="Save an outfit")
async def create_outfit(payload: OutfitCreate, _: PrincipalDep, session: SessionDep) -> OutfitOut:
    outfit = await service.create_outfit(session, payload)
    await session.commit()
    return await service.to_outfit_out(session, outfit)


@router.get("/outfits", response_model=OutfitList, summary="List outfits")
async def list_outfits(
    _: PrincipalDep,
    session: SessionDep,
    occasion: Annotated[str | None, Query(max_length=60)] = None,
    include_archived: bool = False,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> OutfitList:
    outfits, total = await service.list_outfits(
        session,
        occasion=occasion,
        include_archived=include_archived,
        limit=limit,
        offset=offset,
    )
    return OutfitList(
        items=[await service.to_outfit_out(session, outfit) for outfit in outfits], total=total
    )


@router.get("/outfits/{outfit_id}", response_model=OutfitOut, summary="One outfit")
async def get_outfit(outfit_id: str, _: PrincipalDep, session: SessionDep) -> OutfitOut:
    return await service.to_outfit_out(session, await service.get_outfit(session, outfit_id))


@router.post(
    "/outfits/{outfit_id}/wear",
    response_model=OutfitOut,
    summary="Record that an outfit was worn",
    description="Ages every garment in the outfit and moves any that hit their wear limit.",
)
async def record_wear(
    outfit_id: str,
    payload: WearRecord,
    _: PrincipalDep,
    session: SessionDep,
    settings: SettingsDep,
) -> OutfitOut:
    await service.record_wear(
        session,
        outfit_id,
        settings=settings,
        worn_on=payload.worn_on,
        occasion=payload.occasion,
        rating=payload.rating,
        notes=payload.notes,
    )
    await session.commit()
    return await service.to_outfit_out(session, await service.get_outfit(session, outfit_id))


# --------------------------------------------------------- recommendations
@router.get(
    "/recommend",
    response_model=RecommendationResponse,
    summary="What to wear",
    description=(
        "Scores saved outfits against the forecast and returns the reasoning behind "
        "every number, including why an outfit was excluded."
    ),
)
async def recommend(
    _: PrincipalDep,
    session: SessionDep,
    settings: SettingsDep,
    occasion: Annotated[str | None, Query(max_length=60)] = None,
    formality: Annotated[int | None, Query(ge=1, le=5)] = None,
    day_offset: Annotated[int, Query(ge=0, le=6)] = 0,
    limit: Annotated[int, Query(ge=1, le=10)] = 3,
) -> RecommendationResponse:
    report = await get_weather(settings).get_weather(days=day_offset + 1)
    guidance = advise(report, day_index=day_offset)
    day = report.daily[day_offset] if day_offset < len(report.daily) else None
    temperature = (day.high_c + day.low_c) / 2 if day else report.current.feels_like_c

    outfits, _total = await service.list_outfits(session, occasion=occasion, limit=200)
    views = [await service.to_outfit_view(session, outfit) for outfit in outfits]

    today = local_today(settings.app_timezone)
    all_scores = [
        score_outfit(
            view,
            temperature_c=temperature,
            advice=guidance,
            occasion=occasion,
            required_formality=formality,
            today=today,
        )
        for view in views
    ]
    best = rank(
        views,
        temperature_c=temperature,
        advice=guidance,
        occasion=occasion,
        required_formality=formality,
        today=today,
        limit=limit,
    )

    recommendations: list[RecommendationOut] = []
    for score in best:
        payload = score.as_dict()
        recommendations.append(
            RecommendationOut(
                outfit_id=score.outfit_id,
                outfit_name=score.outfit_name,
                score=payload["score"],
                explanation=payload["explanation"],
                components=payload["components"],
                warnings=payload["warnings"],
                items=[
                    await service.to_item_out(session, item)
                    for item in await _items_for(session, score.outfit_id)
                ],
            )
        )

    return RecommendationResponse(
        recommendations=recommendations,
        considered=len(views),
        excluded=[
            {"outfit_name": score.outfit_name, "reason": score.disqualified_reason or ""}
            for score in all_scores
            if score.disqualified
        ],
        weather_summary=guidance.summary(),
        temperature_c=round(temperature, 1),
    )


async def _items_for(session: AsyncSession, outfit_id: str) -> list[WardrobeItem]:
    from database.models import OutfitItem

    links = list(await session.scalars(select(OutfitItem).where(OutfitItem.outfit_id == outfit_id)))
    items: list[WardrobeItem] = []
    for link in links:
        item = await session.get(WardrobeItem, link.item_id)
        if item is not None:
            items.append(item)
    return items


@router.get(
    "/laundry",
    summary="What needs washing",
    description="Garments at or past their wear limit, with the count that got them there.",
)
async def laundry(_: PrincipalDep, session: SessionDep, settings: SettingsDep) -> dict[str, Any]:
    items, _total = await service.list_items(session, limit=500)
    due = suggest_laundry(
        [service.to_item_view(item) for item in items], today=local_today(settings.app_timezone)
    )
    return {"items": due, "total": len(due)}


@router.post("/laundry/done", summary="Mark garments as washed")
async def laundry_done(
    payload: LaundryRequest, _: PrincipalDep, session: SessionDep
) -> dict[str, Any]:
    count = await service.mark_laundered(session, payload.item_ids)
    await session.commit()
    return {"updated": count}


@router.get(
    "/gaps",
    summary="Garments that have fallen out of rotation",
)
async def gaps(
    _: PrincipalDep,
    session: SessionDep,
    settings: SettingsDep,
    days: Annotated[int, Query(ge=7, le=365)] = 90,
) -> dict[str, Any]:
    items, _total = await service.list_items(session, limit=500)
    forgotten = rotation_gaps(
        [service.to_item_view(item) for item in items],
        today=local_today(settings.app_timezone),
        days=days,
    )
    return {"items": forgotten, "total": len(forgotten)}


@router.post(
    "/packing-list",
    summary="Build a packing list for a trip",
    description="Uses the same forecast interpretation as the daily recommendation.",
)
async def packing_list(
    payload: PackingRequest, _: PrincipalDep, session: SessionDep, settings: SettingsDep
) -> dict[str, Any]:
    if payload.end_on < payload.start_on:
        raise ValidationError("The trip ends before it starts.")
    if (payload.end_on - payload.start_on).days > 60:
        raise ValidationError("Packing lists are limited to 60 days.")

    days = (payload.end_on - payload.start_on).days + 1
    report = await get_weather(settings).get_weather(
        latitude=payload.latitude,
        longitude=payload.longitude,
        location=payload.destination,
        days=min(days, 16),
    )
    items, _total = await service.list_items(session, limit=500)

    return build_packing_list(
        destination=payload.destination,
        start_on=payload.start_on,
        end_on=payload.end_on,
        forecast=report,
        items=[service.to_item_view(item) for item in items],
        formality=payload.formality,
    ).as_dict()


@router.get("/stats", summary="Wardrobe overview")
async def stats(_: PrincipalDep, session: SessionDep) -> dict[str, Any]:
    items, total = await service.list_items(session, limit=1000)
    _outfits, outfit_total = await service.list_outfits(session, limit=1000)

    by_category: dict[str, int] = {}
    for item in items:
        by_category[item.category] = by_category.get(item.category, 0) + 1

    return {
        "items": total,
        "outfits": outfit_total,
        "by_category": dict(sorted(by_category.items(), key=lambda pair: -pair[1])),
        "in_laundry": sum(1 for item in items if item.laundry_status == "in_laundry"),
        "never_worn": sum(1 for item in items if item.times_worn == 0),
        "as_of": date.today().isoformat(),
    }
