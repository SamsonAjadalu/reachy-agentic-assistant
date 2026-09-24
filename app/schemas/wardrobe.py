"""Wardrobe request and response models."""

from __future__ import annotations

from datetime import date
from typing import Annotated, Any

from pydantic import Field

from app.schemas.common import ApiModel
from shared.enums import LaundryStatus

Formality = Annotated[int, Field(ge=1, le=5, description="1 loungewear to 5 formal")]
Warmth = Annotated[int, Field(ge=1, le=5, description="1 very light to 5 very warm")]


class ItemCreate(ApiModel):
    name: Annotated[str, Field(min_length=1, max_length=200)]
    category: Annotated[str, Field(min_length=1, max_length=40)]
    primary_color: Annotated[str, Field(min_length=1, max_length=40)]
    subcategory: Annotated[str | None, Field(default=None, max_length=60)] = None
    secondary_colors: list[str] = Field(default_factory=list)
    pattern: Annotated[str, Field(max_length=40)] = "solid"
    material: Annotated[str | None, Field(default=None, max_length=60)] = None
    seasons: list[str] = Field(default_factory=lambda: ["all"])
    formality: Formality = 3
    warmth: Warmth = 3
    water_resistant: bool = False
    fit: Annotated[str | None, Field(default=None, max_length=40)] = None
    size: Annotated[str | None, Field(default=None, max_length=30)] = None
    brand: Annotated[str | None, Field(default=None, max_length=80)] = None
    notes: Annotated[str | None, Field(default=None, max_length=2000)] = None
    max_wears_before_wash: Annotated[int, Field(ge=1, le=50)] = 3
    favourite_rating: Annotated[int | None, Field(default=None, ge=1, le=5)] = None
    purchase_date: date | None = None
    purchase_price: Annotated[float | None, Field(default=None, ge=0)] = None


class ItemUpdate(ApiModel):
    name: Annotated[str | None, Field(default=None, min_length=1, max_length=200)] = None
    category: Annotated[str | None, Field(default=None, max_length=40)] = None
    primary_color: Annotated[str | None, Field(default=None, max_length=40)] = None
    formality: Formality | None = None
    warmth: Warmth | None = None
    water_resistant: bool | None = None
    available: bool | None = None
    laundry_status: LaundryStatus | None = None
    favourite_rating: Annotated[int | None, Field(default=None, ge=1, le=5)] = None
    notes: Annotated[str | None, Field(default=None, max_length=2000)] = None
    archived: bool | None = None


class ImageOut(ApiModel):
    id: str
    relative_path: str
    thumbnail_path: str | None = None
    width: int | None = None
    height: int | None = None
    dominant_colors: list[dict[str, Any]] = Field(default_factory=list)
    is_primary: bool = False


class ItemOut(ApiModel):
    id: str
    name: str
    category: str
    subcategory: str | None = None
    primary_color: str
    secondary_colors: list[str] = Field(default_factory=list)
    pattern: str
    material: str | None = None
    seasons: list[str] = Field(default_factory=list)
    formality: int
    warmth: int
    water_resistant: bool
    available: bool
    laundry_status: str
    wears_since_wash: int
    max_wears_before_wash: int
    times_worn: int
    last_worn_on: date | None = None
    favourite_rating: int | None = None
    archived: bool
    images: list[ImageOut] = Field(default_factory=list)


class ItemList(ApiModel):
    items: list[ItemOut]
    total: int


class ColourProposal(ApiModel):
    """A suggestion for the owner to confirm, not a decision."""

    proposed_primary_color: str
    colors: list[dict[str, Any]]
    image: ImageOut
    note: str = "Colours are proposed from the image. Correct them if they are wrong."


class OutfitItemRef(ApiModel):
    item_id: str
    role: Annotated[str | None, Field(default=None, max_length=40)] = None


class OutfitCreate(ApiModel):
    name: Annotated[str, Field(min_length=1, max_length=200)]
    items: Annotated[list[OutfitItemRef], Field(min_length=1, max_length=12)]
    occasion: Annotated[str | None, Field(default=None, max_length=60)] = None
    seasons: list[str] = Field(default_factory=lambda: ["all"])
    formality: Formality = 3
    min_temperature_c: float | None = None
    max_temperature_c: float | None = None
    notes: Annotated[str | None, Field(default=None, max_length=2000)] = None


class OutfitOut(ApiModel):
    id: str
    name: str
    occasion: str | None = None
    formality: int
    seasons: list[str] = Field(default_factory=list)
    min_temperature_c: float | None = None
    max_temperature_c: float | None = None
    user_rating: int | None = None
    times_worn: int
    last_worn_on: date | None = None
    archived: bool
    items: list[ItemOut] = Field(default_factory=list)


class OutfitList(ApiModel):
    items: list[OutfitOut]
    total: int


class WearRecord(ApiModel):
    worn_on: date | None = None
    occasion: Annotated[str | None, Field(default=None, max_length=60)] = None
    rating: Annotated[int | None, Field(default=None, ge=1, le=5)] = None
    notes: Annotated[str | None, Field(default=None, max_length=1000)] = None


class RecommendationOut(ApiModel):
    outfit_id: str
    outfit_name: str
    score: float
    explanation: str
    components: list[dict[str, Any]]
    warnings: list[str] = Field(default_factory=list)
    items: list[ItemOut] = Field(default_factory=list)


class RecommendationResponse(ApiModel):
    recommendations: list[RecommendationOut]
    considered: int
    excluded: list[dict[str, str]] = Field(default_factory=list)
    weather_summary: str
    temperature_c: float


class PackingRequest(ApiModel):
    destination: Annotated[str, Field(min_length=1, max_length=100)]
    start_on: date
    end_on: date
    formality: Formality | None = None
    latitude: Annotated[float | None, Field(default=None, ge=-90, le=90)] = None
    longitude: Annotated[float | None, Field(default=None, ge=-180, le=180)] = None
