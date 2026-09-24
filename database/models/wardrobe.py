"""Wardrobe items, images, outfits and wear history."""

from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import (
    Boolean,
    Date,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database.base import Base, TimestampMixin, UtcDateTime, UUIDPrimaryKeyMixin
from shared.enums import LaundryStatus


class WardrobeItem(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "wardrobe_items"
    __table_args__ = (
        Index("ix_wardrobe_items_category_archived", "category", "archived"),
        Index("ix_wardrobe_items_primary_color", "primary_color"),
    )

    name: Mapped[str] = mapped_column(String(200), nullable=False)
    category: Mapped[str] = mapped_column(String(40), nullable=False, index=True)
    subcategory: Mapped[str | None] = mapped_column(String(60))
    primary_color: Mapped[str] = mapped_column(String(40), nullable=False)
    secondary_colors: Mapped[str | None] = mapped_column(Text, comment="JSON array")
    pattern: Mapped[str] = mapped_column(String(40), nullable=False, default="solid")
    material: Mapped[str | None] = mapped_column(String(60))
    seasons: Mapped[str] = mapped_column(
        Text, nullable=False, default='["all"]', comment="JSON array"
    )
    formality: Mapped[int] = mapped_column(
        Integer, nullable=False, default=3, comment="1 loungewear to 5 formal"
    )
    fit: Mapped[str | None] = mapped_column(String(40))
    size: Mapped[str | None] = mapped_column(String(30))
    brand: Mapped[str | None] = mapped_column(String(80))
    warmth: Mapped[int] = mapped_column(
        Integer, nullable=False, default=3, comment="1 very light to 5 very warm"
    )
    water_resistant: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    notes: Mapped[str | None] = mapped_column(Text)
    available: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    laundry_status: Mapped[str] = mapped_column(
        String(20), nullable=False, default=LaundryStatus.CLEAN.value, index=True
    )
    wears_since_wash: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    max_wears_before_wash: Mapped[int] = mapped_column(Integer, nullable=False, default=3)

    last_worn_on: Mapped[date | None] = mapped_column(Date, index=True)
    last_posted_on: Mapped[date | None] = mapped_column(Date)
    times_worn: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    favourite_rating: Mapped[int | None] = mapped_column(Integer)

    purchase_date: Mapped[date | None] = mapped_column(Date)
    purchase_price: Mapped[float | None] = mapped_column(Float)
    archived: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    embedding_ref: Mapped[str | None] = mapped_column(
        Text, comment="Optional sidecar path produced by a pluggable embedder"
    )

    images: Mapped[list[WardrobeImage]] = relationship(
        back_populates="item", cascade="all, delete-orphan", lazy="selectin"
    )
    outfit_links: Mapped[list[OutfitItem]] = relationship(
        back_populates="item", cascade="all, delete-orphan"
    )
    availability_events: Mapped[list[WardrobeAvailabilityEvent]] = relationship(
        back_populates="item", cascade="all, delete-orphan"
    )


class WardrobeImage(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Image metadata only.

    Bytes live under ``WARDROBE_IMAGE_ROOT`` inside APP_DATA_DIR, never in the
    database and never in a model prompt.
    """

    __tablename__ = "wardrobe_images"

    item_id: Mapped[str] = mapped_column(
        ForeignKey("wardrobe_items.id", ondelete="CASCADE"), nullable=False, index=True
    )
    relative_path: Mapped[str] = mapped_column(String(400), nullable=False)
    thumbnail_path: Mapped[str | None] = mapped_column(String(400))
    width: Mapped[int | None] = mapped_column(Integer)
    height: Mapped[int | None] = mapped_column(Integer)
    bytes: Mapped[int | None] = mapped_column(Integer)
    content_hash: Mapped[str | None] = mapped_column(String(64), index=True)
    dominant_colors: Mapped[str | None] = mapped_column(Text, comment="JSON array of hex strings")
    is_primary: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    item: Mapped[WardrobeItem] = relationship(back_populates="images")


class Outfit(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "outfits"

    name: Mapped[str] = mapped_column(String(200), nullable=False)
    occasion: Mapped[str | None] = mapped_column(String(60), index=True)
    seasons: Mapped[str] = mapped_column(Text, nullable=False, default='["all"]')
    formality: Mapped[int] = mapped_column(Integer, nullable=False, default=3)
    min_temperature_c: Mapped[float | None] = mapped_column(Float)
    max_temperature_c: Mapped[float | None] = mapped_column(Float)
    weather_conditions: Mapped[str | None] = mapped_column(Text, comment="JSON array")
    image_path: Mapped[str | None] = mapped_column(String(400))
    user_rating: Mapped[int | None] = mapped_column(Integer)
    assistant_rating: Mapped[float | None] = mapped_column(Float)
    notes: Mapped[str | None] = mapped_column(Text)
    archived: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    last_worn_on: Mapped[date | None] = mapped_column(Date, index=True)
    last_posted_on: Mapped[date | None] = mapped_column(Date)
    times_worn: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    items: Mapped[list[OutfitItem]] = relationship(
        back_populates="outfit", cascade="all, delete-orphan", lazy="selectin"
    )
    history: Mapped[list[OutfitHistory]] = relationship(
        back_populates="outfit", cascade="all, delete-orphan"
    )


class OutfitItem(Base):
    """Join table.

    A composite primary key here is what makes "the same garment twice in one
    outfit" unrepresentable rather than merely discouraged.
    """

    __tablename__ = "outfit_items"

    outfit_id: Mapped[str] = mapped_column(
        ForeignKey("outfits.id", ondelete="CASCADE"), primary_key=True
    )
    item_id: Mapped[str] = mapped_column(
        ForeignKey("wardrobe_items.id", ondelete="CASCADE"), primary_key=True
    )
    role: Mapped[str | None] = mapped_column(String(40), comment="top, bottom, outer, shoes, ...")

    outfit: Mapped[Outfit] = relationship(back_populates="items")
    item: Mapped[WardrobeItem] = relationship(back_populates="outfit_links", lazy="selectin")


class OutfitHistory(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "outfit_history"
    __table_args__ = (Index("ix_outfit_history_worn_on", "worn_on"),)

    outfit_id: Mapped[str] = mapped_column(
        ForeignKey("outfits.id", ondelete="CASCADE"), nullable=False, index=True
    )
    worn_on: Mapped[date] = mapped_column(Date, nullable=False)
    occasion: Mapped[str | None] = mapped_column(String(60))
    weather_summary: Mapped[str | None] = mapped_column(String(200))
    temperature_c: Mapped[float | None] = mapped_column(Float)
    rating: Mapped[int | None] = mapped_column(Integer)
    notes: Mapped[str | None] = mapped_column(Text)

    outfit: Mapped[Outfit] = relationship(back_populates="history")


class WardrobeAvailabilityEvent(UUIDPrimaryKeyMixin, Base):
    """Laundry and availability transitions, kept as history rather than a flag."""

    __tablename__ = "wardrobe_availability_events"

    item_id: Mapped[str] = mapped_column(
        ForeignKey("wardrobe_items.id", ondelete="CASCADE"), nullable=False, index=True
    )
    occurred_at: Mapped[datetime] = mapped_column(UtcDateTime, nullable=False)
    from_status: Mapped[str | None] = mapped_column(String(20))
    to_status: Mapped[str] = mapped_column(String(20), nullable=False)
    note: Mapped[str | None] = mapped_column(Text)

    item: Mapped[WardrobeItem] = relationship(back_populates="availability_events")


class SocialPostHistory(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """What was posted publicly, so the same look is not posted twice."""

    __tablename__ = "social_post_history"
    __table_args__ = (
        UniqueConstraint("outfit_id", "posted_on", "platform", name="uq_social_post_history_entry"),
    )

    outfit_id: Mapped[str | None] = mapped_column(
        ForeignKey("outfits.id", ondelete="SET NULL"), index=True
    )
    posted_on: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    platform: Mapped[str] = mapped_column(String(40), nullable=False, default="unspecified")
    caption: Mapped[str | None] = mapped_column(Text)
    external_url: Mapped[str | None] = mapped_column(Text)
