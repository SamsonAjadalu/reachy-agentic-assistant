"""Packing lists.

Built from the same forecast interpretation the daily recommendation uses, so a
trip list and a morning suggestion never disagree about what "cold" means.

The quantities are the obvious ones - a top per day, a bottom per two days -
with the reasoning attached so the owner can see why the list is the length it
is before they disagree with it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from integrations.weather.advice import advise
from integrations.weather.models import WeatherReport
from shared.enums import LaundryStatus
from wardrobe.recommend import ItemView, target_warmth

TOPS_PER_DAY = 1.0
BOTTOMS_PER_DAY = 0.5
UNDERWEAR_PER_DAY = 1.0
MAX_PER_CATEGORY = 10

CATEGORY_RATES: dict[str, float] = {
    "top": TOPS_PER_DAY,
    "bottom": BOTTOMS_PER_DAY,
    "underwear": UNDERWEAR_PER_DAY,
    "socks": UNDERWEAR_PER_DAY,
}


@dataclass
class PackingLine:
    category: str
    quantity: int
    items: list[dict[str, str]] = field(default_factory=list)
    reason: str = ""


@dataclass
class PackingList:
    destination: str
    start_on: date
    end_on: date
    nights: int
    lines: list[PackingLine] = field(default_factory=list)
    essentials: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "destination": self.destination,
            "start_on": self.start_on.isoformat(),
            "end_on": self.end_on.isoformat(),
            "nights": self.nights,
            "lines": [
                {
                    "category": line.category,
                    "quantity": line.quantity,
                    "reason": line.reason,
                    "items": line.items,
                }
                for line in self.lines
            ],
            "essentials": self.essentials,
            "notes": self.notes,
        }


def build_packing_list(
    *,
    destination: str,
    start_on: date,
    end_on: date,
    forecast: WeatherReport,
    items: list[ItemView],
    formality: int | None = None,
) -> PackingList:
    days = max(1, (end_on - start_on).days + 1)
    packing = PackingList(
        destination=destination,
        start_on=start_on,
        end_on=end_on,
        nights=max(0, (end_on - start_on).days),
    )

    daily = forecast.daily[:days] if forecast.daily else []
    if daily:
        low = min(day.low_c for day in daily)
        high = max(day.high_c for day in daily)
        wet_days = sum(
            1 for index in range(len(daily)) if advise(forecast, day_index=index).needs_umbrella
        )
        packing.notes.append(
            f"{low:.0f}C to {high:.0f}C across {days} day{'s' if days > 1 else ''}"
        )
    else:
        low = high = forecast.current.temperature_c
        wet_days = 0
        packing.notes.append("No forecast available; packing for current conditions.")

    wanted_warmth = target_warmth(low)
    candidates = [
        item for item in items if item.available and item.laundry_status != LaundryStatus.IN_LAUNDRY
    ]

    for category, rate in CATEGORY_RATES.items():
        quantity = min(MAX_PER_CATEGORY, max(1, math.ceil(days * rate)))
        chosen = _pick(candidates, category, quantity, wanted_warmth, formality)
        packing.lines.append(
            PackingLine(
                category=category,
                quantity=quantity,
                items=chosen,
                reason=f"{rate:g} per day over {days} days",
            )
        )

    if wanted_warmth >= 4:
        outer = _pick(candidates, "outerwear", 1, wanted_warmth, formality)
        packing.lines.append(
            PackingLine(
                category="outerwear",
                quantity=1,
                items=outer,
                reason=f"lows around {low:.0f}C",
            )
        )
        packing.essentials.extend(["gloves", "warm hat"])

    if wet_days:
        water_resistant = [
            {"item_id": item.id, "name": item.name} for item in candidates if item.water_resistant
        ][:1]
        packing.lines.append(
            PackingLine(
                category="rain layer",
                quantity=1,
                items=water_resistant,
                reason=f"rain expected on {wet_days} of {days} days",
            )
        )
        if not water_resistant:
            packing.notes.append("Nothing in the wardrobe is marked water resistant.")

    if high >= 25:
        packing.essentials.extend(["sunglasses", "sunscreen"])

    shoes = _pick(candidates, "shoes", 1 if days <= 3 else 2, wanted_warmth, formality)
    packing.lines.append(
        PackingLine(
            category="shoes",
            quantity=len(shoes) or 1,
            items=shoes,
            reason="one pair, plus a second for trips over three days",
        )
    )

    return packing


def _pick(
    items: list[ItemView],
    category: str,
    quantity: int,
    wanted_warmth: int,
    formality: int | None,
) -> list[dict[str, str]]:
    """Choose garments for a category, closest to the wanted warmth first."""
    matching = [item for item in items if item.category.lower() == category.lower()]

    def sort_key(item: ItemView) -> tuple[int, int, str]:
        warmth_gap = abs(item.warmth - wanted_warmth)
        formality_gap = abs(item.formality - formality) if formality is not None else 0
        return (warmth_gap, formality_gap, item.name)

    return [
        {"item_id": item.id, "name": item.name}
        for item in sorted(matching, key=sort_key)[:quantity]
    ]
