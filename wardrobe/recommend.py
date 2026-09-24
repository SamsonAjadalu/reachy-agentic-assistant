"""Outfit scoring.

Every number this module produces comes with the reason it was produced. That
is the whole design constraint: the assistant says a recommendation out loud,
and the owner has to be able to disagree with it usefully - "no, that jacket is
warm enough" is only possible if the system said why it rejected the jacket.

Scoring is additive over independent components, each bounded, each with a
stated weight. There is no model here and no training data; a rule that can be
read is a rule that can be corrected.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

from integrations.weather.advice import WeatherAdvice
from shared.enums import LaundryStatus

# Component weights. They sum to 100 so a total reads as a percentage.
WEIGHT_WEATHER = 35
WEIGHT_FORMALITY = 25
WEIGHT_COLOUR = 15
WEIGHT_FRESHNESS = 15
WEIGHT_PREFERENCE = 10

# Warmth ratings run 1 (very light) to 5 (very warm). These are the target
# ratings for the outer layer at a given temperature.
WARMTH_TARGETS: list[tuple[float, int]] = [
    (-10.0, 5),
    (0.0, 5),
    (8.0, 4),
    (15.0, 3),
    (22.0, 2),
    (100.0, 1),
]

# Colours that sit badly together often enough to be worth a deduction. Neutrals
# are deliberately absent: they go with everything and flagging them would make
# the assistant tediously opinionated.
CLASHING_PAIRS = frozenset(
    {
        frozenset({"red", "pink"}),
        frozenset({"red", "orange"}),
        frozenset({"purple", "orange"}),
        frozenset({"green", "red"}),
        frozenset({"brown", "black"}),
        frozenset({"navy", "black"}),
    }
)

NEUTRALS = frozenset({"black", "white", "grey", "light grey", "charcoal", "beige", "tan", "navy"})

REPEAT_WINDOW_DAYS = 14


@dataclass
class ScoreComponent:
    name: str
    score: float
    weight: int
    reason: str

    @property
    def weighted(self) -> float:
        return self.score * self.weight


@dataclass
class OutfitScore:
    outfit_id: str
    outfit_name: str
    total: float
    components: list[ScoreComponent] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    disqualified: bool = False
    disqualified_reason: str | None = None

    def explain(self) -> str:
        if self.disqualified:
            return self.disqualified_reason or "Not suitable."
        return "; ".join(component.reason for component in self.components)

    def as_dict(self) -> dict[str, Any]:
        return {
            "outfit_id": self.outfit_id,
            "outfit_name": self.outfit_name,
            "score": round(self.total, 1),
            "explanation": self.explain(),
            "components": [
                {
                    "name": component.name,
                    "score": round(component.score, 3),
                    "weight": component.weight,
                    "contribution": round(component.weighted, 1),
                    "reason": component.reason,
                }
                for component in self.components
            ],
            "warnings": self.warnings,
            "disqualified": self.disqualified,
            "disqualified_reason": self.disqualified_reason,
        }


@dataclass
class ItemView:
    """The garment fields scoring needs, decoupled from the ORM row."""

    id: str
    name: str
    category: str
    primary_color: str
    formality: int
    warmth: int
    water_resistant: bool
    seasons: list[str]
    laundry_status: str
    available: bool
    wears_since_wash: int
    max_wears_before_wash: int
    favourite_rating: int | None = None
    last_worn_on: date | None = None


@dataclass
class OutfitView:
    id: str
    name: str
    formality: int
    items: list[ItemView]
    occasion: str | None = None
    min_temperature_c: float | None = None
    max_temperature_c: float | None = None
    user_rating: int | None = None
    last_worn_on: date | None = None
    times_worn: int = 0


def target_warmth(temperature_c: float) -> int:
    for ceiling, warmth in WARMTH_TARGETS:
        if temperature_c <= ceiling:
            return warmth
    return 1


def score_outfit(
    outfit: OutfitView,
    *,
    temperature_c: float,
    advice: WeatherAdvice,
    occasion: str | None = None,
    required_formality: int | None = None,
    today: date,
) -> OutfitScore:
    """Score one outfit, or explain why it is out of the running."""
    result = OutfitScore(outfit_id=outfit.id, outfit_name=outfit.name, total=0.0)

    # Laundry is checked first because it is the more actionable answer: a
    # garment in the wash is also unavailable, but only one of those two
    # sentences tells the owner what to do about it.
    in_laundry = [item for item in outfit.items if item.laundry_status == LaundryStatus.IN_LAUNDRY]
    if in_laundry:
        result.disqualified = True
        result.disqualified_reason = f"{in_laundry[0].name} is in the laundry"
        return result

    unavailable = [item for item in outfit.items if not item.available]
    if unavailable:
        result.disqualified = True
        result.disqualified_reason = f"{unavailable[0].name} is not available" + (
            f" and {len(unavailable) - 1} other items are not" if len(unavailable) > 1 else ""
        )
        return result

    result.components.append(_weather_component(outfit, temperature_c, advice))
    result.components.append(_formality_component(outfit, required_formality, occasion))
    result.components.append(_colour_component(outfit))
    result.components.append(_freshness_component(outfit, today))
    result.components.append(_preference_component(outfit))

    result.total = sum(component.weighted for component in result.components)

    needs_wash = [
        item for item in outfit.items if item.wears_since_wash >= item.max_wears_before_wash
    ]
    if needs_wash:
        result.warnings.append(f"{needs_wash[0].name} is due for a wash")
    if advice.needs_umbrella and not any(item.water_resistant for item in outfit.items):
        result.warnings.append("Nothing in this outfit is water resistant")

    return result


def _weather_component(
    outfit: OutfitView, temperature_c: float, advice: WeatherAdvice
) -> ScoreComponent:
    target = target_warmth(temperature_c)
    warmest = max((item.warmth for item in outfit.items), default=3)
    gap = warmest - target

    if gap == 0:
        score, reason = 1.0, f"warmth suits {temperature_c:.0f}C"
    elif gap > 0:
        score = max(0.0, 1 - gap * 0.25)
        reason = f"warmer than needed for {temperature_c:.0f}C"
    else:
        # Being underdressed for the cold is worse than being overdressed.
        score = max(0.0, 1 + gap * 0.35)
        reason = f"lighter than {temperature_c:.0f}C calls for"

    if outfit.min_temperature_c is not None and temperature_c < outfit.min_temperature_c:
        score *= 0.5
        reason = f"below this outfit's {outfit.min_temperature_c:.0f}C floor"
    if outfit.max_temperature_c is not None and temperature_c > outfit.max_temperature_c:
        score *= 0.5
        reason = f"above this outfit's {outfit.max_temperature_c:.0f}C ceiling"

    if advice.needs_umbrella:
        if any(item.water_resistant for item in outfit.items):
            score = min(1.0, score + 0.1)
            reason += ", and it handles rain"
        else:
            score = max(0.0, score - 0.2)
            reason += ", but nothing sheds rain"

    return ScoreComponent("weather", score, WEIGHT_WEATHER, reason)


def _formality_component(
    outfit: OutfitView, required: int | None, occasion: str | None
) -> ScoreComponent:
    if occasion and outfit.occasion and occasion.lower() == outfit.occasion.lower():
        return ScoreComponent("formality", 1.0, WEIGHT_FORMALITY, f"kept for {occasion} occasions")

    if required is None:
        # Without a stated occasion, internal consistency is the only signal:
        # a blazer with gym shorts is wrong at any formality level.
        spread = max(item.formality for item in outfit.items) - min(
            item.formality for item in outfit.items
        )
        score = max(0.0, 1 - spread * 0.25)
        reason = "consistent formality" if spread <= 1 else "mixes dressy and casual pieces"
        return ScoreComponent("formality", score, WEIGHT_FORMALITY, reason)

    gap = abs(outfit.formality - required)
    score = max(0.0, 1 - gap * 0.3)
    if gap == 0:
        reason = "matches the formality asked for"
    elif outfit.formality > required:
        reason = "dressier than needed"
    else:
        reason = "more casual than the occasion wants"
    return ScoreComponent("formality", score, WEIGHT_FORMALITY, reason)


def _colour_component(outfit: OutfitView) -> ScoreComponent:
    colours = [item.primary_color.lower() for item in outfit.items]
    non_neutral = [colour for colour in colours if colour not in NEUTRALS]

    clashes = [pair for pair in CLASHING_PAIRS if len(pair) == 2 and pair.issubset(set(colours))]
    if clashes:
        pair = sorted(next(iter(clashes)))
        return ScoreComponent(
            "colour", 0.3, WEIGHT_COLOUR, f"{pair[0]} with {pair[1]} is a hard pairing"
        )

    if len(non_neutral) > 2:
        return ScoreComponent("colour", 0.6, WEIGHT_COLOUR, "three or more strong colours")
    if not non_neutral:
        return ScoreComponent("colour", 0.85, WEIGHT_COLOUR, "all neutrals")
    return ScoreComponent("colour", 1.0, WEIGHT_COLOUR, "colours work together")


def _freshness_component(outfit: OutfitView, today: date) -> ScoreComponent:
    """Penalise repeats, most strongly for the most recent ones."""
    if outfit.last_worn_on is None:
        return ScoreComponent("freshness", 1.0, WEIGHT_FRESHNESS, "not worn before")

    days = (today - outfit.last_worn_on).days
    if days < 0:
        days = 0
    if days >= REPEAT_WINDOW_DAYS:
        return ScoreComponent("freshness", 1.0, WEIGHT_FRESHNESS, f"last worn {days} days ago")

    score = days / REPEAT_WINDOW_DAYS
    when = "today" if days == 0 else ("yesterday" if days == 1 else f"{days} days ago")
    return ScoreComponent("freshness", score, WEIGHT_FRESHNESS, f"worn {when}")


def _preference_component(outfit: OutfitView) -> ScoreComponent:
    ratings = [item.favourite_rating for item in outfit.items if item.favourite_rating is not None]
    if outfit.user_rating is not None:
        score = (outfit.user_rating - 1) / 4
        return ScoreComponent(
            "preference", score, WEIGHT_PREFERENCE, f"you rated this {outfit.user_rating}/5"
        )
    if ratings:
        average = sum(ratings) / len(ratings)
        return ScoreComponent(
            "preference",
            (average - 1) / 4,
            WEIGHT_PREFERENCE,
            f"favourite items average {average:.1f}/5",
        )
    return ScoreComponent("preference", 0.5, WEIGHT_PREFERENCE, "no rating yet")


def rank(
    outfits: list[OutfitView],
    *,
    temperature_c: float,
    advice: WeatherAdvice,
    occasion: str | None = None,
    required_formality: int | None = None,
    today: date | None = None,
    limit: int = 3,
) -> list[OutfitScore]:
    """Score every outfit and return the best, ties broken deterministically."""
    today = today or date.today()
    scored = [
        score_outfit(
            outfit,
            temperature_c=temperature_c,
            advice=advice,
            occasion=occasion,
            required_formality=required_formality,
            today=today,
        )
        for outfit in outfits
    ]
    eligible = [score for score in scored if not score.disqualified]
    # Name is the final tiebreak so the same wardrobe always produces the same
    # ordering; a recommendation that changes between identical calls is a bug
    # the owner cannot report.
    eligible.sort(key=lambda score: (-score.total, score.outfit_name))
    return eligible[:limit]


def suggest_laundry(items: list[ItemView], *, today: date) -> list[dict[str, Any]]:
    """Which garments are at or past their wear limit."""
    due: list[dict[str, Any]] = []
    for item in items:
        if item.laundry_status == LaundryStatus.IN_LAUNDRY:
            continue
        if item.wears_since_wash >= item.max_wears_before_wash:
            due.append(
                {
                    "item_id": item.id,
                    "name": item.name,
                    "wears_since_wash": item.wears_since_wash,
                    "limit": item.max_wears_before_wash,
                    "reason": f"worn {item.wears_since_wash} times since the last wash",
                }
            )
    return sorted(due, key=lambda entry: -int(entry["wears_since_wash"]))


def rotation_gaps(items: list[ItemView], *, today: date, days: int = 90) -> list[dict[str, Any]]:
    """Garments that have not been worn in a long time."""
    cutoff = today - timedelta(days=days)
    forgotten = [
        {
            "item_id": item.id,
            "name": item.name,
            "category": item.category,
            "last_worn_on": item.last_worn_on.isoformat() if item.last_worn_on else None,
            "reason": (
                "unworn"
                if item.last_worn_on is None
                else f"not worn since {item.last_worn_on.isoformat()}"
            ),
        }
        for item in items
        if item.available and (item.last_worn_on is None or item.last_worn_on < cutoff)
    ]
    return sorted(forgotten, key=lambda entry: str(entry["last_worn_on"] or ""))
