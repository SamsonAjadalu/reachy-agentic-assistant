"""Templated spatial phrases. No free text, no metric length units."""

from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import datetime, timedelta

from shared.errors import ValidationError
from vision.enums import FrameClass, Presence, SpatialPredicate

# Number + length unit is an overclaim. Time units are not length.
METRIC_LENGTH = re.compile(
    r"(?i)(?<![A-Za-z])\d+(?:[.,]\d+)?\s*(?:"
    r"cm|mm|km|m|metres?|meters?|centimetres?|centimeters?|"
    r"millimetres?|millimeters?|inches?|inch|in|ft|feet|foot|yards?|yd"
    r")\b"
)

PREDICATE_PHRASE: dict[SpatialPredicate, str] = {
    SpatialPredicate.LEFT_OF: "to the left of",
    SpatialPredicate.RIGHT_OF: "to the right of",
    SpatialPredicate.ABOVE: "above",
    SpatialPredicate.BELOW: "below",
    SpatialPredicate.IN_FRONT_OF: "in front of",
    SpatialPredicate.BEHIND: "behind",
    SpatialPredicate.OCCLUDES: "occluding",
    SpatialPredicate.SUPPORTED_BY: "supported by",
    SpatialPredicate.ON: "on",
    SpatialPredicate.INSIDE: "inside",
    SpatialPredicate.NEAR: "near",
    SpatialPredicate.PART_OF: "part of",
    SpatialPredicate.IN_REGION: "in",
    SpatialPredicate.CO_VISIBLE_WITH: "visible with",
}

TEMPLATES: dict[str, str] = {
    "disabled": "Visual memory is switched off.",
    "degraded": "I could not use the camera just now ({reason}).",
    "nothing_detected": (
        "I looked, but I did not detect a {label} in this frame. "
        "That is not the same as it being gone."
    ),
    "nothing_present": (
        "This viewpoint does not show a {label}. That is not the same as knowing the area is empty."
    ),
    "unknown_presence": "I cannot tell whether a {label} is here from this frame.",
    "abstain_identical": (
        "I see more than one similar {label} and I cannot tell them apart from here."
    ),
    "present_label": "I can see a {label} from this viewpoint.",
    "present_count_hedge": "I can see {label} in this viewpoint.",
    "last_seen_zone": "I last saw a {label} in {zone} {when}.",
    "last_seen_no_zone": "I last saw a {label} {when}.",
    "never_seen": "I have no memory of a {label}.",
    "relation_egocentric": (
        "From this viewpoint, the {subject} is {relation} the {object} (not a measured distance)."
    ),
    "relation_region": "The {subject} is {relation} {object}.",
    "scan_queued": "I am looking from another angle. I will follow up when that finishes.",
    "scan_rejected": "I could not look from another angle ({reason}).",
    "scan_blocked": "A scan is not available ({reason}).",
    "watch_created": "I will watch for {query} and notify you if it changes.",
    "watch_cancelled": "That visual watch is cancelled.",
    "watch_fired": "Visual watch: a {label} {event} in {zone}.",
    "compare_changed": "The later scene differs: {summary}.",
    "compare_same": "The scenes look materially similar.",
    "empty_search": "Nothing in visual memory matches {query}.",
    "search_hit": "The closest match is a {label} last seen {when}.",
}


def assert_no_metric_language(text: str) -> str:
    match = METRIC_LENGTH.search(text)
    if match:
        raise ValidationError(
            "Spatial phrase contains a metric length claim.",
            details={"span": match.group(0)},
        )
    return text


def render(template_id: str, **fields: str) -> str:
    try:
        template = TEMPLATES[template_id]
    except KeyError as exc:
        raise ValidationError(f"Unknown phrase template: {template_id}") from exc
    phrase = template.format_map(_Blank(fields))
    return assert_no_metric_language(phrase)


def relation_phrase(
    subject: str,
    predicate: SpatialPredicate | str,
    obj: str,
    *,
    egocentric: bool = True,
) -> str:
    if isinstance(predicate, str):
        predicate = SpatialPredicate(predicate)
    relation = PREDICATE_PHRASE[predicate]
    template_id = "relation_egocentric" if egocentric else "relation_region"
    return render(template_id, subject=subject, relation=relation, object=obj)


def qualitative_when(captured_at: datetime, now: datetime) -> str:
    delta = now - captured_at
    if delta < timedelta(minutes=2):
        return "just now"
    if delta < timedelta(hours=1):
        return "a short while ago"
    if delta < timedelta(hours=18):
        return "earlier today"
    if delta < timedelta(days=2):
        return "yesterday"
    return "a few days ago"


def presence_phrase(presence: Presence, label: str) -> str:
    if presence is Presence.PRESENT:
        return render("present_label", label=label)
    if presence is Presence.NOTHING_DETECTED:
        return render("nothing_detected", label=label)
    if presence is Presence.NOTHING_PRESENT:
        return render("nothing_present", label=label)
    if presence is Presence.ABSTAINED:
        return render("abstain_identical", label=label)
    return render("unknown_presence", label=label)


def event_verb(event_type: str) -> str:
    mapping = {
        "appeared": "appeared",
        "disappeared": "disappeared",
        "moved": "moved",
        "changed": "changed",
    }
    return mapping.get(event_type, "changed")


class _Blank(dict[str, str]):
    def __missing__(self, key: str) -> str:
        return ""


def lint_all_templates() -> None:
    samples: Mapping[str, dict[str, str]] = {
        "disabled": {},
        "degraded": {"reason": "the camera was busy"},
        "nothing_detected": {"label": "mug"},
        "nothing_present": {"label": "mug"},
        "unknown_presence": {"label": "mug"},
        "abstain_identical": {"label": "mug"},
        "present_label": {"label": "mug"},
        "present_count_hedge": {"label": "mugs"},
        "last_seen_zone": {"label": "mug", "zone": "the desk", "when": "earlier today"},
        "last_seen_no_zone": {"label": "mug", "when": "a short while ago"},
        "never_seen": {"label": "mug"},
        "relation_egocentric": {
            "subject": "mug",
            "relation": "to the left of",
            "object": "keyboard",
        },
        "relation_region": {"subject": "mug", "relation": "in", "object": "the desk"},
        "scan_queued": {},
        "scan_rejected": {"reason": "voice_active"},
        "scan_blocked": {"reason": "hourly cap"},
        "watch_created": {"query": "a mug appearing"},
        "watch_cancelled": {},
        "watch_fired": {"label": "mug", "event": "appeared", "zone": "the desk"},
        "compare_changed": {"summary": "a mug appeared"},
        "compare_same": {},
        "empty_search": {"query": "mug"},
        "search_hit": {"label": "mug", "when": "earlier today"},
    }
    for template_id in TEMPLATES:
        render(template_id, **samples.get(template_id, {}))


def contains_metric_claim(text: str) -> bool:
    return METRIC_LENGTH.search(text) is not None


def assert_no_metric_claim(text: str) -> str:
    return assert_no_metric_language(text)


def all_template_strings() -> list[str]:
    lint_all_templates()
    return list(TEMPLATES.values()) + list(PREDICATE_PHRASE.values())


def render_presence(presence: object, *, label: str, region: str | None = None) -> str:
    from vision.enums import Presence, PresenceKind

    _ = region
    if presence in {PresenceKind.ABSENT, Presence.NOTHING_PRESENT, "absent"}:
        return render("nothing_present", label=label)
    if presence in {PresenceKind.NOTHING_DETECTED, Presence.NOTHING_DETECTED, "nothing_detected"}:
        return render("nothing_detected", label=label)
    if presence in {PresenceKind.ABSTAINED, Presence.ABSTAINED, "abstained"}:
        return render("abstain_identical", label=label)
    if presence in {PresenceKind.PRESENT, Presence.PRESENT, "present"}:
        return render("present_label", label=label)
    return render("unknown_presence", label=label)


def render_last_seen(
    *,
    label: str,
    zone_name: str | None,
    when_phrase: str,
    abstained: bool = False,
) -> str:
    if abstained:
        return render("abstain_identical", label=label)
    if not when_phrase:
        return render("never_seen", label=label)
    if zone_name:
        return render("last_seen_zone", label=label, zone=zone_name, when=when_phrase)
    return render("last_seen_no_zone", label=label, when=when_phrase)


def render_relation(
    *,
    subject_label: str,
    predicate: SpatialPredicate | str,
    object_label: str | None,
    frame_class: FrameClass | str,
) -> str:
    egocentric = (
        frame_class is FrameClass.EGOCENTRIC
        or frame_class == FrameClass.EGOCENTRIC.value
        or str(frame_class) == "egocentric"
    )
    return relation_phrase(
        subject_label,
        predicate,
        object_label or "the scene",
        egocentric=egocentric,
    )
