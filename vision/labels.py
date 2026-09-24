"""Label normalisation and the person-detection gate."""

from __future__ import annotations

PERSON_LABELS = frozenset(
    {
        "person",
        "human",
        "people",
        "man",
        "woman",
        "child",
        "face",
        "pedestrian",
    }
)


def normalise_label(label: str) -> str:
    cleaned = " ".join(label.strip().lower().replace("_", " ").split())
    if "#" in cleaned:
        cleaned = cleaned.split("#", 1)[0].strip()
    return cleaned


def is_person_label(label: str) -> bool:
    tokens = set(normalise_label(label).split())
    return bool(tokens & PERSON_LABELS) or normalise_label(label) in PERSON_LABELS


def labels_compatible(left: str, right: str) -> bool:
    return normalise_label(left) == normalise_label(right)
