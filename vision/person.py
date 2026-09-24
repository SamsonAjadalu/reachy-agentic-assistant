"""Person-detection policy: no face ID, no instance embeddings."""

from __future__ import annotations

from vision.enums import PERSON_LABELS


def is_person_label(label: str) -> bool:
    return label.strip().lower() in PERSON_LABELS
