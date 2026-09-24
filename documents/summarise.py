"""Extractive summarisation.

Deliberately not generative. The assistant already has a language model for
phrasing; what this needs to provide is a faithful selection of sentences that
are actually in the document, so a summary can never invent a finding that the
paper does not contain.

The ranking is classic TF-IDF-ish scoring over sentences, with a small position
bonus because the first sentences of a document usually carry its subject.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

MIN_SENTENCE_CHARS = 25
MAX_SENTENCE_CHARS = 400
POSITION_BONUS = 0.15

# Words that appear everywhere and would otherwise dominate the scoring.
STOP_WORDS = frozenset(
    """
    a an and are as at be been but by for from had has have he her his i if in into is it its
    of on or our that the their them there these they this to was we were what when where which
    who will with would you your not no can could should may might do does did done than then
    """.split()
)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
_WORD = re.compile(r"[a-zA-Z][a-zA-Z'-]+")


@dataclass
class Summary:
    sentences: list[str]
    text: str
    keywords: list[str]
    source_sentence_count: int

    @property
    def is_empty(self) -> bool:
        return not self.sentences


def split_sentences(text: str) -> list[str]:
    candidates = _SENTENCE_SPLIT.split(text.replace("\n", " "))
    return [
        sentence.strip()
        for sentence in candidates
        if MIN_SENTENCE_CHARS <= len(sentence.strip()) <= MAX_SENTENCE_CHARS
    ]


def keywords(text: str, *, limit: int = 8) -> list[str]:
    counts = Counter(
        word.lower()
        for word in _WORD.findall(text)
        if len(word) > 3 and word.lower() not in STOP_WORDS
    )
    return [word for word, _ in counts.most_common(limit)]


def summarise(text: str, *, max_sentences: int = 3) -> Summary:
    """Pick the most representative sentences, in their original order.

    Keeping source order matters: reordering by score produces summaries that
    read as non-sequiturs when spoken aloud.
    """
    sentences = split_sentences(text)
    if not sentences:
        stripped = " ".join(text.split())
        return Summary(
            sentences=[stripped[:300]] if stripped else [],
            text=stripped[:300],
            keywords=keywords(text),
            source_sentence_count=0,
        )

    frequencies = Counter(
        word.lower()
        for word in _WORD.findall(text)
        if len(word) > 3 and word.lower() not in STOP_WORDS
    )
    if not frequencies:
        chosen = sentences[:max_sentences]
        return Summary(
            sentences=chosen,
            text=" ".join(chosen),
            keywords=[],
            source_sentence_count=len(sentences),
        )

    highest = max(frequencies.values())
    scored: list[tuple[float, int, str]] = []
    for index, sentence in enumerate(sentences):
        words = [
            word.lower()
            for word in _WORD.findall(sentence)
            if len(word) > 3 and word.lower() not in STOP_WORDS
        ]
        if not words:
            continue
        # Length-normalised so a long sentence does not win on volume alone.
        score = sum(frequencies[word] / highest for word in words) / math.sqrt(len(words))
        if index < 3:
            score *= 1 + POSITION_BONUS
        scored.append((score, index, sentence))

    scored.sort(key=lambda item: item[0], reverse=True)
    picked = sorted(scored[:max_sentences], key=lambda item: item[1])
    chosen = [sentence for _, _, sentence in picked]

    return Summary(
        sentences=chosen,
        text=" ".join(chosen),
        keywords=keywords(text),
        source_sentence_count=len(sentences),
    )
