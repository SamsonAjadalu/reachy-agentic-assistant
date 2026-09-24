"""Turning a briefing into words.

A formatter takes the assembled sections and renders them for one destination.
The interface exists so a future formatter - a language model on the Pi, say -
can be swapped in without the assembly logic changing, and so the default
remains a template that produces the same output for the same input every time.

Untrusted text (email subjects, calendar titles from shared invitations) is
escaped by the channel, not here; these formatters emit plain text and let the
Telegram layer do its own escaping.
"""

from __future__ import annotations

from typing import Protocol

from proactive.sections import Section

MAX_SPOKEN_ITEMS = 3


class BriefingFormatter(Protocol):
    name: str

    def render(self, greeting: str, sections: list[Section]) -> str: ...


class TemplateFormatter:
    """The default. Deterministic, readable, and safe to diff in tests."""

    name = "template"

    def render(self, greeting: str, sections: list[Section]) -> str:
        lines = [greeting, ""]

        for section in sections:
            if not section.available:
                if section.note:
                    lines.append(f"{section.title}: {section.note}")
                continue

            lines.append(f"{section.title}")
            if section.is_empty:
                lines.append(f"  {section.summary}")
            else:
                lines.extend(f"  - {line}" for line in _describe(section))
            if section.note:
                lines.append(f"  ({section.note})")
            lines.append("")

        return "\n".join(lines).rstrip() + "\n"


class SpokenFormatter:
    """For Reachy.

    Reads as continuous prose with no headings or bullets, and stops after the
    few things worth saying out loud: a list of twelve items is a screen, not a
    sentence.
    """

    name = "spoken"

    def render(self, greeting: str, sections: list[Section]) -> str:
        parts = [greeting.rstrip(".") + "."]
        for section in sections:
            if not section.available or not section.summary:
                continue
            parts.append(section.summary)
            if section.items and section.name in {"calendar", "tasks"}:
                highlights = _describe(section)[:MAX_SPOKEN_ITEMS]
                if highlights:
                    parts.append("That is " + _join_naturally(highlights) + ".")
        return " ".join(parts)


class DigestFormatter:
    """Summaries only. Suitable for a notification preview."""

    name = "digest"

    def render(self, greeting: str, sections: list[Section]) -> str:
        lines = [greeting]
        lines.extend(
            f"{section.title}: {section.summary}"
            for section in sections
            if section.available and section.summary
        )
        return "\n".join(lines)


_FORMATTERS: dict[str, BriefingFormatter] = {
    formatter.name: formatter
    for formatter in (TemplateFormatter(), SpokenFormatter(), DigestFormatter())
}


def get_formatter(name: str) -> BriefingFormatter:
    """Fall back to the template rather than failing a briefing over a typo."""
    return _FORMATTERS.get(name, _FORMATTERS["template"])


def register_formatter(formatter: BriefingFormatter) -> None:
    _FORMATTERS[formatter.name] = formatter


def formatter_names() -> list[str]:
    return sorted(_FORMATTERS)


def _describe(section: Section) -> list[str]:
    """One line per item, shaped by what the section holds."""
    lines: list[str] = []
    for item in section.items:
        if section.name == "calendar":
            location = f" at {item['location']}" if item.get("location") else ""
            lines.append(f"{item['local_time']}: {item['title']}{location}")
        elif section.name == "tasks":
            marker = " (overdue)" if item.get("overdue") else ""
            lines.append(f"{item['title']}{marker}")
        elif section.name == "reminders":
            lines.append(f"{item['local_time']}: {item['title']}")
        elif section.name == "email":
            lines.append(f"{item['from']}: {item['subject']}")
        elif section.name == "laundry":
            lines.append(str(item["name"]))
        elif section.name == "outfit":
            lines.append(f"{item['outfit_name']} - {item['explanation']}")
        elif section.name == "weather":
            lines.append(f"{item['description']}, {item['low_c']} to {item['high_c']} degrees")
        else:
            lines.append(section.summary)
            break
    return lines


def _join_naturally(parts: list[str]) -> str:
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + f", and {parts[-1]}"
