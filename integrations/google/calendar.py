"""Google Calendar provider.

All timestamps cross this boundary as UTC. Google returns local times with
offsets and all-day events as bare dates; both are normalised on the way in, and
outbound events carry an explicit IANA timezone so an event created during a DST
change lands on the right hour.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta
from typing import Any

from app.config import Settings, get_settings
from app.logging_config import get_logger
from integrations.google.client import CALENDAR_BASE, GoogleClient
from integrations.google.models import CalendarEventModel, FreeBusySlot
from shared.errors import IntegrationError, ValidationError
from shared.timeutils import ensure_utc, get_zone

logger = get_logger(__name__)

CALENDAR_READ_SCOPE = "https://www.googleapis.com/auth/calendar.readonly"
CALENDAR_WRITE_SCOPE = "https://www.googleapis.com/auth/calendar.events"

MAX_EVENTS = 100
MAX_RANGE_DAYS = 366


class CalendarService:
    name = "calendar"

    def __init__(self, client: GoogleClient | None = None, settings: Settings | None = None):
        self._settings = settings or get_settings()
        self._client = client or GoogleClient(self._settings)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def list_events(
        self,
        *,
        starts_after: datetime,
        ends_before: datetime,
        calendar_id: str = "primary",
        limit: int = 25,
    ) -> list[CalendarEventModel]:
        self._client.tokens.require_scope(CALENDAR_READ_SCOPE)
        start, end = _validate_range(starts_after, ends_before)
        limit = max(1, min(limit, MAX_EVENTS))

        events: list[CalendarEventModel] = []
        async for item in self._client.paginate(
            f"{CALENDAR_BASE}/calendars/{calendar_id}/events",
            params={
                "timeMin": start.isoformat().replace("+00:00", "Z"),
                "timeMax": end.isoformat().replace("+00:00", "Z"),
                "singleEvents": "true",
                "orderBy": "startTime",
            },
            items_key="items",
            limit=limit,
            page_size_param="maxResults",
        ):
            if item.get("status") != "cancelled":
                events.append(_to_event(item, calendar_id))
        return events

    async def get_event(self, event_id: str, *, calendar_id: str = "primary") -> CalendarEventModel:
        self._client.tokens.require_scope(CALENDAR_READ_SCOPE)
        payload = await self._client.request(
            "GET", f"{CALENDAR_BASE}/calendars/{calendar_id}/events/{event_id}"
        )
        return _to_event(payload, calendar_id)

    async def free_busy(
        self, *, starts_at: datetime, ends_at: datetime, calendar_id: str = "primary"
    ) -> list[FreeBusySlot]:
        self._client.tokens.require_scope(CALENDAR_READ_SCOPE)
        start, end = _validate_range(starts_at, ends_at)
        body = await self._client.request(
            "POST",
            f"{CALENDAR_BASE}/freeBusy",
            json_body={
                "timeMin": start.isoformat().replace("+00:00", "Z"),
                "timeMax": end.isoformat().replace("+00:00", "Z"),
                "items": [{"id": calendar_id}],
            },
        )
        entries = ((body.get("calendars") or {}).get(calendar_id) or {}).get("busy") or []
        return [
            FreeBusySlot(
                starts_at=_parse_timestamp(entry["start"]),
                ends_at=_parse_timestamp(entry["end"]),
            )
            for entry in entries
        ]

    async def create_event(self, payload: dict[str, Any]) -> CalendarEventModel:
        """Create an event. Reached only through an approved action."""
        self._client.tokens.require_scope(CALENDAR_WRITE_SCOPE)
        calendar_id = str(payload.get("calendar_id") or "primary")
        body = build_event_body(payload, self._settings.app_timezone)

        created = await self._client.request(
            "POST", f"{CALENDAR_BASE}/calendars/{calendar_id}/events", json_body=body
        )
        logger.info("Created a calendar event", extra={"event_id": created.get("id")})
        return _to_event(created, calendar_id)

    async def update_event(
        self, event_id: str, payload: dict[str, Any], *, calendar_id: str = "primary"
    ) -> CalendarEventModel:
        """Patch an event. Reached only through an approved action."""
        self._client.tokens.require_scope(CALENDAR_WRITE_SCOPE)
        body = build_event_patch(payload, self._settings.app_timezone)
        updated = await self._client.request(
            "PATCH",
            f"{CALENDAR_BASE}/calendars/{calendar_id}/events/{event_id}",
            json_body=body,
        )
        logger.info("Updated a calendar event", extra={"event_id": event_id})
        return _to_event(updated, calendar_id)

    async def delete_event(self, event_id: str, *, calendar_id: str = "primary") -> None:
        self._client.tokens.require_scope(CALENDAR_WRITE_SCOPE)
        await self._client.request(
            "DELETE",
            f"{CALENDAR_BASE}/calendars/{calendar_id}/events/{event_id}",
            expect_json=False,
        )

    async def health(self) -> dict[str, Any]:
        if not self._client.tokens.is_authorised():
            return {"ok": False, "detail": "not authorised"}
        try:
            body = await self._client.request(
                "GET", f"{CALENDAR_BASE}/users/me/calendarList", params={"maxResults": 1}
            )
        except IntegrationError as exc:
            return {"ok": False, "detail": str(exc)}
        return {"ok": True, "calendars_visible": len(body.get("items") or [])}


def build_event_body(payload: dict[str, Any], default_timezone: str) -> dict[str, Any]:
    """Translate an approved payload into Google's event shape."""
    title = str(payload.get("title") or "").strip()
    if not title:
        raise ValidationError("An event needs a title.")

    starts_at = ensure_utc(_coerce_datetime(payload["starts_at"]))
    ends_at = ensure_utc(_coerce_datetime(payload["ends_at"]))
    if ends_at <= starts_at:
        raise ValidationError("The event must end after it starts.")

    timezone = str(payload.get("timezone") or default_timezone)
    body: dict[str, Any] = {
        "summary": title[:500],
        "start": {"dateTime": starts_at.isoformat().replace("+00:00", "Z"), "timeZone": timezone},
        "end": {"dateTime": ends_at.isoformat().replace("+00:00", "Z"), "timeZone": timezone},
    }
    if payload.get("description"):
        body["description"] = str(payload["description"])[:8000]
    if payload.get("location"):
        body["location"] = str(payload["location"])[:1000]
    if payload.get("attendees"):
        body["attendees"] = [{"email": str(email)} for email in payload["attendees"][:20]]
    return body


def build_event_patch(payload: dict[str, Any], default_timezone: str) -> dict[str, Any]:
    """Partial update body for Calendar events.patch."""
    body: dict[str, Any] = {}
    if payload.get("title") is not None:
        title = str(payload["title"]).strip()
        if not title:
            raise ValidationError("An event title cannot be empty.")
        body["summary"] = title[:500]

    timezone = str(payload.get("timezone") or default_timezone)
    starts = payload.get("starts_at")
    ends = payload.get("ends_at")
    if starts is not None or ends is not None:
        if starts is None or ends is None:
            raise ValidationError("Updating the time requires both starts_at and ends_at.")
        starts_at = ensure_utc(_coerce_datetime(starts))
        ends_at = ensure_utc(_coerce_datetime(ends))
        if ends_at <= starts_at:
            raise ValidationError("The event must end after it starts.")
        body["start"] = {
            "dateTime": starts_at.isoformat().replace("+00:00", "Z"),
            "timeZone": timezone,
        }
        body["end"] = {
            "dateTime": ends_at.isoformat().replace("+00:00", "Z"),
            "timeZone": timezone,
        }

    if "description" in payload:
        body["description"] = (
            str(payload["description"])[:8000] if payload["description"] is not None else ""
        )
    if "location" in payload:
        body["location"] = (
            str(payload["location"])[:1000] if payload["location"] is not None else ""
        )
    if "attendees" in payload and payload["attendees"] is not None:
        body["attendees"] = [{"email": str(email)} for email in payload["attendees"][:20]]

    if not body:
        raise ValidationError("No fields to update.")
    return body


def suggest_event_slots(
    *,
    busy: list[FreeBusySlot],
    search_from: datetime,
    search_to: datetime,
    duration: timedelta,
    preferred_start: datetime | None = None,
    max_suggestions: int = 3,
    step: timedelta | None = None,
) -> list[tuple[datetime, datetime, list[str]]]:
    """Uses the configured workflow."""
    start = ensure_utc(search_from)
    end = ensure_utc(search_to)
    if end <= start:
        raise ValidationError("The search range must end after it starts.")
    if duration <= timedelta(0) or duration > timedelta(hours=24):
        raise ValidationError("Duration must be between 1 minute and 24 hours.")

    step = step or timedelta(minutes=30)
    busy_sorted = sorted(
        ((ensure_utc(slot.starts_at), ensure_utc(slot.ends_at)) for slot in busy),
        key=lambda pair: pair[0],
    )

    def overlaps(candidate_start: datetime, candidate_end: datetime) -> list[str]:
        hits: list[str] = []
        for busy_start, busy_end in busy_sorted:
            if candidate_start < busy_end and candidate_end > busy_start:
                hits.append(f"{busy_start.isoformat()}-{busy_end.isoformat()}")
        return hits

    candidates: list[tuple[datetime, datetime, list[str]]] = []
    cursor = ensure_utc(preferred_start) if preferred_start is not None else start
    if cursor < start:
        cursor = start

    while cursor + duration <= end and len(candidates) < max_suggestions:
        candidate_end = cursor + duration
        conflicts = overlaps(cursor, candidate_end)
        if not conflicts:
            candidates.append((cursor, candidate_end, []))
        cursor += step

    # If the preferred start was busy, keep scanning from search_from.
    if not candidates and preferred_start is not None:
        return suggest_event_slots(
            busy=busy,
            search_from=start,
            search_to=end,
            duration=duration,
            preferred_start=None,
            max_suggestions=max_suggestions,
            step=step,
        )
    return candidates


def _validate_range(start: datetime, end: datetime) -> tuple[datetime, datetime]:
    start_utc, end_utc = ensure_utc(start), ensure_utc(end)
    if end_utc <= start_utc:
        raise ValidationError("The end of the range must be after its start.")
    if end_utc - start_utc > timedelta(days=MAX_RANGE_DAYS):
        raise ValidationError(f"The range supports up to {MAX_RANGE_DAYS} days.")
    return start_utc, end_utc


def _coerce_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        return _parse_timestamp(value)
    raise ValidationError(f"{value!r} is not a timestamp.")


def _parse_timestamp(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValidationError(f"{value!r} is not a valid ISO-8601 timestamp.") from exc
    return ensure_utc(parsed)


def _to_event(payload: dict[str, Any], calendar_id: str) -> CalendarEventModel:
    start_raw = payload.get("start") or {}
    end_raw = payload.get("end") or {}
    all_day = "date" in start_raw

    if all_day:
        zone = get_zone(str(start_raw.get("timeZone") or "UTC"))
        starts = datetime.combine(date.fromisoformat(start_raw["date"]), time.min, tzinfo=zone)
        ends = datetime.combine(date.fromisoformat(end_raw["date"]), time.min, tzinfo=zone)
    else:
        starts = _parse_timestamp(start_raw.get("dateTime", ""))
        ends = _parse_timestamp(end_raw.get("dateTime", ""))

    conference = None
    for entry in (payload.get("conferenceData") or {}).get("entryPoints") or []:
        if entry.get("entryPointType") == "video":
            conference = entry.get("uri")
            break

    return CalendarEventModel(
        id=str(payload.get("id", "")),
        calendar_id=calendar_id,
        title=str(payload.get("summary") or "(untitled)"),
        description=payload.get("description"),
        location=payload.get("location"),
        starts_at=starts.astimezone(UTC),
        ends_at=ends.astimezone(UTC),
        all_day=all_day,
        attendees=[
            str(person.get("email"))
            for person in payload.get("attendees") or []
            if person.get("email")
        ],
        organiser=(payload.get("organizer") or {}).get("email"),
        status=str(payload.get("status") or "confirmed"),
        conference_url=conference or payload.get("hangoutLink"),
        html_link=payload.get("htmlLink"),
    )
