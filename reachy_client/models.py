"""Request and response models for the personal assistant HTTP API."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ApiModel(BaseModel):
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)


# --------------------------------------------------------------------------- system
class PingResponse(ApiModel):
    pong: bool = True
    message: str
    server_time_utc: str
    server_time_local: str
    timezone: str
    request_id: str | None = None


class StatusResponse(ApiModel):
    version: str
    environment: str
    mock_mode: bool
    timezone: str
    server_time_utc: str
    started_at: str | None = None
    uptime_seconds: int | None = None
    database: dict[str, Any] = Field(default_factory=dict)
    scheduler: dict[str, Any] = Field(default_factory=dict)
    worker: dict[str, Any] = Field(default_factory=dict)


# --------------------------------------------------------------------------- reminders
class Delay(ApiModel):
    days: int = 0
    hours: int = 0
    minutes: int = 0
    seconds: int = 0


class ScheduleInput(ApiModel):
    at: datetime | None = None
    delay: Delay | None = None


class ReminderCreate(ApiModel):
    text: str
    schedule: ScheduleInput
    channel: str = "telegram"
    notes: str | None = None
    task_id: str | None = None


class ReminderUpdate(ApiModel):
    text: str | None = None
    schedule: ScheduleInput | None = None
    channel: str | None = None
    notes: str | None = None


class ReminderSnooze(ApiModel):
    delay: Delay = Field(default_factory=lambda: Delay(minutes=10))


class ReminderOut(ApiModel):
    id: str
    text: str
    status: str
    channel: str
    timezone: str
    trigger_at: datetime
    trigger_at_local: str
    original_trigger_at: datetime
    created_at: datetime
    fired_at: datetime | None = None
    completed_at: datetime | None = None
    cancelled_at: datetime | None = None
    snooze_count: int = 0
    task_id: str | None = None
    notes: str | None = None


class ReminderList(ApiModel):
    items: list[ReminderOut]
    total: int
    limit: int
    offset: int


# ------------------------------------------------------------------------------ tasks
class TaskCreate(ApiModel):
    description: str
    detail: str | None = None
    due_at: datetime | None = None
    priority: str = "normal"
    tags: list[str] = Field(default_factory=list)


class TaskUpdate(ApiModel):
    description: str | None = None
    detail: str | None = None
    due_at: datetime | None = None
    clear_due_at: bool = False
    priority: str | None = None
    status: str | None = None
    tags: list[str] | None = None


class TaskOut(ApiModel):
    id: str
    description: str
    detail: str | None = None
    status: str
    priority: str
    due_at: datetime | None = None
    due_at_local: str | None = None
    is_overdue: bool = False
    tags: list[str] = Field(default_factory=list)
    created_at: datetime
    completed_at: datetime | None = None
    cancelled_at: datetime | None = None


class TaskList(ApiModel):
    items: list[TaskOut]
    total: int
    limit: int
    offset: int


# ----------------------------------------------------------------------------- gmail
class EmailSummary(ApiModel):
    id: str
    thread_id: str
    sender: str
    sender_name: str | None = None
    recipients: list[str] = Field(default_factory=list)
    subject: str
    snippet: str
    received_at: datetime
    is_unread: bool = False
    is_important: bool = False
    has_attachments: bool = False
    labels: list[str] = Field(default_factory=list)


class EmailBody(EmailSummary):
    body_text: str = ""
    truncated: bool = False


class EmailListResponse(ApiModel):
    items: list[EmailSummary]
    total: int
    unread_count: int | None = None


class EmailDetailResponse(ApiModel):
    message: EmailBody
    content_is_untrusted: bool = True


class CreateDraftRequest(ApiModel):
    to: list[str]
    subject: str
    body: str
    cc: list[str] = Field(default_factory=list)


class CreateReplyDraftRequest(ApiModel):
    body: str


SendEmailRequest = CreateDraftRequest


class EmailDraft(ApiModel):
    id: str
    message_id: str | None = None
    thread_id: str | None = None
    to: list[str] = Field(default_factory=list)
    cc: list[str] = Field(default_factory=list)
    subject: str
    body: str
    content_hash: str
    created_at: datetime | None = None
    sent: bool = False
    in_reply_to_message_id: str | None = None
    in_reply_to: str | None = None
    references: str | None = None


class DraftListResponse(ApiModel):
    items: list[EmailDraft]
    total: int


class DraftResponse(ApiModel):
    draft: EmailDraft


class SendDraftRequest(ApiModel):
    draft_id: str
    content_hash: str | None = None


class ApprovalTicket(ApiModel):
    approval_id: str
    action_id: str
    status: str = "awaiting_approval"
    summary: str
    preview: str
    expires_at: datetime
    message: str = "Waiting for your approval in Telegram."


class ActionStatus(ApiModel):
    approval_id: str
    action_id: str
    action_type: str
    status: str
    created_at: datetime
    approved_at: datetime | None = None
    rejected_at: datetime | None = None
    executed_at: datetime | None = None
    result_summary: str | None = None


# --------------------------------------------------------------------------- calendar
class CalendarEvent(ApiModel):
    id: str
    calendar_id: str = "primary"
    title: str
    description: str | None = None
    location: str | None = None
    starts_at: datetime
    ends_at: datetime
    all_day: bool = False
    attendees: list[str] = Field(default_factory=list)
    organiser: str | None = None
    status: str = "confirmed"
    conference_url: str | None = None
    html_link: str | None = None


class EventListResponse(ApiModel):
    items: list[CalendarEvent]
    total: int


class FreeBusySlot(ApiModel):
    starts_at: datetime
    ends_at: datetime


class ProposedSlot(ApiModel):
    starts_at: datetime
    ends_at: datetime
    conflicts: list[str] = Field(default_factory=list)


class CreateEventRequest(ApiModel):
    title: str
    starts_at: datetime
    ends_at: datetime
    description: str | None = None
    location: str | None = None
    attendees: list[str] = Field(default_factory=list)
    timezone: str | None = None
    calendar_id: str = "primary"


class UpdateEventRequest(ApiModel):
    title: str | None = None
    starts_at: datetime | None = None
    ends_at: datetime | None = None
    description: str | None = None
    location: str | None = None
    attendees: list[str] | None = None
    timezone: str | None = None
    calendar_id: str = "primary"


class ProposeEventRequest(ApiModel):
    title: str
    duration_minutes: int = 60
    starts_at: datetime | None = None
    attendees: list[str] = Field(default_factory=list)
    description: str | None = None
    location: str | None = None
    timezone: str | None = None
    calendar_id: str = "primary"
    search_from: datetime | None = None
    search_to: datetime | None = None
    max_suggestions: int = 3


class EventProposalResponse(ApiModel):
    title: str
    duration_minutes: int
    suggestions: list[ProposedSlot]
    preview_text: str
    draft_create_payload: CreateEventRequest


class EmailAttachment(ApiModel):
    id: str
    filename: str
    mime_type: str
    size_bytes: int | None = None


class AttachmentListResponse(ApiModel):
    items: list[EmailAttachment]
    total: int
    content_is_untrusted: bool = True


class AttachmentDownload(ApiModel):
    attachment_id: str
    filename: str
    mime_type: str
    size_bytes: int
    path: str
    text: str | None = None
    truncated: bool = False


class AttachmentDownloadResponse(ApiModel):
    download: AttachmentDownload
    content_is_untrusted: bool = True


class ModifyLabelsRequest(ApiModel):
    add_label_ids: list[str] = Field(default_factory=list)
    remove_label_ids: list[str] = Field(default_factory=list)


class MessageLabelsResponse(ApiModel):
    message_id: str
    labels: list[str]


class FreeBusyResponse(ApiModel):
    busy: list[FreeBusySlot]
    free_slots: list[FreeBusySlot]
    queried_from: datetime
    queried_to: datetime


# --------------------------------------------------------------------------- contacts
class ContactRecord(ApiModel):
    id: str
    display_name: str
    given_name: str | None = None
    family_name: str | None = None
    emails: list[str] = Field(default_factory=list)
    phones: list[str] = Field(default_factory=list)
    organisation: str | None = None
    photo_url: str | None = None


class ContactListResponse(ApiModel):
    items: list[ContactRecord]
    total: int


# ----------------------------------------------------------------------------- notion
class NotionPageSummary(ApiModel):
    id: str
    title: str
    url: str | None = None
    last_edited_at: str | None = None
    parent_kind: str = "unknown"
    archived: bool = False


class NotionPageList(ApiModel):
    items: list[NotionPageSummary]
    total: int


class NotionBlockText(ApiModel):
    id: str
    kind: str
    text: str


class NotionPageContent(ApiModel):
    page: NotionPageSummary
    blocks: list[NotionBlockText] = Field(default_factory=list)
    text: str = ""
    truncated: bool = False
    content_is_untrusted: bool = True


# ------------------------------------------------------------------------- documents
class DocumentSearchResult(ApiModel):
    path: str
    name: str
    kind: str
    snippet: str
    score: float
    size_bytes: int
    modified_at: str
    page_count: int | None = None


class DocumentSearchResponse(ApiModel):
    items: list[DocumentSearchResult]
    total: int
    query: str


class DocumentTextResponse(ApiModel):
    path: str
    name: str
    kind: str
    text: str
    truncated: bool
    page_count: int | None = None
    content_is_untrusted: bool = True


# --------------------------------------------------------------------------- weather
class WeatherCurrent(ApiModel):
    location: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    temperature_c: float
    feels_like_c: float
    condition: str
    description: str | None = None
    humidity_percent: int | None = None
    wind_kph: float | None = None
    observed_at: datetime | None = None


class WeatherDaily(ApiModel):
    day: str
    high_c: float
    low_c: float
    condition: str
    description: str | None = None
    precipitation_probability: int | None = None


class WeatherReport(ApiModel):
    location: str
    current: WeatherCurrent
    daily: list[WeatherDaily] = Field(default_factory=list)
    from_cache: bool = False
    retrieved_at: datetime | None = None


class WeatherAdvice(ApiModel):
    warmth_band: str
    needs_umbrella: bool
    needs_winter_layers: bool
    needs_sun_protection: bool
    windy: bool
    reasons: list[str]
    summary: str


class WeatherResponse(ApiModel):
    report: WeatherReport
    advice: WeatherAdvice
    notable: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- wardrobe
class WardrobeItem(ApiModel):
    id: str
    name: str
    category: str
    primary_color: str
    subcategory: str | None = None
    formality: int = 3
    warmth: int = 3
    available: bool = True
    laundry_status: str = "clean"


class WardrobeItemList(ApiModel):
    items: list[WardrobeItem]
    total: int


class OutfitRecommendation(ApiModel):
    outfit_id: str
    outfit_name: str
    score: float
    explanation: str
    warnings: list[str] = Field(default_factory=list)
    items: list[WardrobeItem] = Field(default_factory=list)


class OutfitRecommendationResponse(ApiModel):
    recommendations: list[OutfitRecommendation]
    considered: int
    excluded: list[dict[str, str]] = Field(default_factory=list)
    weather_summary: str
    temperature_c: float


# ----------------------------------------------------------------------- workstation
class WorkstationStatus(ApiModel):
    hostname: str
    checked_at: datetime
    uptime_seconds: int
    cpu_percent: float
    cpu_count: int
    memory_total_gb: float
    memory_used_gb: float
    memory_percent: float
    swap_percent: float
    warnings: list[str] = Field(default_factory=list)


class WorkstationStatusResponse(ApiModel):
    status: WorkstationStatus
    spoken_summary: str


# ----------------------------------------------------------------------------- reachy
class PendingNotification(ApiModel):
    id: str
    kind: str
    severity: str
    spoken_text: str
    detail: str | None = None
    created_at: datetime
    resource_type: str | None = None
    resource_id: str | None = None


class PendingNotificationList(ApiModel):
    items: list[PendingNotification]
    total: int
    spoken_text: str = ""


class AcknowledgeResponse(ApiModel):
    acknowledged: int


# ------------------------------------------------------------------------ vision
class VisionCameraStatus(ApiModel):
    adapter_version: str
    camera_health: str
    image_width: int | None = None
    image_height: int | None = None
    stream_active: bool | None = None
    time_since_last_frame_ms: int | None = None
    mocked: bool = True
    phrase: str = ""
    may_move_head: bool = False


class VisionObserveOut(ApiModel):
    presence: str
    phrase: str
    deferred: bool = False
    task_id: str | None = None
    poll_url: str | None = None
    observation_id: str | None = None
    detections: list[dict[str, Any]] = Field(default_factory=list)
    relations: list[dict[str, Any]] = Field(default_factory=list)
    need_look: float | None = None
    scan_outcome: str | None = None
    world_frame_status: str = "unknown"
    may_move_head: bool = False


class VisionLastSeen(ApiModel):
    presence: str
    phrase: str
    items: list[dict[str, Any]] = Field(default_factory=list)
    abstained: bool = False
    may_move_head: bool = False
    relations: list[dict[str, Any]] = Field(default_factory=list)


class VisionSceneOut(ApiModel):
    snapshot_id: str
    state: str
    kind: str
    viewpoint_count: int
    completeness: float | None = None
    labels: list[str] = Field(default_factory=list)
    phrase: str
    may_move_head: bool = False


class VisionCompareOut(ApiModel):
    appeared: list[str] = Field(default_factory=list)
    disappeared: list[str] = Field(default_factory=list)
    phrase: str
    left: dict[str, Any] = Field(default_factory=dict)
    right: dict[str, Any] = Field(default_factory=dict)
    may_move_head: bool = False


class VisionWatchOut(ApiModel):
    id: str
    status: str
    trigger: str
    query: dict[str, Any] = Field(default_factory=dict)
    next_check_at: datetime | None = None
    cooldown_seconds: int = 3600
    last_fired_at: datetime | None = None
    phrase: str = ""
    may_move_head: bool = False


class VisionWatchList(ApiModel):
    items: list[VisionWatchOut] = Field(default_factory=list)
    total: int = 0
    limit: int = 20
    offset: int = 0
    phrase: str = ""
    may_move_head: bool = False
