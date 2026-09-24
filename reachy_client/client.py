"""Sync and async HTTP clients for the personal assistant API."""

from __future__ import annotations

from datetime import datetime
from typing import Any, TypeVar

import httpx

from reachy_client import models
from reachy_client._transport import (
    CORRELATION_HEADER,
    build_headers,
    map_error,
    raise_for_status,
    request_with_retry,
    should_retry,
)
from reachy_client.exceptions import ApiError, TimeoutError

T = TypeVar("T", bound=models.ApiModel)


class _BaseClient:
    """Shared connection settings for the sync and async clients."""

    def __init__(
        self,
        base_url: str,
        token: str,
        *,
        connect_timeout: float = 3.0,
        read_timeout: float = 10.0,
        write_read_timeout: float | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.connect_timeout = connect_timeout
        self.read_timeout = read_timeout
        self.write_read_timeout = write_read_timeout or max(read_timeout, 30.0)

    def _read_timeout(self, *, write: bool = False) -> httpx.Timeout:
        return httpx.Timeout(
            connect=self.connect_timeout,
            read=self.write_read_timeout if write else self.read_timeout,
            write=self.write_read_timeout if write else self.read_timeout,
            pool=self.connect_timeout,
        )


class PersonalAssistantClient(_BaseClient):
    """Synchronous client using httpx."""

    def __init__(
        self,
        base_url: str,
        token: str,
        *,
        connect_timeout: float = 3.0,
        read_timeout: float = 10.0,
        write_read_timeout: float | None = None,
    ) -> None:
        super().__init__(
            base_url,
            token,
            connect_timeout=connect_timeout,
            read_timeout=read_timeout,
            write_read_timeout=write_read_timeout,
        )
        self._client = httpx.Client(base_url=self.base_url)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> PersonalAssistantClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _get(
        self,
        path: str,
        model: type[T],
        *,
        params: dict[str, Any] | None = None,
    ) -> T:
        correlation_id, response = self._request("GET", path, params=params)
        raise_for_status(response, correlation_id=correlation_id)
        return model.model_validate(response.json())

    def _post(
        self,
        path: str,
        body: models.ApiModel | dict[str, Any] | None,
        model: type[T],
        *,
        idempotency_key: str | None = None,
        write: bool = False,
        expected_status: tuple[int, ...] = (200, 201),
    ) -> T:
        correlation_id, response = self._request(
            "POST",
            path,
            json_body=_json_body(body),
            idempotency_key=idempotency_key,
            write=write,
        )
        raise_for_status(response, correlation_id=correlation_id)
        if response.status_code not in expected_status:
            raise map_error(response, correlation_id=correlation_id)
        return model.model_validate(response.json())

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
        idempotency_key: str | None = None,
        write: bool = False,
    ) -> tuple[str, httpx.Response]:
        headers = build_headers(self.token, idempotency_key=idempotency_key)
        correlation_id = headers[CORRELATION_HEADER]

        def send() -> httpx.Response:
            return self._client.request(
                method,
                path,
                params=params,
                json=json_body,
                headers=headers,
                timeout=self._read_timeout(write=write),
            )

        try:
            response = request_with_retry(send, method=method, idempotency_key=idempotency_key)
        except TimeoutError as exc:
            exc.correlation_id = correlation_id
            raise
        except ApiError as exc:
            exc.correlation_id = correlation_id
            raise
        return correlation_id, response

    # ------------------------------------------------------------------ system
    def ping(self) -> models.PingResponse:
        return self._get("/api/v1/ping", models.PingResponse)

    def status(self) -> models.StatusResponse:
        return self._get("/api/v1/status", models.StatusResponse)

    # --------------------------------------------------------------- reminders
    def create_reminder(
        self,
        payload: models.ReminderCreate,
        *,
        idempotency_key: str | None = None,
    ) -> models.ReminderOut:
        return self._post(
            "/api/v1/reminders",
            payload,
            models.ReminderOut,
            idempotency_key=idempotency_key,
            write=True,
        )

    def list_reminders(
        self,
        *,
        status: str | None = None,
        upcoming_only: bool = False,
        limit: int = 50,
        offset: int = 0,
    ) -> models.ReminderList:
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if status is not None:
            params["status"] = status
        if upcoming_only:
            params["upcoming_only"] = True
        return self._get("/api/v1/reminders", models.ReminderList, params=params)

    def update_reminder(
        self, reminder_id: str, payload: models.ReminderUpdate
    ) -> models.ReminderOut:
        correlation_id, response = self._request(
            "PATCH",
            f"/api/v1/reminders/{reminder_id}",
            json_body=_json_body(payload),
            write=True,
        )
        raise_for_status(response, correlation_id=correlation_id)
        return models.ReminderOut.model_validate(response.json())

    def snooze_reminder(
        self, reminder_id: str, payload: models.ReminderSnooze
    ) -> models.ReminderOut:
        return self._post(
            f"/api/v1/reminders/{reminder_id}/snooze",
            payload,
            models.ReminderOut,
            write=True,
        )

    # --------------------------------------------------------------------- tasks
    def create_task(
        self,
        payload: models.TaskCreate,
        *,
        idempotency_key: str | None = None,
    ) -> models.TaskOut:
        return self._post(
            "/api/v1/tasks",
            payload,
            models.TaskOut,
            idempotency_key=idempotency_key,
            write=True,
        )

    def list_tasks(
        self,
        *,
        status: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> models.TaskList:
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if status is not None:
            params["status"] = status
        return self._get("/api/v1/tasks", models.TaskList, params=params)

    def update_task(self, task_id: str, payload: models.TaskUpdate) -> models.TaskOut:
        correlation_id, response = self._request(
            "PATCH",
            f"/api/v1/tasks/{task_id}",
            json_body=_json_body(payload),
            write=True,
        )
        raise_for_status(response, correlation_id=correlation_id)
        return models.TaskOut.model_validate(response.json())

    def complete_task(self, task_id: str) -> models.TaskOut:
        return self._post(f"/api/v1/tasks/{task_id}/complete", None, models.TaskOut, write=True)

    # --------------------------------------------------------------------- gmail
    def search_gmail(self, query: str, *, limit: int = 10) -> models.EmailListResponse:
        return self._get(
            "/api/v1/gmail/search",
            models.EmailListResponse,
            params={"q": query, "limit": limit},
        )

    def read_email(self, message_id: str, *, max_chars: int = 4000) -> models.EmailDetailResponse:
        return self._get(
            f"/api/v1/gmail/messages/{message_id}",
            models.EmailDetailResponse,
            params={"max_chars": max_chars},
        )

    def create_gmail_draft(
        self,
        payload: models.CreateDraftRequest,
        *,
        idempotency_key: str | None = None,
    ) -> models.DraftResponse:
        """Create a Gmail draft only. Nothing is transmitted."""
        return self._post(
            "/api/v1/gmail/drafts",
            payload,
            models.DraftResponse,
            idempotency_key=idempotency_key,
            write=True,
            expected_status=(201,),
        )

    def create_gmail_reply_draft(
        self,
        message_id: str,
        payload: models.CreateReplyDraftRequest,
        *,
        idempotency_key: str | None = None,
    ) -> models.DraftResponse:
        """Create a threaded reply draft only. Nothing is transmitted."""
        return self._post(
            f"/api/v1/gmail/messages/{message_id}/reply-draft",
            payload,
            models.DraftResponse,
            idempotency_key=idempotency_key,
            write=True,
            expected_status=(201,),
        )

    def list_gmail_drafts(self, *, limit: int = 20) -> models.DraftListResponse:
        return self._get(
            "/api/v1/gmail/drafts",
            models.DraftListResponse,
            params={"limit": limit},
        )

    def get_gmail_draft(self, draft_id: str) -> models.DraftResponse:
        return self._get(f"/api/v1/gmail/drafts/{draft_id}", models.DraftResponse)

    def send_existing_draft(
        self,
        payload: models.SendDraftRequest,
        *,
        idempotency_key: str | None = None,
    ) -> models.ApprovalTicket:
        """Request owner approval to send an existing draft revision."""
        return self._post(
            "/api/v1/gmail/actions/send-draft",
            payload,
            models.ApprovalTicket,
            idempotency_key=idempotency_key,
            write=True,
            expected_status=(202,),
        )

    def get_action_status(self, action_id: str) -> models.ActionStatus:
        return self._get(f"/api/v1/actions/{action_id}/status", models.ActionStatus)

    def get_approval_status(self, approval_id: str) -> models.ActionStatus:
        return self._get(f"/api/v1/approvals/{approval_id}/status", models.ActionStatus)

    def list_gmail_attachments(self, message_id: str) -> models.AttachmentListResponse:
        return self._get(
            f"/api/v1/gmail/messages/{message_id}/attachments",
            models.AttachmentListResponse,
        )

    def download_gmail_attachment(
        self, message_id: str, attachment_id: str
    ) -> models.AttachmentDownloadResponse:
        return self._post(
            f"/api/v1/gmail/messages/{message_id}/attachments/{attachment_id}/download",
            None,
            models.AttachmentDownloadResponse,
            write=True,
        )

    def archive_gmail_message(self, message_id: str) -> models.MessageLabelsResponse:
        return self._post(
            f"/api/v1/gmail/messages/{message_id}/actions/archive",
            None,
            models.MessageLabelsResponse,
            write=True,
        )

    def modify_gmail_labels(
        self, message_id: str, payload: models.ModifyLabelsRequest
    ) -> models.MessageLabelsResponse:
        return self._post(
            f"/api/v1/gmail/messages/{message_id}/actions/modify-labels",
            payload,
            models.MessageLabelsResponse,
            write=True,
        )

    # ------------------------------------------------------------------- calendar
    def get_calendar_events(
        self,
        *,
        days: int = 7,
        limit: int = 25,
        starts_after: datetime | None = None,
        ends_before: datetime | None = None,
    ) -> models.EventListResponse:
        params: dict[str, Any] = {"days": days, "limit": limit}
        if starts_after is not None:
            params["starts_after"] = starts_after.isoformat()
        if ends_before is not None:
            params["ends_before"] = ends_before.isoformat()
        return self._get("/api/v1/calendar/events", models.EventListResponse, params=params)

    def check_free_busy(self, starts_at: datetime, ends_at: datetime) -> models.FreeBusyResponse:
        return self._get(
            "/api/v1/calendar/free-busy",
            models.FreeBusyResponse,
            params={"starts_at": starts_at.isoformat(), "ends_at": ends_at.isoformat()},
        )

    def propose_calendar_event(
        self, payload: models.ProposeEventRequest
    ) -> models.EventProposalResponse:
        return self._post(
            "/api/v1/calendar/events/propose",
            payload,
            models.EventProposalResponse,
            write=True,
        )

    def request_create_calendar_event(
        self,
        payload: models.CreateEventRequest,
        *,
        idempotency_key: str | None = None,
    ) -> models.ApprovalTicket:
        return self._post(
            "/api/v1/calendar/events",
            payload,
            models.ApprovalTicket,
            idempotency_key=idempotency_key,
            write=True,
            expected_status=(202,),
        )

    def request_update_calendar_event(
        self,
        event_id: str,
        payload: models.UpdateEventRequest,
        *,
        idempotency_key: str | None = None,
    ) -> models.ApprovalTicket:
        correlation_id, response = self._request(
            "PATCH",
            f"/api/v1/calendar/events/{event_id}",
            json_body=_json_body(payload),
            idempotency_key=idempotency_key,
            write=True,
        )
        raise_for_status(response, correlation_id=correlation_id)
        if response.status_code != 202:
            raise map_error(response, correlation_id=correlation_id)
        return models.ApprovalTicket.model_validate(response.json())

    def request_delete_calendar_event(
        self,
        event_id: str,
        *,
        calendar_id: str = "primary",
        idempotency_key: str | None = None,
    ) -> models.ApprovalTicket:
        correlation_id, response = self._request(
            "DELETE",
            f"/api/v1/calendar/events/{event_id}",
            params={"calendar_id": calendar_id},
            idempotency_key=idempotency_key,
            write=True,
        )
        raise_for_status(response, correlation_id=correlation_id)
        if response.status_code != 202:
            raise map_error(response, correlation_id=correlation_id)
        return models.ApprovalTicket.model_validate(response.json())

    # ------------------------------------------------------------------- contacts
    def search_contacts(self, query: str, *, limit: int = 10) -> models.ContactListResponse:
        return self._get(
            "/api/v1/contacts/search",
            models.ContactListResponse,
            params={"q": query, "limit": limit},
        )

    # ---------------------------------------------------------------------- notion
    def search_notion(self, query: str, *, limit: int = 10) -> models.NotionPageList:
        return self._get(
            "/api/v1/notion/search",
            models.NotionPageList,
            params={"q": query, "limit": limit},
        )

    def read_notion_page(self, page_id: str, *, max_chars: int = 8000) -> models.NotionPageContent:
        return self._get(
            f"/api/v1/notion/pages/{page_id}/content",
            models.NotionPageContent,
            params={"max_chars": max_chars},
        )

    # ------------------------------------------------------------------- documents
    def search_documents(self, query: str, *, limit: int = 10) -> models.DocumentSearchResponse:
        return self._get(
            "/api/v1/documents/search",
            models.DocumentSearchResponse,
            params={"q": query, "limit": limit},
        )

    def read_document(self, path: str, *, max_chars: int = 20000) -> models.DocumentTextResponse:
        return self._get(
            "/api/v1/documents/content",
            models.DocumentTextResponse,
            params={"path": path, "max_chars": max_chars},
        )

    # ---------------------------------------------------------------------- weather
    def get_weather(
        self,
        *,
        location: str | None = None,
        latitude: float | None = None,
        longitude: float | None = None,
    ) -> models.WeatherResponse:
        params: dict[str, Any] = {}
        if location is not None:
            params["location"] = location
        if latitude is not None:
            params["latitude"] = latitude
        if longitude is not None:
            params["longitude"] = longitude
        return self._get("/api/v1/weather/current", models.WeatherResponse, params=params)

    # ---------------------------------------------------------------------- wardrobe
    def search_wardrobe(
        self,
        *,
        category: str | None = None,
        color: str | None = None,
        available_only: bool = False,
        limit: int = 50,
    ) -> models.WardrobeItemList:
        params: dict[str, Any] = {"limit": limit}
        if category is not None:
            params["category"] = category
        if color is not None:
            params["color"] = color
        if available_only:
            params["available_only"] = True
        return self._get("/api/v1/wardrobe/items", models.WardrobeItemList, params=params)

    def recommend_outfit(
        self,
        *,
        occasion: str | None = None,
        formality: int | None = None,
        day_offset: int = 0,
        limit: int = 3,
    ) -> models.OutfitRecommendationResponse:
        params: dict[str, Any] = {"day_offset": day_offset, "limit": limit}
        if occasion is not None:
            params["occasion"] = occasion
        if formality is not None:
            params["formality"] = formality
        return self._get(
            "/api/v1/wardrobe/recommend",
            models.OutfitRecommendationResponse,
            params=params,
        )

    # ----------------------------------------------------------------- workstation
    def get_workstation_status(self) -> models.WorkstationStatusResponse:
        return self._get("/api/v1/workstation/status", models.WorkstationStatusResponse)

    # ----------------------------------------------------------------------- reachy
    def list_pending_notifications(self, *, limit: int = 5) -> models.PendingNotificationList:
        return self._get(
            "/api/v1/reachy/pending",
            models.PendingNotificationList,
            params={"limit": limit},
        )

    def acknowledge_notification(self, notification_ids: list[str]) -> models.AcknowledgeResponse:
        return self._post(
            "/api/v1/reachy/pending/acknowledge",
            {"ids": notification_ids},
            models.AcknowledgeResponse,
            write=True,
        )

    # ----------------------------------------------------------------------- vision
    def get_camera_status(self) -> models.VisionCameraStatus:
        return self._get("/api/v1/vision/camera/status", models.VisionCameraStatus)

    def look_now(
        self,
        *,
        query: str | None = None,
        zone: str | None = None,
        preset_id: str | None = None,
        allow_scan: bool = False,
        idempotency_key: str | None = None,
    ) -> models.VisionObserveOut:
        return self._post(
            "/api/v1/vision/look",
            {
                "query": query,
                "zone": zone,
                "preset_id": preset_id,
                "allow_scan": allow_scan,
            },
            models.VisionObserveOut,
            idempotency_key=idempotency_key,
            expected_status=(200, 202),
        )

    def find_last_seen(self, query: str, *, limit: int = 5) -> models.VisionLastSeen:
        return self._get(
            "/api/v1/vision/memory/last-seen",
            models.VisionLastSeen,
            params={"q": query, "limit": limit},
        )

    def search_visual_memory(self, query: str, *, limit: int = 8) -> models.VisionLastSeen:
        return self._get(
            "/api/v1/vision/memory/search",
            models.VisionLastSeen,
            params={"q": query, "limit": limit},
        )

    def describe_previous_scene(self, snapshot_id: str) -> models.VisionSceneOut:
        return self._get(f"/api/v1/vision/scenes/{snapshot_id}", models.VisionSceneOut)

    def compare_visual_scenes(
        self, left_snapshot_id: str, right_snapshot_id: str
    ) -> models.VisionCompareOut:
        return self._post(
            "/api/v1/vision/scenes/compare",
            {"left_snapshot_id": left_snapshot_id, "right_snapshot_id": right_snapshot_id},
            models.VisionCompareOut,
        )

    def create_visual_watch(
        self,
        *,
        label: str,
        event: str = "change",
        zone: str | None = None,
        cooldown_seconds: int = 3600,
        idempotency_key: str | None = None,
    ) -> models.VisionWatchOut:
        return self._post(
            "/api/v1/vision/watches",
            {
                "label": label,
                "event": event,
                "zone": zone,
                "cooldown_seconds": cooldown_seconds,
            },
            models.VisionWatchOut,
            idempotency_key=idempotency_key,
            write=True,
            expected_status=(201,),
        )

    def list_visual_watches(self, *, limit: int = 10) -> models.VisionWatchList:
        return self._get("/api/v1/vision/watches", models.VisionWatchList, params={"limit": limit})

    def cancel_visual_watch(self, watch_id: str) -> models.VisionWatchOut:
        return self._post(
            f"/api/v1/vision/watches/{watch_id}/cancel",
            None,
            models.VisionWatchOut,
            write=True,
        )


class AsyncPersonalAssistantClient(_BaseClient):
    """Asynchronous client using httpx."""

    def __init__(
        self,
        base_url: str,
        token: str,
        *,
        connect_timeout: float = 3.0,
        read_timeout: float = 10.0,
        write_read_timeout: float | None = None,
    ) -> None:
        super().__init__(
            base_url,
            token,
            connect_timeout=connect_timeout,
            read_timeout=read_timeout,
            write_read_timeout=write_read_timeout,
        )
        self._client = httpx.AsyncClient(base_url=self.base_url)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> AsyncPersonalAssistantClient:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    async def ping(self) -> models.PingResponse:
        return await self._get("/api/v1/ping", models.PingResponse)

    async def status(self) -> models.StatusResponse:
        return await self._get("/api/v1/status", models.StatusResponse)

    async def create_reminder(
        self,
        payload: models.ReminderCreate,
        *,
        idempotency_key: str | None = None,
    ) -> models.ReminderOut:
        return await self._post(
            "/api/v1/reminders",
            payload,
            models.ReminderOut,
            idempotency_key=idempotency_key,
            write=True,
        )

    async def list_reminders(
        self,
        *,
        status: str | None = None,
        upcoming_only: bool = False,
        limit: int = 50,
        offset: int = 0,
    ) -> models.ReminderList:
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if status is not None:
            params["status"] = status
        if upcoming_only:
            params["upcoming_only"] = True
        return await self._get("/api/v1/reminders", models.ReminderList, params=params)

    async def update_reminder(
        self, reminder_id: str, payload: models.ReminderUpdate
    ) -> models.ReminderOut:
        correlation_id, response = await self._request(
            "PATCH",
            f"/api/v1/reminders/{reminder_id}",
            json_body=_json_body(payload),
            write=True,
        )
        raise_for_status(response, correlation_id=correlation_id)
        return models.ReminderOut.model_validate(response.json())

    async def snooze_reminder(
        self, reminder_id: str, payload: models.ReminderSnooze
    ) -> models.ReminderOut:
        return await self._post(
            f"/api/v1/reminders/{reminder_id}/snooze",
            payload,
            models.ReminderOut,
            write=True,
        )

    async def create_task(
        self,
        payload: models.TaskCreate,
        *,
        idempotency_key: str | None = None,
    ) -> models.TaskOut:
        return await self._post(
            "/api/v1/tasks",
            payload,
            models.TaskOut,
            idempotency_key=idempotency_key,
            write=True,
        )

    async def list_tasks(
        self,
        *,
        status: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> models.TaskList:
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if status is not None:
            params["status"] = status
        return await self._get("/api/v1/tasks", models.TaskList, params=params)

    async def update_task(self, task_id: str, payload: models.TaskUpdate) -> models.TaskOut:
        correlation_id, response = await self._request(
            "PATCH",
            f"/api/v1/tasks/{task_id}",
            json_body=_json_body(payload),
            write=True,
        )
        raise_for_status(response, correlation_id=correlation_id)
        return models.TaskOut.model_validate(response.json())

    async def complete_task(self, task_id: str) -> models.TaskOut:
        return await self._post(
            f"/api/v1/tasks/{task_id}/complete", None, models.TaskOut, write=True
        )

    async def search_gmail(self, query: str, *, limit: int = 10) -> models.EmailListResponse:
        return await self._get(
            "/api/v1/gmail/search",
            models.EmailListResponse,
            params={"q": query, "limit": limit},
        )

    async def read_email(
        self, message_id: str, *, max_chars: int = 4000
    ) -> models.EmailDetailResponse:
        return await self._get(
            f"/api/v1/gmail/messages/{message_id}",
            models.EmailDetailResponse,
            params={"max_chars": max_chars},
        )

    async def create_gmail_draft(
        self,
        payload: models.CreateDraftRequest,
        *,
        idempotency_key: str | None = None,
    ) -> models.DraftResponse:
        return await self._post(
            "/api/v1/gmail/drafts",
            payload,
            models.DraftResponse,
            idempotency_key=idempotency_key,
            write=True,
            expected_status=(201,),
        )

    async def create_gmail_reply_draft(
        self,
        message_id: str,
        payload: models.CreateReplyDraftRequest,
        *,
        idempotency_key: str | None = None,
    ) -> models.DraftResponse:
        return await self._post(
            f"/api/v1/gmail/messages/{message_id}/reply-draft",
            payload,
            models.DraftResponse,
            idempotency_key=idempotency_key,
            write=True,
            expected_status=(201,),
        )

    async def list_gmail_drafts(self, *, limit: int = 20) -> models.DraftListResponse:
        return await self._get(
            "/api/v1/gmail/drafts",
            models.DraftListResponse,
            params={"limit": limit},
        )

    async def get_gmail_draft(self, draft_id: str) -> models.DraftResponse:
        return await self._get(f"/api/v1/gmail/drafts/{draft_id}", models.DraftResponse)

    async def send_existing_draft(
        self,
        payload: models.SendDraftRequest,
        *,
        idempotency_key: str | None = None,
    ) -> models.ApprovalTicket:
        return await self._post(
            "/api/v1/gmail/actions/send-draft",
            payload,
            models.ApprovalTicket,
            idempotency_key=idempotency_key,
            write=True,
            expected_status=(202,),
        )

    async def get_action_status(self, action_id: str) -> models.ActionStatus:
        return await self._get(f"/api/v1/actions/{action_id}/status", models.ActionStatus)

    async def get_approval_status(self, approval_id: str) -> models.ActionStatus:
        return await self._get(f"/api/v1/approvals/{approval_id}/status", models.ActionStatus)

    async def get_calendar_events(
        self,
        *,
        days: int = 7,
        limit: int = 25,
        starts_after: datetime | None = None,
        ends_before: datetime | None = None,
    ) -> models.EventListResponse:
        params: dict[str, Any] = {"days": days, "limit": limit}
        if starts_after is not None:
            params["starts_after"] = starts_after.isoformat()
        if ends_before is not None:
            params["ends_before"] = ends_before.isoformat()
        return await self._get("/api/v1/calendar/events", models.EventListResponse, params=params)

    async def check_free_busy(
        self, starts_at: datetime, ends_at: datetime
    ) -> models.FreeBusyResponse:
        return await self._get(
            "/api/v1/calendar/free-busy",
            models.FreeBusyResponse,
            params={"starts_at": starts_at.isoformat(), "ends_at": ends_at.isoformat()},
        )

    async def propose_calendar_event(
        self, payload: models.ProposeEventRequest
    ) -> models.EventProposalResponse:
        return await self._post(
            "/api/v1/calendar/events/propose",
            payload,
            models.EventProposalResponse,
            write=True,
        )

    async def request_create_calendar_event(
        self,
        payload: models.CreateEventRequest,
        *,
        idempotency_key: str | None = None,
    ) -> models.ApprovalTicket:
        return await self._post(
            "/api/v1/calendar/events",
            payload,
            models.ApprovalTicket,
            idempotency_key=idempotency_key,
            write=True,
            expected_status=(202,),
        )

    async def request_update_calendar_event(
        self,
        event_id: str,
        payload: models.UpdateEventRequest,
        *,
        idempotency_key: str | None = None,
    ) -> models.ApprovalTicket:
        correlation_id, response = await self._request(
            "PATCH",
            f"/api/v1/calendar/events/{event_id}",
            json_body=_json_body(payload),
            idempotency_key=idempotency_key,
            write=True,
        )
        raise_for_status(response, correlation_id=correlation_id)
        if response.status_code != 202:
            raise map_error(response, correlation_id=correlation_id)
        return models.ApprovalTicket.model_validate(response.json())

    async def search_contacts(self, query: str, *, limit: int = 10) -> models.ContactListResponse:
        return await self._get(
            "/api/v1/contacts/search",
            models.ContactListResponse,
            params={"q": query, "limit": limit},
        )

    async def search_notion(self, query: str, *, limit: int = 10) -> models.NotionPageList:
        return await self._get(
            "/api/v1/notion/search",
            models.NotionPageList,
            params={"q": query, "limit": limit},
        )

    async def read_notion_page(
        self, page_id: str, *, max_chars: int = 8000
    ) -> models.NotionPageContent:
        return await self._get(
            f"/api/v1/notion/pages/{page_id}/content",
            models.NotionPageContent,
            params={"max_chars": max_chars},
        )

    async def search_documents(
        self, query: str, *, limit: int = 10
    ) -> models.DocumentSearchResponse:
        return await self._get(
            "/api/v1/documents/search",
            models.DocumentSearchResponse,
            params={"q": query, "limit": limit},
        )

    async def read_document(
        self, path: str, *, max_chars: int = 20000
    ) -> models.DocumentTextResponse:
        return await self._get(
            "/api/v1/documents/content",
            models.DocumentTextResponse,
            params={"path": path, "max_chars": max_chars},
        )

    async def get_weather(
        self,
        *,
        location: str | None = None,
        latitude: float | None = None,
        longitude: float | None = None,
    ) -> models.WeatherResponse:
        params: dict[str, Any] = {}
        if location is not None:
            params["location"] = location
        if latitude is not None:
            params["latitude"] = latitude
        if longitude is not None:
            params["longitude"] = longitude
        return await self._get("/api/v1/weather/current", models.WeatherResponse, params=params)

    async def search_wardrobe(
        self,
        *,
        category: str | None = None,
        color: str | None = None,
        available_only: bool = False,
        limit: int = 50,
    ) -> models.WardrobeItemList:
        params: dict[str, Any] = {"limit": limit}
        if category is not None:
            params["category"] = category
        if color is not None:
            params["color"] = color
        if available_only:
            params["available_only"] = True
        return await self._get("/api/v1/wardrobe/items", models.WardrobeItemList, params=params)

    async def recommend_outfit(
        self,
        *,
        occasion: str | None = None,
        formality: int | None = None,
        day_offset: int = 0,
        limit: int = 3,
    ) -> models.OutfitRecommendationResponse:
        params: dict[str, Any] = {"day_offset": day_offset, "limit": limit}
        if occasion is not None:
            params["occasion"] = occasion
        if formality is not None:
            params["formality"] = formality
        return await self._get(
            "/api/v1/wardrobe/recommend",
            models.OutfitRecommendationResponse,
            params=params,
        )

    async def get_workstation_status(self) -> models.WorkstationStatusResponse:
        return await self._get("/api/v1/workstation/status", models.WorkstationStatusResponse)

    async def list_pending_notifications(self, *, limit: int = 5) -> models.PendingNotificationList:
        return await self._get(
            "/api/v1/reachy/pending",
            models.PendingNotificationList,
            params={"limit": limit},
        )

    async def acknowledge_notification(
        self, notification_ids: list[str]
    ) -> models.AcknowledgeResponse:
        return await self._post(
            "/api/v1/reachy/pending/acknowledge",
            {"ids": notification_ids},
            models.AcknowledgeResponse,
            write=True,
        )

    async def get_camera_status(self) -> models.VisionCameraStatus:
        return await self._get("/api/v1/vision/camera/status", models.VisionCameraStatus)

    async def look_now(
        self,
        *,
        query: str | None = None,
        zone: str | None = None,
        preset_id: str | None = None,
        allow_scan: bool = False,
        idempotency_key: str | None = None,
    ) -> models.VisionObserveOut:
        return await self._post(
            "/api/v1/vision/look",
            {
                "query": query,
                "zone": zone,
                "preset_id": preset_id,
                "allow_scan": allow_scan,
            },
            models.VisionObserveOut,
            idempotency_key=idempotency_key,
            expected_status=(200, 202),
        )

    async def find_last_seen(self, query: str, *, limit: int = 5) -> models.VisionLastSeen:
        return await self._get(
            "/api/v1/vision/memory/last-seen",
            models.VisionLastSeen,
            params={"q": query, "limit": limit},
        )

    async def search_visual_memory(self, query: str, *, limit: int = 8) -> models.VisionLastSeen:
        return await self._get(
            "/api/v1/vision/memory/search",
            models.VisionLastSeen,
            params={"q": query, "limit": limit},
        )

    async def describe_previous_scene(self, snapshot_id: str) -> models.VisionSceneOut:
        return await self._get(f"/api/v1/vision/scenes/{snapshot_id}", models.VisionSceneOut)

    async def compare_visual_scenes(
        self, left_snapshot_id: str, right_snapshot_id: str
    ) -> models.VisionCompareOut:
        return await self._post(
            "/api/v1/vision/scenes/compare",
            {"left_snapshot_id": left_snapshot_id, "right_snapshot_id": right_snapshot_id},
            models.VisionCompareOut,
        )

    async def create_visual_watch(
        self,
        *,
        label: str,
        event: str = "change",
        zone: str | None = None,
        cooldown_seconds: int = 3600,
        idempotency_key: str | None = None,
    ) -> models.VisionWatchOut:
        return await self._post(
            "/api/v1/vision/watches",
            {
                "label": label,
                "event": event,
                "zone": zone,
                "cooldown_seconds": cooldown_seconds,
            },
            models.VisionWatchOut,
            idempotency_key=idempotency_key,
            write=True,
            expected_status=(201,),
        )

    async def list_visual_watches(self, *, limit: int = 10) -> models.VisionWatchList:
        return await self._get(
            "/api/v1/vision/watches", models.VisionWatchList, params={"limit": limit}
        )

    async def cancel_visual_watch(self, watch_id: str) -> models.VisionWatchOut:
        return await self._post(
            f"/api/v1/vision/watches/{watch_id}/cancel",
            None,
            models.VisionWatchOut,
            write=True,
        )

    async def _get(
        self,
        path: str,
        model: type[T],
        *,
        params: dict[str, Any] | None = None,
    ) -> T:
        correlation_id, response = await self._request("GET", path, params=params)
        raise_for_status(response, correlation_id=correlation_id)
        return model.model_validate(response.json())

    async def _post(
        self,
        path: str,
        body: models.ApiModel | dict[str, Any] | None,
        model: type[T],
        *,
        idempotency_key: str | None = None,
        write: bool = False,
        expected_status: tuple[int, ...] = (200, 201),
    ) -> T:
        correlation_id, response = await self._request(
            "POST",
            path,
            json_body=_json_body(body),
            idempotency_key=idempotency_key,
            write=write,
        )
        raise_for_status(response, correlation_id=correlation_id)
        if response.status_code not in expected_status:
            raise map_error(response, correlation_id=correlation_id)
        return model.model_validate(response.json())

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
        idempotency_key: str | None = None,
        write: bool = False,
    ) -> tuple[str, httpx.Response]:
        headers = build_headers(self.token, idempotency_key=idempotency_key)
        correlation_id = headers[CORRELATION_HEADER]

        async def send() -> httpx.Response:
            return await self._client.request(
                method,
                path,
                params=params,
                json=json_body,
                headers=headers,
                timeout=self._read_timeout(write=write),
            )

        last_exc: Exception | None = None
        for attempt in range(2):
            try:
                response = await send()
                if response.is_success or not should_retry(
                    method, idempotency_key, response.status_code
                ):
                    return correlation_id, response
            except (httpx.ConnectError, httpx.TimeoutException) as exc:
                last_exc = exc
                if not should_retry(method, idempotency_key, None) or attempt == 1:
                    raise TimeoutError(str(exc), correlation_id=correlation_id) from exc
                continue

        if last_exc is not None:
            raise TimeoutError(str(last_exc), correlation_id=correlation_id) from last_exc
        return correlation_id, response


def _json_body(body: models.ApiModel | dict[str, Any] | None) -> dict[str, Any] | None:
    if body is None:
        return None
    if isinstance(body, models.ApiModel):
        return body.model_dump(mode="json", exclude_none=True)
    return body
