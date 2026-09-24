"""Uses the configured workflow."""

from __future__ import annotations

from shared.errors import AssistantError, IntegrationDisabledError, IntegrationError


class ProviderUnavailableError(IntegrationError):
    """Uses the configured workflow."""

    code = "provider_unavailable"
    http_status = 503


class QueueFullError(AssistantError):
    code = "queue_full"
    http_status = 429


class JobCancelledError(AssistantError):
    code = "job_cancelled"
    http_status = 409


class StageTimeoutError(AssistantError):
    code = "stage_timeout"
    http_status = 504


class GpuOomError(AssistantError):
    code = "gpu_oom"
    http_status = 503


class VisionDisabledError(IntegrationDisabledError):
    code = "vision_disabled"
    http_status = 503
