# API overview

Reachy Agentic Assistant HTTP API for Reachy Mini. Interactive schema:

- Swagger UI: `/docs`
- OpenAPI JSON: `/openapi.json`

Base URL default: `http://127.0.0.1:8080` (`APP_HOST`, `APP_PORT`).

## Authentication

Every route except **`/health`** and **`/ready`** requires:

```
Authorization: Bearer <PA_API_TOKEN>
```

Requests must originate from an address in `PA_API_ALLOWED_NETWORKS` (CIDR list). Probes are exempt so systemd and monitors can distinguish *down* from *blocked*.

## Probes (no auth)

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Liveness — process running |
| GET | `/ready` | Readiness — database OK; scheduler/worker status included |

## System

| Method | Path | Summary |
|--------|------|---------|
| GET | `/api/v1/ping` | Authenticated connectivity check |
| GET | `/api/v1/status` | Uptime, DB integrity, queue depth, scheduler jobs |
| GET | `/api/v1/integrations` | Provider inventory (real / mock / disabled) |
| GET | `/api/v1/integrations/google` | OAuth grant status (no tokens returned) |

## Core assistant

| Prefix | Tags | Notes |
|--------|------|-------|
| `/api/v1/reminders` | reminders | Create, list, snooze, cancel |
| `/api/v1/tasks` | tasks | Task list and lifecycle |
| `/api/v1/background-tasks` | background | Poll long-running work |
| `/api/v1/approvals` | approvals | Pending approval queue |

## Proactive

| Prefix | Notes |
|--------|-------|
| `/api/v1/briefings` | Assemble (`/today`), send, preferences |
| `/api/v1/alerts` | Alert rules and evaluation |

## Integrations

| Prefix | Provider |
|--------|----------|
| `/api/v1/gmail` | Drafts; attachment list/download; archive/labels (sync); send only via `/actions/send-draft` + approval |
| `/api/v1/calendar` | Reads; `POST /events/propose` dry-run; create/update/delete approval-gated |
| `/api/v1/contacts` | Google Contacts |
| `/api/v1/drive` | Google Drive |
| `/api/v1/notion` | Notion |
| `/api/v1/weather` | Weather + derived advice |
| `/api/v1/documents` | Search, read, summarise, reindex (202 + task ticket) |
| `/api/v1/wardrobe` | Items, outfits, recommendations |
| `/api/v1/workstation` | Allowlisted local scripts and service status |
| `/api/v1/reachy` | Reachy-specific hooks |

Exact request bodies, query parameters, and response models are in `/openapi.json`.

## Sync vs async behaviour

Fast local reads and writes return immediately. Operations that can take minutes (document re-index, some workstation runs) return **202 Accepted** with a `task_id` and `poll_url` under `/api/v1/background-tasks/{id}`.

## Idempotency

Write endpoints that matter for voice retries accept `Idempotency-Key` (see OpenAPI per route).

## CLI

The package declares a `pa-cli` entry point in `pyproject.toml`; when installed, use it for operator tasks. HTTP remains the primary integration surface for Reachy tools.

## Related

- [Operations](operations.md) — install, health checks, backups
- Integration setup: [Telegram](telegram_setup.md), [Google](google_setup.md), [Notion](notion_setup.md), [Wardrobe](wardrobe.md)
