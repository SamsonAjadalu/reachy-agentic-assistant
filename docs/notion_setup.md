# Notion setup

Notion integration uses an **internal integration token**. The assistant only reads pages explicitly shared with that integration.

## Create an integration

1. [Notion integrations](https://www.notion.so/my-integrations) → **New integration**.
2. Copy the **Internal Integration Secret** (this is `NOTION_TOKEN`).
3. Open each page or database the assistant should see → **⋯** → **Connect to** → your integration.

## Configuration

In `.env`:

```env
NOTION_ENABLED=true
NOTION_TOKEN=<integration secret>
NOTION_ALLOWED_PAGE_IDS=<page-or-database-id>,<another-id>
```

`NOTION_ALLOWED_PAGE_IDS` is a comma-separated allowlist. Even if more content is shared, only listed ids are reachable through the API.

Optional:

```env
NOTION_CACHE_TTL_SECONDS=300
```

## Verify

```bash
python scripts/check_configuration.py
curl -s -H "Authorization: Bearer $PA_API_TOKEN" \
  http://127.0.0.1:8080/api/v1/integrations | jq '.[] | select(.name=="notion")'
```

## Write operations

Creating or updating Notion content goes through the approvals flow when configured as externally visible. See `integrations/notion/executors.py`.

## Mock mode

With `MOCK_MODE=true` or `NOTION_ENABLED=false`, the mock provider is used.

## Related

- [API overview](api.md) — `/api/v1/notion`
- [Operations](operations.md)
