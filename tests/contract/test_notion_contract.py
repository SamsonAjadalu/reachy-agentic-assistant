"""Notion contract and allowlist behaviour.

The allowlist is the security control that matters here, so most of these tests
are about what the client refuses rather than what it fetches.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
import respx

from app.config import Settings
from integrations.notion.real import (
    API_ROOT,
    NOTION_VERSION,
    NotionClient,
    normalise_id,
)
from shared.errors import (
    IntegrationError,
    PermissionDeniedError,
    ProviderRateLimitError,
    ValidationError,
)

PAGE_ID = "1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d"
HYPHENATED = "1a2b3c4d-5e6f-7a8b-9c0d-1e2f3a4b5c6d"
OTHER_PAGE = "ffffffffffffffffffffffffffffffff"
TOKEN = "secret_FakeNotionTokenForTests1234567890"


@pytest.fixture
def notion_settings(settings: Settings) -> Settings:
    settings.notion_enabled = True
    settings.notion_token = type(settings.notion_token)(TOKEN)
    settings.notion_allowed_page_ids = [HYPHENATED]
    settings.notion_cache_ttl_seconds = 300
    return settings


@pytest.fixture
def client(notion_settings: Settings) -> NotionClient:
    return NotionClient(notion_settings)


def ok(payload: Any) -> httpx.Response:
    return httpx.Response(200, json=payload)


PAGE_PAYLOAD = {
    "id": HYPHENATED,
    "url": "https://notion.so/research-log",
    "last_edited_time": "2026-03-09T18:20:00.000Z",
    "parent": {"type": "workspace"},
    "properties": {
        "Name": {
            "type": "title",
            "title": [{"plain_text": "Manipulation research log"}],
        }
    },
}


class TestIdNormalisation:
    @pytest.mark.parametrize(
        "value",
        [
            PAGE_ID,
            HYPHENATED,
            f"https://www.notion.so/workspace/Research-Log-{PAGE_ID}",
            f"  {HYPHENATED.upper()}  ",
        ],
    )
    def test_every_spelling_reduces_to_the_same_id(self, value: str) -> None:
        """An allowlist that matches only one spelling has a hole in it."""
        assert normalise_id(value) == PAGE_ID

    @pytest.mark.parametrize("value", ["", "not-an-id", "../../etc/passwd", "12345"])
    def test_a_non_id_is_refused(self, value: str) -> None:
        with pytest.raises(ValidationError):
            normalise_id(value)


class TestAllowlist:
    async def test_a_page_outside_the_list_is_refused_without_a_request(
        self, client: NotionClient
    ) -> None:
        with respx.mock:
            route = respx.get(f"{API_ROOT}/pages/{OTHER_PAGE}").mock(return_value=ok({}))
            with pytest.raises(PermissionDeniedError):
                await client.get_page(OTHER_PAGE)
            assert route.call_count == 0

    async def test_an_allowlisted_page_is_fetched(self, client: NotionClient) -> None:
        with respx.mock:
            respx.get(f"{API_ROOT}/pages/{PAGE_ID}").mock(return_value=ok(PAGE_PAYLOAD))
            page = await client.get_page(HYPHENATED)
        assert page.title == "Manipulation research log"

    async def test_an_empty_allowlist_blocks_everything(self, notion_settings: Settings) -> None:
        notion_settings.notion_allowed_page_ids = []
        with pytest.raises(PermissionDeniedError, match="No Notion pages are allowlisted"):
            await NotionClient(notion_settings).get_page(PAGE_ID)

    @respx.mock
    async def test_search_results_outside_the_list_are_dropped(self, client: NotionClient) -> None:
        """Notion's search covers everything shared with the integration."""
        respx.post(f"{API_ROOT}/search").mock(
            return_value=ok(
                {
                    "results": [
                        PAGE_PAYLOAD,
                        {**PAGE_PAYLOAD, "id": OTHER_PAGE, "properties": {}},
                    ]
                }
            )
        )

        results = await client.search("log")

        assert [page.id for page in results] == [PAGE_ID]

    async def test_a_write_outside_the_list_is_refused(self, client: NotionClient) -> None:
        with pytest.raises(PermissionDeniedError):
            await client.append_block(OTHER_PAGE, "text")


class TestRequests:
    @respx.mock
    async def test_the_pinned_api_version_is_sent(self, client: NotionClient) -> None:
        """Notion changes response shapes between versions."""
        route = respx.get(f"{API_ROOT}/pages/{PAGE_ID}").mock(return_value=ok(PAGE_PAYLOAD))
        await client.get_page(PAGE_ID)
        assert route.calls.last.request.headers["Notion-Version"] == NOTION_VERSION

    @respx.mock
    async def test_blocks_are_flattened_to_readable_text(self, client: NotionClient) -> None:
        respx.get(f"{API_ROOT}/pages/{PAGE_ID}").mock(return_value=ok(PAGE_PAYLOAD))
        respx.get(f"{API_ROOT}/blocks/{PAGE_ID}/children").mock(
            return_value=ok(
                {
                    "results": [
                        {
                            "id": "b1",
                            "type": "heading_2",
                            "heading_2": {"rich_text": [{"plain_text": "Week 11"}]},
                        },
                        {
                            "id": "b2",
                            "type": "to_do",
                            "to_do": {
                                "rich_text": [{"plain_text": "Add error bars"}],
                                "checked": False,
                            },
                        },
                        {"id": "b3", "type": "divider", "divider": {}},
                    ],
                    "has_more": False,
                }
            )
        )

        content = await client.get_page_content(PAGE_ID)

        assert "## Week 11" in content.text
        assert "[ ] Add error bars" in content.text
        assert len(content.blocks) == 2  # the divider carries no text

    @respx.mock
    async def test_paging_follows_the_cursor(self, client: NotionClient) -> None:
        respx.get(f"{API_ROOT}/pages/{PAGE_ID}").mock(return_value=ok(PAGE_PAYLOAD))
        route = respx.get(f"{API_ROOT}/blocks/{PAGE_ID}/children").mock(
            side_effect=[
                ok(
                    {
                        "results": [
                            {
                                "id": "b1",
                                "type": "paragraph",
                                "paragraph": {"rich_text": [{"plain_text": "first"}]},
                            }
                        ],
                        "has_more": True,
                        "next_cursor": "cursor-2",
                    }
                ),
                ok(
                    {
                        "results": [
                            {
                                "id": "b2",
                                "type": "paragraph",
                                "paragraph": {"rich_text": [{"plain_text": "second"}]},
                            }
                        ],
                        "has_more": False,
                    }
                ),
            ]
        )

        content = await client.get_page_content(PAGE_ID)

        assert route.call_count == 2
        assert route.calls[1].request.url.params["start_cursor"] == "cursor-2"
        assert content.text == "first\nsecond"

    @respx.mock
    async def test_database_properties_are_flattened(self, client: NotionClient) -> None:
        respx.post(f"{API_ROOT}/databases/{PAGE_ID}/query").mock(
            return_value=ok(
                {
                    "results": [
                        {
                            "id": HYPHENATED,
                            "properties": {
                                "Name": {
                                    "type": "title",
                                    "title": [{"plain_text": "Rerun ablation"}],
                                },
                                "Status": {"type": "status", "status": {"name": "In progress"}},
                                "Tags": {
                                    "type": "multi_select",
                                    "multi_select": [{"name": "paper"}, {"name": "urgent"}],
                                },
                                "Done": {"type": "checkbox", "checkbox": False},
                                "Formula": {"type": "formula", "formula": {"number": 3}},
                            },
                        }
                    ]
                }
            )
        )

        rows = await client.query_database(PAGE_ID)

        assert rows[0].title == "Rerun ablation"
        assert rows[0].properties["Status"] == "In progress"
        assert rows[0].properties["Tags"] == ["paper", "urgent"]
        assert rows[0].properties["Done"] is False
        # Unsupported property types are skipped rather than guessed at.
        assert "Formula" not in rows[0].properties

    @respx.mock
    async def test_appending_sends_a_paragraph_block(self, client: NotionClient) -> None:
        route = respx.patch(f"{API_ROOT}/blocks/{PAGE_ID}/children").mock(
            return_value=ok({"results": [{"id": "new-block"}]})
        )

        result = await client.append_block(PAGE_ID, "Grasp success improved to 94%.")

        body = json.loads(route.calls.last.request.read())
        block = body["children"][0]
        assert block["type"] == "paragraph"
        assert block["paragraph"]["rich_text"][0]["text"]["content"].startswith("Grasp success")
        assert result["block_id"] == "new-block"


class TestCaching:
    @respx.mock
    async def test_a_repeated_read_is_served_from_cache(self, client: NotionClient) -> None:
        route = respx.get(f"{API_ROOT}/pages/{PAGE_ID}").mock(return_value=ok(PAGE_PAYLOAD))

        await client.get_page(PAGE_ID)
        await client.get_page(PAGE_ID)

        assert route.call_count == 1

    @respx.mock
    async def test_a_write_invalidates_the_cache(self, client: NotionClient) -> None:
        """Reading a stale page straight after editing it would be worse than a refetch."""
        route = respx.get(f"{API_ROOT}/pages/{PAGE_ID}").mock(return_value=ok(PAGE_PAYLOAD))
        respx.patch(f"{API_ROOT}/blocks/{PAGE_ID}/children").mock(
            return_value=ok({"results": [{"id": "new-block"}]})
        )

        await client.get_page(PAGE_ID)
        await client.append_block(PAGE_ID, "note")
        await client.get_page(PAGE_ID)

        assert route.call_count == 2

    @respx.mock
    async def test_caching_can_be_switched_off(self, notion_settings: Settings) -> None:
        notion_settings.notion_cache_ttl_seconds = 0
        client = NotionClient(notion_settings)
        route = respx.get(f"{API_ROOT}/pages/{PAGE_ID}").mock(return_value=ok(PAGE_PAYLOAD))

        await client.get_page(PAGE_ID)
        await client.get_page(PAGE_ID)

        assert route.call_count == 2


class TestFailures:
    @pytest.fixture(autouse=True)
    def _no_backoff(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Skip the real backoff; the retry count is what these tests care about."""

        async def instant(_seconds: float) -> None:
            return None

        monkeypatch.setattr("integrations.notion.real.asyncio.sleep", instant)

    @respx.mock
    async def test_a_404_is_explained_as_a_sharing_problem(self, client: NotionClient) -> None:
        """Notion returns 404 for a page that exists but was never shared."""
        respx.get(f"{API_ROOT}/pages/{PAGE_ID}").mock(return_value=httpx.Response(404, json={}))

        with pytest.raises(PermissionDeniedError, match="Share it with the integration"):
            await client.get_page(PAGE_ID)

    @respx.mock
    async def test_rate_limiting_is_retried_then_surfaced(self, client: NotionClient) -> None:
        route = respx.get(f"{API_ROOT}/pages/{PAGE_ID}").mock(
            return_value=httpx.Response(429, headers={"Retry-After": "0"}, json={})
        )

        with pytest.raises(ProviderRateLimitError):
            await client.get_page(PAGE_ID)

        assert route.call_count == 3

    @respx.mock
    async def test_a_server_error_is_retried_then_reported(self, client: NotionClient) -> None:
        respx.get(f"{API_ROOT}/pages/{PAGE_ID}").mock(return_value=httpx.Response(500, json={}))
        with pytest.raises(IntegrationError):
            await client.get_page(PAGE_ID)

    async def test_a_missing_token_is_reported_clearly(self, notion_settings: Settings) -> None:
        notion_settings.notion_token = type(notion_settings.notion_token)("")
        with pytest.raises(IntegrationError, match="NOTION_TOKEN"):
            await NotionClient(notion_settings).get_page(PAGE_ID)

    @respx.mock
    async def test_the_token_is_not_in_an_error(self, client: NotionClient) -> None:
        respx.get(f"{API_ROOT}/pages/{PAGE_ID}").mock(return_value=httpx.Response(400, json={}))
        with pytest.raises(IntegrationError) as caught:
            await client.get_page(PAGE_ID)
        assert TOKEN not in str(caught.value)
