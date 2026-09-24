"""Fake Pi end-to-end: capture → persist → retrieve without a live robot."""

from __future__ import annotations

from httpx import AsyncClient

from vision.phrases import contains_metric_claim


async def test_fake_pi_look_creates_observation_and_relations(client: AsyncClient) -> None:
    response = await client.post(
        "/api/v1/vision/look",
        json={"query": "mug", "zone": "desk", "inject_labels": ["mug", "book"]},
    )
    assert response.status_code == 200
    body = response.json()
    assert not contains_metric_claim(body["phrase"])
    assert body["observation_id"]
    assert body["world_frame_status"] == "unknown"
    assert body["relations"] is not None


async def test_memory_read_is_synchronous(client: AsyncClient) -> None:
    await client.post("/api/v1/vision/look", json={"inject_labels": ["keys"]})
    response = await client.get("/api/v1/vision/memory/last-seen", params={"q": "keys"})
    assert response.status_code == 200
    assert response.json()["may_move_head"] is False
