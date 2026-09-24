"""Mock providers: blake2s embeddings, not hash(); deterministic duplicates."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

from vision_sidecar.hashing import DUPLICATE_COSINE, cosine_similarity, deterministic_embedding
from vision_sidecar.providers.mock import (
    MockDepthProvider,
    MockDetectionProvider,
    MockEmbeddingProvider,
    MockSegmentationProvider,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _frame(color: tuple[int, int, int] = (40, 80, 120)) -> Image.Image:
    return Image.new("RGB", (64, 48), color=color)


class TestMockDetection:
    def test_duplicate_mugs_for_mug_query(self) -> None:
        provider = MockDetectionProvider()
        provider.load("cpu")
        detections = provider.detect(_frame(), ["white mug", "keys"])
        mug_hits = [item for item in detections if "mug" in item.query.lower()]
        assert len(mug_hits) >= 2


class TestMockDepth:
    def test_swap_inverts_ordinal_preview(self) -> None:
        provider = MockDepthProvider()
        provider.load("cpu")
        frame = _frame()
        normal = provider.depth(frame, swap=False)["depth"]
        swapped = provider.depth(frame, swap=True)["depth"]
        assert normal.relative_min < normal.relative_max
        assert swapped.preview_png_b64 != normal.preview_png_b64


class TestMockEmbeddings:
    def test_not_python_hash(self) -> None:
        source = Path(__file__).resolve().parents[2] / "vision_sidecar" / "providers" / "mock.py"
        text = source.read_text(encoding="utf-8")
        assert "blake2s" in text
        assert "hash(frame" not in text
        assert "hash(payload" not in text

    def test_controlled_duplicate_cosine(self) -> None:
        provider = MockEmbeddingProvider()
        provider.load("cpu")
        left = _frame((10, 20, 30))
        right = _frame((200, 180, 10))
        vectors = provider.embed_images([left, right])
        assert len(vectors) == 2
        cosine = cosine_similarity(vectors[0].values, vectors[1].values)
        assert cosine == pytest.approx(DUPLICATE_COSINE, abs=1e-6)

    def test_stable_across_pythonhashseed(self) -> None:
        code = (
            "from vision_sidecar.hashing import deterministic_embedding\n"
            "print(deterministic_embedding(b'mug-seed')[:8])\n"
        )

        def run(seed: str) -> str:
            env = os.environ.copy()
            env["PYTHONHASHSEED"] = seed
            env["PYTHONPATH"] = str(REPO_ROOT)
            env.pop("PYTHONSAFEPATH", None)
            return subprocess.check_output(
                [sys.executable, "-c", code],
                env=env,
                text=True,
            )

        assert run("1") == run("2")
        assert deterministic_embedding(b"mug-seed")[0] != 0.0


class TestMockSegment:
    def test_box_prompt_produces_mask(self) -> None:
        provider = MockSegmentationProvider()
        provider.load("cpu")
        masks = provider.segment(_frame(), boxes=[(0.1, 0.1, 0.4, 0.5)], points=[])
        assert len(masks) == 1
        assert masks[0].area_fraction > 0
