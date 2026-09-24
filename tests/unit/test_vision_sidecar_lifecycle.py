"""Serial load/unload and real providers that refuse to fake."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from vision_sidecar.config import SidecarSettings
from vision_sidecar.errors import ProviderUnavailableError, VisionDisabledError
from vision_sidecar.lifecycle import ModelLifecycle
from vision_sidecar.providers.da3 import Da3MonoProvider
from vision_sidecar.providers.dinov2 import Dinov2Provider
from vision_sidecar.providers.grounding_dino import GroundingDinoProvider
from vision_sidecar.providers.mock import MockDetectionProvider, MockEmbeddingProvider
from vision_sidecar.providers.registry import build_providers
from vision_sidecar.providers.sam2 import Sam2Provider
from vision_sidecar.providers.yolo_world import YoloWorldProvider


def _settings(tmp_path: Path) -> SidecarSettings:
    data = tmp_path / "pa-data"
    data.mkdir()
    return SidecarSettings(
        app_env="test",
        app_data_dir=data,
        mock_mode=False,
        vision_allow_downloads=False,
        vision_sidecar_token="a" * 48,
        yolo_world_enabled=True,
    )


class TestLifecycle:
    async def test_unloads_between_providers_when_not_kept(self) -> None:
        lifecycle = ModelLifecycle(serial=True, keep_loaded=False)
        detect = MockDetectionProvider()
        embed = MockEmbeddingProvider()
        frame = Image.new("RGB", (16, 16), color=1)

        await lifecycle.run(
            detect, lambda: detect.detect(frame, ["cup"]), device="cpu", load_timeout=5
        )
        assert detect.loaded is False
        await lifecycle.run(embed, lambda: embed.embed_scene(frame), device="cpu", load_timeout=5)
        assert lifecycle.load_thrash >= 1
        assert embed.loaded is False


class TestRealProvidersRefuseFake:
    def test_grounding_dino_without_weights_does_not_invent_boxes(self, tmp_path: Path) -> None:
        provider = GroundingDinoProvider(_settings(tmp_path))
        frame = Image.new("RGB", (32, 24), color=2)
        with pytest.raises(ProviderUnavailableError):
            provider.detect(frame, ["mug"])

    def test_grounding_dino_load_without_cache_fails_honestly(self, tmp_path: Path) -> None:
        provider = GroundingDinoProvider(_settings(tmp_path))
        with pytest.raises(ProviderUnavailableError):
            provider.load("cpu")

    def test_yolo_world_without_weights_does_not_invent_boxes(self, tmp_path: Path) -> None:
        provider = YoloWorldProvider(_settings(tmp_path))
        frame = Image.new("RGB", (32, 24), color=3)
        with pytest.raises(ProviderUnavailableError):
            provider.detect(frame, ["mug"])

    def test_sam2_da3_dinov2_without_load_do_not_invent_output(self, tmp_path: Path) -> None:
        settings = _settings(tmp_path)
        frame = Image.new("RGB", (32, 24), color=4)
        with pytest.raises(ProviderUnavailableError):
            Sam2Provider(settings).segment(frame, boxes=[(0.1, 0.1, 0.4, 0.5)], points=[])
        with pytest.raises(ProviderUnavailableError):
            Da3MonoProvider(settings).depth(frame)
        with pytest.raises(ProviderUnavailableError):
            Dinov2Provider(settings).embed_scene(frame)
        with pytest.raises(ProviderUnavailableError):
            Dinov2Provider(settings).embed_texts(["mug"])

    def test_yolo_world_stays_disabled_until_licence_opt_in(self, tmp_path: Path) -> None:
        settings = SidecarSettings(
            app_env="test",
            app_data_dir=tmp_path / "pa-data",
            mock_mode=False,
            vision_allow_downloads=False,
            vision_sidecar_token="a" * 48,
            detection_provider="yolo_world",
            yolo_world_enabled=False,
        )
        with pytest.raises(VisionDisabledError, match="licence"):
            build_providers(settings)
