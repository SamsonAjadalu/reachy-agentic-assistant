import sys
import types

import numpy as np

from reachy_mini_conversation_app.wake_word import WakeWordDetector


def test_wake_word_detector_uses_onnx_model_and_threshold(monkeypatch, tmp_path) -> None:
    """The detector gates audio on the configured openWakeWord score."""
    model_calls: list[tuple[list[str], str]] = []

    class FakeModel:
        def __init__(self, *, wakeword_models, inference_framework, **kwargs):
            model_calls.append((wakeword_models, inference_framework))

        def predict(self, audio):
            assert audio.dtype == np.int16
            return {"hey_reachy_exact_original": 0.31}

    package = types.ModuleType("openwakeword")
    model_module = types.ModuleType("openwakeword.model")
    model_module.Model = FakeModel  # type: ignore[attr-defined]
    package.model = model_module  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openwakeword", package)
    monkeypatch.setitem(sys.modules, "openwakeword.model", model_module)

    model_path = tmp_path / "wake.onnx"
    model_path.write_bytes(b"test")
    detector = WakeWordDetector(model_path=model_path, threshold=0.3)

    assert detector.process(np.zeros(480, dtype=np.int16), 16000) is True
    assert model_calls == [([str(model_path)], "onnx")]
