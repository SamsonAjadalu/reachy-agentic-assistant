"""Optional local openWakeWord gate for the Conversation App microphone."""

from __future__ import annotations
import os
import time
import logging
from typing import Any
from pathlib import Path
from logging.handlers import RotatingFileHandler

import numpy as np
from numpy.typing import NDArray
from scipy.signal import resample_poly

from reachy_mini_conversation_app.config import (
    WAKE_WORD_LOG_FILE_ENV,
    get_wake_word_threshold,
    get_wake_word_model_paths,
)
from reachy_mini_conversation_app.streaming import audio_to_int16


logger = logging.getLogger(__name__)

WAKE_WORD_RESOURCES_ENV = "REACHY_MINI_WAKE_WORD_RESOURCES"


def _onnx_resource_kwargs() -> dict[str, str]:
    """Find the shared ONNX preprocessor models when a wheel omitted them."""
    candidates: list[Path] = []
    configured = os.getenv(WAKE_WORD_RESOURCES_ENV, "").strip()
    if configured:
        candidates.append(Path(configured).expanduser())
    try:
        import openwakeword

        module_file = getattr(openwakeword, "__file__", None)
        if module_file:
            candidates.append(Path(module_file).resolve().parent / "resources" / "models")
    except Exception:
        pass
    # The Pi setup keeps the verified source checkout beside this app. This
    # fallback is only used when the installed wheel lacks package resources.
    candidates.append(Path.home() / "projects/openWakeWord_source/openwakeword/resources/models")
    for directory in candidates:
        mel = directory / "melspectrogram.onnx"
        embedding = directory / "embedding_model.onnx"
        if mel.is_file() and embedding.is_file():
            return {"melspec_model_path": str(mel), "embedding_model_path": str(embedding)}
    return {}


def _configure_file_logging() -> None:
    """Write low-volume wake-word diagnostics to a dedicated file, not the console."""
    if logger.handlers:
        return
    path = Path(os.getenv(WAKE_WORD_LOG_FILE_ENV, "logs/wakeword.log")).expanduser()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(path, maxBytes=2_000_000, backupCount=2)
    except OSError:
        logger.warning("Unable to create wake-word log file at %s", path)
        return
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s | %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


class WakeWordDetector:
    """Run configured openWakeWord ONNX models over the existing microphone frames."""

    def __init__(self, model_path: Path | list[Path] | None = None, threshold: float | None = None) -> None:
        """Load the configured openWakeWord model and threshold."""
        _configure_file_logging()
        self.model_paths = (
            [model_path] if isinstance(model_path, Path) else (model_path or get_wake_word_model_paths())
        )
        self.threshold = get_wake_word_threshold() if threshold is None else threshold
        missing = [path for path in self.model_paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Wake-word model not found: {missing[0]}")
        try:
            from openwakeword.model import Model
        except ImportError as error:
            raise RuntimeError(
                "Wake-word mode requires openwakeword; install the project dependencies before enabling it."
            ) from error
        model_paths = [str(path) for path in self.model_paths]
        # The configured files are ONNX.  OpenWakeWord currently defaults to
        # TFLite, so the framework must be explicit; otherwise construction
        # fails before MovementManager can start.
        resource_kwargs = _onnx_resource_kwargs()
        try:
            self._model: Any = Model(wakeword_models=model_paths, inference_framework="onnx", **resource_kwargs)
        except TypeError:
            # Compatibility with the older keyword name used by the pinned
            # source implementation.
            self._model = Model(wakeword_model_paths=model_paths, inference_framework="onnx", **resource_kwargs)
        self._last_score_log = 0.0
        self._last_score = 0.0
        self._interval_max = 0.0
        self._overall_peak = 0.0
        self._detected = False
        logger.info(
            "Wake-word detector ready: models=%s threshold=%.2f backend=onnx", self.model_paths, self.threshold
        )

    @property
    def last_score(self) -> float:
        """Return the most recent model score."""
        return self._last_score

    @property
    def model_path(self) -> Path:
        """Return the first configured model path for legacy callers."""
        return self.model_paths[0]

    def process(self, audio_frame: NDArray[Any], sample_rate: int) -> bool:
        """Process one existing PCM frame and return True once the threshold is reached."""
        if self._detected or audio_frame.size == 0:
            return False
        audio = audio_to_int16(np.asarray(audio_frame))
        if audio.ndim == 2:
            if audio.shape[1] > audio.shape[0]:
                audio = audio.T
            if audio.shape[1] > 1:
                audio = audio[:, 0]
        audio = audio.reshape(-1)
        if sample_rate != 16000:
            audio = np.asarray(
                np.rint(resample_poly(audio.astype(np.float32), 16000, sample_rate)),
                dtype=np.int16,
            )
        scores = self._model.predict(audio.reshape(-1))
        values = [float(value) for value in scores.values()]
        self._last_score = max(values, default=0.0)
        self._interval_max = max(self._interval_max, self._last_score)
        self._overall_peak = max(self._overall_peak, self._last_score)
        now = time.monotonic()
        if now - self._last_score_log >= 1.0:
            logger.info("Wake-word score=%.4f", self._last_score)
            self._last_score_log = now
        if self._last_score >= self.threshold:
            self._detected = True
            logger.info("Wake word detected: score=%.4f threshold=%.2f", self._last_score, self.threshold)
            return True
        return False

    def console_status(self) -> tuple[float, float, float]:
        """Return current, interval-maximum, and overall-peak scores."""
        status = (self._last_score, self._interval_max, self._overall_peak)
        self._interval_max = 0.0
        return status

    @property
    def overall_peak(self) -> float:
        """Return the highest score observed since the detector was reset."""
        return self._overall_peak

    def reset(self) -> None:
        """Resume detection after the app is muted again."""
        reset = getattr(self._model, "reset", None)
        if callable(reset):
            reset()
        self._detected = False
        self._last_score = 0.0
        self._interval_max = 0.0
        self._overall_peak = 0.0
