"""Uses the configured workflow."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class QualityReport:
    mean_luma: float
    laplacian_variance: float
    blur_flag: bool
    exposure_flag: bool
    saturated_fraction: float
    low_texture: bool
    quality_score: float
    flags: tuple[str, ...]

    @property
    def may_originate_relation(self) -> bool:
        return not self.blur_flag and not self.exposure_flag and not self.low_texture


def assess_image(
    image_bytes: bytes,
    *,
    settled: bool = True,
    stale: bool = False,
    blur_threshold: float = 12.0,
) -> QualityReport:
    image = Image.open(BytesIO(image_bytes)).convert("L")
    pixels = np.asarray(image, dtype=np.float64)
    mean_luma = float(pixels.mean())
    saturated = float(np.mean((pixels <= 5.0) | (pixels >= 250.0)))
    lap = _laplacian_var(pixels)
    flags: list[str] = []
    blur_flag = lap < blur_threshold
    exposure_flag = mean_luma < 18.0 or mean_luma > 240.0 or saturated > 0.35
    low_texture = lap < blur_threshold * 0.5
    if blur_flag:
        flags.append("blur")
    if exposure_flag:
        flags.append("exposure")
    if low_texture:
        flags.append("low_texture")
    if not settled:
        flags.append("pose_unsettled")
    if stale:
        flags.append("stale_frame")
    score = 1.0
    if blur_flag:
        score -= 0.35
    if exposure_flag:
        score -= 0.25
    if not settled:
        score -= 0.2
    if stale:
        score -= 0.4
    score = max(0.0, min(1.0, score))
    return QualityReport(
        mean_luma=mean_luma,
        laplacian_variance=lap,
        blur_flag=blur_flag,
        exposure_flag=exposure_flag,
        saturated_fraction=saturated,
        low_texture=low_texture,
        quality_score=score,
        flags=tuple(flags),
    )


def _laplacian_var(pixels: np.ndarray) -> float:
    kernel = np.array([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])
    padded = np.pad(pixels, 1, mode="edge")
    acc = np.zeros_like(pixels)
    for i in range(3):
        for j in range(3):
            acc += kernel[i, j] * padded[i : i + pixels.shape[0], j : j + pixels.shape[1]]
    return float(acc.var())
