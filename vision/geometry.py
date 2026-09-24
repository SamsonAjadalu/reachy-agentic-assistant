"""Normalised-box geometry. No metric units anywhere."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Box:
    """Normalised xyxy in ``[0, 1]``."""

    x0: float
    y0: float
    x1: float
    y1: float

    @classmethod
    def from_xyxy(cls, box: tuple[float, float, float, float]) -> Box:
        x0, y0, x1, y1 = box
        return cls(float(x0), float(y0), float(x1), float(y1))

    def as_xyxy(self) -> tuple[float, float, float, float]:
        return (self.x0, self.y0, self.x1, self.y1)

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2.0

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2.0

    @property
    def width(self) -> float:
        return max(0.0, self.x1 - self.x0)

    @property
    def height(self) -> float:
        return max(0.0, self.y1 - self.y0)

    @property
    def area(self) -> float:
        return self.width * self.height

    def iou(self, other: Box) -> float:
        ix0 = max(self.x0, other.x0)
        iy0 = max(self.y0, other.y0)
        ix1 = min(self.x1, other.x1)
        iy1 = min(self.y1, other.y1)
        iw = max(0.0, ix1 - ix0)
        ih = max(0.0, iy1 - iy0)
        inter = iw * ih
        union = self.area + other.area - inter
        if union <= 0.0:
            return 0.0
        return inter / union

    def y_overlap_fraction(self, other: Box) -> float:
        overlap = max(0.0, min(self.y1, other.y1) - max(self.y0, other.y0))
        denom = min(self.height, other.height)
        if denom <= 0.0:
            return 0.0
        return overlap / denom

    def x_overlap_fraction(self, other: Box) -> float:
        overlap = max(0.0, min(self.x1, other.x1) - max(self.x0, other.x0))
        denom = min(self.width, other.width)
        if denom <= 0.0:
            return 0.0
        return overlap / denom

    def centroid_shift(self, other: Box) -> float:
        dx = self.cx - other.cx
        dy = self.cy - other.cy
        return (dx * dx + dy * dy) ** 0.5
