"""Evaluation manifest. Point estimates only — no superiority claims."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class EvalBox(BaseModel):
    label: str
    box_xyxy: tuple[float, float, float, float]
    identity: str | None = None
    ambiguous: bool = False


class EvalSample(BaseModel):
    external_key: str
    suite: str
    scene: str
    split: str = "dev"
    ground_truth: dict[str, Any] = Field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        if key == "identical_set":
            return bool(
                self.ground_truth.get("identical_set") or self.ground_truth.get("ambiguous")
            )
        if key in self.model_fields:
            return getattr(self, key)
        return self.ground_truth.get(key, default)


class EvalManifest(BaseModel):
    name: str
    version: str
    description: str = (
        "Phase 1 measurement harness. Report point estimates and scene-clustered "
        "intervals. Report measured comparisons."
    )
    scenes: list[str] = Field(default_factory=list)
    samples: list[EvalSample] = Field(default_factory=list)
    parent: str | None = None
    notes: str | None = None

    def sample_count(self) -> int:
        return len(self.samples)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "scenes": list(self.scenes),
            "n_samples": self.sample_count(),
            "claim_policy": "no_superiority",
            "parent": self.parent,
            "notes": self.notes,
        }
