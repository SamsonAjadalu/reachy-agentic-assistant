"""Visual evaluation harness. Point estimates only — no superiority claims."""

from __future__ import annotations

from typing import Any

from vision.eval.harness import run_eval, toy_manifest
from vision.eval.manifest import EvalManifest, EvalSample
from vision.eval.metrics import MetricValue, abstention_rates, scene_cluster_bootstrap
from vision.eval.metrics import detection_pr as detection_pr_counts
from vision.geometry import Box
from vision.phrases import contains_metric_claim

SUPERIORITY_TERMS = (
    "significantly better",
    "statistically superior",
    "outperforms",
    "state of the art",
    "sota",
)


def report_header() -> str:
    return (
        "Phase 1 visual evaluation. Point estimates with scene-clustered intervals. "
        "This is not a claim of statistical superiority."
    )


def assert_no_superiority_language(text: str) -> str:
    blob = text.lower()
    for term in SUPERIORITY_TERMS:
        if term in blob:
            raise AssertionError(
                f"Eval text reports measured results without superiority claims ({term!r})."
            )
    return text


def cluster_bootstrap(
    values_by_scene: dict[str, list[float]],
    n_resample: int = 200,
) -> tuple[float | None, float | None, float | None]:
    return scene_cluster_bootstrap(values_by_scene, draws=n_resample)


def detection_pr(
    predicted: list[Any],
    truth: list[Any],
    *,
    iou_threshold: float = 0.5,
) -> MetricValue:
    if not predicted and not truth:
        return MetricValue("detection_pr", None, 0, note="undefined: no predictions and no truth")
    boxes_p = _as_boxes(predicted)
    boxes_t = _as_boxes(truth)
    precision, _recall, _tp, _fp, _fn = detection_pr_counts(
        boxes_p, boxes_t, iou_threshold=iou_threshold
    )
    return MetricValue("detection_pr", precision, len(predicted) + len(truth))


def _as_boxes(items: list[Any]) -> list[tuple[str, Box]]:
    out: list[tuple[str, Box]] = []
    for item in items:
        if isinstance(item, tuple) and len(item) == 2:
            label, box = item
            if isinstance(box, Box):
                out.append((str(label), box))
            else:
                out.append((str(label), Box.from_xyxy(tuple(box))))
        elif isinstance(item, dict):
            out.append((str(item["label"]), Box.from_xyxy(tuple(item["box_xyxy"]))))
    return out


def identity_with_abstain(decisions: list[str], truth_ok: list[bool]) -> dict[str, MetricValue]:
    abstain, _wrong_ambiguous = abstention_rates(
        decisions, truth_ambiguous=[not ok for ok in truth_ok]
    )
    n = len(decisions)
    confident_wrong = 0
    for decision, ok in zip(decisions, truth_ok, strict=True):
        if decision in {"matched", "new"} and not ok:
            confident_wrong += 1
    return {
        "abstention_rate": MetricValue("abstention_rate", abstain, n),
        "confident_wrong_rate": MetricValue(
            "confident_wrong_rate", confident_wrong / n if n else 0.0, n
        ),
    }


def faithfulness_audit(phrases: list[str], cited: list[bool]) -> dict[str, MetricValue]:
    _ = cited
    n = len(phrases)
    metric_hits = sum(1 for phrase in phrases if contains_metric_claim(phrase))
    return {
        "metric_distance_claims": MetricValue("metric_distance_claims", float(metric_hits), n),
        "faithfulness": MetricValue(
            "faithfulness", 1.0 if metric_hits == 0 else 0.0, n, note="Invariant: must be 1.0"
        ),
    }


def sample_dev_manifest() -> EvalManifest:
    manifest = EvalManifest(
        name="phase1-dev",
        version="v0",
        scenes=["tidy_desk"],
        samples=[
            EvalSample(
                external_key="desk-1",
                suite="detection",
                scene="tidy_desk",
                ground_truth={
                    "boxes": [{"label": "mug", "box_xyxy": [0.1, 0.1, 0.3, 0.3]}],
                    "identical_set": False,
                    "ambiguous": False,
                },
            ),
            EvalSample(
                external_key="mugs-identical",
                suite="identity",
                scene="tidy_desk",
                ground_truth={
                    "boxes": [
                        {"label": "mug", "box_xyxy": [0.1, 0.1, 0.25, 0.25]},
                        {"label": "mug", "box_xyxy": [0.4, 0.1, 0.55, 0.25]},
                    ],
                    "identical_set": True,
                    "ambiguous": True,
                },
            ),
        ],
        notes="Synthetic only. Live Reachy capture is a later operator session.",
    )
    return manifest


__all__ = [
    "EvalManifest",
    "EvalSample",
    "MetricValue",
    "assert_no_superiority_language",
    "cluster_bootstrap",
    "detection_pr",
    "faithfulness_audit",
    "identity_with_abstain",
    "report_header",
    "run_eval",
    "sample_dev_manifest",
    "toy_manifest",
]
