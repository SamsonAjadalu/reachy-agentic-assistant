"""Run a visual evaluation from a manifest. No superiority language."""

from __future__ import annotations

from typing import Any

from vision.eval.manifest import EvalManifest, EvalSample
from vision.eval.metrics import MetricValue, abstention_rates, detection_pr, scene_cluster_bootstrap
from vision.faithfulness import audit_payload
from vision.geometry import Box


def run_eval(
    manifest: EvalManifest,
    predictions: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """``predictions`` keyed by ``external_key``.

    Each prediction: ``{boxes: [{label, box_xyxy}], decision, payload}``.
    """
    precisions: dict[str, list[float]] = {}
    recalls: dict[str, list[float]] = {}
    decisions: list[str] = []
    ambiguous: list[bool] = []
    faithfulness_fail = 0
    for sample in manifest.samples:
        pred = predictions.get(sample.external_key, {})
        truth_boxes = [
            (str(item["label"]), Box.from_xyxy(tuple(item["box_xyxy"])))
            for item in sample.ground_truth.get("boxes") or []
        ]
        pred_boxes = [
            (str(item["label"]), Box.from_xyxy(tuple(item["box_xyxy"])))
            for item in pred.get("boxes") or []
        ]
        precision, recall, _tp, _fp, _fn = detection_pr(pred_boxes, truth_boxes)
        precisions.setdefault(sample.scene, []).append(precision)
        recalls.setdefault(sample.scene, []).append(recall)
        decisions.append(str(pred.get("decision") or "new"))
        ambiguous.append(bool(sample.ground_truth.get("ambiguous")))
        payload = pred.get("payload") or {}
        if payload and not audit_payload(payload)["ok"]:
            faithfulness_fail += 1

    p_point, p_lo, p_hi = scene_cluster_bootstrap(precisions)
    r_point, r_lo, r_hi = scene_cluster_bootstrap(recalls)
    abstain, confident_wrong = abstention_rates(decisions, truth_ambiguous=ambiguous)
    metrics = [
        MetricValue("detection_precision", p_point, manifest.sample_count(), p_lo, p_hi),
        MetricValue("detection_recall", r_point, manifest.sample_count(), r_lo, r_hi),
        MetricValue("abstention_rate", abstain, len(decisions)),
        MetricValue(
            "confident_wrong_rate",
            confident_wrong,
            len(decisions),
            note="Must stay low; identical objects should abstain.",
        ),
        MetricValue(
            "faithfulness_violations",
            float(faithfulness_fail),
            manifest.sample_count(),
            note="Invariant: must be 0.",
        ),
    ]
    result = {
        "dataset": manifest.name,
        "version": manifest.version,
        "n_samples": manifest.sample_count(),
        "n_scenes": len(manifest.scenes),
        "disclaimer": (
            "Point estimates with scene-clustered intervals. "
            "This harness does not support claims of statistical superiority."
        ),
        "metrics": [item.as_dict() for item in metrics],
    }
    forbidden = (
        "significantly better",
        "statistically superior",
        "outperforms",
        "state of the art",
        "sota",
    )
    blob = str(result).lower()
    if any(term in blob for term in forbidden):
        raise ValueError("Eval report reports measured intervals without superiority claims.")
    return result


def toy_manifest() -> EvalManifest:
    return EvalManifest(
        name="phase1-synthetic",
        version="v0",
        scenes=["tidy_desk"],
        samples=[
            EvalSample(
                external_key="desk-1",
                suite="detection",
                scene="tidy_desk",
                ground_truth={
                    "boxes": [{"label": "mug", "box_xyxy": [0.1, 0.1, 0.3, 0.3]}],
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
                    "ambiguous": True,
                },
            ),
        ],
        notes="Synthetic only. Live Reachy capture is a later operator session.",
    )
