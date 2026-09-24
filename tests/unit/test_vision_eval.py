"""Evaluation harness: no superiority claims, abstain metrics, faithfulness."""

from __future__ import annotations

from vision.eval import (
    assert_no_superiority_language,
    cluster_bootstrap,
    detection_pr,
    faithfulness_audit,
    identity_with_abstain,
    report_header,
    sample_dev_manifest,
)
from vision.phrases import all_template_strings


def test_report_header_forbids_superiority() -> None:
    header = report_header()
    assert_no_superiority_language(header)
    assert "not a claim of statistical superiority" in header


def test_identical_set_abstain_is_not_confident_wrong() -> None:
    metrics = identity_with_abstain(
        ["abstain", "abstain", "matched"],
        [False, False, True],
    )
    assert metrics["abstention_rate"].value and metrics["abstention_rate"].value > 0.5
    assert metrics["confident_wrong_rate"].value == 0.0


def test_faithfulness_on_templates() -> None:
    phrases = all_template_strings()
    audit = faithfulness_audit(phrases, [True] * len(phrases))
    assert audit["metric_distance_claims"].value == 0.0
    assert audit["faithfulness"].value == 1.0


def test_scene_clustered_bootstrap_width() -> None:
    point, lo, hi = cluster_bootstrap(
        {"desk": [0.9, 0.8], "floor": [0.2, 0.3]},
        n_resample=200,
    )
    assert point is not None and lo is not None and hi is not None
    assert lo <= point <= hi


def test_undefined_when_no_truth() -> None:
    point = detection_pr([], [])
    assert point.undefined
    assert point.value is None


def test_dev_manifest_has_identical_mugs() -> None:
    manifest = sample_dev_manifest()
    assert any(sample.get("identical_set") for sample in manifest.samples)
    assert manifest.to_dict()["claim_policy"] == "no_superiority"
