"""Uses the configured workflow."""

from __future__ import annotations

from reachy_tools.proposed import adapters
from vision.enums import PresenceKind
from vision.phrases import contains_metric_claim, render_presence


def test_adapter_error_shape() -> None:
    result = adapters.find_last_seen
    assert callable(result)


def test_template_adapters_have_no_metric_units() -> None:
    spoken = render_presence(PresenceKind.ABSTAINED, label="mug")
    assert not contains_metric_claim(spoken)
