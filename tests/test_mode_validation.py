"""Mode-isolated directional validation contracts."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from rr_vfiqa.calibration import mode_validation


def _report(score: float, mode: str):
    meta = {
        "score_schema": (
            "nr-stability-risk-v2"
            if mode == "no-reference" else "fr-same-rate-fidelity-v2"),
        "metric_contract": (
            "nr-metrics-v2"
            if mode == "no-reference" else "fr-metrics-v2"),
        "preset_contract": (
            "nr-fast-v1" if mode == "no-reference" else "fr-fast-v1"),
        "backend_contract": {"flow": {"backend": "farneback"}},
    }
    return SimpleNamespace(
        overall_score=score,
        meta=meta,
        to_dict=lambda: {"overall_score": score},
    )


@pytest.mark.parametrize("mode", ["no-reference", "full-reference"])
def test_directional_validation_is_mode_local(monkeypatch, mode):
    def fake_evaluate(*, candidate_video, **kwargs):
        return _report(90.0 if candidate_video == "better.mp4" else 60.0, mode)

    monkeypatch.setattr(mode_validation, "evaluate", fake_evaluate)
    case = {"name": "blur", "better": "better.mp4", "worse": "worse.mp4"}
    if mode == "full-reference":
        case["reference"] = "reference.mp4"
    result = mode_validation.validate_manifest({"mode": mode, "cases": [case]})
    assert result["directional_accuracy"] == 1.0
    assert result["mode"] == mode
    assert result["production_gate"] is False
    assert result["metric_contract"].startswith(
        "nr-" if mode == "no-reference" else "fr-")


def test_directional_validation_rejects_endpoint_mode():
    with pytest.raises(ValueError, match="endpoint calibration"):
        mode_validation.validate_manifest(
            {"mode": "endpoint-2x", "cases": []})
