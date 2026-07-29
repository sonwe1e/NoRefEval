"""Contracts introduced by the USERPLAN multi-mode split."""

from __future__ import annotations

import numpy as np
import pytest

from rr_vfiqa import (
    EvaluationMode,
    evaluate,
    evaluate_full_reference,
    evaluate_no_reference,
)
from rr_vfiqa.config import EvalConfig
from rr_vfiqa.testing.synth import render_scene, write_video


@pytest.fixture(scope="module")
def mode_videos(tmp_path_factory):
    root = tmp_path_factory.mktemp("modes")
    frames = render_scene(n_frames=48, w=160, h=96, seed=13)
    bad = frames.copy()
    bad[16:22] = bad[15]
    return {
        "reference": write_video(root / "reference_60.mp4", frames, 60),
        "bad_same_rate": write_video(root / "bad_60.mp4", bad, 60),
        "unrelated_same_rate": write_video(
            root / "unrelated_60.mp4",
            render_scene(n_frames=48, w=160, h=96, seed=91),
            60,
        ),
        "hfr": write_video(root / "candidate_120.mp4",
                           render_scene(n_frames=64, w=160, h=96, seed=17), 120),
    }


def test_mode_input_contracts():
    nr = EvalConfig.build_mode("candidate.mp4", mode="no-reference", preset="fast")
    assert nr.mode is EvaluationMode.NO_REFERENCE
    assert nr.reference_video is None
    with pytest.raises(ValueError, match="does not accept"):
        EvalConfig.build_mode(
            "candidate.mp4", reference_video="reference.mp4",
            mode="no-reference")
    with pytest.raises(ValueError, match="requires"):
        EvalConfig.build_mode("candidate.mp4", mode="full-reference")


def test_no_reference_60_and_120_are_time_normalized(mode_videos):
    reports = [
        evaluate_no_reference(
            str(mode_videos[key]),
            preset="fast",
            device="cpu",
            flow_backend="farneback",
            vqa_backend="none",
            export_clips=False,
        )
        for key in ("reference", "hfr")
    ]
    for report in reports:
        assert report.meta["mode"] == "no-reference"
        assert report.meta["score_schema"] == "nr-stability-risk-v1"
        assert report.meta["status"] == "ok"
        assert 0.0 <= report.overall_score <= 100.0
        assert report.confidence <= 0.75
        assert report.meta["time_scales_seconds"] == [0.016667, 0.033333]
        assert "spatial_fidelity" not in report.scores


def test_full_reference_is_separate_and_detects_same_rate_error(mode_videos):
    common = dict(
        preset="fast",
        device="cpu",
        flow_backend="farneback",
        export_clips=False,
    )
    good = evaluate_full_reference(
        str(mode_videos["reference"]),
        str(mode_videos["reference"]),
        **common,
    )
    bad = evaluate_full_reference(
        str(mode_videos["reference"]),
        str(mode_videos["bad_same_rate"]),
        **common,
    )
    assert good.meta["mode"] == "full-reference"
    assert good.meta["score_schema"] == "fr-same-rate-fidelity-v1"
    assert good.meta["alignment"]["reliable"] is True
    assert np.isclose(good.overall_score, 100.0)
    assert bad.overall_score < good.overall_score
    assert bad.scores["temporal_fidelity"] < good.scores["temporal_fidelity"]
    assert "phase_consistency" not in good.scores


def test_public_dispatch_requires_explicit_reference(mode_videos):
    with pytest.raises(ValueError, match="requires"):
        evaluate(
            candidate_video=str(mode_videos["reference"]),
            mode="endpoint-2x",
            preset="fast",
        )
    report = evaluate(
        candidate_video=str(mode_videos["reference"]),
        mode="no-reference",
        preset="fast",
        device="cpu",
        flow_backend="farneback",
        vqa_backend="none",
        export_clips=False,
    )
    assert report.meta["reference"] is None


def test_full_reference_fails_closed_for_unrelated_capture(mode_videos):
    report = evaluate_full_reference(
        str(mode_videos["reference"]),
        str(mode_videos["unrelated_same_rate"]),
        preset="fast",
        device="cpu",
        flow_backend="farneback",
        export_clips=False,
    )
    assert report.meta["status"] == "failed"
    assert report.meta["alignment"]["reliable"] is False
    assert report.to_dict()["overall_score"] is None
    assert report.confidence <= 0.01
