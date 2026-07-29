"""Contracts introduced by the USERPLAN multi-mode split."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from rr_vfiqa import (
    EvaluationMode,
    compare,
    evaluate,
    evaluate_full_reference,
    evaluate_no_reference,
)
from rr_vfiqa.config import EvalConfig
from rr_vfiqa.fusion.feature_registry import (
    definitions,
    feature_contract_hash,
    required_features,
)
from rr_vfiqa.fusion.score_schema import build_category_errors
from rr_vfiqa.metrics.full_reference import _local_ssim
from rr_vfiqa.multimode import _unique_coverage
from rr_vfiqa.sampling.temporal_plan import FlowPairPlan, TemporalLagPlan
from rr_vfiqa.schema import MetricResult, Window, WindowFeatures
from rr_vfiqa.testing.synth import render_scene, write_video


@pytest.fixture(scope="module")
def mode_videos(tmp_path_factory):
    root = tmp_path_factory.mktemp("modes")
    frames = render_scene(n_frames=48, w=160, h=96, seed=13)
    bad = frames.copy()
    bad[16:22] = bad[15]
    hfr = render_scene(n_frames=64, w=160, h=96, seed=17)
    hfr_duplicates = hfr.copy()
    hfr_duplicates[1::2] = hfr_duplicates[::2]
    return {
        "reference": write_video(root / "reference_60.mp4", frames, 60),
        "bad_same_rate": write_video(root / "bad_60.mp4", bad, 60),
        "unrelated_same_rate": write_video(
            root / "unrelated_60.mp4",
            render_scene(n_frames=48, w=160, h=96, seed=91),
            60,
        ),
        "hfr": write_video(root / "candidate_120.mp4", hfr, 120),
        "hfr_duplicates": write_video(
            root / "candidate_duplicates_120.mp4", hfr_duplicates, 120),
        "reference_half": write_video(
            root / "reference_half_60.mp4",
            np.stack([
                cv2.resize(frame, (80, 48))
                for frame in frames
            ]),
            60,
        ),
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
    with pytest.raises(ValueError, match="explicitly"):
        evaluate("candidate.mp4")


def test_feature_and_metric_contracts_are_versioned():
    required = required_features("no-reference")
    assert "nr_common_self_comp" in required
    result = MetricResult.from_scalars(
        {"nr_common_self_comp": 0.1}, required=required)
    assert result.status == "failed"
    assert result.confidence == 0.0
    assert result.warnings
    assert len(feature_contract_hash("no-reference")) == 16
    assert any(row.required for row in definitions("full-reference"))


def test_temporal_lag_plan_uses_physical_time():
    plan_60 = TemporalLagPlan.build(np.arange(7) / 60.0)
    plan_120 = TemporalLagPlan.build(np.arange(9) / 120.0)
    assert plan_60.lag_1_60[0] == (0, 1)
    assert plan_120.lag_1_60[0] == (0, 2)
    assert plan_120.lag_1_30[0] == (0, 4)
    assert np.isclose(plan_60.common_triplets[0].span_seconds, 1 / 30)
    assert np.isclose(plan_120.common_triplets[0].span_seconds, 1 / 30)
    flow_plan = FlowPairPlan.for_no_reference(plan_120)
    assert (0, 4) in flow_plan.unique_pairs()


def test_unique_coverage_does_not_double_count_overlap():
    windows = [
        WindowFeatures(Window(2, np.asarray([0, 1, 2, 3, 4]), -1)),
        WindowFeatures(Window(4, np.asarray([2, 3, 4, 5, 6]), -1)),
    ]
    assert np.isclose(_unique_coverage(windows, 10), 0.7)


def test_endpoint_category_aggregation_excludes_failed_stage():
    valid = WindowFeatures(
        Window(2, np.asarray([0, 1, 2, 3, 4]), 0),
        scalars={"mct_lag1_mean": 10.0},
    )
    failed = WindowFeatures(
        Window(4, np.asarray([2, 3, 4, 5, 6]), 1),
        scalars={"mct_lag1_mean": 100.0},
        labels={"error_temporal": "backend failed"},
    )
    errors = build_category_errors([valid, failed], {})
    assert len(errors["temporal"]) == 1


def test_local_ssim_is_standard_identity_case():
    image = np.arange(64 * 64, dtype=np.float32).reshape(64, 64) % 255
    assert np.isclose(_local_ssim(image, image), 1.0)


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
        assert report.meta["score_schema"] == "nr-stability-risk-v2"
        assert report.meta["status"] == "ok"
        assert 0.0 <= report.overall_score <= 100.0
        assert report.confidence <= 0.75
        assert report.meta["time_scales_seconds"] == [0.016667, 0.033333]
        assert report.meta["production_gate"] is False
        assert isinstance(report.meta["working_tree_dirty"], bool)
        assert report.meta["feature_contract_hash"]
        assert "spatial_fidelity" not in report.scores
        assert "nr_flow_fold_fraction" in report.features
        assert report.features["nr_track_points"] >= 8
        assert "nr_phase_sharp_energy" in report.features
        assert "nr_mct_native_mean" in report.features
        assert "nr_mct_1_60_mean" in report.features
        assert "nr_mct_1_30_mean" in report.features
        assert "nr_common_self_comp" in report.features


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
    assert good.meta["score_schema"] == "fr-same-rate-fidelity-v2"
    assert good.meta["alignment"]["reliable"] is True
    assert np.isclose(good.overall_score, 100.0)
    assert bad.overall_score < good.overall_score
    assert bad.scores["temporal_fidelity"] < good.scores["temporal_fidelity"]
    assert "phase_consistency" not in good.scores
    assert good.features["fr_l1_rgb"] == 0.0
    assert good.features["fr_charbonnier_rgb"] == 0.0
    assert good.features["fr_edge_recall"] == 1.0
    assert good.features["fr_edge_precision"] == 1.0
    assert "fr_salient_roi_l1" in bad.features
    assert bad.meta["roi_branches"]["text"] == "dense-stroke-proxy"
    assert good.features["fr_ssim"] == pytest.approx(1.0, abs=1e-5)
    assert "fr_global_ssim_proxy" in good.features
    assert "fr_multiscale_perceptual" not in good.features
    assert "fr_multiscale_luma_gradient_l1" in good.features
    assert good.meta["full_reference_scan"]["fr_scan_y_l1"] == 0.0


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


def test_no_reference_120_native_duplicate_signal(mode_videos):
    common = dict(
        preset="fast",
        device="cpu",
        flow_backend="farneback",
        vqa_backend="none",
        export_clips=False,
    )
    clean = evaluate_no_reference(str(mode_videos["hfr"]), **common)
    duplicated = evaluate_no_reference(
        str(mode_videos["hfr_duplicates"]), **common)
    assert duplicated.features["nr_native_duplicate_fraction"] > 0.4
    assert duplicated.features["nr_native_duplicate_fraction"] > \
        clean.features["nr_native_duplicate_fraction"]
    assert duplicated.scores["temporal_stability"] < \
        clean.scores["temporal_stability"]


def test_nr_compare_rejects_different_content(mode_videos):
    with pytest.raises(ValueError, match="not safely comparable"):
        compare(
            [
                str(mode_videos["reference"]),
                str(mode_videos["unrelated_same_rate"]),
            ],
            mode="no-reference",
            preset="fast",
            device="cpu",
            flow_backend="farneback",
            vqa_backend="none",
        )


def test_full_reference_geometry_policy_is_explicit(mode_videos):
    strict = evaluate_full_reference(
        str(mode_videos["reference_half"]),
        str(mode_videos["reference"]),
        preset="fast",
        device="cpu",
        flow_backend="farneback",
        geometry_policy="strict",
        export_clips=False,
    )
    assert strict.meta["status"] == "failed"
    assert strict.meta["alignment"]["geometry_policy"] == "strict"

    common = evaluate_full_reference(
        str(mode_videos["reference_half"]),
        str(mode_videos["reference"]),
        preset="fast",
        device="cpu",
        flow_backend="farneback",
        geometry_policy="common-resolution",
        export_clips=False,
    )
    assert common.meta["status"] == "ok"
    assert common.meta["score_schema"].endswith("+common-resolution")

    resized = evaluate_full_reference(
        str(mode_videos["reference_half"]),
        str(mode_videos["reference"]),
        preset="fast",
        device="cpu",
        flow_backend="farneback",
        geometry_policy="resize-candidate",
        export_clips=False,
    )
    assert resized.meta["status"] == "ok"
    assert resized.meta["score_schema"].endswith("+resize-candidate")
