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
from rr_vfiqa.fusion.mode_score_schemas import (
    NR_FEATURES,
    compute_mode_scores,
)
from rr_vfiqa.fusion.score_schema import build_category_errors
from rr_vfiqa.metrics.full_reference import _local_ssim
from rr_vfiqa.multimode import (
    _fr_target_size,
    _nr_compare_issues,
    _unique_coverage,
)
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
    hfr_odd_blur = hfr.copy()
    hfr_odd_blur[1::2] = np.stack([
        cv2.GaussianBlur(frame, (15, 15), 3.0)
        for frame in hfr_odd_blur[1::2]
    ])
    color_shift = np.clip(
        frames.astype(np.int16) + np.asarray([28, -12, 18], np.int16),
        0,
        255,
    ).astype(np.uint8)
    spatial_shift = np.roll(frames, 3, axis=2)
    local_delete = frames.copy()
    local_delete[:, 28:62, 62:104] = 0
    return {
        "reference": write_video(root / "reference_60.mp4", frames, 60),
        "bad_same_rate": write_video(root / "bad_60.mp4", bad, 60),
        "blurred_same_rate": write_video(
            root / "blurred_60.mp4",
            np.stack([
                cv2.GaussianBlur(frame, (21, 21), 5.0)
                for frame in frames
            ]),
            60,
        ),
        "color_shift_same_rate": write_video(
            root / "color_shift_60.mp4", color_shift, 60),
        "spatial_shift_same_rate": write_video(
            root / "spatial_shift_60.mp4", spatial_shift, 60),
        "local_delete_same_rate": write_video(
            root / "local_delete_60.mp4", local_delete, 60),
        "codec_same_rate": write_video(
            root / "codec_60.mp4", frames, 60, crf=42),
        "unrelated_same_rate": write_video(
            root / "unrelated_60.mp4",
            render_scene(n_frames=48, w=160, h=96, seed=91),
            60,
        ),
        "hfr": write_video(root / "candidate_120.mp4", hfr, 120),
        "hfr_duplicates": write_video(
            root / "candidate_duplicates_120.mp4", hfr_duplicates, 120),
        "hfr_odd_blur": write_video(
            root / "candidate_odd_blur_120.mp4", hfr_odd_blur, 120),
        "reference_half": write_video(
            root / "reference_half_60.mp4",
            np.stack([
                cv2.resize(frame, (80, 48))
                for frame in frames
            ]),
            60,
        ),
    }


@pytest.fixture(scope="module")
def fr_direction_reports(mode_videos):
    common = dict(
        preset="fast",
        device="cpu",
        flow_backend="farneback",
        export_clips=False,
    )
    keys = (
        "reference",
        "blurred_same_rate",
        "color_shift_same_rate",
        "spatial_shift_same_rate",
        "local_delete_same_rate",
        "codec_same_rate",
    )
    return {
        key: evaluate_full_reference(
            str(mode_videos["reference"]),
            str(mode_videos[key]),
            **common,
        )
        for key in keys
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
    native = next(
        row for row in definitions("no-reference")
        if row.name == "nr_mct_native_mean")
    assert native.required is False
    assert "diagnostic" in native.category
    assert "nr_mct_native_mean" not in NR_FEATURES["temporal"]


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
        scalars={"mct_lag1_mean": 10.0, "cycle_resid_mean": 5.0},
    )
    failed = WindowFeatures(
        Window(4, np.asarray([2, 3, 4, 5, 6]), 1),
        scalars={"mct_lag1_mean": 100.0},
        labels={"error_temporal": "backend failed"},
    )
    errors = build_category_errors([valid, failed], {})
    assert len(errors["temporal"]) == 1


def test_endpoint_category_requires_complete_core_evidence():
    partial = WindowFeatures(
        Window(2, np.asarray([0, 1, 2, 3, 4]), 0),
        scalars={"mct_lag1_mean": 10.0},
    )
    assert build_category_errors([partial], {})["temporal"] == []


def test_nr_native_diagnostics_do_not_change_public_score():
    base = {
        "nr_mct_1_60_mean": 10.0,
        "nr_mct_1_30_mean": 12.0,
        "nr_common_self_cycle": 8.0,
        "nr_duplicate_fraction": 0.0,
        "nr_freeze_fraction": 0.0,
        "nr_common_self_comp": 0.05,
        "nr_phase_sharp_gap": 0.02,
    }
    a = WindowFeatures(
        Window(2, np.asarray([0, 1, 2, 3, 4]), -1),
        scalars={**base, "nr_mct_native_mean": 0.0},
    )
    b = WindowFeatures(
        Window(2, np.asarray([0, 1, 2, 3, 4]), -1),
        scalars={**base, "nr_mct_native_mean": 1000.0},
    )
    score_a = compute_mode_scores(EvaluationMode.NO_REFERENCE, [a])
    score_b = compute_mode_scores(EvaluationMode.NO_REFERENCE, [b])
    assert score_a[0] == score_b[0]
    assert score_a[2]["temporal"] == score_b[2]["temporal"]
    assert score_a[2]["motion"] == score_b[2]["motion"]


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
        assert report.meta["score_schema"] == \
            "nr-stability-risk-v4"
        assert report.meta["status"] == "ok"
        assert 0.0 <= report.overall_score <= 100.0
        assert report.confidence <= 0.75
        assert report.meta["time_scales_seconds"] == [0.016667, 0.033333]
        assert report.meta["production_gate"] is False
        assert isinstance(report.meta["working_tree_dirty"], bool)
        assert report.meta["feature_contract_hash"]
        assert len(report.meta["build_contract"]["package_source_sha256"]) == 64
        assert report.meta["metric_diagnostics"]["failed_windows"] == 0
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
    assert good.meta["score_schema"] == "fr-same-rate-fidelity-v3"
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
    assert good.meta["full_reference_scan_contract"]["streaming"] is True
    assert good.meta["full_reference_scan_contract"]["gradient_operator"] == \
        "sobel-magnitude-3x3"
    assert "fr_tile_flow_error_p90" in good.features
    assert "fr_camera_residual_flow_error" in good.features


@pytest.mark.parametrize(
    ("candidate_key", "subscore"),
    [
        ("blurred_same_rate", "spatial_fidelity"),
        ("color_shift_same_rate", "spatial_fidelity"),
        ("spatial_shift_same_rate", "structural_fidelity"),
        ("local_delete_same_rate", "structural_fidelity"),
        ("codec_same_rate", "spatial_fidelity"),
    ],
)
def test_full_reference_defect_directions(
    fr_direction_reports,
    candidate_key,
    subscore,
):
    good = fr_direction_reports["reference"]
    degraded = fr_direction_reports[candidate_key]
    assert degraded.overall_score < good.overall_score
    assert degraded.scores[subscore] < good.scores[subscore]


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
    # Native cadence remains diagnostic-only in the common-time score schema;
    # do not silently make 60 and 120 FPS totals depend on different spans.
    assert duplicated.meta["score_schema"] == \
        "nr-stability-risk-v4"


def test_no_reference_60_freeze_lowers_common_time_score(mode_videos):
    common = dict(
        preset="fast",
        device="cpu",
        flow_backend="farneback",
        vqa_backend="none",
        export_clips=False,
    )
    clean = evaluate_no_reference(str(mode_videos["reference"]), **common)
    frozen = evaluate_no_reference(
        str(mode_videos["bad_same_rate"]), **common)
    assert frozen.overall_score < clean.overall_score
    assert frozen.scores["temporal_stability"] < \
        clean.scores["temporal_stability"]


def test_no_reference_odd_blur_lowers_phase_score(mode_videos):
    common = dict(
        preset="fast",
        device="cpu",
        flow_backend="farneback",
        vqa_backend="none",
        export_clips=False,
    )
    clean = evaluate_no_reference(str(mode_videos["hfr"]), **common)
    blurred = evaluate_no_reference(
        str(mode_videos["hfr_odd_blur"]), **common)
    assert blurred.scores["phase_consistency"] < \
        clean.scores["phase_consistency"]


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


def test_nr_compare_fingerprint_tolerates_severe_blur(mode_videos):
    issues = _nr_compare_issues([
        str(mode_videos["reference"]),
        str(mode_videos["blurred_same_rate"]),
    ])
    assert not any("fingerprint" in issue for issue in issues)
    assert _nr_compare_issues([
        str(mode_videos["reference"]),
        str(mode_videos["unrelated_same_rate"]),
    ], trusted_content=True) == []


def test_common_resolution_uses_one_explicit_grid():
    class Meta:
        def __init__(self, width, height):
            self.width = width
            self.height = height

    target = _fr_target_size(
        Meta(1920, 1080),
        Meta(1280, 718),
        max_width=480,
        geometry_policy="common-resolution",
    )
    assert target == (480, 270)


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
