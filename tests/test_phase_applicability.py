"""Tests for two-phase applicability gating (USERPLAN §6).

A Phase-Consistency subscore may only enter the total when the clip genuinely
has a stable two-phase structure: strong sharpness/edge alternation whose
direction is coherent across the film.  True 60 FPS combat flicker is local
and direction-unstable -> phase is gated out (N/A) and the fusion weights are
renormalised over the remaining categories.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from rr_vfiqa.fusion.mode_score_schemas import (
    MODE_WEIGHTS,
    compute_mode_scores,
    compute_mode_confidence,
)
from rr_vfiqa.metrics.parity_frequency import (
    scan_phase_applicability,
    _signed_phase_gap,
    _coherence,
)
from rr_vfiqa.config import EvaluationMode
from rr_vfiqa.sampling.cheap_scan import CheapScan
from rr_vfiqa.schema import Window, WindowFeatures


def _scan(sharp, edge=None, indices=None):
    n = len(sharp)
    edge = edge if edge is not None else np.full(n, 0.3, np.float64)
    indices = indices if indices is not None else np.arange(n, dtype=np.int32)
    return CheapScan(
        width=32,
        indices=np.asarray(indices, np.int32),
        luma_mean=np.zeros(n, np.float32),
        luma_std=np.zeros(n, np.float32),
        sharpness=np.asarray(sharp, np.float32),
        grad_energy=np.zeros(n, np.float32),
        edge_frac=np.asarray(edge, np.float32),
        hash64=np.zeros(n, np.int64),
        hash_dist=np.zeros(n, np.float32),
        frame_diff=np.zeros(n, np.float32),
        hist_dist=np.zeros(n, np.float32),
        frame_diff_parent=np.zeros(n, np.float32),
        moving_frac_native=np.zeros(n, np.float32),
        moving_frac_parent=np.zeros(n, np.float32),
    )


def _uniform_sharp_scan():
    """True 60 FPS: sharpness roughly uniform, no parity structure."""
    n = 192
    sharp = np.abs(np.random.RandomState(0).normal(200.0, 8.0, n))
    return _scan(sharp)


def _odd_blur_scan():
    """Blended interpolation: odd frames consistently blurrier."""
    n = 192
    rng = np.random.RandomState(1)
    even = np.abs(rng.normal(210.0, 8.0, n))
    odd = np.abs(rng.normal(70.0, 10.0, n))
    sharp = np.where(np.arange(n) % 2 == 0, even, odd)
    return _scan(sharp)


def _flipping_blur_scan():
    """Combat flicker: strong local blur alternation but direction flips."""
    n = 256
    rng = np.random.RandomState(2)
    sharp = np.abs(rng.normal(200.0, 8.0, n))
    for bs in range(0, n, 32):
        be = min(bs + 32, n)
        idx = np.arange(bs, be)
        even = (idx % 2 == 0)
        if (bs // 32) % 2 == 0:
            sharp[idx[even]] = 60.0          # even phase blurrier
        else:
            sharp[idx[~even]] = 60.0         # odd phase blurrier
    return _scan(sharp)


def test_clean_video_is_not_two_phase():
    meta = scan_phase_applicability(_uniform_sharp_scan())
    assert meta["applicable"] is False
    assert meta["source_phase_likelihood"] < 0.7
    assert meta["sharpness_ratio"] > 0.9


def test_odd_blur_is_two_phase_and_interpretable():
    meta = scan_phase_applicability(_odd_blur_scan())
    assert meta["applicable"] is True
    assert meta["source_phase_likelihood"] >= 0.7
    assert meta["phase_coherence"] >= 0.6
    assert meta["sharpness_ratio"] < 0.5
    assert meta["dominant_bad_phase"] == "B"     # odd phase is blurrier
    assert meta["phase_a_sharpness"] > meta["phase_b_sharpness"]


def test_flipping_direction_is_not_coherent():
    meta = scan_phase_applicability(_flipping_blur_scan())
    # Even though the blur is strong, the direction flips -> not coherent.
    assert meta["source_phase_likelihood"] >= 0.7
    assert meta["phase_coherence"] < 0.6
    assert meta["applicable"] is False


def test_signed_phase_gap_and_coherence():
    k = np.arange(16)
    values = np.where(k % 2 == 0, 200.0, 60.0)
    g = _signed_phase_gap(values, k)
    assert g < -0.5                     # odd phase smaller -> negative gap
    assert abs(_coherence([0.8, 0.8, 0.8])) > 0.9
    assert _coherence([0.8, -0.8, 0.8]) < 0.5


# ------------------------------------------------------------ fusion gating
def _wf(features: dict[str, float]) -> WindowFeatures:
    return WindowFeatures(
        window=Window(center=0, indices=np.array([0, 1, 2, 3, 4]),
                      pair=-1, risk=0.0, source="uniform"),
        scalars=features,
    )


def _clean_window_features() -> list[WindowFeatures]:
    base = {
        "nr_mct_1_60_mean": 5.0,
        "nr_mct_1_30_mean": 8.0,
        "nr_common_self_cycle": 12.0,
        "nr_duplicate_fraction": 0.02,
        "nr_freeze_fraction": 0.03,
        "nr_common_self_comp": 0.05,
        "nr_flow_accel_ratio": 0.1,
        "nr_flow_jerk_ratio": 0.15,
        "nr_flow_fold_fraction": 0.01,
        "nr_flow_jdet_low_fraction": 0.02,
        "nr_flow_divergence_std": 0.4,
        "nr_flow_curl_std": 0.4,
        "nr_track_accel_p90": 0.2,
        "nr_track_jerk_p90": 0.15,
        "nr_track_turn_p90": 0.3,
        "nr_tile_accel_p90": 0.2,
        "nr_tile_jerk_p90": 0.15,
        "nr_local_reversal_fraction": 0.03,
        # clean phase: tiny gaps
        "nr_phase_sharp_gap": 0.03,
        "nr_phase_edge_gap": 0.04,
        "nr_phase_sharp_energy": 0.05,
        "nr_phase_edge_energy": 0.05,
        "nr_ui_edge_instability": 0.02,
        "nr_text_stroke_instability": 0.03,
        "nr_ui_component_instability": 0.05,
        "gtq_sharpness": 260.0,
        "gtq_blockiness": 0.2,
        "gtq_noise": 3.0,
    }
    return [_wf(dict(base)) for _ in range(4)]


def test_gating_phase_renormalizes_weights():
    windows = _clean_window_features()
    overall_with_phase, subs, _ = compute_mode_scores(
        EvaluationMode.NO_REFERENCE, windows)
    overall_gated, subs_gated, _ = compute_mode_scores(
        EvaluationMode.NO_REFERENCE, windows, gated_categories={"phase"})
    # phase_consistency becomes N/A when gated
    assert math.isnan(subs_gated["phase_consistency"])
    assert np.isfinite(subs["phase_consistency"])
    # a clean window's phase barely contributes, so the total barely moves
    assert abs(overall_with_phase - overall_gated) < 5.0
    # weights are renormalised: both totals are on the same 0..100 scale
    assert 0.0 < overall_gated <= 100.0


def test_gating_phase_with_bad_phase_raises_overall():
    bad_phase = dict(_clean_window_features()[0].scalars)
    bad_phase.update(nr_phase_sharp_gap=0.6, nr_phase_edge_gap=0.7,
                     nr_phase_sharp_energy=0.8, nr_phase_edge_energy=0.8)
    windows = [_wf(bad_phase) for _ in range(4)]
    _, with_phase, _ = compute_mode_scores(EvaluationMode.NO_REFERENCE, windows)
    _, gated, _ = compute_mode_scores(
        EvaluationMode.NO_REFERENCE, windows, gated_categories={"phase"})
    # When phase IS applicable its heavy errors drag the total down; gating it
    # out removes that drag (the phase axis is reported separately as N/A).
    assert with_phase["phase_consistency"] < 60.0
    assert math.isnan(gated["phase_consistency"])


def test_confidence_api_unaffected():
    c = compute_mode_confidence(valid_windows=4, total_windows=4,
                                coverage_fraction=0.1)
    assert 0.0 < c <= 1.0
