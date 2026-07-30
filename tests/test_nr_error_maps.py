"""Tests for NR dense error-map generation (USERPLAN §8).

These build a small synthetic 120-FPS-like frame bundle (with a static UI
band and small translation) and assert that ``compute_window_maps`` returns
the full set of named fields, all aligned to the luma grid, plus the
issue->map association logic.
"""

from __future__ import annotations

import numpy as np
import pytest

from rr_vfiqa.config import EvalConfig, EvaluationMode
from rr_vfiqa.metrics.no_reference import compute_window_maps
from rr_vfiqa.schema import FrameBundle
from rr_vfiqa.metrics.window_flows import WindowFlows
from rr_vfiqa.motion.flow_estimator import get_flow_backend
from rr_vfiqa.sampling.temporal_plan import FlowPairPlan, TemporalLagPlan


EXPECTED_MAPS = {
    "mct_residual_map",
    "composition_error_map",
    "self_cycle_residual_map",
    "flow_fold_map",
    "jacobian_determinant_map",
    "ui_edge_instability_map",
    "phase_sharpness_map",
    "duplicate_frame_indicator",
}


def _bundle(seed: int = 7, n: int = 6, h: int = 96, w: int = 128,
            fps: float = 120.0) -> FrameBundle:
    rng = np.random.RandomState(seed)
    rgb = np.empty((n, h, w, 3), np.uint8)
    base = rng.randint(40, 200, (h, w, 3), np.uint8).astype(np.float32)
    for i in range(n):
        # small horizontal translation + a static bright UI band at the top
        shift = i * 2
        frame = np.roll(base, shift, axis=1)
        frame[:10, :, :] = 220.0
        rgb[i] = frame.astype(np.uint8)
    times = np.array([i / fps for i in range(n)])
    return FrameBundle(indices=np.arange(n), times=times, rgb=rgb,
                       width=w, height=h)


def _cfg() -> EvalConfig:
    return EvalConfig.build_mode(
        candidate_video="fake_nr.mp4",
        mode=EvaluationMode.NO_REFERENCE,
        preset="standard",
        flow_backend="farneback",
        device="cpu",
    )


def _maps_for(bundle: FrameBundle, cfg: EvalConfig):
    backend = get_flow_backend("farneback", "cpu")
    flows = WindowFlows(bundle, backend)
    lag = TemporalLagPlan.build(bundle.times)
    flows.precompute(FlowPairPlan.for_no_reference(lag).unique_pairs())
    return compute_window_maps(bundle, flows, cfg)


def test_returns_full_map_set():
    maps = _maps_for(_bundle(), _cfg())
    assert set(maps.keys()) == EXPECTED_MAPS


def test_all_maps_align_to_luma_grid():
    h, w = 96, 128
    maps = _maps_for(_bundle(h=h, w=w), _cfg())
    for name, arr in maps.items():
        assert arr.shape == (h, w), f"{name} has shape {arr.shape}"
        assert arr.dtype == np.float32


def test_maps_are_finite():
    maps = _maps_for(_bundle(), _cfg())
    for name, arr in maps.items():
        assert np.all(np.isfinite(arr)), f"{name} has non-finite values"


def test_duplicate_indicator_spots_static_region():
    # USERPLAN P1: duplicate_frame_indicator is now oriented so HIGH value =
    # high duplication risk (hot = bad).  A fully static frame (consecutive
    # identical frames -> |Δluma| ≈ 0) should yield risk ≈ 1.0.
    bundle = _bundle()
    bundle.rgb[:] = bundle.rgb[0]
    maps = _maps_for(bundle, _cfg())
    assert float(maps["duplicate_frame_indicator"].mean()) == pytest.approx(1.0)


def test_jacobian_determinant_near_rigid():
    # Pure translation should keep det(I + ∇F) close to 1 almost everywhere.
    maps = _maps_for(_bundle(), _cfg())
    jdet = maps["jacobian_determinant_map"]
    assert float(np.mean(np.abs(jdet - 1.0))) < 0.2


def test_empty_bundle_returns_empty():
    # Too few samples -> no maps (function guards on n >= 5).
    bundle = _bundle(n=3)
    maps = _maps_for(bundle, _cfg())
    assert maps == {}
