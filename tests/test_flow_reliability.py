"""Tests for the Flow Reliability Gate (USERPLAN §7).

Particles, flashes, explosions and occlusion make optical flow unreliable.
These appearance changes must raise *uncertainty* (`nr_effect_transient_*`),
never be scored as motion smoothness failure.  When flow coverage is too low
(`nr_flow_valid_fraction < 0.25`) the motion evidence is N/A.
"""

from __future__ import annotations

import numpy as np
import pytest

from rr_vfiqa.config import EvalConfig, EvaluationMode
from rr_vfiqa.metrics.no_reference import (
    FLOW_VALID_FRACTION_MIN,
    _flow_reliability,
    _tile_motion_dynamics,
    _visibility_masks,
)
from rr_vfiqa.schema import FrameBundle


class _FakeFlows:
    """Minimal WindowFlows stand-in returning the flows handed to it."""

    def __init__(self, flows: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]):
        self._flows = flows

    def forward(self, a: int, b: int) -> np.ndarray:
        return self._flows[(a, b)][0]

    def pair(self, a: int, b: int) -> tuple[np.ndarray, np.ndarray]:
        return self._flows[(a, b)]


def _cfg() -> EvalConfig:
    return EvalConfig.build_mode(
        candidate_video="fake.mp4",
        mode=EvaluationMode.NO_REFERENCE,
        preset="fast",
        cache_dir=".",
        device="cpu",
        flow_backend="farneback",
    )


def _translation_flows(h: int, w: int, dx: float, n: int = 5) -> dict:
    """Smooth pure-translation flows (a -> b all dx), highly reliable."""
    flows = {}
    for a in range(n):
        for b in range(a + 1, n):
            d = dx * (b - a)
            f_ab = np.zeros((h, w, 2), np.float32)
            f_ab[..., 0] = d
            f_ba = -f_ab
            flows[(a, b)] = (f_ab, f_ba)
    return flows


def test_clean_translation_flow_is_reliable():
    h, w = 64, 96
    y = np.zeros((5, h, w), np.float32)
    for t in range(5):
        y[t] = np.roll(
            np.arange(w, dtype=np.float32)[None, :].repeat(h, 0), 3 * t, axis=1)
    flows = _FakeFlows(_translation_flows(h, w, 3.0))
    cfg = _cfg()
    rel = _flow_reliability(y, flows, [(0, 1), (1, 2), (2, 3), (3, 4)], cfg)
    assert rel["nr_flow_valid_fraction"] > 0.8
    assert rel["nr_forward_backward_consistency"] > 0.8
    assert rel["nr_photometric_support_fraction"] > 0.8
    assert rel["nr_effect_transient_fraction"] < 0.2


def test_appearance_change_is_uncertainty_not_motion():
    # Frame 1 suddenly replaces the content (a flash / particle burst): the
    # change is NOT explained by any smooth flow -> high effect-transience,
    # and the flow-valid fraction drops.
    h, w = 64, 96
    y = np.zeros((3, h, w), np.float32)
    y[0] = np.tile(np.arange(w, dtype=np.float32), (h, 1))
    y[1] = np.random.RandomState(0).rand(h, w).astype(np.float32) * 255.0
    y[2] = y[0] + 3.0
    flows = _FakeFlows(_translation_flows(h, w, 3.0))
    cfg = _cfg()
    rel = _flow_reliability(y, flows, [(0, 1), (1, 2)], cfg)
    # The un-explainable burst dominates the warping residual.
    assert rel["nr_effect_transient_fraction"] > 0.3
    assert rel["nr_photometric_support_fraction"] < 0.9


def test_visibility_masks_flag_occlusion():
    h, w = 64, 96
    y = np.zeros((3, h, w), np.float32)
    # zero flow everywhere: cycle error 0 -> fully visible.
    flows = _FakeFlows(_translation_flows(h, w, 0.0))
    masks = _visibility_masks(y, flows, [(0, 1)], _cfg())
    assert masks[0].mean() > 0.95


def test_tile_dynamics_exclude_unreliable_tiles():
    h, w = 64, 96
    bundle = FrameBundle(
        rgb=np.zeros((3, h, w, 3), np.uint8),
        times=np.asarray([0.0, 1.0 / 60.0, 2.0 / 60.0]),
        indices=np.asarray([0, 1, 2]),
        width=w,
        height=h,
    )
    flows = _FakeFlows(_translation_flows(h, w, 3.0))
    # Only a tiny corner is reliable -> most tiles become NaN, and the feature
    # degrades to NaN rather than reporting junk motion.
    mask = np.zeros((h, w), bool)
    mask[:10, :10] = True
    out = _tile_motion_dynamics(
        bundle, flows, [(0, 1)], grid=4, visibility_masks=[mask])
    assert not np.isfinite(out["nr_flow_accel_ratio"])


def test_flow_valid_threshold_constant():
    assert FLOW_VALID_FRACTION_MIN == 0.25
