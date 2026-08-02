"""Tests for Cadence Integrity v2 hard gates (USERPLAN §4/§5).

v2 declares a cadence collapse only when ALL three hard gates hold, estimated
from the full-film uniform scan:

1. parent-scale (t -> t+2, FPS-adaptive) motion is present — a genuinely
   static scene is NOT a cadence failure;
2. native adjacent-frame diffs are parity-asymmetric — one phase carries
   (near-)zero new information;
3. the asymmetry direction is coherent across the film — transient combat
   effects that flip phase are suppressed.

Low MCT / self-composition / self-cycle are only auxiliary corroboration.
"""

from __future__ import annotations

import numpy as np
import pytest

from rr_vfiqa.diagnosis.cadence import (
    cadence_integrity,
    cadence_penalty,
    scan_phase_stats,
    _window_cadence_risk,
)
from rr_vfiqa.sampling.cheap_scan import CheapScan


def _static_scene_scalars() -> dict[str, float]:
    """Scalars of a fully static frame: duplicate/freeze high, no parent motion."""
    return {
        "nr_native_duplicate_fraction": 0.95,
        "nr_native_freeze_fraction": 0.98,
        "nr_mct_native_mean": 0.5,
        "nr_native_self_comp": 0.02,
        "nr_native_self_cycle": 0.5,
    }


def _true_60fps_combat_scalars() -> dict[str, float]:
    """Scalars of a true 60 FPS combat clip: real parent motion, UNIFORM
    adjacent diffs (parity asymmetry ~0), smooth flow (low residuals)."""
    return {
        "nr_native_duplicate_fraction": 0.05,
        "nr_native_freeze_fraction": 0.10,
        "nr_mct_native_mean": 12.0,
        "nr_native_self_comp": 0.06,
        "nr_native_self_cycle": 10.0,
        "nr_parent_motion_p90": 22.0,
        "nr_parent_moving_pixel_fraction": 0.45,
        "nr_phase_asymmetry": 0.05,
        "nr_phase_gap_signed": 0.03,
        "nr_native_motion_alternation": 0.06,
    }


def _collapsed_cadence_scalars() -> dict[str, float]:
    """Scalars of a real odd-frame duplication (30->60 copy): strong parent
    motion, MAXIMAL parity asymmetry, one phase carrying no new content."""
    return {
        "nr_native_duplicate_fraction": 0.5,
        "nr_native_freeze_fraction": 0.5,
        "nr_mct_native_mean": 25.0,
        "nr_native_self_comp": 0.2,
        "nr_native_self_cycle": 18.0,
        "nr_parent_motion_p90": 26.0,
        "nr_parent_moving_pixel_fraction": 0.5,
        "nr_phase_asymmetry": 0.9,
        "nr_phase_gap_signed": 0.85,
        "nr_native_motion_alternation": 0.7,
    }


def _transient_combat_scalars() -> dict[str, float]:
    """Scalars with strong LOCAL parity asymmetry but UNSTABLE direction
    across windows (flicker / particles): must be suppressed by coherence."""
    return dict(
        _collapsed_cadence_scalars(),
        nr_phase_gap_signed=np.nan,   # no stable sign at the window level
    )


def _fake_scan(d, d_parent, moving_native, moving_parent, indices=None):
    """Build a synthetic CheapScan for the full-film cadence path."""
    n = len(d)
    indices = indices if indices is not None else np.arange(n, dtype=np.int32)
    return CheapScan(
        width=32,
        indices=np.asarray(indices, np.int32),
        luma_mean=np.zeros(n, np.float32),
        luma_std=np.zeros(n, np.float32),
        sharpness=np.zeros(n, np.float32),
        grad_energy=np.zeros(n, np.float32),
        edge_frac=np.zeros(n, np.float32),
        hash64=np.zeros(n, np.int64),
        hash_dist=np.zeros(n, np.float32),
        frame_diff=np.asarray(d, np.float32),
        hist_dist=np.zeros(n, np.float32),
        frame_diff_parent=np.asarray(d_parent, np.float32),
        moving_frac_native=np.asarray(moving_native, np.float32),
        moving_frac_parent=np.asarray(moving_parent, np.float32),
    )


def _uniform_motion_scan():
    """True 60 FPS: uniform adjacent diffs, no parity structure."""
    n = 192
    d = np.zeros(n)
    d[1:] = np.abs(np.random.RandomState(0).normal(6.0, 2.0, n - 1))
    d_parent = np.abs(np.random.RandomState(1).normal(12.0, 2.0, n))
    return _fake_scan(d, d_parent, np.full(n, 0.5), np.full(n, 0.5))


def _duplicate_scan():
    """30->60 copy: adjacent diffs alternate 0 / real-motion, parent diffs
    uniformly real.  Even-frame diffs carry real motion (source transition),
    odd-frame diffs are ~0 (copy frame)."""
    n = 200
    motion = np.abs(np.random.RandomState(1).normal(8.0, 2.0, (n - 1) // 2))
    d = np.zeros(n)
    d[2::2] = motion
    d_parent = np.abs(np.random.RandomState(2).normal(12.0, 2.0, n))
    return _fake_scan(d, d_parent, np.full(n, 0.5), np.full(n, 0.5))


# ------------------------------------------------------------- unit: gates 1+2
def test_static_scene_is_not_cadence_collapse():
    # Gate 1 fails: no parent-scale motion.
    risk, ev = _window_cadence_risk(
        _static_scene_scalars(), parent_motion_p90=0.5,
        parent_moving_fraction=0.01, phase_asymmetry=0.05)
    assert risk < 0.15, f"static scene cadence risk too high: {risk}"
    assert any(e.get("verdict") == "static_scene" for e in ev)


def test_true_60fps_combat_is_not_cadence_collapse():
    # Gate 2 fails: uniform adjacent diffs -> no parity asymmetry.
    risk, ev = _window_cadence_risk(
        _true_60fps_combat_scalars(), parent_motion_p90=22.0,
        parent_moving_fraction=0.45, phase_asymmetry=0.05)
    assert risk < 0.15, f"true 60 FPS combat cadence risk too high: {risk}"
    assert any(e.get("verdict") == "no_parity_structure" for e in ev)


def test_collapsed_cadence_with_motion_is_penalised():
    # Gates 1+2 pass: parent motion present AND strong asymmetry.
    risk, _ = _window_cadence_risk(
        _collapsed_cadence_scalars(), parent_motion_p90=26.0,
        parent_moving_fraction=0.5, phase_asymmetry=0.9)
    assert risk > 0.4, f"real cadence collapse risk too low: {risk}"


def test_low_residual_alone_never_fires():
    # USERPLAN §4.2: low MCT/comp/cycle is what SMOOTH MOTION also produces;
    # without parent motion + asymmetry it must yield zero risk.
    scalars = dict(_static_scene_scalars())
    risk, _ = _window_cadence_risk(
        scalars, parent_motion_p90=0.5, parent_moving_fraction=0.0,
        phase_asymmetry=0.0)
    assert risk == 0.0


# ---------------------------------------------------- full-film scan path
def test_scan_true_60fps_risk_low():
    stats = scan_phase_stats(_uniform_motion_scan())
    assert stats["phase_coherence"] >= 0.0
    rep = cadence_integrity(
        [_true_60fps_combat_scalars()], 80.0, scan_stats=stats)
    # Gate 2 (asymmetry) fails for uniform motion -> ~0 risk.
    assert rep.cadence_risk <= 0.15, rep.evidence
    # Separate reporting: overall is NOT crushed by the exponential penalty.
    assert rep.overall == pytest.approx(80.0)


def test_scan_duplicate_risk_high():
    stats = scan_phase_stats(_duplicate_scan())
    assert stats["phase_asymmetry"] > 0.5, stats
    assert stats["phase_coherence"] > 0.5, stats
    rep = cadence_integrity(
        [_collapsed_cadence_scalars()], 80.0, scan_stats=stats)
    assert rep.cadence_risk > 0.4, rep.evidence
    assert rep.gates == {
        "parent_motion": True,
        "phase_asymmetry": True,
        "phase_coherence": True,
    }


def test_scan_unstable_direction_suppresses_risk():
    # Combat flicker: strong local asymmetry but the *direction* flips
    # block-to-block -> coherence below the gate -> risk must be suppressed
    # even though auxiliary signals agree.
    n = 256
    rng = np.random.RandomState(3)
    d = np.zeros(n)
    motion = np.abs(rng.normal(8.0, 2.0, n // 2))
    for bs in range(2, n, 32):
        be = min(bs + 32, n)
        even_idx = np.arange(bs, be, 2)
        odd_idx = np.arange(bs + 1, be, 2)
        if (bs // 32) % 2 == 0:
            d[even_idx] = motion[:len(even_idx)]
        else:
            d[odd_idx] = motion[:len(odd_idx)]
    d_parent = np.abs(rng.normal(12.0, 2.0, n))
    scan = _fake_scan(d, d_parent, np.full(n, 0.5), np.full(n, 0.5))
    stats = scan_phase_stats(scan)
    assert stats["phase_coherence"] < 0.5, stats
    rep = cadence_integrity(
        [_collapsed_cadence_scalars()], 80.0, scan_stats=stats)
    assert rep.cadence_risk == 0.0, rep.evidence
    assert rep.gates["phase_coherence"] is False


# ----------------------------------------------------------------- penalty
def test_separate_reporting_keeps_overall():
    stats = scan_phase_stats(_duplicate_scan())
    rep = cadence_integrity(
        [_collapsed_cadence_scalars()], 80.0, scan_stats=stats,
        penalty_mode="separate")
    # Even a high-risk collapse does NOT crush the artifact-quality overall
    # before calibration (USERPLAN §5); cadence is reported as its own axis.
    assert rep.overall == pytest.approx(80.0)
    assert rep.cadence_integrity < 60.0


def test_cap15_penalty_is_bounded():
    stats = scan_phase_stats(_duplicate_scan())
    rep = cadence_integrity(
        [_collapsed_cadence_scalars()], 80.0, scan_stats=stats,
        penalty_mode="cap15")
    # S_final = S * (0.85 + 0.15 * cadence/100): worst case loses 15 points.
    assert rep.overall >= 80.0 * 0.85
    assert rep.overall < 80.0


def test_cadence_penalty_formula_unchanged():
    # The isolated legacy penalty still behaves monotonically in risk.
    assert cadence_penalty(80.0, 0.0) == pytest.approx(80.0)
    assert cadence_penalty(80.0, 1.0) == pytest.approx(80.0 * np.exp(-2.2))
