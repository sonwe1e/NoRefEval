"""Tests for the Cadence Integrity motion gate (USERPLAN P0-R1).

A genuinely static scene (pause menu, loading screen, idle character) has
native duplicate/freeze fractions near 1, but is NOT a cadence collapse —
there is no motion at ANY scale.  The motion gate must suppress the cadence
risk in that case while still penalising a real odd-frame duplication where
the common-time scale shows motion or a stable odd/even alternation.
"""

from __future__ import annotations

import numpy as np
import pytest

from rr_vfiqa.diagnosis.cadence import (
    cadence_integrity,
    cadence_penalty,
    _window_cadence_risk,
)


def _static_scene_scalars() -> dict[str, float]:
    """Scalars of a fully static frame: duplicate/freeze high, MCT low."""
    return {
        "nr_native_duplicate_fraction": 0.95,
        "nr_native_freeze_fraction": 0.98,
        "nr_mct_native_mean": 0.5,
        "nr_native_self_comp": 0.02,
        "nr_native_self_cycle": 0.5,
    }


def _collapsed_cadence_scalars() -> dict[str, float]:
    """Scalars of a real odd-frame duplication (duplicate high AND common-scale
    motion present, i.e. the even frames keep progressing)."""
    return {
        "nr_native_duplicate_fraction": 0.9,
        "nr_native_freeze_fraction": 0.5,
        "nr_mct_native_mean": 25.0,   # common-scale motion is clearly present
        "nr_native_self_comp": 0.2,
        "nr_native_self_cycle": 18.0,
    }


def test_static_scene_is_not_cadence_collapse():
    # No common-time motion, no alternation -> motion gate suppresses risk.
    risk, ev = _window_cadence_risk(
        _static_scene_scalars(), alternation=0.05, common_time_motion=0.3)
    assert risk < 0.15, f"static scene cadence risk too high: {risk}"


def test_collapsed_cadence_with_motion_is_penalised():
    # Common-time motion present -> real collapse, risk stays high.
    risk, _ = _window_cadence_risk(
        _collapsed_cadence_scalars(), alternation=0.1, common_time_motion=25.0)
    assert risk > 0.4, f"real cadence collapse risk too low: {risk}"


def test_alternation_bypasses_motion_gate():
    # Stable odd/even alternation is itself evidence of native duplication,
    # so even with low common-time motion the risk is NOT suppressed.
    risk, _ = _window_cadence_risk(
        _static_scene_scalars(), alternation=0.6, common_time_motion=0.3)
    assert risk > 0.3, f"alternation should keep cadence risk high: {risk}"


def test_cadence_penalty_mild_for_static_scene():
    # End-to-end: a static scene's overall score should barely move.
    common = 80.0
    rep = cadence_integrity(
        [_static_scene_scalars()], common,
        alternation_per_window=[0.05],
        common_time_motion_per_window=[0.3])
    # Static scene: penalty should be small (overall close to common score).
    assert rep.overall >= common * 0.7, \
        f"static scene overall too low: {rep.overall} vs common {common}"


def test_cadence_penalty_strong_for_collapse():
    # End-to-end: a real collapse with motion should be heavily penalised.
    common = 80.0
    rep = cadence_integrity(
        [_collapsed_cadence_scalars()], common,
        alternation_per_window=[0.1],
        common_time_motion_per_window=[25.0])
    assert rep.overall < common * 0.5, \
        f"collapse overall too high: {rep.overall} vs common {common}"


def test_cadence_penalty_formula_unchanged():
    # The isolated penalty formula still behaves monotonically in risk.
    assert cadence_penalty(80.0, 0.0) == pytest.approx(80.0)
    assert cadence_penalty(80.0, 1.0) == pytest.approx(80.0 * np.exp(-2.2))
