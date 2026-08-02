"""Minimal real-video-like Cadence regression set (USERPLAN §11).

Constructs the True-60 / Duplicate-30→60 / Blend-30→60 gradient on a synthetic
combat-like scene and verifies the v2 hard gates behave as specified:

* a TRUE 60 FPS stream must not be flagged as a cadence collapse (risk ≤ 0.10);
* a 30→60 duplicate stream (odd frames copy the previous even frame) must be
  flagged strongly (risk ≥ 0.70);
* combat-style flashes / particles on a true 60 FPS stream must NOT trip the
  cadence gates (transient, direction-unstable phase).

Linear-blend 30→60 is deliberately NOT a duplicate collapse — every frame
carries new (half-blended) content — so its cadence risk stays low; its quality
cost is a *phase* artifact (blurred one phase), which the Phase-Consistency
branch reports.  This is the USERPLAN §4.2 design: cadence = frames with zero
new information, not every interpolation artifact.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from rr_vfiqa.diagnosis.cadence import cadence_integrity, scan_phase_stats
from rr_vfiqa.io.video_reader import VideoReader
from rr_vfiqa.sampling.cheap_scan import scan_candidate
from rr_vfiqa.testing.synth import render_scene, write_video


def _combat_scene(n_frames: int, seed: int, w: int = 160, h: int = 96):
    """A busy moving scene with occasional flashes/particles (combat-like).

    Used ONLY as a true-60 negative control: its flashes produce large,
    direction-unstable appearance changes, but the cadence gates must NOT fire
    (the video is genuinely 60 FPS, every frame carries new content).
    """
    rng = np.random.RandomState(seed)
    frames = []
    base = np.zeros((h, w, 3), np.uint8)
    for t in range(n_frames):
        y = int(20 + 30 * (0.5 + 0.5 * np.sin(t * 0.15)))
        x = int(20 + 40 * (0.5 + 0.5 * np.cos(t * 0.11)))
        frame = base.copy()
        cv2_blob = np.zeros((h, w), np.uint8)
        cv2.circle(cv2_blob, (x, y), 8, 255, -1)
        frame[..., 0] = cv2_blob.astype(np.uint8)
        frame[..., 1] = cv2_blob.astype(np.uint8) * 0.6
        if t % 12 in (3, 9):
            frame += rng.randint(0, 90, frame.shape, np.uint8)
        frames.append(frame)
    return np.stack(frames)


def _true60() -> np.ndarray:
    # Smooth real motion between every adjacent pair (proper 60 FPS).
    return render_scene(n_frames=160, w=160, h=96, seed=7)


def _dup30_to_60(source30: np.ndarray) -> np.ndarray:
    return np.repeat(source30, 2, axis=0)


def _blend30_to_60(source30: np.ndarray) -> np.ndarray:
    out = []
    for i in range(len(source30)):
        out.append(source30[i])
        if i + 1 < len(source30):
            mid = (source30[i].astype(np.float32)
                   + source30[i + 1].astype(np.float32)) * 0.5
            out.append(mid.astype(np.uint8))
    return np.stack(out)


def _duplicate_window_scalars() -> dict[str, float]:
    """Auxiliary corroboration that a real duplicate window would emit."""
    return {
        "nr_native_duplicate_fraction": 0.5,
        "nr_native_freeze_fraction": 0.5,
        "nr_mct_native_mean": 22.0,
        "nr_native_self_comp": 0.2,
        "nr_native_self_cycle": 16.0,
        "nr_native_motion_alternation": 0.7,
    }


def _scan_stats(video_path: str) -> dict:
    scan = scan_candidate(VideoReader(video_path))
    return scan_phase_stats(scan)


@pytest.fixture(scope="module")
def cadence_gradient(tmp_path_factory):
    root = tmp_path_factory.mktemp("cadence_regression")
    true60 = _true60()
    source30 = true60[::2]
    variants = {
        "true60": _true60(),
        "dup60": _dup30_to_60(source30),
        "blend60": _blend30_to_60(source30),
        "combat60": _combat_scene(n_frames=160, seed=7),
    }
    paths = {}
    for name, arr in variants.items():
        p = root / f"{name}.mp4"
        write_video(p, arr, 60)
        paths[name] = str(p)
    return paths


def test_true_60_fps_is_not_a_cadence_collapse(cadence_gradient):
    stats = _scan_stats(cadence_gradient["true60"])
    rep = cadence_integrity([_duplicate_window_scalars()], 80.0,
                            scan_stats=stats)
    # Uniform motion -> no parity asymmetry -> gates close -> risk ~0.
    assert rep.cadence_risk <= 0.10, rep.evidence
    assert rep.gates["phase_asymmetry"] is False
    # Separate reporting: the artifact-quality overall is NOT crushed.
    assert rep.overall == pytest.approx(80.0)


def test_duplicate_30_to_60_is_flagged(cadence_gradient):
    stats = _scan_stats(cadence_gradient["dup60"])
    rep = cadence_integrity([_duplicate_window_scalars()], 80.0,
                            scan_stats=stats)
    assert rep.gates == {
        "parent_motion": True,
        "phase_asymmetry": True,
        "phase_coherence": True,
    }
    assert rep.cadence_risk >= 0.70, rep.evidence
    assert rep.cadence_integrity < 30.0


def test_blend_30_to_60_is_not_a_duplicate_collapse(cadence_gradient):
    stats = _scan_stats(cadence_gradient["blend60"])
    rep = cadence_integrity([_duplicate_window_scalars()], 80.0,
                            scan_stats=stats)
    # Every frame carries new (half-blended) content -> low cadence risk.
    assert rep.cadence_risk <= 0.30, rep.evidence


def test_combat_flashes_do_not_trip_cadence(cadence_gradient):
    # A genuine-60 combat-style scene WITH flashes / particle bursts: its
    # large appearance changes are direction-unstable, so the cadence gates
    # must not fire (this is the USERPLAN §4.3 false-positive case).
    stats = _scan_stats(cadence_gradient["combat60"])
    rep = cadence_integrity([_duplicate_window_scalars()], 80.0, scan_stats=stats)
    assert rep.cadence_risk <= 0.10, rep.evidence
    assert rep.gates["phase_asymmetry"] is False or \
        rep.gates["phase_coherence"] is False
