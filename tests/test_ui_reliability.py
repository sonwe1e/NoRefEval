"""Tests for the UI/Text reliability gate (USERPLAN §8).

UI instability is only trustworthy when the detected "UI" is a compact
structure persisting at FIXED screen coordinates.  Combat flashes and
particles produce erratic, moving masks and must be rejected (N/A) rather than
scored as UI instability.
"""

from __future__ import annotations

import numpy as np
import pytest

from rr_vfiqa.metrics.no_reference import _ui_reliability


def _edge(frame: np.ndarray) -> np.ndarray:
    import cv2
    return cv2.Canny(frame.astype(np.uint8), 60, 160) > 0


def _static_hud_frames(n: int = 6, h: int = 96, w: int = 160):
    """Scene + a fixed corner HUD box: the HUD persists at fixed coords."""
    frames = []
    for t in range(n):
        frame = np.zeros((h, w), np.uint8)
        frame[:, :] = np.random.RandomState(10 + t).randint(60, 120)
        frame[80:92, 120:150] = 200          # fixed HUD box
        frame[10:20, 20:40] = 60             # scene content moves slightly
        frames.append(frame)
    return [_edge(f) for f in frames]


def _erratic_mask_frames(n: int = 6, h: int = 96, w: int = 160):
    """Flashes / particles: bright regions appear at RANDOM positions each
    frame -> the detected "UI" mask moves around."""
    rng = np.random.RandomState(0)
    frames = []
    for t in range(n):
        frame = np.full((h, w), 60, np.uint8)
        for _ in range(8):
            x, y = rng.randint(0, w - 8), rng.randint(0, h - 8)
            frame[y:y + 6, x:x + 6] = 255    # particle burst
        frames.append(frame)
    return [_edge(f) for f in frames]


def test_static_hud_is_trustworthy():
    edges = _static_hud_frames()
    rel, per_frame = _ui_reliability(edges, (96, 160))
    assert rel["ui_detection_confidence"] >= 0.8
    assert rel["ui_screen_motion"] <= 0.3
    assert 0.0 < rel["ui_component_area_ratio"] < 0.1
    assert len(per_frame) == len(edges)


def test_erratic_masks_are_rejected():
    edges = _erratic_mask_frames()
    rel, _ = _ui_reliability(edges, (96, 160))
    # Particle bursts have a large, moving mask -> not a trustworthy HUD.
    assert rel["ui_screen_motion"] > 0.5 or \
        rel["ui_component_area_ratio"] > 0.1 or \
        rel["ui_detection_confidence"] < 0.5
