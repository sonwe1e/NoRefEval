"""Tests for resolution-independent spatial distances (USERPLAN §6.2 / P2).

The core property: a pixel distance reported as a fraction of the frame
diagonal must be (approximately) invariant when the same content is evaluated
at different working resolutions.  We also assert the helper and the Chamfer
tolerance threshold scale correctly.
"""

from __future__ import annotations

import numpy as np
import pytest

from rr_vfiqa.imutils import mask_chamfer, spatial_norm_factor


def _mask_pair(h: int, w: int, shift: int = 6):
    """Two binary masks offset by ``shift`` pixels — a controllable defect."""
    a = np.zeros((h, w), np.uint8)
    b = np.zeros((h, w), np.uint8)
    a[h // 4:3 * h // 4, w // 4:3 * w // 4] = 1
    b[h // 4:3 * h // 4, w // 4 + shift:3 * w // 4 + shift] = 1
    return a, b


def test_spatial_norm_factor_is_diagonal():
    assert spatial_norm_factor(96, 128) == pytest.approx(np.hypot(96, 128))


def test_chamfer_scales_linearly_with_resolution():
    # Same relative defect (shift = 3% of width) at two resolutions → the raw
    # Chamfer distance doubles with the resolution, but the NORMALIZED
    # distance (fraction of diagonal) stays the same.
    a64, b64 = _mask_pair(64, 64, shift=2)
    a128, b128 = _mask_pair(128, 128, shift=4)
    raw_small = mask_chamfer(a64, b64)
    raw_large = mask_chamfer(a128, b128)
    assert raw_large == pytest.approx(2 * raw_small, rel=0.05)

    norm_small = raw_small / spatial_norm_factor(64, 64)
    norm_large = raw_large / spatial_norm_factor(128, 128)
    assert norm_large == pytest.approx(norm_small, rel=0.05)


def test_normalized_chamfer_is_resolution_independent():
    # A fixed relative defect across several resolutions yields the same
    # normalized Chamfer distance (the property USERPLAN §6.2 requires).
    norms = []
    for scale in (1, 2, 3, 4):
        a, b = _mask_pair(64 * scale, 96 * scale, shift=3 * scale)
        raw = mask_chamfer(a, b)
        norms.append(raw / spatial_norm_factor(64 * scale, 96 * scale))
    for n in norms[1:]:
        assert n == pytest.approx(norms[0], rel=0.02)


def test_chamfer_tolerance_threshold_scales_with_resolution():
    # The Chamfer match tolerance (2 px at the reference resolution) must
    # shrink at lower resolution so the same *relative* edge displacement is
    # still counted as a match.  ``spatial_norm_factor`` is the conversion.
    norm = spatial_norm_factor(480, 288)
    assert (2.0 / norm) == pytest.approx(2.0 / np.hypot(480, 288))


def test_empty_mask_chamfer_is_nan():
    a = np.zeros((64, 64), np.uint8)
    b = np.zeros((64, 64), np.uint8)
    assert np.isnan(mask_chamfer(a, b))
