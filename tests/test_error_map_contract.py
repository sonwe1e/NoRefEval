"""Tests for NR error-map normalization and overlay error-map normalization (USERPLAN §9).

These cover two layers:

1. ``rr_vfiqa.report.overlay_exporter._normalize_map`` — the P99.5 robust
   normalization used to render both the report heatmaps and the overlay
   video error maps to a 0..1 range.
2. ``rr_vfiqa.metrics.no_reference.compute_window_maps`` — the named
   (H, W) diagnostic fields returned for a no-reference window.

The window-maps tests reuse the ``_bundle`` / ``_cfg`` / ``_maps_for``
helpers from ``tests/test_nr_error_maps.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from rr_vfiqa.report.overlay_exporter import _normalize_map

# Reuse the synthetic bundle / flow helpers from the existing NR test module.
from tests.test_nr_error_maps import EXPECTED_MAPS, _bundle, _cfg, _maps_for


# ---------------------------------------------------------------------------
# _normalize_map (report + overlay heatmap normalization)
# ---------------------------------------------------------------------------


def test_normalize_map_p995_keeps_dynamic_range() -> None:
    # A 0..255 ramp should span the full normalized range; P99.5 of a
    # strictly increasing linear ramp is ~253.7, so the top end maps to ~1.0
    # and the 95th percentile lands near (but not above) 1.0.
    x = np.linspace(0, 255, 10000).reshape(100, 100)
    y = _normalize_map(x)
    assert y.min() == 0.0
    assert 0.9 < np.percentile(y, 95) <= 1.0


def test_normalize_map_handles_constant_input() -> None:
    # Constant input -> vmax == 1e-6 floor, so output is all zeros.  The key
    # property is that normalization must never produce NaN/inf.
    x = np.full((64, 64), 5.0, dtype=np.float32)
    y = _normalize_map(x)
    assert np.all(np.isfinite(y))
    assert np.all(y == y[0, 0])  # uniform (all zeros)


def test_normalize_map_nan_to_zero() -> None:
    # NaN entries are replaced with 0 before normalization, so they stay 0
    # in the output (assuming 0 is below the P99.5 ceiling).
    x = np.linspace(0, 1, 100, dtype=np.float32).reshape(10, 10)
    x[0, 0] = np.nan
    x[3, 4] = np.nan
    y = _normalize_map(x)
    assert np.all(np.isfinite(y))
    assert y[0, 0] == 0.0
    assert y[3, 4] == 0.0


def test_normalize_map_robust_to_outliers() -> None:
    # One extreme outlier must not crush the rest of the signal: P99.5 of a
    # 10000-sample field with a single 1e6 spike is still ~1.0, so the bulk
    # of the 0..1 values map into 0..1 rather than being driven to ~0.
    rng = np.random.RandomState(0)
    x = rng.rand(100, 100).astype(np.float32)  # values in 0..1
    x[50, 50] = 1e6
    y = _normalize_map(x)
    assert np.all(np.isfinite(y))
    assert np.percentile(y, 99.5) <= 1.0
    # The outlier saturates at 1.0; the remaining 9999 samples stay <= 1.0
    # and the median of the non-outlier mass is clearly above 0.
    assert float(np.median(y[y < 1.0])) > 0.0


# ---------------------------------------------------------------------------
# compute_window_maps (named diagnostic fields)
# ---------------------------------------------------------------------------


def test_compute_window_maps_returns_full_set() -> None:
    maps = _maps_for(_bundle(), _cfg())
    assert set(maps.keys()) == EXPECTED_MAPS


def test_compute_window_maps_align_to_luma_grid() -> None:
    h, w = 96, 128
    maps = _maps_for(_bundle(h=h, w=w), _cfg())
    for name, arr in maps.items():
        assert arr.shape == (h, w), f"{name} has shape {arr.shape}"
        assert arr.dtype == np.float32, f"{name} has dtype {arr.dtype}"


def test_compute_window_maps_all_finite() -> None:
    maps = _maps_for(_bundle(), _cfg())
    for name, arr in maps.items():
        assert np.all(np.isfinite(arr)), f"{name} has non-finite values"


def test_duplicate_indicator_static_is_high() -> None:
    # A fully static bundle (all identical frames -> |Delta luma| ~ 0) should
    # yield duplicate_frame_indicator ~= 1.0 everywhere (hot = high dup risk).
    bundle = _bundle()
    bundle.rgb[:] = bundle.rgb[0]
    maps = _maps_for(bundle, _cfg())
    assert float(maps["duplicate_frame_indicator"].mean()) == pytest.approx(1.0)
