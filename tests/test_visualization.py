"""Tests for the spatial diagnostic visualization module (USERPLAN §8)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from rr_vfiqa.schema import robust_z
from rr_vfiqa.visualization import (
    blend_heat,
    flow_to_bgr,
    heat_bgr,
    heatmap_to_bgr,
    normalize_map,
    save_heatmap,
    fit_normalization,
)
from rr_vfiqa.visualization.flow_overlay import flow_to_bgr_with_legend
from rr_vfiqa.visualization.maps import _mad


def _field(seed: int = 0, shape=(48, 64), peak: float = 5.0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    f = rng.rand(*shape).astype(np.float32)
    f[20:24, 30:34] = peak
    return f


# --------------------------------------------------------------------------- #
# maps
# --------------------------------------------------------------------------- #
def test_normalize_map_percentile_range():
    f = _field()
    out = normalize_map(f, method="percentile")
    assert out.shape == f.shape
    assert out.min() == pytest.approx(0.0)
    assert out.max() == pytest.approx(1.0)


def test_normalize_map_minmax():
    f = _field()
    out = normalize_map(f, method="minmax")
    assert out.min() == pytest.approx(0.0)
    assert out.max() == pytest.approx(1.0)


def test_normalize_map_robust_z_squashes_outliers():
    f = _field(peak=100.0)   # one huge outlier
    out = normalize_map(f, method="robust_z")
    assert out.shape == f.shape
    # sigmoid output lives in (0, 1]; the bulk stays well below the outlier
    assert 0.0 < float(out.min()) <= float(out.max()) <= 1.0
    # the outlier no longer dominates: median bulk value is modest
    assert float(np.median(out)) < 0.95


def test_normalize_map_constant_field_is_zero():
    f = np.full((20, 20), 3.0, np.float32)
    for method in ("percentile", "minmax"):
        out = normalize_map(f, method=method)
        assert float(out.max()) == pytest.approx(0.0)


def test_normalize_map_rejects_unknown_method():
    with pytest.raises(ValueError):
        normalize_map(_field(), method="bogus")


def test_robust_z_zero_for_constant():
    f = np.ones((10, 10), np.float32)
    assert float(np.max(np.abs(robust_z(f)))) == pytest.approx(0.0)


def test_mad_matches_definition():
    f = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
    med = float(np.median(f))
    expected = float(np.median(np.abs(f - med)))
    assert _mad(f) == pytest.approx(expected)


def test_fit_normalization_apply_roundtrip():
    f = _field()
    for method in ("percentile", "minmax", "robust_z"):
        norm = fit_normalization(f, method=method)
        out = norm.apply(f)
        assert out.shape == f.shape
        assert float(out.min()) >= 0.0
        if method != "robust_z":   # robust_z is not clipped to [0,1]
            assert float(out.max()) <= 1.0 + 1e-6


# --------------------------------------------------------------------------- #
# heatmaps
# --------------------------------------------------------------------------- #
def test_heat_bgr_ramp_endpoints():
    low = heat_bgr(np.zeros((4, 4), np.float32))
    high = heat_bgr(np.ones((4, 4), np.float32))
    assert low.shape == (4, 4, 3)
    assert high.shape == (4, 4, 3)
    # 0 -> blue channel dominant, 1 -> red channel dominant
    assert float(low[..., 0].mean()) > float(low[..., 2].mean())
    assert float(high[..., 2].mean()) > float(high[..., 0].mean())


def test_heatmap_to_bgr_shape_and_type():
    heat = heatmap_to_bgr(_field())
    assert heat.shape == (48, 64, 3)
    assert heat.dtype == np.uint8


def test_blend_heat_only_tints_strong_regions():
    frame = np.full((48, 64, 3), 128, np.uint8)
    # A field that is genuinely quiet (0) over the corner and hot in the
    # centre, normalized with minmax so the quiet region stays below threshold.
    f = np.zeros((48, 64), np.float32)
    f[20:28, 28:36] = 5.0
    blended = blend_heat(frame, f, alpha=0.6, threshold=0.25, method="minmax")
    assert blended.shape == frame.shape
    # quiet corner (0,0) stays untouched
    assert np.array_equal(blended[0, 0], frame[0, 0])
    # hot region changes
    assert not np.array_equal(blended[24, 32], frame[24, 32])


def test_blend_heat_rejects_non_rgb():
    with pytest.raises(ValueError):
        blend_heat(np.zeros((10, 10), np.uint8), np.zeros((10, 10)))


def test_save_heatmap_writes_png_with_label():
    with tempfile.TemporaryDirectory() as d:
        p = save_heatmap(_field(), Path(d) / "h.png", label="hello")
        assert p.exists()
        assert p.suffix == ".png"
        assert p.stat().st_size > 0


# --------------------------------------------------------------------------- #
# flow overlay
# --------------------------------------------------------------------------- #
def test_flow_to_bgr_shape():
    flow = np.random.randn(48, 64, 2).astype(np.float32)
    bgr = flow_to_bgr(flow)
    assert bgr.shape == (48, 64, 3)
    assert bgr.dtype == np.uint8


def test_flow_to_bgr_zero_flow_is_grey():
    flow = np.zeros((32, 32, 2), np.float32)
    bgr = flow_to_bgr(flow)
    # magnitude 0 -> value 0 -> black
    assert float(bgr[..., 2].mean()) == pytest.approx(0.0)


def test_flow_to_bgr_rejects_bad_shape():
    with pytest.raises(ValueError):
        flow_to_bgr(np.zeros((10, 10), np.float32))


def test_flow_to_bgr_with_legend_writes_png():
    flow = np.random.randn(64, 64, 2).astype(np.float32) * 3
    with tempfile.TemporaryDirectory() as d:
        p = flow_to_bgr_with_legend(flow, Path(d) / "flow.png", step=16)
        assert p.exists()
        assert p.stat().st_size > 0
