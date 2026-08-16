"""Heatmap rendering and frame overlay (USERPLAN §8 / §9).

A normalized (H, W) field in 0..1 becomes a jet heatmap PNG, or a translucent
overlay blended onto the source frame so the viewer sees *where* the metric
lights up.  No matplotlib — pure OpenCV ``applyColorMap``.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .maps import normalize_map

# 0..1 -> blue->cyan->yellow->red jet approximation.  Columns are BGR, so
# ``out[..., 0]`` is the blue channel and ``out[..., 2]`` the red channel.
_JET = np.array([
    [0.50, 0.0, 0.0], [1.00, 0.0, 0.0], [1.00, 0.5, 0.0], [1.00, 1.0, 0.0],
    [0.50, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.5], [0.0, 1.0, 1.0],
], dtype=np.float32)


def heat_bgr(x: np.ndarray) -> np.ndarray:
    """Map a 0..1 field to a blue->red BGR ramp (no OpenCV dependency)."""
    x = np.clip(x, 0.0, 1.0)
    idx = x * (_JET.shape[0] - 1)
    lo = np.floor(idx).astype(np.int32)
    hi = np.clip(lo + 1, 0, _JET.shape[0] - 1)
    frac = (idx - lo)[..., None]
    ramp = _JET[lo] * (1.0 - frac) + _JET[hi] * frac
    return ramp


def heatmap_to_bgr(field: np.ndarray, method: str = "percentile",
                   p_low: float = 1.0, p_high: float = 99.0,
                   vmin: float | None = None,
                   vmax: float | None = None) -> np.ndarray:
    """Normalize ``field`` and return a (H, W, 3) jet BGR image."""
    norm = normalize_map(field, method=method, p_low=p_low, p_high=p_high,
                         vmin=vmin, vmax=vmax)
    return cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)


def blend_heat(frame: np.ndarray, field: np.ndarray, *,
               alpha: float = 0.5, method: str = "percentile",
               p_low: float = 1.0, p_high: float = 99.0,
               vmin: float | None = None,
               vmax: float | None = None,
               threshold: float = 0.25) -> np.ndarray:
    """Overlay a normalized heat field on ``frame`` (BGR or RGB uint8).

    Pixels whose normalized value is below ``threshold`` are left untouched so
    quiet regions do not get tinted.  ``alpha`` scales with the field strength
    (stronger error -> more opaque), matching the overlay exporter's look.
    """
    if frame.ndim != 3:
        raise ValueError("frame must be (H,W,3) uint8")
    norm = normalize_map(field, method=method, p_low=p_low, p_high=p_high,
                         vmin=vmin, vmax=vmax)
    heat = (heat_bgr(norm) * 255).astype(np.uint8)
    out = frame.copy()
    mask = norm > threshold
    if not mask.any():
        return out
    a = (np.clip(norm, 0.0, 1.0) * alpha)[..., None]
    blended = (out.astype(np.float32) * (1.0 - a) + heat.astype(np.float32) * a)
    out[mask] = blended[mask].astype(np.uint8)
    return out


def save_heatmap(field: np.ndarray, path: str | Path, *,
                 method: str = "percentile", p_low: float = 1.0,
                 p_high: float = 99.0, vmin: float | None = None,
                 vmax: float | None = None, label: str = "") -> Path:
    """Render ``field`` as a jet heatmap PNG (optional white label)."""
    heat = heatmap_to_bgr(field, method=method, p_low=p_low, p_high=p_high,
                          vmin=vmin, vmax=vmax)
    if label:
        cv2.putText(heat, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2, cv2.LINE_AA)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), heat)
    return path
