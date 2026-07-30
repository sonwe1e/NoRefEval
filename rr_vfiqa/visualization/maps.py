"""Unified error-map normalization (USERPLAN §8 / §6.4).

Different metrics emit dense fields on wildly different scales (a flow-error
field lives in pixels, a Jacobian determinant is dimensionless, an MCT
residual is in luma units).  To render them on a common heat scale and to feed
them into multi-evidence rules, every field is normalized to 0..1 by one of:

* ``percentile`` — clip to ``[p_low, p_high]`` percentile range then stretch;
* ``robust_z`` — ``(x - median) / (1.4826 * MAD)`` then sigmoid-squash, the
  same robust statistic USERPLAN §6.4 uses for the absolute/relative risk
  blend, so a field's "how unusual is this pixel" matches the scalar risk;
* ``minmax`` — plain min/max stretch (only for already-bounded fields).

``MapNormalization`` records the parameters so a field can be normalized at
train time and rendered with identical scaling at report time.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MapNormalization:
    """Enough information to reproduce a normalization on a new field."""

    method: str                   # "percentile" | "robust_z" | "minmax"
    vmin: float = 0.0
    vmax: float = 1.0
    median: float = 0.0
    mad: float = 1.0
    p_low: float = 1.0
    p_high: float = 99.0

    def apply(self, field: np.ndarray) -> np.ndarray:
        return normalize_map(field, method=self.method, vmin=self.vmin,
                             vmax=self.vmax, median=self.median, mad=self.mad,
                             p_low=self.p_low, p_high=self.p_high)


def normalize_map(
    field: np.ndarray,
    method: str = "percentile",
    vmin: float | None = None,
    vmax: float | None = None,
    *,
    median: float | None = None,
    mad: float | None = None,
    p_low: float = 1.0,
    p_high: float = 99.0,
    clip: bool = True,
) -> np.ndarray:
    """Return ``field`` normalized to 0..1 (float32).

    Parameters only used by the relevant ``method`` are ignored, so callers can
    pass a full ``MapNormalization`` via ``.apply`` without trimming.
    """
    f = np.nan_to_num(field.astype(np.float32, copy=False), nan=0.0)
    if method == "minmax":
        lo = float(np.min(f)) if vmin is None else float(vmin)
        hi = float(np.max(f)) if vmax is None else float(vmax)
        scale = hi - lo
        if scale < 1e-9:
            return np.zeros_like(f)
        out = (f - lo) / scale
    elif method == "robust_z":
        med = float(np.median(f)) if median is None else float(median)
        m = _mad(f) if mad is None else float(mad)
        z = (f - med) / (1.4826 * m + 1e-6)
        # sigmoid squash so extreme outliers saturate instead of dominating
        out = 1.0 / (1.0 + np.exp(-z))
        if not clip:
            return out.astype(np.float32)
        return out.astype(np.float32)
    elif method == "percentile":
        if vmin is not None and vmax is not None:
            lo, hi = float(vmin), float(vmax)
        else:
            lo = float(np.percentile(f, p_low))
            hi = float(np.percentile(f, p_high))
        scale = hi - lo
        if scale < 1e-9:
            return np.zeros_like(f)
        out = (f - lo) / scale
    else:
        raise ValueError(f"unknown normalization method: {method!r}")
    if clip:
        out = np.clip(out, 0.0, 1.0)
    return out.astype(np.float32)


def robust_z(field: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """(x - median) / (1.4826 * MAD + eps) — USERPLAN §6.4."""
    f = np.nan_to_num(field.astype(np.float32, copy=False), nan=0.0)
    med = float(np.median(f))
    mad = _mad(f)
    return (f - med) / (1.4826 * mad + eps)


def fit_normalization(field: np.ndarray, method: str = "percentile",
                      p_low: float = 1.0,
                      p_high: float = 99.0) -> MapNormalization:
    """Learn the normalization parameters from ``field`` for later ``.apply``."""
    f = np.nan_to_num(field.astype(np.float32, copy=False), nan=0.0)
    if method == "minmax":
        return MapNormalization(method="minmax", vmin=float(np.min(f)),
                                vmax=float(np.max(f)))
    if method == "robust_z":
        return MapNormalization(method="robust_z", median=float(np.median(f)),
                                mad=_mad(f))
    if method == "percentile":
        return MapNormalization(method="percentile",
                                vmin=float(np.percentile(f, p_low)),
                                vmax=float(np.percentile(f, p_high)),
                                p_low=p_low, p_high=p_high)
    raise ValueError(f"unknown normalization method: {method!r}")


def _mad(f: np.ndarray) -> float:
    med = float(np.median(f))
    return float(np.median(np.abs(f - med)))
