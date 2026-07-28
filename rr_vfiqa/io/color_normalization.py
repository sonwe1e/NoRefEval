"""Global color/gamma mismatch estimation between candidate anchors and source.

Re-encoding rarely preserves pixels exactly; before interpreting anchor error as
an interpolation problem we fit a per-channel affine transform and report it
(USERPLAN.md §2.2: color range / RGB-YUV / gamma drift detection).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ColorTransform:
    """Fitted cand ≈ gain · src + offset (source as independent variable).

    ``apply`` maps CANDIDATE pixels into the source color space — the
    inverse — which is what anchor comparison needs.
    """

    gain: np.ndarray      # (3,) float32
    offset: np.ndarray    # (3,) float32
    residual: float       # fit residual in 0..255 units

    @staticmethod
    def identity() -> "ColorTransform":
        return ColorTransform(np.ones(3, np.float32), np.zeros(3, np.float32), 0.0)

    def apply(self, cand_rgb: np.ndarray) -> np.ndarray:
        """Normalize candidate → source space: (cand − offset) / gain."""
        out = (cand_rgb.astype(np.float32) - self.offset) / np.clip(self.gain, 0.25, 4.0)
        return np.clip(out, 0, 255).astype(np.uint8)

    def apply_forward(self, src_rgb: np.ndarray) -> np.ndarray:
        """Source → candidate direction (the fitted relationship itself)."""
        out = src_rgb.astype(np.float32) * self.gain + self.offset
        return np.clip(out, 0, 255).astype(np.uint8)

    def is_trivial(self, tol: float = 0.5) -> bool:
        return (np.all(np.abs(self.gain - 1.0) < 0.02)
                and np.all(np.abs(self.offset) < tol))


def estimate_color_transform(src_rgb: np.ndarray, cand_rgb: np.ndarray,
                             sample_px: int = 20000) -> ColorTransform:
    """Fit cand ≈ gain * src + offset per channel on a pixel subsample."""
    if src_rgb.shape != cand_rgb.shape:
        import cv2
        cand_rgb = cv2.resize(cand_rgb, (src_rgb.shape[1], src_rgb.shape[0]),
                              interpolation=cv2.INTER_AREA)
    a = src_rgb.reshape(-1, 3).astype(np.float32)
    b = cand_rgb.reshape(-1, 3).astype(np.float32)
    if a.shape[0] > sample_px:
        sel = np.linspace(0, a.shape[0] - 1, sample_px).astype(int)
        a, b = a[sel], b[sel]
    gain = np.ones(3, np.float32)
    offset = np.zeros(3, np.float32)
    resid = []
    for c in range(3):
        x = np.stack([a[:, c], np.ones_like(a[:, c])], axis=1)
        coef, *_ = np.linalg.lstsq(x, b[:, c], rcond=None)
        gain[c] = float(np.clip(coef[0], 0.5, 2.0))
        offset[c] = float(np.clip(coef[1], -64.0, 64.0))
        resid.append(float(np.mean(np.abs(x @ coef - b[:, c]))))
    return ColorTransform(gain, offset, float(np.mean(resid)))


def estimate_from_anchor_pairs(pairs: list[tuple[np.ndarray, np.ndarray]]) -> ColorTransform:
    """Median-combine transforms estimated on several (src, cand) anchor pairs."""
    if not pairs:
        return ColorTransform.identity()
    tfs = [estimate_color_transform(s, c) for s, c in pairs]
    return ColorTransform(
        gain=np.median(np.stack([t.gain for t in tfs]), axis=0).astype(np.float32),
        offset=np.median(np.stack([t.offset for t in tfs]), axis=0).astype(np.float32),
        residual=float(np.median([t.residual for t in tfs])),
    )
