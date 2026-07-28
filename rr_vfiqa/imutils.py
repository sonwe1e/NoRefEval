"""Small image/mask utilities shared by regions and model backends.

Kept dependency-free of other rr_vfiqa modules so backends can use it without
circular imports.
"""

from __future__ import annotations

import cv2
import numpy as np


def luma(rgb: np.ndarray) -> np.ndarray:
    """BT.601 luma, float32 0..255, from RGB uint8/float."""
    return (0.299 * rgb[..., 0].astype(np.float32)
            + 0.587 * rgb[..., 1].astype(np.float32)
            + 0.114 * rgb[..., 2].astype(np.float32))


def clean_mask(mask: np.ndarray, min_area: int = 64, close_k: int = 5) -> np.ndarray:
    m = mask.astype(np.uint8)
    if close_k > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    out = np.zeros_like(m)
    for lab in range(1, n):
        if stats[lab, cv2.CC_STAT_AREA] >= min_area:
            out[labels == lab] = 1
    return out


def mask_chamfer(a: np.ndarray, b: np.ndarray) -> float:
    """Mean distance from mask a boundary pixels to mask b (pixels)."""
    a = a.astype(np.uint8)
    b = b.astype(np.uint8)
    if a.sum() == 0 or b.sum() == 0:
        return float("nan")
    ba = a.astype(bool)
    eroded = cv2.erode(a, np.ones((3, 3), np.uint8))
    boundary = ba & (eroded == 0)
    if boundary.sum() == 0:
        boundary = ba
    dist = cv2.distanceTransform((b == 0).astype(np.uint8), cv2.DIST_L2, 3)
    return float(dist[boundary].mean())


def bbox_of(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def phash64(gray: np.ndarray) -> np.int64:
    """64-bit DCT perceptual hash (motion-tolerant, structure-sensitive)."""
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(small)[:8, :8]
    med = np.median(dct[1:, 1:])
    bits = (dct > med).ravel()
    h = np.uint64(0)
    for b in bits:
        h = np.uint64((np.uint64(h) << np.uint64(1)) | np.uint64(int(b)))
    return np.int64(np.uint64(h) >> np.uint64(1))


def hamming64(a: np.int64, b: np.int64) -> int:
    return int(np.uint64(np.uint64(a) ^ np.uint64(b))).bit_count()


def alpha_blend_fit(xi: np.ndarray, xm: np.ndarray, xj: np.ndarray
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel optimal α of M ≈ α·X_i + (1−α)·X_j via RGB least squares.

    Returns (alpha clipped to [0,1], residual mean-abs error per pixel).
    Real alpha mixing (crossfade ghosts, partial blends) shows α strictly
    inside (0,1) with a SMALL residual; a correct mid frame has new content
    (large residual) and a hard switch sits at α≈0/1 (USERPLAN §8.6).
    """
    d = xi.astype(np.float32) - xj.astype(np.float32)
    m = xm.astype(np.float32) - xj.astype(np.float32)
    denom = (d * d).sum(-1) + 1e-6
    alpha = np.clip((m * d).sum(-1) / denom, 0.0, 1.0)
    fitted = alpha[..., None] * xi.astype(np.float32) \
        + (1.0 - alpha[..., None]) * xj.astype(np.float32)
    resid = np.abs(xm.astype(np.float32) - fitted).mean(-1)
    return alpha.astype(np.float32), resid.astype(np.float32)
