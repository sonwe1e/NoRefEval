"""Global technical quality prior (USERPLAN §9 — deliberately low weight).

Coarse compression / blur / noise / flicker statistics. Cannot judge
interpolation correctness; only provides context for the fusion stage.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..sampling.cheap_scan import CheapScan
from ..schema import FrameBundle


def _blockiness(gray: np.ndarray) -> float:
    """8x8 block boundary gradient vs non-boundary gradient (>1 = blocking)."""
    g = gray.astype(np.float32)
    dx = np.abs(np.diff(g, axis=1))
    dy = np.abs(np.diff(g, axis=0))
    bx = dx[:, 7::8].mean() if dx.shape[1] > 8 else dx.mean()
    by = dy[7::8, :].mean() if dy.shape[0] > 8 else dy.mean()
    nx = np.delete(dx, np.s_[7::8], axis=1).mean() if dx.shape[1] > 8 else dx.mean()
    ny = np.delete(dy, np.s_[7::8], axis=0).mean() if dy.shape[0] > 8 else dy.mean()
    return float((bx + by) / (nx + ny + 1e-6))


def _noise_sigma(gray: np.ndarray) -> float:
    """Robust high-frequency noise estimate (Laplacian MAD)."""
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    return float(np.median(np.abs(lap - np.median(lap))) / 0.6745 / np.sqrt(20.0))


def compute_window(bundle: FrameBundle) -> dict[str, float]:
    out: dict[str, float] = {}
    sharp, block, noise = [], [], []
    for t in range(bundle.rgb.shape[0]):
        g = cv2.cvtColor(bundle.rgb[t], cv2.COLOR_RGB2GRAY)
        sharp.append(float(cv2.Laplacian(g, cv2.CV_32F).var()))
        block.append(_blockiness(g))
        noise.append(_noise_sigma(g))
    sh = np.asarray(sharp)
    idx = bundle.indices.astype(np.int64)
    even, odd = sh[idx % 2 == 0], sh[idx % 2 == 1]
    out["gtq_sharpness"] = float(np.median(sh))
    out["gtq_sharp_odd_even_ratio"] = float(odd.mean() / max(even.mean(), 1e-6)) \
        if len(even) and len(odd) else 1.0
    out["gtq_blockiness"] = float(np.median(block))
    out["gtq_noise"] = float(np.median(noise))
    return out


def compute_global(scan: CheapScan) -> dict[str, float]:
    k = scan.indices
    sh = scan.sharpness.astype(np.float64)
    even, odd = sh[k % 2 == 0], sh[k % 2 == 1]
    return {
        "gtq_global_sharp_p50": float(np.median(sh)),
        "gtq_global_sharp_p10": float(np.percentile(sh, 10)),
        "gtq_global_odd_even_sharp": float(odd.mean() / max(even.mean(), 1e-6))
        if len(even) and len(odd) else 1.0,
        "gtq_global_luma_flicker": float(np.std(np.diff(scan.luma_mean))),
    }
