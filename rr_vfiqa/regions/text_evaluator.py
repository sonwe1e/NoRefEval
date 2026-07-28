"""Text stability branch (USERPLAN §8.4).

OCR is deliberately NOT the workhorse — its own instability would be
misread as interpolation error. We evaluate stroke *edges and topology*
inside static UI regions; component-count change catches stroke merging
(粘连) and breaking.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..config import EvalConfig
from ..schema import FrameBundle
from ._common import luma
from .ui_detector import UIDetector, _edge_fscore

_BLOCK = 16
_EDGE_DENSITY = 0.12


def _text_roi(gray: np.ndarray, ui_mask: np.ndarray) -> np.ndarray:
    """UI pixels that sit inside high-edge-density blocks (text-like)."""
    e = cv2.Canny(gray, 60, 180) > 0
    h, w = gray.shape
    bh, bw = h // _BLOCK, w // _BLOCK
    dense = np.zeros((h, w), bool)
    for by in range(bh):
        for bx in range(bw):
            patch = e[by * _BLOCK:(by + 1) * _BLOCK, bx * _BLOCK:(bx + 1) * _BLOCK]
            if patch.mean() > _EDGE_DENSITY:
                dense[by * _BLOCK:(by + 1) * _BLOCK, bx * _BLOCK:(bx + 1) * _BLOCK] = True
    return dense & (ui_mask > 0)


def _stroke_components(gray: np.ndarray, roi: np.ndarray) -> int:
    """Ink-pixel connected components inside the ROI (binarized strokes)."""
    if roi.sum() < 32:
        return 0
    vals = gray[roi]
    # Otsu within the ROI; ink = whichever side is smaller (text is usually a
    # minority of its bounding UI panel).
    thr, binv = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ink = binv.astype(bool)
    if ink[roi].mean() > 0.5:
        ink = ~ink
    n, _ = cv2.connectedComponents((ink & roi).astype(np.uint8), 8)
    return n - 1


def compute_window(bundle: FrameBundle, ui: UIDetector, cfg: EvalConfig
                   ) -> dict[str, float]:
    h, w = bundle.height, bundle.width
    mask = ui.mask_at(h, w)
    if mask.mean() < 0.004:
        return {"text_coverage": 0.0, "text_edge_f": float("nan"),
                "text_comp_ratio": float("nan")}

    g_anchor = cv2.cvtColor(bundle.rgb[1], cv2.COLOR_RGB2GRAY)
    g_mid = cv2.cvtColor(bundle.rgb[2], cv2.COLOR_RGB2GRAY)
    roi = _text_roi(g_anchor, mask)
    coverage = float(roi.mean())
    out = {"text_coverage": coverage}
    if coverage < 0.002:
        out["text_edge_f"] = float("nan")
        out["text_comp_ratio"] = float("nan")
        return out

    out["text_edge_f"] = _edge_fscore(g_mid, g_anchor, roi.astype(np.uint8))

    n_a = _stroke_components(g_anchor, roi)
    n_m = _stroke_components(g_mid, roi)
    out["text_comp_ratio"] = float(n_m / max(n_a, 1))
    # Blur of strokes: gradient energy loss inside the ROI.
    def grad_energy(g: np.ndarray) -> float:
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        return float(np.sqrt(gx * gx + gy * gy)[roi].mean())
    ga, gm = grad_energy(g_anchor), grad_energy(g_mid)
    out["text_grad_loss"] = float(np.clip(1.0 - gm / max(ga, 1e-6), 0, 1))
    return out
