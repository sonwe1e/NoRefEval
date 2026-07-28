"""Segmentation backends (USERPLAN §8.2).

Production path: SAM 2 teacher labels → distilled lightweight game-domain
segmenter. Until that model exists, the default backend is a classical
motion+contrast segmenter that yields character-proxy masks — good enough for
missing-area / ghost checks, clearly flagged as approximate.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import cv2
import numpy as np

from ..imutils import clean_mask, luma


class SegmentationBackend(ABC):
    name = "base"

    @abstractmethod
    def segment(self, rgb: np.ndarray, hint: dict | None = None) -> np.ndarray:
        """(H, W) uint8 foreground/character mask."""


class ClassicMotionSegmenter(SegmentationBackend):
    """Foreground from camera-compensated flow + local contrast."""

    name = "classic_motion"

    def __init__(self, flow_thr_px: float | None = None):
        self.flow_thr_px = flow_thr_px

    def segment(self, rgb: np.ndarray, hint: dict | None = None) -> np.ndarray:
        """hint may contain ``residual_flow`` (H, W, 2) — motion minus camera."""
        h, w = rgb.shape[:2]
        if hint is None or "residual_flow" not in hint:
            return np.zeros((h, w), np.uint8)
        f = hint["residual_flow"]
        mag = np.sqrt(f[..., 0] ** 2 + f[..., 1] ** 2)
        thr = self.flow_thr_px if self.flow_thr_px is not None \
            else max(2.0, float(np.percentile(mag, 80)))
        motion = mag > thr

        # Contrast against the surrounding background: local stddev boost.
        y = luma(rgb)
        local = cv2.blur(y, (31, 31))
        contrast = np.abs(y - local) > max(8.0, float(np.percentile(np.abs(y - local), 85)))

        mask = motion | (contrast & _near(motion, 15))
        mask = clean_mask(mask.astype(np.uint8), min_area=int(0.0008 * h * w), close_k=9)
        # Keep the largest few blobs (characters, not speckle).
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        if n <= 1:
            return mask
        areas = [(stats[lab, cv2.CC_STAT_AREA], lab) for lab in range(1, n)]
        areas.sort(reverse=True)
        keep = {lab for _, lab in areas[:3] if _ > 0.001 * h * w}
        out = np.zeros_like(mask)
        for lab in keep:
            out[labels == lab] = 1
        return out


def _near(mask: np.ndarray, radius: int) -> np.ndarray:
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2)
    return cv2.dilate(mask.astype(np.uint8), k) > 0


def get_segmentation_backend(name: str = "auto", **kw) -> SegmentationBackend:
    if name in ("auto", "classic_motion", "classic"):
        return ClassicMotionSegmenter(**kw)
    raise ValueError(f"unknown segmentation backend {name!r}; trained game-domain "
                     f"segmenter not installed")
