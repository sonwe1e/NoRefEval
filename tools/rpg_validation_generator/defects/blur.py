"""Blur-family defects (NR02 odd-frame, FR01 global, endpoint01 motion)."""
from __future__ import annotations

import cv2
import numpy as np

from .base import (DefectOperatorBase, LID, bbox_of, dilate, label_mask)


def _gblur(img, k, sigma):
    k = k | 1
    return cv2.GaussianBlur(img, (k, k), sigma)


class GlobalBlur(DefectOperatorBase):
    defect_type = "global_blur"

    def run(self, frames, a, b, ctx, p):
        k, sigma = p.get("kernel", 21), p.get("sigma", 5.0)
        affected, masks, rois = [], [], []
        h, w = frames.shape[1:3]
        for f in range(a, b):
            frames[f] = _gblur(frames[f], k, sigma)
            affected.append(f)
            m = np.ones((h, w), dtype=bool)
            masks.append(m)
            rois.append([0, 0, w, h])
        return affected, masks, rois

    def report_params(self, p):
        return {"kernel": p.get("kernel", 21), "sigma": p.get("sigma", 5.0)}


class OddFrameBlur(DefectOperatorBase):
    defect_type = "odd_frame_blur"

    def run(self, frames, a, b, ctx, p):
        k, sigma = p.get("kernel", 15), p.get("sigma", 3.0)
        affected, masks, rois = [], [], []
        h, w = frames.shape[1:3]
        for f in range(a, b):
            if f % 2 == 1:
                frames[f] = _gblur(frames[f], k, sigma)
                affected.append(f)
                masks.append(np.ones((h, w), dtype=bool))
                rois.append([0, 0, w, h])
        return affected, masks, rois

    def report_params(self, p):
        return {"kernel": p.get("kernel", 15), "sigma": p.get("sigma", 3.0),
                "parity": "odd"}


class GeneratedMotionBlur(DefectOperatorBase):
    defect_type = "generated_motion_blur"

    def run(self, frames, a, b, ctx, p):
        k, sigma = p.get("kernel", 11), p.get("sigma", 2.5)
        affected, masks, rois = [], [], []
        for f in range(a, b):
            if f % 2 != 1:
                continue
            m = label_mask(ctx.labels[f], (LID["character"], LID["enemy"]))
            m = dilate(m, 5) > 0
            if not np.any(m):
                continue
            blurred = _gblur(frames[f], k, sigma)
            frames[f][m] = blurred[m]
            affected.append(f)
            masks.append(m)
            rois.append(bbox_of(m))
        return affected, masks, rois

    def report_params(self, p):
        return {"kernel": p.get("kernel", 11), "sigma": p.get("sigma", 2.5),
                "parity": "odd", "semantic_targets": ["character", "enemy"]}
