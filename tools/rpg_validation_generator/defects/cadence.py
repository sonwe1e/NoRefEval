"""Cadence collapse: Y[2i+1] = Y[2i] inside the window (NR03)."""
from __future__ import annotations

import numpy as np

from .base import DefectOperatorBase


class CadenceCollapse(DefectOperatorBase):
    defect_type = "cadence_collapse"

    def run(self, frames, a, b, ctx, p):
        affected, masks, rois = [], [], []
        h, w = frames.shape[1:3]
        # align pairs to even/odd within the window
        start = a if a % 2 == 0 else a + 1
        for f in range(start + 1, b, 2):
            frames[f] = frames[f - 1]
            affected.append(f)
            masks.append(np.ones((h, w), dtype=bool))
            rois.append([0, 0, w, h])
        return affected, masks, rois
