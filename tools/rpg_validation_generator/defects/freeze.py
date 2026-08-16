"""Freeze-family defects (NR01 freeze, endpoint05 freeze-copy + particle blend)."""
from __future__ import annotations

import numpy as np

from .base import DefectOperatorBase, LID, bbox_of, label_mask


class Freeze(DefectOperatorBase):
    defect_type = "freeze"

    def run(self, frames, a, b, ctx, p):
        src = max(0, a - 1)
        frozen = frames[src].copy()
        affected, masks, rois = [], [], []
        h, w = frames.shape[1:3]
        for f in range(a, b):
            frames[f] = frozen
            affected.append(f)
            masks.append(np.ones((h, w), dtype=bool))
            rois.append([0, 0, w, h])
        return affected, masks, rois


class GeneratedFreezeCopy(DefectOperatorBase):
    defect_type = "generated_freeze_copy"

    def run(self, frames, a, b, ctx, p):
        affected, masks, rois = [], [], []
        for f in range(a, b):
            if f % 2 != 1 or f < 1:
                continue
            pair = (f // 2) % 2
            if pair == 0:
                # freeze-copy this odd frame from its preceding even endpoint
                frames[f] = frames[f - 1]
                m = np.ones(frames.shape[1:3], dtype=bool)
                roi = [0, 0, frames.shape[2], frames.shape[1]]
            else:
                # blend particle region with the preceding endpoint
                pm = label_mask(ctx.labels[f], (LID["particle"], LID["projectile"]))
                if not np.any(pm):
                    continue
                prev = frames[f - 1]
                frames[f][pm] = (0.5 * frames[f][pm].astype(np.float32)
                                 + 0.5 * prev[pm].astype(np.float32)).astype(np.uint8)
                m = pm
                roi = bbox_of(pm)
            affected.append(f)
            masks.append(m)
            rois.append(roi)
        return affected, masks, rois

    def report_params(self, p):
        return {"parity": "odd",
                "semantic_targets": ["particle", "projectile"]}
