"""Thin-object deletion with neighbourhood inpainting (FR03)."""
from __future__ import annotations

import numpy as np

from .base import (DefectOperatorBase, LID, bbox_of, dilate, inpaint,
                   label_mask)


class ThinObjectDelete(DefectOperatorBase):
    defect_type = "thin_object_delete"

    def run(self, frames, a, b, ctx, p):
        affected, masks, rois = [], [], []
        for f in range(a, b):
            weap = label_mask(ctx.labels[f], (LID["weapon"],))
            if weap.sum() < 24:
                continue
            ys, xs = np.where(weap)
            cx, cy = float(xs.mean()), float(ys.mean())
            coords = np.stack([xs - cx, ys - cy], axis=1).astype(np.float64)
            cov = np.cov(coords.T)
            evals, evecs = np.linalg.eigh(cov)
            axis = evecs[:, -1]
            proj = coords @ axis
            pmin, pmax = proj.min(), proj.max()
            span = max(pmax - pmin, 1e-6)
            norm = (proj - pmin) / span                 # 0 at grip, 1 at tip
            norm_map = np.full(frames.shape[1:3], -1.0)
            norm_map[ys, xs] = norm
            # delete the outer blade + tip (alternate a mid gap every other frame)
            if f % 2 == 0:
                delete = weap & (norm_map > 0.5)
            else:
                delete = weap & ((norm_map > 0.62) |
                                 ((norm_map > 0.2) & (norm_map < 0.34)))
            if delete.sum() < 6:
                delete = weap & (norm_map > 0.5)
            if not np.any(delete):
                continue
            frames[f] = inpaint(frames[f], delete, 4)
            m = dilate(delete, 3) > 0
            affected.append(f)
            masks.append(m)
            rois.append(bbox_of(m))
        return affected, masks, rois

    def report_params(self, p):
        return {"semantic_targets": ["weapon"], "fill": "neighbourhood_inpaint"}
