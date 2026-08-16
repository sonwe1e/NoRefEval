"""Ghost / flicker defects tied to specific semantic regions."""
from __future__ import annotations

import numpy as np

from .base import (DefectOperatorBase, LID, bbox_of, dilate, inpaint,
                   label_mask)


class DisocclusionGhost(DefectOperatorBase):
    defect_type = "disocclusion_ghost"

    def run(self, frames, a, b, ctx, p):
        alpha = p.get("alpha", 0.42)
        affected, masks, rois = [], [], []
        for f in range(a, b):
            if f % 2 != 1 or f < 1:
                continue
            occ = label_mask(ctx.labels[f], (LID["foreground_occluder"],))
            # a ring just outside the occluder = the freshly-revealed edge
            edges = (dilate(occ, 9) > 0) & (occ == 0)
            char = label_mask(ctx.labels[f], (LID["character"],))
            region = edges | (dilate(char, 5) > 0)
            if not np.any(region):
                continue
            prev = frames[f - 1]
            cur = frames[f].astype(np.float32)
            cur[region] = (1 - alpha) * cur[region] + alpha * prev[region].astype(np.float32)
            frames[f] = np.clip(cur, 0, 255).astype(np.uint8)
            affected.append(f)
            masks.append(region)
            rois.append(bbox_of(region))
        return affected, masks, rois

    def report_params(self, p):
        return {"parity": "odd", "alpha": p.get("alpha", 0.42),
                "semantic_targets": ["foreground_occluder", "character"]}


class ProjectileGhostFlicker(DefectOperatorBase):
    defect_type = "projectile_ghost_flicker"

    def run(self, frames, a, b, ctx, p):
        affected, masks, rois = [], [], []
        for f in range(a, b):
            proj = label_mask(ctx.labels[f], (LID["projectile"],))
            part = label_mask(ctx.labels[f], (LID["particle"],))
            m = proj | part
            if not np.any(m):
                continue
            cur = frames[f].astype(np.float32)
            if f % 2 == 1 and f >= 1 and np.any(proj):
                prev = frames[f - 1].astype(np.float32)
                cur[proj] = 0.5 * cur[proj] + 0.5 * prev[proj]
            if f % 4 == 0 and np.any(part):
                # reduce particle alpha by blending toward an inpainted bg
                bg = inpaint(frames[f], part, 3).astype(np.float32)
                cur[part] = 0.5 * cur[part] + 0.5 * bg[part]
            frames[f] = np.clip(cur, 0, 255).astype(np.uint8)
            affected.append(f)
            masks.append(m)
            rois.append(bbox_of(m))
        return affected, masks, rois

    def report_params(self, p):
        return {"semantic_targets": ["projectile", "particle"],
                "odd_ghost": True, "particle_flicker_every": 4}
