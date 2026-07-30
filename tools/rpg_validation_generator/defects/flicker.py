"""Temporal freeze + luminance flicker (FR05)."""
from __future__ import annotations

import numpy as np

from .base import (DefectOperatorBase, LID, bbox_of, dilate, label_mask)

COMBAT_IDS = (LID["character"], LID["enemy"], LID["projectile"], LID["particle"])


class TemporalFreezeFlicker(DefectOperatorBase):
    defect_type = "temporal_freeze_flicker"

    def run(self, frames, a, b, ctx, p):
        freeze_n = int(p.get("freeze_frames", 4))
        amp = int(p.get("luma_amp", 12))
        affected, masks, rois = [], [], []
        h, w = frames.shape[1:3]
        f = a
        block = 0
        while f < b:
            # freeze chunk: hold the first frame of the chunk
            hold = frames[f].copy()
            end_freeze = min(b, f + freeze_n)
            for g in range(f + 1, end_freeze):
                frames[g] = hold
                affected.append(g)
                masks.append(np.ones((h, w), dtype=bool))
                rois.append([0, 0, w, h])
            f = end_freeze
            # flicker chunk: alternating luminance on the combat region
            end_flick = min(b, f + 3)
            for g in range(f, end_flick):
                region = dilate(label_mask(ctx.labels[g], COMBAT_IDS), 7) > 0
                if np.any(region):
                    sign = amp if (g + block) % 2 == 0 else -amp
                    v = frames[g].astype(np.int16)
                    v[region] = np.clip(v[region] + sign, 0, 255)
                    frames[g] = v.astype(np.uint8)
                    affected.append(g)
                    masks.append(region)
                    rois.append(bbox_of(region))
                else:
                    affected.append(g)
                    masks.append(np.ones((h, w), dtype=bool))
                    rois.append([0, 0, w, h])
            f = end_flick
            block += 1
        return affected, masks, rois

    def report_params(self, p):
        return {"freeze_frames": int(p.get("freeze_frames", 4)),
                "luma_amp": int(p.get("luma_amp", 12)),
                "semantic_targets": ["character", "enemy", "projectile",
                                     "particle"]}
