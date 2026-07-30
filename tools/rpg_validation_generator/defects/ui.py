"""UI / text defects.  These move screen-space elements only; the world
region (outside the moving-UI footprint) stays bit-identical to the clean
master by refilling uncovered HUD pixels from the world-only buffer."""
from __future__ import annotations

import numpy as np

from ..motion import sinusoidal
from .base import (DefectOperatorBase, LID, bbox_of, dilate, label_mask,
                   paste_shifted, shifted_mask, spatial_box)

SKILL_BOX = (0.62, 1.0, 0.66, 1.0)
SHOP_BOX = (0.74, 1.0, 0.10, 0.82)
FLOAT_BOX = (0.0, 1.0, 0.12, 0.55)


def _sample_color(frame, mask):
    px = frame[mask]
    if len(px) == 0:
        return (255, 255, 255)
    return tuple(int(v) for v in np.median(px, axis=0))


class UIDrift(DefectOperatorBase):
    """NR04: skill region periodic drift + floating-name parity jitter."""
    defect_type = "ui_drift"

    def run(self, frames, a, b, ctx, p):
        assert ctx.world_only is not None, "ui_drift needs world_only buffer"
        affected, masks, rois = [], [], []
        h, w = frames.shape[1:3]
        skill_box = spatial_box(h, w, *SKILL_BOX)
        float_box = spatial_box(h, w, *FLOAT_BOX)
        for f in range(a, b):
            t = f / ctx.fps
            dx = int(round(3 * sinusoidal(t, 1.7, 1.0)))
            dy = int(round(2 * sinusoidal(t, 1.3, 1.0, phase=1.0)))
            jt = 1 if f % 2 == 0 else -1
            skill = label_mask(ctx.labels[f], (LID["ui"],)) & skill_box
            ftext = label_mask(ctx.labels[f], (LID["text"],)) & float_box
            clean = frames[f].copy()
            base = clean.copy()
            erase = skill | ftext
            base[erase] = ctx.world_only[f][erase]
            paste_shifted(base, clean, skill, dx, dy)
            paste_shifted(base, clean, ftext, 0, jt)
            frames[f] = base
            m = erase | shifted_mask(skill, dx, dy) | shifted_mask(ftext, 0, jt)
            affected.append(f)
            masks.append(m)
            rois.append(bbox_of(m))
        return affected, masks, rois

    def report_params(self, p):
        return {"skill_drift_px": 3, "name_jitter_px": 1,
                "world_unchanged": True}


class UITextDrift(DefectOperatorBase):
    """Endpoint04 (odd frames): shop panel +3px, price text parity jitter."""
    defect_type = "ui_text_drift"

    def run(self, frames, a, b, ctx, p):
        assert ctx.world_only is not None
        affected, masks, rois = [], [], []
        h, w = frames.shape[1:3]
        shop_box = spatial_box(h, w, *SHOP_BOX)
        for f in range(a, b):
            if f % 2 != 1:
                continue
            jt = 1 if (f // 2) % 2 == 0 else -1
            shop = label_mask(ctx.labels[f], (LID["ui"],)) & shop_box
            price = label_mask(ctx.labels[f], (LID["text"],)) & shop_box
            if not (np.any(shop) or np.any(price)):
                continue
            clean = frames[f].copy()
            base = clean.copy()
            erase = shop | price
            base[erase] = ctx.world_only[f][erase]
            paste_shifted(base, clean, shop, 3, 0)
            paste_shifted(base, clean, price, 0, jt)
            frames[f] = base
            m = erase | shifted_mask(shop, 3, 0) | shifted_mask(price, 0, jt)
            affected.append(f)
            masks.append(m)
            rois.append(bbox_of(m))
        return affected, masks, rois

    def report_params(self, p):
        return {"parity": "odd", "shop_shift_px": 3, "price_jitter_px": 1,
                "world_unchanged": True}


class UITextCorruption(DefectOperatorBase):
    """FR04: shop +3px, price strokes thickened, skill buttons hue shift."""
    defect_type = "ui_text_corruption"

    def run(self, frames, a, b, ctx, p):
        assert ctx.world_only is not None
        affected, masks, rois = [], [], []
        h, w = frames.shape[1:3]
        shop_box = spatial_box(h, w, *SHOP_BOX)
        skill_box = spatial_box(h, w, *SKILL_BOX)
        for f in range(a, b):
            shop = label_mask(ctx.labels[f], (LID["ui"],)) & shop_box
            price = label_mask(ctx.labels[f], (LID["text"],)) & shop_box
            skill = label_mask(ctx.labels[f], (LID["ui"],)) & skill_box
            clean = frames[f].copy()
            base = clean.copy()
            erase = shop | price
            if np.any(erase):
                base[erase] = ctx.world_only[f][erase]
                paste_shifted(base, clean, shop, 3, 0)
                paste_shifted(base, clean, price, 3, 0)
            # thicken price strokes (in the shifted position)
            if np.any(price):
                price_color = _sample_color(clean, price)
                shifted_price = shifted_mask(price, 3, 0)
                thick = (dilate(shifted_price, 3) > 0) & ~shifted_price & shop_box
                base[thick] = price_color
            # slight colour shift on skill buttons
            if np.any(skill):
                delta = np.array([14, -6, -10], dtype=np.int16)
                v = base[skill].astype(np.int16) + delta
                base[skill] = np.clip(v, 0, 255).astype(np.uint8)
            frames[f] = base
            m = (erase | shifted_mask(shop, 3, 0) | shifted_mask(price, 3, 0)
                 | ((dilate(shifted_mask(price, 3, 0), 3) > 0) & shop_box)
                 | skill)
            affected.append(f)
            masks.append(m)
            rois.append(bbox_of(m))
        return affected, masks, rois

    def report_params(self, p):
        return {"shop_shift_px": 3, "price_thicken": True,
                "skill_color_shift": True, "world_unchanged": True}
