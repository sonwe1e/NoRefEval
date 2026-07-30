"""Spatial-shift defects (FR02 local shift, endpoint03 thin-weapon motion)."""
from __future__ import annotations

import cv2
import numpy as np

from ..motion import sinusoidal
from .base import (DefectOperatorBase, LID, bbox_of, dilate, inpaint,
                   label_mask, shifted_mask)


class LocalSpatialShift(DefectOperatorBase):
    defect_type = "local_spatial_shift"

    def run(self, frames, a, b, ctx, p):
        dx = int(p.get("dx", 4))
        affected, masks, rois = [], [], []
        for f in range(a, b):
            char = dilate(label_mask(ctx.labels[f], (LID["character"],)), 5) > 0
            occ = dilate(label_mask(ctx.labels[f], (LID["foreground_occluder"],)), 5) > 0
            roi = char & occ
            if roi.sum() < 16:
                # fall back to the character region if no junction
                roi = char
            if not np.any(roi):
                continue
            bg = inpaint(frames[f], roi, 5)
            out = bg
            ys, xs = np.where(roi)
            nx = xs + dx
            h, w = frames.shape[1:3]
            ok = (nx >= 0) & (nx < w)
            out[ys[ok], nx[ok]] = frames[f][ys[ok], xs[ok]]
            frames[f] = out
            m = roi | shifted_mask(roi, dx, 0)
            affected.append(f)
            masks.append(m)
            rois.append(bbox_of(m))
        return affected, masks, rois

    def report_params(self, p):
        return {"dx": int(p.get("dx", 4)),
                "semantic_targets": ["character", "foreground_occluder"]}


class ThinWeaponWrongMotion(DefectOperatorBase):
    defect_type = "thin_weapon_wrong_motion"

    def run(self, frames, a, b, ctx, p):
        ang_err = float(p.get("angle_error_degrees", 8))
        tip_off = float(p.get("tip_offset_pixels", 6))
        affected, masks, rois = [], [], []
        for f in range(a, b):
            if f % 2 != 1:
                continue
            weap = label_mask(ctx.labels[f], (LID["weapon"],))
            if weap.sum() < 24:
                continue
            t = f / ctx.fps
            ys, xs = np.where(weap)
            cx, cy = float(xs.mean()), float(ys.mean())
            coords = np.stack([xs - cx, ys - cy], axis=1).astype(np.float64)
            cov = np.cov(coords.T)
            evals, evecs = np.linalg.eigh(cov)
            axis = evecs[:, -1]                       # principal (blade) axis
            err = ang_err * sinusoidal(t, 1.6, 1.0)
            tip = tip_off * (0.5 + 0.5 * sinusoidal(t, 0.9, 1.0))
            M = cv2.getRotationMatrix2D((cx, cy), float(err), 1.0)
            M[0, 2] += tip * float(axis[0])
            M[1, 2] += tip * float(axis[1])
            layer = np.zeros_like(frames[f])
            layer[weap] = frames[f][weap]
            alpha = (weap.astype(np.uint8)) * 255
            wlayer = cv2.warpAffine(layer, M, (frames.shape[2], frames.shape[1]),
                                    flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
            walpha = cv2.warpAffine(alpha, M, (frames.shape[2], frames.shape[1]),
                                    flags=cv2.INTER_NEAREST, borderValue=0)
            bg = inpaint(frames[f], weap, 3)
            sel = walpha > 128
            bg[sel] = wlayer[sel]
            frames[f] = bg
            m = weap | sel
            affected.append(f)
            masks.append(m)
            rois.append(bbox_of(m))
        return affected, masks, rois

    def report_params(self, p):
        return {"parity": "odd",
                "angle_error_degrees": float(p.get("angle_error_degrees", 8)),
                "tip_offset_pixels": float(p.get("tip_offset_pixels", 6)),
                "semantic_targets": ["weapon"]}
