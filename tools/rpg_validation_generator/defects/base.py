"""Shared defect infrastructure (PICPLAN §12, §13).

A DefectOperator mutates a writable candidate frame buffer in place and
returns a DefectResult describing exactly which frames / pixels changed so
the manifest and validator can reason about it.  The per-frame effect mask
is saved as a 320x180 PNG (only inside the defect window) — that mask is
the *footprint*: outside it the candidate must equal the clean reference.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ..config import GeneratorConfig
from ..motion import frame_range
from ..schema import DefectResult, SEMANTIC_CLASSES

MASK_SIZE = (320, 180)

LID = {k: SEMANTIC_CLASSES[k] for k in
       ("character", "enemy", "weapon", "projectile", "particle",
        "ui", "text", "foreground_occluder")}
UI_TEXT_IDS = (LID["ui"], LID["text"])


@dataclass
class DefectContext:
    fps: int
    n_frames: int
    labels: np.ndarray                 # (n, h, w) uint8 at candidate fps
    world_only: np.ndarray | None      # (n, h, w, 3) uint8 at candidate fps
    config: GeneratorConfig
    rng: np.random.Generator
    case_dir: Path


class DefectOperator:
    defect_type: str = "base"

    def apply(self, frames: np.ndarray, fps: int, start_time: float,
              end_time: float, context: DefectContext,
              **params) -> DefectResult:  # pragma: no cover - abstract
        raise NotImplementedError


# ---------------------------------------------------------------- utilities
def clamp_window(ctx: DefectContext, start: float, end: float) -> tuple[int, int]:
    a, b = frame_range(ctx.fps, start, end, ctx.n_frames)
    b = min(b, ctx.n_frames)
    a = max(0, min(a, ctx.n_frames - 1))
    return a, b


def label_mask(label_frame: np.ndarray, ids) -> np.ndarray:
    ids = np.asarray(ids, dtype=np.uint8)
    return np.isin(label_frame, ids)


def dilate(mask: np.ndarray, k: int = 3, iters: int = 1) -> np.ndarray:
    if k <= 1:
        return mask.astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    return cv2.dilate(mask.astype(np.uint8) * 255, kernel, iterations=iters)


def bbox_of(mask: np.ndarray) -> list[int]:
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return [0, 0, 0, 0]
    x, y = int(xs.min()), int(ys.min())
    return [x, y, int(xs.max()) - x + 1, int(ys.max()) - y + 1]


def inpaint(frame: np.ndarray, mask: np.ndarray, radius: int = 3) -> np.ndarray:
    m = (mask > 0).astype(np.uint8) * 255
    if not np.any(m):
        return frame.copy()
    return cv2.inpaint(frame, m, radius, cv2.INPAINT_TELEA)


def paste_shifted(dst: np.ndarray, src: np.ndarray, mask: np.ndarray,
                  dx: int, dy: int) -> np.ndarray:
    """Copy ``src`` pixels (where mask) into ``dst`` shifted by (dx, dy)."""
    h, w = dst.shape[:2]
    if dx == 0 and dy == 0:
        dst[mask] = src[mask]
        return dst
    ys, xs = np.where(mask)
    ny, nx = ys + dy, xs + dx
    ok = (ny >= 0) & (ny < h) & (nx >= 0) & (nx < w)
    dst[ny[ok], nx[ok]] = src[ys[ok], xs[ok]]
    return dst


def shifted_mask(mask: np.ndarray, dx: int, dy: int) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros((h, w), dtype=bool)
    ys, xs = np.where(mask)
    ny, nx = ys + dy, xs + dx
    ok = (ny >= 0) & (ny < h) & (nx >= 0) & (nx < w)
    out[ny[ok], nx[ok]] = True
    return out


def spatial_box(h: int, w: int, x0f, x1f, y0f, y1f) -> np.ndarray:
    m = np.zeros((h, w), dtype=bool)
    m[int(y0f * h):int(y1f * h), int(x0f * w):int(x1f * w)] = True
    return m


def save_masks(case_dir: Path, masks: list[np.ndarray],
               frame_indices: list[int]) -> list[str]:
    """Persist per-frame footprint masks.

    Two copies are written: a 320x180 PNG (the manifest/visualisation
    artifact required by PICPLAN §13) and a full-resolution PNG under
    ``_contract_masks/`` used for exact pre-encode contract checks (the
    downscaled mask would quantise boundaries and spuriously fail the
    outside-mask equality assertion).  Returned paths point at the small
    masks.
    """
    out_dir = case_dir / "defect_masks"
    contract_dir = case_dir / "_contract_masks"
    out_dir.mkdir(parents=True, exist_ok=True)
    contract_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    kern = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # 2-px halo
    for m, fi in zip(masks, frame_indices):
        full = (m > 0).astype(np.uint8) * 255
        cv2.imwrite(str(contract_dir / f"{fi:06d}.png"), full)
        halo = cv2.dilate(full, kern, iterations=1)
        small = cv2.resize(halo, MASK_SIZE, interpolation=cv2.INTER_NEAREST)
        name = f"{fi:06d}.png"
        cv2.imwrite(str(out_dir / name), small)
        paths.append(f"defect_masks/{name}")
    return paths


def make_result(frames, affected, interval, masks, rois, params,
                case_dir) -> DefectResult:
    return DefectResult(
        frames=frames,
        affected_indices=list(affected),
        affected_interval_seconds=(round(interval[0], 4), round(interval[1], 4)),
        roi_boxes=rois,
        mask_paths=save_masks(case_dir, masks, list(affected)),
        parameters=params,
    )


class DefectOperatorBase(DefectOperator):
    """Convenience: subclasses implement ``run`` returning per-frame data."""

    def apply(self, frames, fps, start_time, end_time, context, **params):
        a, b = clamp_window(context, start_time, end_time)
        interval = (a / fps, b / fps)
        affected, masks, rois = self.run(frames, a, b, context, params)
        return make_result(frames, affected, interval, masks, rois,
                           {**params, **self.report_params(params)},
                           context.case_dir)

    def report_params(self, params) -> dict:
        return {}

    def run(self, frames, a, b, ctx, params):  # pragma: no cover
        raise NotImplementedError
