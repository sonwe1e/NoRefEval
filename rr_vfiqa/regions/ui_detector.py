"""Screen-space static UI detection and evaluation (USERPLAN §8.4).

UI is the region where endpoint reference is strongest: it sits still in
screen coordinates while the world moves. Detection uses multi-frame
screen-space stability + persistent edges, then windows are scored as either

* static UI  → M_i must match the endpoints (L1 / gradient / edge F), or
* dynamic UI → only ghosting, out-of-range mixing and non-monotonicity are
  penalized; the exact switch timing is unknowable (§8.6 event ambiguity).
"""

from __future__ import annotations

import cv2
import numpy as np

from ..config import EvalConfig
from ..io.video_reader import VideoReader
from ..schema import FrameBundle
from ._common import clean_mask, luma

_UI_WIDTH = 480
_N_SAMPLE = 14
_STATIC_DEV_FRAC = 0.35       # below this fraction of the median deviation


class UIDetector:
    """Detects screen-static UI regions once per source video."""

    def __init__(self, source: VideoReader, cfg: EvalConfig):
        self.mask_small: np.ndarray | None = None     # at _UI_WIDTH
        self.small_size: tuple[int, int] = (0, 0)
        self._build(source, cfg)

    def _build(self, source: VideoReader, cfg: EvalConfig) -> None:
        n = source.meta.n_frames
        pick = np.linspace(0, n - 1, _N_SAMPLE).astype(int)
        frames = [source.read_one(int(i), width=_UI_WIDTH) for i in pick]
        h, w = frames[0].shape[:2]
        self.small_size = (h, w)
        stack = np.stack([luma(f) for f in frames], 0)          # (T, H, W)
        dev = np.median(np.abs(stack - np.median(stack, 0, keepdims=True)), 0)
        thr = max(1.5, _STATIC_DEV_FRAC * np.median(dev[dev > 0.1]) if (dev > 0.1).any() else 1.5)
        static = dev < thr

        # Persistent edges: present in most frames.
        edge_votes = np.zeros((h, w), np.float32)
        for f in frames:
            g = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
            edge_votes += (cv2.Canny(g, 60, 160) > 0).astype(np.float32)
        persistent = edge_votes > 0.5 * len(frames)
        # UI = static region whose neighbourhood carries persistent structure.
        nb = cv2.dilate(persistent.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        mask = static & nb
        # Screen-position prior is implicit; drop huge blobs (usually walls/sky).
        self.mask_small = clean_mask(mask, min_area=80, close_k=7)
        frac = self.mask_small.mean()
        if frac > 0.35 or frac < 0.002:
            # unreliable either way — keep but flag through coverage features
            if frac > 0.35:
                self.mask_small = np.zeros_like(self.mask_small)

    def mask_at(self, h: int, w: int) -> np.ndarray:
        if self.mask_small is None or self.mask_small.sum() == 0:
            return np.zeros((h, w), np.uint8)
        return (cv2.resize(self.mask_small.astype(np.float32), (w, h),
                           interpolation=cv2.INTER_NEAREST) > 0.5).astype(np.uint8)


def _edge_fscore(a_gray: np.ndarray, b_gray: np.ndarray, mask: np.ndarray,
                 tol_px: float = 2.0) -> float:
    ea = cv2.Canny(a_gray, 60, 160) > 0
    eb = cv2.Canny(b_gray, 60, 160) > 0
    da = cv2.distanceTransform((ea == 0).astype(np.uint8), cv2.DIST_L2, 3)
    db = cv2.distanceTransform((eb == 0).astype(np.uint8), cv2.DIST_L2, 3)
    m = mask.astype(bool)
    if m.sum() < 16 or eb[m].sum() == 0:
        return float("nan")
    prec = float(np.mean(db[eb & m] <= tol_px)) if (eb & m).any() else float("nan")
    rec = float(np.mean(da[ea & m] <= tol_px)) if (ea & m).any() else float("nan")
    if prec != prec or rec != rec or prec + rec < 1e-6:
        return float("nan")
    return 2 * prec * rec / (prec + rec)


def compute_window(bundle: FrameBundle, ui: UIDetector, cfg: EvalConfig
                   ) -> dict[str, float]:
    """Positions: 1 = X_i, 2 = M_i, 3 = X_{i+1}; 0/4 used for cross-gen checks."""
    h, w = bundle.height, bundle.width
    mask = ui.mask_at(h, w)
    out: dict[str, float] = {"ui_coverage": float(mask.mean())}
    if mask.mean() < 0.004:
        out.update({"ui_mode": float("nan"), "ui_static_l1": float("nan")})
        return out

    xi = bundle.rgb[1].astype(np.float32)
    xm = bundle.rgb[2].astype(np.float32)
    xj = bundle.rgb[3].astype(np.float32)
    m = mask.astype(bool)

    endpoint_change = float(np.abs(xi - xj)[m].mean())
    out["ui_endpoint_change"] = endpoint_change
    static = endpoint_change < 6.0
    out["ui_mode"] = 0.0 if static else 1.0

    if static:
        # §8.4 E_UI-static: M_i should equal the (identical) endpoint content.
        out["ui_static_l1"] = float(np.abs(xm - xi)[m].mean())
        gm = cv2.Sobel(luma(xm.astype(np.uint8)), cv2.CV_32F, 1, 0)
        gi = cv2.Sobel(luma(xi.astype(np.uint8)), cv2.CV_32F, 1, 0)
        out["ui_static_grad"] = float(np.abs(gm - gi)[m].mean())
        out["ui_static_edge_f"] = _edge_fscore(
            cv2.cvtColor(xm.astype(np.uint8), cv2.COLOR_RGB2GRAY),
            cv2.cvtColor(xi.astype(np.uint8), cv2.COLOR_RGB2GRAY), mask)
        # Cross-generated drift: M_{i-1} vs M_i inside UI (persistent drift).
        if bundle.rgb.shape[0] >= 5:
            out["ui_gen_drift"] = float(
                np.abs(bundle.rgb[0].astype(np.float32) - xm)[m].mean())
    else:
        # §8.4 dynamic UI / §8.6 discrete events: only penalize mixing defects.
        d0 = np.abs(xm - xi)
        d1 = np.abs(xm - xj)
        diff01 = np.abs(xi - xj)
        tau = 8.0
        out_of_range = ((xm < np.minimum(xi, xj) - tau) |
                        (xm > np.maximum(xi, xj) + tau)).any(-1)
        out["ui_dyn_out_of_range_frac"] = float(np.mean(out_of_range & m))
        double_exp = ((d0.mean(-1) < tau) & (d1.mean(-1) < tau) &
                      (diff01.mean(-1) > 3 * tau))
        out["ui_dyn_double_exposure"] = float(np.mean(double_exp & m))
        # State regression between consecutive generated frames (goes backward).
        if bundle.rgb.shape[0] >= 5:
            m_prev = bundle.rgb[0].astype(np.float32)
            sim_prev_to_xi = float(np.abs(m_prev - xi)[m].mean())
            sim_cur_to_xi = float(np.abs(xm - xi)[m].mean())
            out["ui_dyn_regression"] = float(
                np.clip((sim_cur_to_xi - sim_prev_to_xi) / max(endpoint_change, 1e-3),
                        0, 2))
    return out
