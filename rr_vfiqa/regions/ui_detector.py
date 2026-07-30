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
from ..imutils import alpha_blend_fit, spatial_norm_factor
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


def _edge_fscore(candidate_gray: np.ndarray, reference_gray: np.ndarray,
                 mask: np.ndarray, tol_px: float = 2.0) -> float:
    """Edge F-score of candidate against reference inside the mask.

    recall    = fraction of REFERENCE edges within tol of a candidate edge;
    precision = fraction of CANDIDATE edges within tol of a reference edge.
    (The previous version measured each edge set against its OWN distance
    transform, which is trivially zero distance — F degraded to a constant 1.)
    """
    e_cand = cv2.Canny(candidate_gray, 60, 160) > 0
    e_ref = cv2.Canny(reference_gray, 60, 160) > 0
    dist_to_cand = cv2.distanceTransform((e_cand == 0).astype(np.uint8),
                                         cv2.DIST_L2, 3)
    dist_to_ref = cv2.distanceTransform((e_ref == 0).astype(np.uint8),
                                        cv2.DIST_L2, 3)
    m = mask.astype(bool)
    if m.sum() < 16 or (e_ref & m).sum() == 0 or (e_cand & m).sum() == 0:
        return float("nan")
    rec = float(np.mean(dist_to_cand[e_ref & m] <= tol_px))
    prec = float(np.mean(dist_to_ref[e_cand & m] <= tol_px))
    if prec + rec < 1e-6:
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

    # USERPLAN §6.2: decision thresholds are resolution-normalized (fraction
    # of frame diagonal) so the static/dynamic UI decision is consistent
    # across resolutions and presets.
    norm = spatial_norm_factor(h, w)
    endpoint_change = float(np.abs(xi - xj)[m].mean()) / norm
    out["ui_endpoint_change"] = endpoint_change
    static = endpoint_change < 6.0 / norm
    out["ui_mode"] = 0.0 if static else 1.0

    if static:
        # §8.4 E_UI-static: M_i should equal the (identical) endpoint content.
        out["ui_static_l1"] = float(np.abs(xm - xi)[m].mean()) / norm
        gm = cv2.Sobel(luma(xm.astype(np.uint8)), cv2.CV_32F, 1, 0)
        gi = cv2.Sobel(luma(xi.astype(np.uint8)), cv2.CV_32F, 1, 0)
        out["ui_static_grad"] = float(np.abs(gm - gi)[m].mean())
        out["ui_static_edge_f"] = _edge_fscore(
            cv2.cvtColor(xm.astype(np.uint8), cv2.COLOR_RGB2GRAY),
            cv2.cvtColor(xi.astype(np.uint8), cv2.COLOR_RGB2GRAY), mask)
        # Per-component drift: a single element (cooldown number, bar) can
        # shift while the mask-wide average L1 stays small (§4: component
        # level, not whole-HUD average).
        n_lab, labels, stats_cc, _ = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), 8)
        drifts = [float(np.abs(xm - xi)[labels == lab].mean()) / norm
                  for lab in range(1, n_lab)
                  if stats_cc[lab, cv2.CC_STAT_AREA] >= 40]
        if drifts:
            out["ui_comp_drift_p90"] = float(np.percentile(drifts, 90))
        # Cross-generated drift: M_{i-1} vs M_i inside UI (persistent drift).
        if bundle.rgb.shape[0] >= 5:
            out["ui_gen_drift"] = float(
                np.abs(bundle.rgb[0].astype(np.float32) - xm)[m].mean()) / norm
    else:
        # §8.4 dynamic UI / §8.6 discrete events: only penalize mixing defects.
        d0 = np.abs(xm - xi).mean(-1) / norm
        d1 = np.abs(xm - xj).mean(-1) / norm
        diff01m = np.abs(xi - xj).mean(-1) / norm
        ch = m & (diff01m > 12.0 / norm)
        n_ch = max(int(ch.sum()), 1)

        # Alpha-mixing evidence: M fits α·Xi + (1−α)·Xj with α strictly inside
        # (0,1) and tiny residual — impossible for a correct hard switch, and
        # not tripped by the triangle-inequality-violating old d0<τ & d1<τ.
        alpha, resid = alpha_blend_fit(xi, xm, xj)
        blend = ch & (resid < 6.0 / norm) & (alpha > 0.15) & (alpha < 0.85)
        out["ui_dyn_blend_frac"] = float(blend.sum() / n_ch)
        out["ui_dyn_blend_resid"] = float(resid[ch].mean()) if ch.any() else float("nan")

        out_of_range = ((xm < np.minimum(xi, xj) - 8.0 / norm) |
                        (xm > np.maximum(xi, xj) + 8.0 / norm)).any(-1)
        out["ui_dyn_out_of_range_frac"] = float((out_of_range & ch).sum() / n_ch)

        # State regression (§3.6 fixed sign): M_i returns TOWARD X_i relative
        # to the previous generated frame — normal forward progression has
        # M_i FURTHER from X_i than M_{i-1}, so only a shrink counts.
        if bundle.rgb.shape[0] >= 5:
            m_prev = bundle.rgb[0].astype(np.float32)
            d0_prev = np.abs(m_prev - xi).mean(-1) / norm
            back = ch & (d0 < d0_prev - 0.2 * diff01m)
            out["ui_dyn_regression"] = float(back.sum() / n_ch)
    return out
