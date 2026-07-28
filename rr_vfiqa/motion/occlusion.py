"""Occlusion / visibility from forward-backward flow cycle consistency.

USERPLAN.md §3: core composition metrics must only run on reliably visible
pixels; occluded / disoccluded regions go to a separate quality branch.
"""

from __future__ import annotations

import numpy as np

from ..schema import OcclusionMasks, warp_flow, flow_magnitude


def cycle_occlusion(f_ab: np.ndarray, f_ba: np.ndarray,
                    threshold_px: float = 3.0, conf_sigma: float = 2.0
                    ) -> OcclusionMasks:
    """Classic fwd-bwd consistency: |F_ab(x) + F_ba(x + F_ab(x))| > thr ⇒ occluded.

    conf maps the cycle error through an exponential so that high-confidence
    regions weight composition metrics more.
    """
    cycle_ab = warp_flow(f_ab, f_ba)          # should be ~0 where visible
    cycle_ba = warp_flow(f_ba, f_ab)
    err_ab = flow_magnitude(cycle_ab)
    err_ba = flow_magnitude(cycle_ba)
    occ_ab = (err_ab > threshold_px).astype(np.uint8)
    occ_ba = (err_ba > threshold_px).astype(np.uint8)
    conf_ab = np.exp(-(err_ab / conf_sigma) ** 2).astype(np.float32)
    conf_ba = np.exp(-(err_ba / conf_sigma) ** 2).astype(np.float32)
    return OcclusionMasks(occ_ab=occ_ab, occ_ba=occ_ba, conf_ab=conf_ab, conf_ba=conf_ba)


def exposure_map(f_01: np.ndarray, f_10: np.ndarray, h: int, w: int) -> dict[str, np.ndarray]:
    """Mid-time visibility coverage from both endpoints (USERPLAN §8.1).

    Forward-warps both endpoint grids to t=0.5 and bins coverage:
      both / left_only / right_only / unknown (no endpoint support).
    """
    def splat(flow_half: np.ndarray) -> np.ndarray:
        h0, w0 = flow_half.shape[:2]
        yy, xx = np.mgrid[0:h0, 0:w0].astype(np.float32)
        sx = (w / w0)
        sy = (h / h0)
        px = ((xx + flow_half[..., 0]) * sx).ravel()
        py = ((yy + flow_half[..., 1]) * sy).ravel()
        acc, = np.histogram2d(py, px, bins=(h, w), range=((0, h), (0, w)))
        return acc

    cov0 = splat(0.5 * f_01)
    cov1 = splat(0.5 * f_10)
    both = (cov0 > 0.5) & (cov1 > 0.5)
    left = (cov0 > 0.5) & (cov1 <= 0.5)
    right = (cov1 > 0.5) & (cov0 <= 0.5)
    unknown = (cov0 <= 0.5) & (cov1 <= 0.5)
    return {"both": both, "left_only": left, "right_only": right, "unknown": unknown}
