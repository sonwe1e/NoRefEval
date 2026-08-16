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
