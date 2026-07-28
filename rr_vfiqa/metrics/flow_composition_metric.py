"""Per-window endpoint flow-composition metric (USERPLAN §3 + §8.1 tearing).

    F_0m(x) + F_m1(x + F_0m(x)) ≈ F_01(x)     (and the reverse direction)

plus residual-flow geometry (folding/tearing) after removing camera motion,
and a low-confidence disocclusion-branch residual.
"""

from __future__ import annotations

import numpy as np

from ..cache.source_cache import SourcePairData
from ..config import EvalConfig
from ..motion.flow_composition import bidirectional_composition
from ..motion.flow_geometry import geometry_stats, residual_flow
from ..schema import flow_magnitude, resize_flow
from .window_flows import WindowFlows


def compute(flow: WindowFlows, pair: SourcePairData, cfg: EvalConfig
            ) -> dict[str, float]:
    """Window positions: 1 = X_i, 2 = M_i, 3 = X_{i+1}."""
    h, w = flow.height(), flow.width()

    # Cached source flows → window resolution.
    f_01 = resize_flow(pair.f_01, h, w)
    f_10 = resize_flow(pair.f_10, h, w)
    conf_01 = _resize_mask(pair.occ.conf_ab, h, w)
    occ_01 = _resize_mask(pair.occ.occ_ab.astype(np.float32), h, w) > 0.5
    conf_10 = _resize_mask(pair.occ.conf_ba, h, w)
    occ_10 = _resize_mask(pair.occ.occ_ba.astype(np.float32), h, w) > 0.5

    # Candidate half-leg flows, both directions.
    f_0m, f_m0 = flow.pair(1, 2)
    f_m1, f_1m = flow.pair(2, 3)

    weight = conf_01 * (1.0 - occ_01.astype(np.float32))
    stats, fwd_map, bwd_map = bidirectional_composition(
        f_01, f_10, f_0m, f_m0, f_m1, f_1m,
        conf_01=conf_01, occ_01=occ_01.astype(np.uint8),
        conf_10=conf_10, occ_10=occ_10.astype(np.uint8),
        tau_px=cfg.charbonnier_tau,
    )

    # Disocclusion branch: residual where the cycle says "unreliable". The
    # composition error there is reported but never allowed to dominate.
    vis = weight > 1e-3
    stats["comp_occ_region_mean"] = float(np.mean(0.5 * (fwd_map + bwd_map)[~vis])) \
        if (~vis).any() else 0.0
    stats["comp_visible_fraction"] = float(np.mean(vis))

    # Tearing / folding: geometry of the candidate residual flow (first leg
    # minus half the global camera motion of the source pair).
    cam_flow = pair.camera.warp_flow(h, w) * pair.scale  # to flow-res pixels
    res = residual_flow(f_0m, 0.5 * cam_flow)
    geom = geometry_stats(res, mask=vis)
    stats.update(geom)

    # Leg-length sanity: |F_0m| should be ~half |F_01| where visible.
    mag01 = flow_magnitude(f_01)[vis]
    mag0m = flow_magnitude(f_0m)[vis]
    if mag01.size > 100 and np.median(mag01) > 2.0:
        ratio = np.median(mag0m) / max(np.median(mag01), 1e-6)
        stats["comp_leg_ratio"] = float(abs(ratio - 0.5))
    else:
        stats["comp_leg_ratio"] = 0.0
    return stats


def _resize_mask(mask: np.ndarray, h: int, w: int) -> np.ndarray:
    import cv2
    if mask.shape[:2] == (h, w):
        return mask.astype(np.float32)
    return cv2.resize(mask.astype(np.float32), (w, h),
                      interpolation=cv2.INTER_LINEAR)
