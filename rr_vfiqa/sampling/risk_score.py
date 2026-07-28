"""Per-frame risk from cheap-scan features (USERPLAN §10, R_w formula).

Robust per-feature z-scores adapt to the video's own statistics instead of
fixed thresholds, then a weighted sum forms the risk timeline.
"""

from __future__ import annotations

import numpy as np

from ..schema import robust_z
from .cheap_scan import CheapScan

RISK_WEIGHTS = {
    "parity_gap": 2.0,       # odd/even sharpness alternation — classic VFI tell
    "edge_flicker": 1.5,     # contours blinking frame to frame
    "sharpness_dip": 1.5,    # isolated soft frames
    "grad_drop": 1.0,        # sudden texture loss (smearing / background loss)
    "luma_flash": 0.5,       # brightness flicker
    "motion_jump": 1.0,      # frame-diff discontinuity (freeze then jump)
}


def compute_risk(scan: CheapScan) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Returns (risk_per_frame, z_components) aligned with scan.indices."""
    n = len(scan.indices)

    parity_gap = scan.parity_sharpness_gap()
    edge_flicker = scan.edge_flicker()

    sharp = scan.sharpness.astype(np.float64)
    # sharpness dip: negative deviation from a 5-frame median trend
    k = np.ones(5) / 5
    trend = np.convolve(sharp, k, mode="same")
    sharp_dip = np.clip(trend - sharp, 0, None)

    grad = scan.grad_energy.astype(np.float64)
    grad_trend = np.convolve(grad, k, mode="same")
    grad_drop = np.clip(grad_trend - grad, 0, None)

    luma = scan.luma_mean.astype(np.float64)
    luma_flash = np.zeros(n)
    luma_flash[1:-1] = np.abs(luma[2:] - 2 * luma[1:-1] + luma[:-2])

    fd = scan.frame_diff.astype(np.float64)
    motion_jump = np.zeros(n)
    motion_jump[1:] = np.abs(np.diff(fd))

    comps = {
        "parity_gap": np.clip(robust_z(parity_gap), 0, None),
        "edge_flicker": np.clip(robust_z(edge_flicker), 0, None),
        "sharpness_dip": np.clip(robust_z(sharp_dip), 0, None),
        "grad_drop": np.clip(robust_z(grad_drop), 0, None),
        "luma_flash": np.clip(robust_z(luma_flash), 0, None),
        "motion_jump": np.clip(robust_z(motion_jump), 0, None),
    }
    risk = np.zeros(n, np.float64)
    for name, z in comps.items():
        risk += RISK_WEIGHTS[name] * z
    # Cut-adjacent frames are handled separately; do not let them eat the
    # risk budget.
    cut_prob = scan.scene_cut_prob()
    risk[cut_prob > 8.0] *= 0.1
    return risk, comps
