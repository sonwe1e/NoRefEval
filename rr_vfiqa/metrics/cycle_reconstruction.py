"""Reverse anchor cycle (USERPLAN.md §4).

Adjacent generated frames M_{i-1} (t = i-0.5) and M_i (t = i+0.5) bracket the
known anchor X_i (t = i). An *independent* reconstructor R rebuilds X_i from
the two generated frames; the residual is a grounded signal of inter-frame
inconsistency, systematic odd-frame blur and flicker.

R here is a plain half-warp blend of the two generated frames using their
mutual flow — deliberately not a learned VFI model, so it cannot reproduce a
candidate model's own biases.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..config import EvalConfig
from ..schema import FrameBundle, charbonnier, forward_splat, percentiles
from .window_flows import WindowFlows

# The reconstructor itself is imperfect; subtract a small baseline measured on
# real HFR data during calibration (USERPLAN §4 note 4). Y units at flow res.
DEFAULT_CYCLE_BASELINE = 2.0


def _half_warp_blend(img_a: np.ndarray, img_b: np.ndarray,
                     f_ab: np.ndarray, f_ba: np.ndarray
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct the mid instant from both sides via FORWARD splatting.

    f_ab / f_ba are forward flows; half of each moves the endpoint content
    toward the mid grid. Splat coverage is the visibility weight — holes
    (occluded/disoccluded mid pixels) get zero weight instead of fabricated
    content. Returns (blend, weight).
    """
    wa, ca = forward_splat(img_a.astype(np.float32), 0.5 * f_ab)
    wb, cb = forward_splat(img_b.astype(np.float32), 0.5 * f_ba)
    va = np.clip(ca, 0, 1)
    vb = np.clip(cb, 0, 1)
    wsum = va + vb + 1e-6
    blend = (wa * va[..., None] + wb * vb[..., None]) / wsum[..., None]
    weight = np.clip(wsum / 2.0, 0, 1)
    return blend.astype(np.float32), weight.astype(np.float32)


def _luma(rgb: np.ndarray) -> np.ndarray:
    return (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2])


def compute(bundle: FrameBundle, flow: WindowFlows, cfg: EvalConfig,
            baseline: float = DEFAULT_CYCLE_BASELINE
            ) -> dict[str, float]:
    """Uses positions 0 (M_{i-1}), 1 (X_i — the known anchor), 2 (M_i)."""
    m_prev = bundle.rgb[0]
    anchor = bundle.rgb[1].astype(np.float32)
    m_cur = bundle.rgb[2]

    f_ab, f_ba = flow.pair(0, 2)              # M_{i-1} <-> M_i
    recon, weight = _half_warp_blend(m_prev, m_cur, f_ab, f_ba)

    resid = charbonnier(np.abs(_luma(recon) - _luma(anchor)), tau=cfg.charbonnier_tau)
    vis = weight > 0.25
    if vis.sum() < 64:
        return {"cycle_resid_mean": float("nan"), "cycle_resid_p90": float("nan"),
                "cycle_coverage": 0.0, "cycle_sharp_loss": float("nan")}

    r = resid[vis]
    base_sub = np.clip(r - baseline, 0, None)
    pct = percentiles(base_sub, (50, 90))
    out = {
        "cycle_resid_mean": float(base_sub.mean()),
        "cycle_resid_p50": pct["p50"],
        "cycle_resid_p90": pct["p90"],
        "cycle_coverage": float(vis.mean()),
    }

    # Systematic odd-frame blur: the reconstruction inherits generated-frame
    # blur; compare Laplacian energy of recon vs anchor in visible regions.
    lap_a = np.abs(cv2.Laplacian(_luma(anchor).astype(np.float32), cv2.CV_32F))
    lap_r = np.abs(cv2.Laplacian(_luma(recon).astype(np.float32), cv2.CV_32F))
    sharp_a = lap_a[vis].mean()
    sharp_r = lap_r[vis].mean()
    out["cycle_sharp_loss"] = float(np.clip(1.0 - sharp_r / max(sharp_a, 1e-6), 0, 1))
    return out
