"""Motion-compensated temporal stability (USERPLAN.md §5).

Warp every frame of the 5-frame window onto the center frame and study the
residual — raw frame differences are meaningless because normal motion also
produces large differences.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..config import EvalConfig
from ..schema import FrameBundle, backward_warp, charbonnier, percentiles, warp_flow
from .window_flows import WindowFlows

_LAG_POSITIONS = {1: (1, 3), 2: (0, 4)}       # lag in candidate frames


def _residual_stats(resid: np.ndarray, vis: np.ndarray, tau: float
                    ) -> dict[str, float]:
    r = charbonnier(resid, tau)
    v = vis.astype(bool)
    if v.sum() < 64:
        return {"mean": float("nan"), "p50": float("nan"), "p90": float("nan"),
                "p99": float("nan"), "large_area": float("nan")}
    rv = r[v]
    pct = percentiles(rv, (50, 90, 99))
    return {
        "mean": float(rv.mean()),
        "p50": pct["p50"],
        "p90": pct["p90"],
        "p99": pct["p99"],
        "large_area": float(np.mean((r > max(3.0 * pct["p50"], 12.0)) & v)),
    }


def compute(bundle: FrameBundle, flow: WindowFlows, cfg: EvalConfig
            ) -> dict[str, float]:
    center = 2
    yc = bundle.y_channel()
    out: dict[str, float] = {}

    for pos in range(5):
        if pos == center:
            continue
        f_cj = flow.forward(center, pos)     # center-grid flow center -> j
        warped = backward_warp(yc[pos], f_cj)    # resample j onto center grid
        cyc = np.linalg.norm(warp_flow(f_cj, flow.forward(pos, center)), axis=-1)
        vis = cyc < cfg.occlusion_cycle_threshold

        y_res = np.abs(warped - yc[center])
        s = _residual_stats(y_res, vis, cfg.charbonnier_tau)
        lag = abs(pos - center)
        tag = f"lag{lag}{'a' if pos < center else 'b'}"
        out.update({f"mct_{tag}_{k}": v for k, v in s.items()})

        # Gradient residual (structure beyond luma).
        gx = cv2.Sobel(warped, cv2.CV_32F, 1, 0) + cv2.Sobel(warped, cv2.CV_32F, 0, 1)
        gc = cv2.Sobel(yc[center], cv2.CV_32F, 1, 0) + cv2.Sobel(yc[center], cv2.CV_32F, 0, 1)
        out[f"mct_{tag}_grad_mean"] = float(np.abs(np.abs(gx) - np.abs(gc))[vis].mean()) \
            if vis.sum() > 64 else float("nan")

    # Aggregate per lag: mean of both sides.
    for lag, (pa, pb) in _LAG_POSITIONS.items():
        for k in ("mean", "p90", "p99"):
            a = out.get(f"mct_lag{lag}a_{k}", float("nan"))
            b = out.get(f"mct_lag{lag}b_{k}", float("nan"))
            vals = [v for v in (a, b) if v == v]
            out[f"mct_lag{lag}_{k}"] = float(np.mean(vals)) if vals else float("nan")

    # Anchor neighbours (lag 1) warp onto the generated center; generated
    # neighbours (lag 2) do the same. If the center M_i is off-trajectory the
    # anchor residuals blow up relative to the generated ones (lag-2 diffs
    # connect same-parity frames — USERPLAN §6.3).
    l1 = out.get("mct_lag1_mean", float("nan"))
    l2 = out.get("mct_lag2_mean", float("nan"))
    if l1 == l1 and l2 == l2 and l2 > 1e-3:
        out["mct_anchor_vs_gen_ratio"] = float(l1 / l2)
    else:
        out["mct_anchor_vs_gen_ratio"] = float("nan")
    return out
