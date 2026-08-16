"""Flow-field geometry: divergence, curl, Jacobian determinant (USERPLAN §8.1).

Anomalies in det(I + ∇F) flag local folding / tearing of the motion field.
Always run on flow *minus global camera motion* to avoid rotation false
positives.
"""

from __future__ import annotations

import numpy as np


def flow_derivatives(flow: np.ndarray) -> dict[str, np.ndarray]:
    """du/dx, du/dy, dv/dx, dv/dy via central differences (px/px)."""
    u, v = flow[..., 0], flow[..., 1]
    dudy, dudx = np.gradient(u.astype(np.float64))
    dvdy, dvdx = np.gradient(v.astype(np.float64))
    return {"dudx": dudx.astype(np.float32), "dudy": dudy.astype(np.float32),
            "dvdx": dvdx.astype(np.float32), "dvdy": dvdy.astype(np.float32)}


def divergence(flow: np.ndarray) -> np.ndarray:
    d = flow_derivatives(flow)
    return d["dudx"] + d["dvdy"]


def curl(flow: np.ndarray) -> np.ndarray:
    d = flow_derivatives(flow)
    return d["dvdx"] - d["dudy"]


def jacobian_det(flow: np.ndarray) -> np.ndarray:
    """det(I + ∇F): 1 = rigid, ~0 = collapse, <0 = fold."""
    d = flow_derivatives(flow)
    return (1.0 + d["dudx"]) * (1.0 + d["dvdy"]) - d["dudy"] * d["dvdx"]


def geometry_stats(flow: np.ndarray, mask: np.ndarray | None = None
                   ) -> dict[str, float]:
    """Summary statistics over non-occluded pixels."""
    if mask is None:
        mask = np.ones(flow.shape[:2], bool)
    mask = mask.astype(bool)
    if mask.sum() < 16:
        return {"flow_fold_frac": float("nan"), "flow_div_std": float("nan"),
                "flow_curl_std": float("nan"), "flow_jdet_low_frac": float("nan")}
    # One gradient pass instead of one per statistic (jacobian/div/curl each
    # used to recompute flow_derivatives). Same expressions, bit-identical.
    d = flow_derivatives(flow)
    jdet = ((1.0 + d["dudx"]) * (1.0 + d["dvdy"]) - d["dudy"] * d["dvdx"])[mask]
    div = (d["dudx"] + d["dvdy"])[mask]
    cur = (d["dvdx"] - d["dudy"])[mask]
    return {
        "flow_fold_frac": float(np.mean(jdet < 0.0)),
        "flow_jdet_low_frac": float(np.mean(jdet < 0.5)),
        "flow_div_std": float(np.std(div)),
        "flow_curl_std": float(np.std(cur)),
    }


def residual_flow(flow: np.ndarray, camera_flow: np.ndarray) -> np.ndarray:
    """Dense flow minus global camera motion — foreground/parallax residual."""
    return flow - camera_flow
