"""Odd/even alternation and temporal-frequency features (USERPLAN §6).

In 120 FPS output even frames are original and odd frames are generated, so a
frame-alternating quality difference is a near-60 Hz temporal component and a
direct fingerprint of systematic interpolation artifacts.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..config import EvalConfig
from ..sampling.cheap_scan import CheapScan
from ..schema import FrameBundle, backward_warp, flow_magnitude, warp_flow
from .window_flows import WindowFlows


def _alternation_energy(q: np.ndarray, k: np.ndarray, block: int = 64) -> dict[str, float]:
    """E_pi globally and the strongest local block (USERPLAN §6.1)."""
    q = q.astype(np.float64)
    med = np.median(q)
    q = q - med
    sign = (-1.0) ** (k.astype(np.int64) % 2)
    denom = np.abs(q).sum() + 1e-6
    e_pi = float(abs((sign * q).sum()) / denom)
    # Block-local version to catch transient alternation.
    n = len(q)
    blocks = max(1, n // block)
    e_local = []
    for b in range(blocks):
        s, e = b * block, min((b + 1) * block, n)
        if e - s < 8:
            continue
        d = np.abs(q[s:e]).sum() + 1e-6
        e_local.append(abs((sign[s:e] * q[s:e]).sum()) / d)
    return {"e_pi": e_pi, "e_pi_block_max": float(max(e_local)) if e_local else e_pi}


def _parity_mean_gap(q: np.ndarray, k: np.ndarray) -> float:
    """E_parity: |mean(q_even) - mean(q_odd)| / mean|q| (USERPLAN §6.1)."""
    q = q.astype(np.float64) - np.median(q)
    even = q[k % 2 == 0]
    odd = q[k % 2 == 1]
    if len(even) < 4 or len(odd) < 4:
        return 0.0
    return float(abs(even.mean() - odd.mean()) / (np.abs(q).mean() + 1e-6))


def compute_global(scan: CheapScan) -> dict[str, float]:
    out: dict[str, float] = {}
    k = scan.indices
    for name, q in (("sharp", scan.sharpness), ("grad", scan.grad_energy),
                    ("edge", scan.edge_frac)):
        ae = _alternation_energy(q, k)
        out[f"parity_{name}_e_pi"] = ae["e_pi"]
        out[f"parity_{name}_e_pi_block_max"] = ae["e_pi_block_max"]
        out[f"parity_{name}_mean_gap"] = _parity_mean_gap(q, k)
    return out


def _sharpness_series(bundle: FrameBundle) -> np.ndarray:
    y = bundle.y_channel()
    return np.asarray([float(cv2.Laplacian(y[t].astype(np.float32), cv2.CV_32F).var())
                       for t in range(y.shape[0])])


def compute_window(bundle: FrameBundle, flow: WindowFlows, cfg: EvalConfig
                   ) -> dict[str, float]:
    out: dict[str, float] = {}

    # Sharpness alternation inside the 5-frame window.
    sh = _sharpness_series(bundle)
    med = np.median(sh) + 1e-6
    k = bundle.indices.astype(np.int64)
    out["parity_window_sharp_gap"] = float(
        abs(sh[k % 2 == 0].mean() - sh[k % 2 == 1].mean()) / med) \
        if (k % 2 == 0).any() and (k % 2 == 1).any() else 0.0

    # Lag-2 same-parity motion-compensated differences (§6.3): the source
    # subsequence (anchors pos 1↔3) should be at least as stable as the
    # generated subsequence (pos 0↔2 and 2↔4).
    def mc_diff(a: int, b: int) -> float:
        ya = bundle.y_channel()[a]
        yb = bundle.y_channel()[b]
        f_ab, f_ba = flow.pair(a, b)
        # Resample b onto a's grid: target a, source b, so the target→source
        # flow is f_ab (a→b) itself.
        warped = backward_warp(yb, f_ab)
        cyc = np.linalg.norm(warp_flow(f_ab, f_ba), axis=-1)
        vis = cyc < cfg.occlusion_cycle_threshold
        if vis.sum() < 64:
            return float("nan")
        return float(np.abs(warped - ya)[vis].mean())

    anchor_lag2 = mc_diff(1, 3)
    gen_parts = [v for v in (mc_diff(0, 2), mc_diff(2, 4)) if v == v]
    gen_lag2 = float(np.mean(gen_parts)) if gen_parts else float("nan")
    out["parity_lag2_anchor"] = anchor_lag2
    out["parity_lag2_gen"] = float(gen_lag2)
    if anchor_lag2 == anchor_lag2 and gen_lag2 == gen_lag2 and anchor_lag2 > 1e-3:
        out["parity_lag2_gen_vs_anchor"] = float(gen_lag2 / anchor_lag2)
    else:
        out["parity_lag2_gen_vs_anchor"] = float("nan")

    # Freeze / copy detection: a frozen M_i equals X_i even in regions that
    # MUST change (|F_{01}| > 2 px), where a true mid differs from X_i by
    # about half the endpoint difference. Window-local statistics alone look
    # plausible for a freeze — this is the direct check.
    f_01 = flow.forward(1, 3)
    moving = flow_magnitude(f_01) > 2.0
    if moving.sum() > 200:
        xi = bundle.rgb[1].astype(np.float32)
        xj = bundle.rgb[3].astype(np.float32)
        xm = bundle.rgb[2].astype(np.float32)
        full = np.abs(xj - xi)[moving].mean()
        prev = np.median(np.abs(xm - xi)[moving])   # robust to edge outliers
        # full > 8 keeps codec noise (~2–3 Y) well below the half-difference
        # of genuinely moving pixels, so a clean mid scores near 0.
        out["freeze_copy_score"] = float(
            np.clip(1.0 - prev / (0.5 * full + 1e-6), 0.0, 1.0)) if full > 8.0 \
            else float("nan")
    return out
