"""Endpoint flow-composition consistency — the central metric (USERPLAN §3).

    F_0m(x) + F_m1(x + F_0m(x))  ≈  F_01(x)

Normalized, robustly penalized (Charbonnier), weighted by visibility and
flow confidence, and computed in both directions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..schema import charbonnier, flow_magnitude, percentiles, warp_flow


@dataclass
class CompositionResult:
    error_map: np.ndarray        # (H, W) float32, normalized composition error
    weight_map: np.ndarray       # (H, W) float32, effective weights used
    weighted_mean: float
    p90: float
    p99: float
    covered_fraction: float      # fraction of pixels with weight > 0

    def stats(self, prefix: str = "comp") -> dict[str, float]:
        return {
            f"{prefix}_mean": self.weighted_mean,
            f"{prefix}_p90": self.p90,
            f"{prefix}_p99": self.p99,
            f"{prefix}_coverage": self.covered_fraction,
        }


def composition_error(f_01: np.ndarray, f_0m: np.ndarray, f_m1: np.ndarray,
                      weight: np.ndarray | None = None, tau_px: float = 4.0,
                      norm_eps_px: float = 2.0) -> CompositionResult:
    """Forward-direction composition error.

    f_01 : source endpoint flow X_i -> X_{i+1}
    f_0m : X_i -> M_i
    f_m1 : M_i -> X_{i+1}
    weight: (H, W) visibility × confidence in [0, 1]; None => all ones.
    """
    composed = warp_flow(f_0m, f_m1)                 # X_i -> X_{i+1} via M_i
    diff = composed - f_01
    mag = flow_magnitude(diff)
    norm = flow_magnitude(f_01) + norm_eps_px
    err = charbonnier(mag, tau_px) / norm            # dimensionless ~fractional error

    if weight is None:
        weight = np.ones(err.shape, np.float32)
    weight = weight.astype(np.float32)
    covered = float(np.mean(weight > 1e-3))
    wsum = weight.sum() + 1e-6
    wmean = float(np.sum(err * weight) / wsum)
    # Percentiles over the (unweighted) supported population.
    supported = err[weight > 1e-3]
    pct = percentiles(supported, (90, 99))
    return CompositionResult(
        error_map=err.astype(np.float32),
        weight_map=weight,
        weighted_mean=wmean,
        p90=pct["p90"],
        p99=pct["p99"],
        covered_fraction=covered,
    )


def bidirectional_composition(f_01: np.ndarray, f_10: np.ndarray,
                              f_0m: np.ndarray, f_m0: np.ndarray,
                              f_m1: np.ndarray, f_1m: np.ndarray,
                              conf_01: np.ndarray | None = None,
                              occ_01: np.ndarray | None = None,
                              tau_px: float = 4.0
                              ) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    """Both composition directions; returns merged stats.

    Forward uses the confidence/occlusion of the 0->1 cycle; backward the same
    masks (symmetric role). Final score = mean of both plus the worst-side P90.
    """
    w = None
    if conf_01 is not None:
        w = conf_01.astype(np.float32)
        if occ_01 is not None:
            w = w * (1.0 - occ_01.astype(np.float32))

    fwd = composition_error(f_01, f_0m, f_m1, weight=w, tau_px=tau_px)
    bwd = composition_error(f_10, f_1m, f_m0, weight=w, tau_px=tau_px)

    out = {}
    out.update(fwd.stats("comp_fwd"))
    out.update(bwd.stats("comp_bwd"))
    out["comp_mean"] = 0.5 * (fwd.weighted_mean + bwd.weighted_mean)
    out["comp_p90_max"] = max(fwd.p90, bwd.p90)
    out["comp_coverage"] = 0.5 * (fwd.covered_fraction + bwd.covered_fraction)
    return out, fwd.error_map, bwd.error_map
