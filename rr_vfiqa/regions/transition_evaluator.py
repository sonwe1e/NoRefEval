"""Discrete-event / transition branch (USERPLAN §8.6).

Card flips, shop popups, menu switches: the true switch instant between two
60 FPS endpoints is *unknowable*, so the system must not judge timing. It
only penalizes mixing defects (double exposure, partial blends, regression,
extra flicker) and reports high event_ambiguity so fusion lowers confidence.
"""

from __future__ import annotations

import numpy as np

from ..config import EvalConfig
from ..metrics.window_flows import WindowFlows
from ..schema import FrameBundle, flow_magnitude, warp_image

_CHANGE_TAU = 24.0     # per-pixel endpoint change that counts as "switched"
_SIDE_TAU = 10.0       # closeness to one endpoint state
_MOTION_GATE = 1.5     # px; low-motion pixels are excluded from "changed"


def compute_window(bundle: FrameBundle, flow: WindowFlows, cfg: EvalConfig
                   ) -> dict[str, float]:
    xi = bundle.rgb[1].astype(np.float32)
    xm = bundle.rgb[2].astype(np.float32)
    xj = bundle.rgb[3].astype(np.float32)

    diff01 = np.abs(xi - xj).mean(-1)
    d0 = np.abs(xm - xi).mean(-1)
    d1 = np.abs(xm - xj).mean(-1)

    # "Changed" requires a large luma delta AND real local motion — otherwise
    # slow/aliasing pixels (bright static texture, codec shimmer) masquerade
    # as state switches.
    moved = flow_magnitude(flow.forward(1, 3)) > _MOTION_GATE
    changed = (diff01 > _CHANGE_TAU) & moved
    changed_frac = float(changed.mean())
    out = {"event_changed_frac": changed_frac}
    if changed_frac < 0.005:
        out.update({"event_discrete_score": 0.0, "event_ghost_frac": 0.0,
                    "event_regression": 0.0, "event_ambiguity": 0.0})
        return out

    c = changed
    # Discrete evidence: changed pixels where M_i already sits on one side.
    close_to_side = np.minimum(d0, d1) < _SIDE_TAU
    out["event_discrete_score"] = float(close_to_side[c].mean())

    # Double exposure: M_i close to BOTH endpoint states at once.
    double = (d0 < _SIDE_TAU) & (d1 < _SIDE_TAU)
    out["event_ghost_frac"] = float(double[c].mean())

    # State regression: M_{i+1} returns toward X_i by more than 20% of the
    # endpoint gap — going backwards after a switch. (Plain "closer to X_i"
    # would penalize normal forward motion, since M_{i+1} is later in time.)
    if bundle.rgb.shape[0] >= 5:
        m_next = bundle.rgb[4].astype(np.float32)
        dn0 = np.abs(m_next - xi).mean(-1)
        back = (dn0 < d0 - 0.2 * diff01) & (diff01 > 1e-3)
        out["event_regression"] = float(back[c].mean())

    # Continuous-motion contrast: a real flip keeps moving monotonically; use
    # motion-compensated continuity of M_{i-1}->M_i->M_{i+1} as counter-signal.
    f_ab, f_ba = flow.pair(0, 2)
    cyc = np.linalg.norm(f_ab + warp_image(f_ba, f_ab), axis=-1)
    out["event_continuity_vis"] = float((cyc < cfg.occlusion_cycle_threshold).mean())

    out["event_ambiguity"] = float(np.clip(
        out["event_discrete_score"] * (0.5 + 0.5 * min(1.0, changed_frac / 0.1)),
        0, 1))
    return out
