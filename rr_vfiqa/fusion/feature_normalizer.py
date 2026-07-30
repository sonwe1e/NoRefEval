"""Raw feature → 0..1+ error normalization (USERPLAN §11).

Forward features (error-like, 0 = perfect) map through 1 - exp(-x / scale).
Inverted features (quality-like, higher = better — recall, F-scores) use a
3-tuple (good_ref, drop_width, True): error = (good_ref - v) / drop_width,
so a healthy value just below the reference stays a small error. Scales are
bootstrap references — the trained calibrator replaces them (§11.4).
"""

from __future__ import annotations

import math

import numpy as np

# USERPLAN §6.2: pixel-distance features are now reported as a fraction of
# the frame diagonal (see ``imutils.spatial_norm_factor``).  Scales for those
# features are expressed in the same normalized unit, i.e. the old pixel
# scale divided by the reference diagonal of the resolution the bootstrap
# scales were tuned on (fast-preset working resolution: 480 x 288).
REFERENCE_DIAGONAL = 559.7713819051488

FEATURE_SCALES: dict[str, tuple] = {
    # motion composition
    "comp_mean": (0.18, False),
    "comp_fwd_mean": (0.18, False),
    "comp_bwd_mean": (0.18, False),
    "comp_p90_max": (0.45, False),
    "comp_fwd_p90": (0.45, False),
    "comp_bwd_p90": (0.45, False),
    "comp_leg_ratio": (0.25, False),
    "comp_occ_region_mean": (0.5, False),
    "flow_fold_frac": (0.10, False),
    "flow_jdet_low_frac": (0.20, False),
    "flow_div_std": (1.5, False),
    "flow_curl_std": (1.5, False),
    # cycle (reconstructor baseline is imperfect — generous scales)
    "cycle_resid_mean": (35.0, False),
    "cycle_resid_p90": (100.0, False),
    "cycle_sharp_loss": (0.30, False),
    # temporal (flow + codec noise floor dominates on clean content)
    "mct_lag1_mean": (30.0, False),
    "mct_lag1_p90": (45.0, False),
    "mct_lag2_mean": (30.0, False),
    "mct_anchor_vs_gen_ratio": (1.5, False),   # >1 means center off-trajectory
    # parity — the strongest tell for systematic odd-frame degradation
    "freeze_copy_score": (0.50, False),
    "parity_window_sharp_gap": (0.25, False),
    "parity_lag2_gen_vs_anchor": (1.5, False),
    "parity_sharp_e_pi": (0.30, False),
    "parity_sharp_e_pi_block_max": (0.50, False),
    "parity_sharp_mean_gap": (0.45, False),
    "parity_grad_mean_gap": (0.30, False),
    "parity_edge_mean_gap": (0.50, False),
    # structure
    "edge_recall": (0.75, 0.45, True),
    "edge_precision": (0.90, 0.30, True),
    "edge_inst_recall_med": (0.30, 0.35, True),
    "edge_inst_recall_p10": (0.15, 0.35, True),
    "edge_chamfer_sup_to_em": (6.0 / REFERENCE_DIAGONAL, False),  # normalized
    "edge_ghost_frac": (0.20, False),
    "edge_count_odd_even_ratio": (0.0, False),  # special-cased below
    # character
    "char_missing_frac": (0.45, False),
    "char_extra_frac": (0.30, False),
    "char_chamfer": (150.0 / REFERENCE_DIAGONAL, False),  # normalized
    "char_components_delta": (3.0, False),
    "char_leak_mean": (10.0, 10.0, True),       # inverted: small leak = copy-like
    "char_ring_edge_frac": (0.35, False),
    # thin objects / weapons
    "thin_layer_err": (0.50, False),
    "thin_inst_err_p90": (0.70, False),
    "thin_attrib_fail_frac": (0.30, False),
    "thin_count_odd_even_ratio": (0.0, False),  # special-cased
    "weapon_dev_p90": (0.60, False),
    "weapon_dir_change_p90": (1.0, False),
    # UI / text
    "ui_static_l1": (6.0, False),    # RGB intensity units (0-255)
    "ui_static_grad": (20.0, False),
    "ui_static_edge_f": (0.85, 0.35, True),
    "ui_comp_drift_p90": (8.0, False),  # RGB intensity units (0-255)
    "ui_gen_drift": (6.0, False),    # RGB intensity units (0-255)
    "ui_dyn_blend_frac": (0.15, False),
    "ui_dyn_out_of_range_frac": (0.15, False),
    "ui_dyn_regression": (0.20, False),
    "text_edge_f": (0.80, 0.40, True),
    "text_grad_loss": (0.30, False),
    "text_comp_ratio": (0.0, False),            # special-cased
    # transition defects
    "event_ghost_frac": (0.15, False),
    "event_regression": (0.20, False),
    "card_flip_err": (0.50, False),
    # global technical
    "gtq_blockiness": (0.60, False),            # ~1.0 is clean; excess is blocking
    "gtq_noise": (16.0, False),
    "gtq_sharp_odd_even_ratio": (0.0, False),   # special-cased
    "gtq_global_odd_even_sharp": (0.0, False),
    "anchor_y_l1_p90": (12.0, False),
    "anchor_shift_better_frac": (0.5, False),
}

# ratio-type features: error = |x - 1| / dev_scale
_RATIO_DEVIATION = {
    "edge_count_odd_even_ratio": 0.35,
    "thin_count_odd_even_ratio": 0.45,
    "gtq_sharp_odd_even_ratio": 0.25,
    "gtq_global_odd_even_sharp": 0.25,
    "text_comp_ratio": 0.60,
}


def normalize_error(key: str, value: float) -> float | None:
    """Map a raw feature value to an error in [0, ~1.5]. None if unknown/NaN."""
    if value is None:
        return None
    v = float(value)
    if math.isnan(v):
        return None
    if key in _RATIO_DEVIATION:
        dev = _RATIO_DEVIATION[key]
        return float(np.clip(abs(v - 1.0) / dev, 0, 1.5))
    if key == "gtq_blockiness":
        v = max(v - 1.0, 0.0)          # ~1.0 is clean; excess is blocking
    spec = FEATURE_SCALES.get(key)
    if spec is None:
        return None
    if len(spec) == 3:
        good_ref, drop_width, _inverted = spec
        return float(np.clip((good_ref - v) / max(drop_width, 1e-6), 0, 1.5))
    scale = spec[0]
    return float(1.0 - math.exp(-max(v, 0.0) / max(scale, 1e-6)))
