"""Per-category aggregation and the bootstrap score formula (USERPLAN §11).

Window-level errors are aggregated with 0.4·P50 + 0.4·P90 + 0.2·P99 so brief
severe defects are not diluted by thousands of good frames (§11.1). The
overall score uses the §11.3 exponential formula until a trained monotonic
calibrator is available.
"""

from __future__ import annotations

import numpy as np

from ..config import EvalConfig
from ..schema import SUBSCORE_KEYS, WindowFeatures
from .feature_normalizer import normalize_error

CATEGORY_FEATURES: dict[str, list[str]] = {
    "motion": [
        "comp_mean", "comp_p90_max", "comp_leg_ratio", "comp_fwd_p90", "comp_bwd_p90",
        "flow_fold_frac", "flow_jdet_low_frac", "comp_occ_region_mean",
    ],
    "temporal": [
        "mct_lag1_mean", "mct_lag1_p90", "mct_anchor_vs_gen_ratio",
        "cycle_resid_mean", "cycle_resid_p90", "cycle_sharp_loss",
        "freeze_copy_score",
        "parity_window_sharp_gap", "parity_lag2_gen_vs_anchor",
        "parity_sharp_e_pi_block_max", "parity_sharp_mean_gap",
        "edge_count_odd_even_ratio", "gtq_sharp_odd_even_ratio",
    ],
    "structure": [
        "edge_recall", "edge_precision", "edge_inst_recall_med", "edge_inst_recall_p10",
        "edge_chamfer_sup_to_em", "edge_ghost_frac",
    ],
    "character": [
        "char_missing_frac", "char_extra_frac", "char_chamfer",
        "char_components_delta", "char_leak_mean", "char_ring_edge_frac",
    ],
    "thin_weapon": [
        "thin_layer_err", "thin_inst_err_p90", "thin_attrib_fail_frac",
        "thin_count_odd_even_ratio", "weapon_dev_p90", "weapon_dir_change_p90",
    ],
    "ui": [
        "ui_static_l1", "ui_static_grad", "ui_static_edge_f",
        "ui_comp_drift_p90", "ui_gen_drift",
        "ui_dyn_blend_frac", "ui_dyn_out_of_range_frac", "ui_dyn_regression",
        "text_edge_f", "text_grad_loss", "text_comp_ratio",
    ],
    "transition": [
        "event_ghost_frac", "event_regression", "card_flip_err",
    ],
    "global": [
        "gtq_blockiness", "gtq_noise",
        "anchor_y_l1_p90", "anchor_shift_better_frac",
    ],
}

CATEGORY_TO_SUBSCORE = {
    "motion": "motion_consistency",
    "temporal": "temporal_stability",
    "structure": "structural_integrity",
    "character": "character_integrity",
    "thin_weapon": "thin_object_weapon",
    "ui": "ui_text",
    "transition": "transition_quality",
    "global": "global_technical_quality",
}

AGG_WEIGHTS = (0.4, 0.4, 0.2)      # P50 / P90 / P99
_SCORE_K = 1.8                     # error→score sensitivity (bootstrap)


def aggregate_errors(errors: list[float]) -> float:
    """§11.1 aggregation: 0.4·P50 + 0.4·P90 + 0.2·P99 of window error values."""
    if not errors:
        return float("nan")
    a = np.asarray(errors, np.float64)
    p50, p90, p99 = np.percentile(a, [50, 90, 99])
    return float(AGG_WEIGHTS[0] * p50 + AGG_WEIGHTS[1] * p90 + AGG_WEIGHTS[2] * p99)


def build_category_errors(windows: list[WindowFeatures],
                          global_features: dict[str, float]
                          ) -> dict[str, list[float]]:
    """Collect normalized per-window error values for each category.

    Global-only features (scan/anchor level) are appended once so they affect
    the P50 branch but cannot alone drive the P99 tail.
    """
    out: dict[str, list[float]] = {c: [] for c in CATEGORY_FEATURES}
    for wf in windows:
        for cat, keys in CATEGORY_FEATURES.items():
            vals = []
            for k in keys:
                raw = wf.scalars.get(k)
                if raw is None:
                    continue
                err = normalize_error(k, raw)
                if err is not None:
                    vals.append(err)
            if vals:
                out[cat].append(float(np.mean(vals)))
    for k, raw in global_features.items():
        for cat, keys in CATEGORY_FEATURES.items():
            if k in keys:
                err = normalize_error(k, raw)
                if err is not None:
                    out[cat].append(err)
    return out


def compute_scores(cfg: EvalConfig, cat_errors: dict[str, list[float]]
                   ) -> tuple[float, dict[str, float], dict[str, float]]:
    """Returns (overall, subscores, per-category A_c).

    Fail-closed (§7): categories without measurements are NaN, never a default
    error. The overall score renormalizes weights over available categories,
    and is NaN outright when the core categories (motion/temporal/structure)
    are missing — an incomplete evaluation must not look like a good result.
    """
    weights = cfg.preset.score_weights
    A: dict[str, float] = {}
    for cat in CATEGORY_FEATURES:
        A[cat] = aggregate_errors(cat_errors[cat])

    subscores = {}
    for cat, key in CATEGORY_TO_SUBSCORE.items():
        a = A[cat]
        subscores[key] = float(100.0 * np.exp(-_SCORE_K * a)) if a == a else float("nan")

    core = ("motion", "temporal", "structure")
    if any(A[c] != A[c] for c in core):
        return float("nan"), subscores, A

    # §11.3 weighted exponent over available categories; character+thin share
    # one bucket when both are available, else whichever exists.
    a_char = A.get("character", float("nan"))
    a_thin = A.get("thin_weapon", float("nan"))
    if a_char == a_char and a_thin == a_thin:
        a_charthin, w_charthin = 0.5 * a_char + 0.5 * a_thin, weights["character_thin"]
    elif a_char == a_char:
        a_charthin, w_charthin = a_char, weights["character_thin"]
    elif a_thin == a_thin:
        a_charthin, w_charthin = a_thin, weights["character_thin"]
    else:
        a_charthin, w_charthin = float("nan"), 0.0

    terms = [("motion", weights["motion"]), ("temporal", weights["temporal"]),
             ("structure", weights["structure"]), ("ui", weights["ui"]),
             ("transition", weights["transition"]), ("global", weights["global"])]
    num = w_num = 0.0
    for cat, w in terms:
        v = A.get(cat, float("nan"))
        if v == v:
            num += w * v
            w_num += w
    if a_charthin == a_charthin:
        num += w_charthin * a_charthin
        w_num += w_charthin
    overall = float(100.0 * np.exp(-num / w_num)) if w_num > 0 else float("nan")
    return overall, subscores, A


SUBSCORE_KEY_ORDER = SUBSCORE_KEYS
