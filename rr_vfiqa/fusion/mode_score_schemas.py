"""Independent fusion schemas for NR and same-rate full-reference modes."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from ..config import EvaluationMode
from ..schema import WindowFeatures
from .score_schema import aggregate_errors


@dataclass(frozen=True)
class FeatureSpec:
    scale: float
    good: float | None = None
    lower_is_better: bool = True

    def error(self, value: float) -> float | None:
        value = float(value)
        if not np.isfinite(value):
            return None
        if self.good is not None:
            if self.lower_is_better:
                raw = max(value - self.good, 0.0) / max(self.scale, 1e-6)
            else:
                raw = max(self.good - value, 0.0) / max(self.scale, 1e-6)
            return float(np.clip(raw, 0.0, 1.5))
        if self.lower_is_better:
            return float(1.0 - math.exp(-max(value, 0.0) / max(self.scale, 1e-6)))
        return float(1.0 - math.exp(-max(1.0 - value, 0.0) / max(self.scale, 1e-6)))


NR_FEATURES: dict[str, dict[str, FeatureSpec]] = {
    "temporal": {
        "nr_mct_native_mean": FeatureSpec(24.0),
        "nr_mct_1_60_mean": FeatureSpec(30.0),
        "nr_mct_1_30_mean": FeatureSpec(45.0),
        "nr_common_self_cycle": FeatureSpec(35.0),
        "nr_native_self_cycle": FeatureSpec(28.0),
        "nr_duplicate_fraction": FeatureSpec(0.20),
        "nr_freeze_fraction": FeatureSpec(0.30),
        "nr_native_duplicate_fraction": FeatureSpec(0.15),
        "nr_native_freeze_fraction": FeatureSpec(0.25),
    },
    "motion": {
        "nr_common_self_comp": FeatureSpec(0.18),
        "nr_native_self_comp": FeatureSpec(0.15),
        "nr_flow_accel_ratio": FeatureSpec(0.50),
        "nr_flow_jerk_ratio": FeatureSpec(0.75),
        "nr_flow_fold_fraction": FeatureSpec(0.08),
        "nr_flow_jdet_low_fraction": FeatureSpec(0.18),
        "nr_flow_divergence_std": FeatureSpec(1.5),
        "nr_flow_curl_std": FeatureSpec(1.5),
        "nr_track_accel_p90": FeatureSpec(0.75),
        "nr_track_jerk_p90": FeatureSpec(0.50),
        "nr_track_turn_p90": FeatureSpec(1.0),
        "nr_tile_accel_p90": FeatureSpec(0.85),
        "nr_tile_jerk_p90": FeatureSpec(0.65),
        "nr_local_reversal_fraction": FeatureSpec(0.20),
    },
    "phase": {
        "nr_phase_sharp_gap": FeatureSpec(0.25),
        "nr_phase_edge_gap": FeatureSpec(0.35),
        "nr_phase_sharp_energy": FeatureSpec(0.45),
        "nr_phase_edge_energy": FeatureSpec(0.45),
    },
    "ui_structure": {
        "nr_ui_edge_instability": FeatureSpec(0.12),
        "nr_text_stroke_instability": FeatureSpec(0.20),
    },
    "technical": {
        "gtq_blockiness": FeatureSpec(0.60, good=1.0),
        "gtq_noise": FeatureSpec(16.0),
        "nr_learned_vqa_error": FeatureSpec(1.0),
    },
}

FR_FEATURES: dict[str, dict[str, FeatureSpec]] = {
    "spatial": {
        "fr_l1_y": FeatureSpec(12.0),
        "fr_l1_rgb": FeatureSpec(12.0),
        "fr_charbonnier_rgb": FeatureSpec(10.0),
        "fr_psnr": FeatureSpec(18.0, good=38.0, lower_is_better=False),
        "fr_ssim": FeatureSpec(0.25, good=0.98, lower_is_better=False),
        "fr_multiscale_luma_gradient_l1": FeatureSpec(0.06),
        "fr_salient_roi_l1": FeatureSpec(16.0),
        "fr_motion_roi_l1": FeatureSpec(18.0),
        "fr_scan_y_l1": FeatureSpec(12.0),
        "fr_scan_chroma_l1": FeatureSpec(10.0),
        "fr_scan_gradient_l1": FeatureSpec(20.0),
        "fr_scan_ssim_proxy_error": FeatureSpec(0.20),
    },
    "structure": {
        "fr_edge_recall": FeatureSpec(
            0.40, good=0.90, lower_is_better=False),
        "fr_edge_precision": FeatureSpec(
            0.40, good=0.90, lower_is_better=False),
        "fr_edge_f1": FeatureSpec(0.40, good=0.90, lower_is_better=False),
        "fr_edge_chamfer": FeatureSpec(4.0),
        "fr_ui_roi_l1": FeatureSpec(12.0),
        "fr_text_roi_edge_f1": FeatureSpec(
            0.45, good=0.85, lower_is_better=False),
        "fr_structure_persistence_error": FeatureSpec(0.12),
        "fr_scan_edge_mismatch": FeatureSpec(0.12),
    },
    "temporal": {
        "fr_temporal_diff_error": FeatureSpec(10.0),
        "fr_mcr_difference": FeatureSpec(10.0),
        "fr_flicker_excess": FeatureSpec(3.0),
        "fr_scan_frame_diff_mismatch": FeatureSpec(10.0),
    },
    "motion": {
        "fr_flow_error": FeatureSpec(0.30),
        "fr_trajectory_deviation": FeatureSpec(2.0),
    },
}

MODE_WEIGHTS = {
    EvaluationMode.NO_REFERENCE: {
        "temporal": 0.35,
        "motion": 0.30,
        "phase": 0.20,
        "ui_structure": 0.10,
        "technical": 0.05,
    },
    EvaluationMode.FULL_REFERENCE: {
        "spatial": 0.35,
        "structure": 0.20,
        "temporal": 0.25,
        "motion": 0.20,
    },
}

SUBSCORE_NAMES = {
    EvaluationMode.NO_REFERENCE: {
        "temporal": "temporal_stability",
        "motion": "motion_smoothness",
        "phase": "phase_consistency",
        "ui_structure": "ui_text_stability",
        "technical": "technical_quality_prior",
    },
    EvaluationMode.FULL_REFERENCE: {
        "spatial": "spatial_fidelity",
        "structure": "structural_fidelity",
        "temporal": "temporal_fidelity",
        "motion": "motion_fidelity",
    },
}

CORE_CATEGORIES = {
    EvaluationMode.NO_REFERENCE: ("temporal", "motion", "phase"),
    EvaluationMode.FULL_REFERENCE: ("spatial", "structure", "temporal", "motion"),
}

SCHEMA_IDS = {
    EvaluationMode.NO_REFERENCE: "nr-stability-risk-v2",
    EvaluationMode.FULL_REFERENCE: "fr-same-rate-fidelity-v2",
}


def _schema(mode: EvaluationMode) -> dict[str, dict[str, FeatureSpec]]:
    if mode is EvaluationMode.NO_REFERENCE:
        return NR_FEATURES
    if mode is EvaluationMode.FULL_REFERENCE:
        return FR_FEATURES
    raise ValueError("endpoint-2x uses fusion.score_schema, not a generic mode schema")


def window_category_errors(
    mode: EvaluationMode,
    wf: WindowFeatures,
) -> dict[str, float]:
    result: dict[str, float] = {}
    for category, features in _schema(mode).items():
        values = [
            error
            for key, spec in features.items()
            if key in wf.scalars
            for error in [spec.error(wf.scalars[key])]
            if error is not None
        ]
        if values:
            result[category] = float(np.mean(values))
    return result


def compute_mode_scores(
    mode: EvaluationMode,
    windows: list[WindowFeatures],
    global_features: dict[str, float] | None = None,
) -> tuple[float, dict[str, float], dict[str, float]]:
    """Fuse using the selected mode's own features, weights, and semantics."""
    schema = _schema(mode)
    per_category: dict[str, list[float]] = {category: [] for category in schema}
    for wf in windows:
        for category, error in window_category_errors(mode, wf).items():
            per_category[category].append(error)
    if global_features:
        for category, features in schema.items():
            values = [
                error
                for key, spec in features.items()
                if key in global_features
                for error in [spec.error(global_features[key])]
                if error is not None
            ]
            if values:
                per_category[category].append(float(np.mean(values)))
    aggregated = {
        category: aggregate_errors(values)
        for category, values in per_category.items()
    }
    subscores = {
        SUBSCORE_NAMES[mode][category]: (
            float(100.0 * np.exp(-1.8 * error))
            if np.isfinite(error) else float("nan")
        )
        for category, error in aggregated.items()
    }
    if any(not np.isfinite(aggregated[c]) for c in CORE_CATEGORIES[mode]):
        return float("nan"), subscores, aggregated

    numerator = denominator = 0.0
    for category, weight in MODE_WEIGHTS[mode].items():
        error = aggregated[category]
        if np.isfinite(error):
            numerator += weight * error
            denominator += weight
    overall = (
        float(100.0 * np.exp(-numerator / denominator))
        if denominator > 0 else float("nan")
    )
    return overall, subscores, aggregated


def compute_mode_confidence(
    *,
    valid_windows: int,
    total_windows: int,
    coverage_fraction: float,
    warning_count: int = 0,
    alignment_fraction: float = 1.0,
    ceiling: float = 1.0,
) -> float:
    if total_windows <= 0 or valid_windows <= 0:
        return 0.02
    validity = valid_windows / total_windows
    density = float(np.clip(coverage_fraction / 0.05, 0.0, 1.0))
    warnings = max(0.0, 1.0 - 0.12 * warning_count)
    confidence = (
        0.45 * validity
        + 0.20 * density
        + 0.25 * float(np.clip(alignment_fraction, 0.0, 1.0))
        + 0.10 * warnings
    )
    return float(np.clip(confidence, 0.02, ceiling))
