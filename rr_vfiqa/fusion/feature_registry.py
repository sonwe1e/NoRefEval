"""Versioned feature definitions shared by execution, fusion and reports."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json

from ..config import EvaluationMode, parse_mode


@dataclass(frozen=True)
class FeatureDefinition:
    name: str
    mode: EvaluationMode
    category: str
    units: str
    direction: str
    resolution_invariant: bool
    required: bool
    scale: float
    version: str = "v2"


_REQUIRED = {
    EvaluationMode.NO_REFERENCE: {
        "nr_common_self_comp", "nr_common_self_cycle",
        "nr_mct_1_60_mean", "nr_phase_sharp_gap",
    },
    EvaluationMode.FULL_REFERENCE: {
        "fr_l1_y", "fr_ssim", "fr_temporal_diff_error", "fr_flow_error",
    },
    EvaluationMode.ENDPOINT_2X: {
        "comp_mean", "mct_lag1_mean", "edge_recall",
    },
}

_UNITS = {
    "psnr": "dB",
    "ssim": "ratio",
    "fraction": "ratio",
    "ratio": "ratio",
    "l1": "luma",
    "charbonnier": "rgb",
    "chamfer": "pixels",
    "trajectory": "pixels",
    "flow": "normalized-flow",
}

_UNIT_OVERRIDES = {
    "fr_l1_y": "luma",
    "fr_l1_rgb": "rgb",
    "fr_charbonnier_rgb": "rgb",
    "fr_scan_chroma_l1": "chroma",
    "fr_scan_gradient_l1": "luma-gradient",
    "fr_multiscale_luma_gradient_l1": "normalized-luma-gradient",
    "nr_mct_native_mean": "luma",
    "nr_native_self_cycle": "luma",
}

# Explicit per-feature contracts for features whose units / resolution semantics
# cannot be reliably inferred from the name. The ``_unit`` heuristic guesses
# "pixels" for any "*chamfer*" name and "luma" for "*l1*", which is wrong for
# the normalized chamfer distances and the RGB-intensity UI drifts below.
# FeatureRegistry 逐项显式定义 units 和 resolution_invariant，不再从名称猜.
# Mapping: feature_name -> (units, resolution_invariant).
_FEATURE_CONTRACT_OVERRIDES: dict[str, tuple[str, bool]] = {
    "edge_chamfer_sup_to_em": ("frame-diagonal-ratio", True),
    "char_chamfer": ("frame-diagonal-ratio", True),
    "fr_edge_chamfer": ("frame-diagonal-ratio", True),
    "fr_trajectory_deviation": ("frame-diagonal-ratio", True),
    "ui_static_l1": ("rgb", True),
    "ui_comp_drift_p90": ("rgb", True),
    "ui_gen_drift": ("rgb", True),
}

_NR_DIAGNOSTICS = {
    "nr_mct_native_mean": (24.0, "temporal_native_diagnostic"),
    "nr_mct_native_p90": (36.0, "temporal_native_diagnostic"),
    "nr_native_self_cycle": (28.0, "temporal_native_diagnostic"),
    "nr_native_self_cycle_p90": (42.0, "temporal_native_diagnostic"),
    "nr_native_self_comp": (0.15, "motion_native_diagnostic"),
    "nr_native_duplicate_fraction": (0.15, "temporal_native_diagnostic"),
    "nr_native_freeze_fraction": (0.25, "temporal_native_diagnostic"),
    "nr_edge_instability_native": (0.12, "ui_native_diagnostic"),
}


def _unit(name: str) -> str:
    if name in _UNIT_OVERRIDES:
        return _UNIT_OVERRIDES[name]
    if any(token in name for token in (
            "mct", "self_cycle", "temporal_diff", "luma")):
        return "luma"
    if "chroma" in name:
        return "chroma"
    if "chamfer" in name:
        return "pixels"
    return next((unit for token, unit in _UNITS.items() if token in name),
                "dimensionless")


def definitions(mode: str | EvaluationMode) -> tuple[FeatureDefinition, ...]:
    parsed = parse_mode(mode)
    if parsed is EvaluationMode.ENDPOINT_2X:
        from .feature_normalizer import FEATURE_SCALES
        from .score_schema import CATEGORY_FEATURES
        rows = []
        for category, names in CATEGORY_FEATURES.items():
            for name in names:
                spec = FEATURE_SCALES.get(name, (1.0, False))
                # Prefer an explicit contract override; fall back to the name
                # heuristic only when no override exists. 逐项显式定义, not guess.
                if name in _FEATURE_CONTRACT_OVERRIDES:
                    units, resolution_invariant = (
                        _FEATURE_CONTRACT_OVERRIDES[name])
                else:
                    units, resolution_invariant = (
                        _unit(name), "chamfer" not in name)
                rows.append(FeatureDefinition(
                    name=name, mode=parsed, category=category,
                    units=units,
                    direction="higher_is_better" if len(spec) == 3
                    else "lower_is_better",
                    resolution_invariant=resolution_invariant,
                    required=name in _REQUIRED[parsed],
                    scale=float(spec[1] if len(spec) == 3 else spec[0]),
                ))
        return tuple(rows)

    from .mode_score_schemas import FR_FEATURES, NR_FEATURES
    schema = NR_FEATURES if parsed is EvaluationMode.NO_REFERENCE else FR_FEATURES
    rows = []
    for category, features in schema.items():
        for name, spec in features.items():
            if name in _FEATURE_CONTRACT_OVERRIDES:
                units, resolution_invariant = (
                    _FEATURE_CONTRACT_OVERRIDES[name])
            else:
                units, resolution_invariant = (
                    _unit(name),
                    not any(token in name
                            for token in ("chamfer", "trajectory")),
                )
            rows.append(FeatureDefinition(
                name=name, mode=parsed, category=category, units=units,
                direction=("lower_is_better" if spec.lower_is_better
                           else "higher_is_better"),
                resolution_invariant=resolution_invariant,
                required=name in _REQUIRED[parsed],
                scale=float(spec.scale),
            ))
    if parsed is EvaluationMode.NO_REFERENCE:
        for name, (scale, category) in _NR_DIAGNOSTICS.items():
            rows.append(FeatureDefinition(
                name=name,
                mode=parsed,
                category=category,
                units=_unit(name),
                direction="lower_is_better",
                resolution_invariant=True,
                required=False,
                scale=scale,
            ))
    return tuple(rows)


def required_features(mode: str | EvaluationMode) -> tuple[str, ...]:
    parsed = parse_mode(mode)
    return tuple(sorted(_REQUIRED.get(parsed, set())))


def feature_contract_hash(mode: str | EvaluationMode) -> str:
    payload = []
    for definition in definitions(mode):
        row = asdict(definition)
        row["mode"] = definition.mode.value
        payload.append(row)
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256(encoded).hexdigest()[:16]
