"""Pure contract tests for mode parsing, preset resolution and EvalConfig
validation (config.py / mode_router.py).

These functions are the CLI/API contract surface: aliases, defaults and
fail-closed errors here are user-visible, so lock them down without needing
any video decode.
"""

from __future__ import annotations

import pytest

from rr_vfiqa.config import EvalConfig, EvaluationMode, PRESETS, parse_mode
from rr_vfiqa.mode_router import SPEED_ALIASES, preset_from_speed


# ------------------------------------------------------------------ parse_mode

def test_parse_mode_passthrough_enum():
    assert parse_mode(EvaluationMode.NO_REFERENCE) is EvaluationMode.NO_REFERENCE


def test_parse_mode_canonical_strings():
    for m in EvaluationMode:
        assert parse_mode(m.value) is m


def test_parse_mode_case_and_space_insensitive():
    assert parse_mode(" No-Reference ") is EvaluationMode.NO_REFERENCE
    assert parse_mode("FULL_REFERENCE") is EvaluationMode.FULL_REFERENCE


def test_parse_mode_aliases():
    for alias in ("no_reference", "nr"):
        assert parse_mode(alias) is EvaluationMode.NO_REFERENCE
    for alias in ("endpoint", "endpoint_reference", "endpoint-reference"):
        assert parse_mode(alias) is EvaluationMode.ENDPOINT_2X
    for alias in ("full_reference", "fr"):
        assert parse_mode(alias) is EvaluationMode.FULL_REFERENCE


def test_parse_mode_unknown_raises_with_choices():
    with pytest.raises(ValueError, match="unknown evaluation mode"):
        parse_mode("bogus")


# ------------------------------------------------------------- preset_from_speed

def test_preset_from_speed_aliases():
    assert preset_from_speed("fast", None) == "fast"
    assert preset_from_speed("balanced", None) == "standard"
    assert preset_from_speed("thorough", None) == "audit"


def test_preset_from_speed_case_insensitive():
    assert preset_from_speed("FAST", None) == "fast"


def test_preset_from_speed_prefers_speed_over_preset():
    # ``--speed`` wins when both are given (CLI resolves speed first).
    assert preset_from_speed("fast", "audit") == "fast"


def test_preset_from_speed_defaults_to_standard():
    assert preset_from_speed(None, None) == "standard"
    assert preset_from_speed(None, "audit") == "audit"


def test_preset_from_speed_unknown_raises():
    with pytest.raises(ValueError, match="unknown --speed"):
        preset_from_speed("slow", None)


def test_speed_aliases_map_to_real_presets():
    # Every documented speed alias must resolve to a preset the pipeline can
    # actually load (guards against a renamed PRESETS entry).
    for alias, preset in SPEED_ALIASES.items():
        assert alias in SPEED_ALIASES
        assert preset in PRESETS


# ------------------------------------------------------- EvalConfig.build_mode

def test_build_mode_unknown_preset():
    with pytest.raises(ValueError, match="unknown preset"):
        EvalConfig.build_mode(
            candidate_video="c.mp4", mode="no-reference", preset="bogus")


def test_build_mode_requires_mode():
    with pytest.raises(ValueError, match="mode must be explicitly specified"):
        EvalConfig.build_mode(candidate_video="c.mp4")


def test_build_mode_nr_rejects_reference():
    with pytest.raises(ValueError, match="does not accept a reference"):
        EvalConfig.build_mode(
            candidate_video="c.mp4", reference_video="r.mp4",
            mode="no-reference")


def test_build_mode_reference_modes_require_reference():
    for mode in ("endpoint-2x", "full-reference"):
        with pytest.raises(ValueError, match="requires a reference"):
            EvalConfig.build_mode(candidate_video="c.mp4", mode=mode)


def test_build_mode_unknown_override():
    with pytest.raises(ValueError, match="unknown config override"):
        EvalConfig.build_mode(
            candidate_video="c.mp4", mode="no-reference", bogus=1)


def test_build_mode_invalid_geometry_policy():
    with pytest.raises(ValueError, match="geometry_policy"):
        EvalConfig.build_mode(
            candidate_video="c.mp4", mode="no-reference",
            geometry_policy="bogus")


def test_build_mode_applies_overrides():
    cfg = EvalConfig.build_mode(
        candidate_video="c.mp4", mode="no-reference", device="cpu")
    assert cfg.device == "cpu"
    assert cfg.mode is EvaluationMode.NO_REFERENCE


def test_build_backward_compatible_endpoint():
    cfg = EvalConfig.build("s.mp4", "c.mp4", "fast")
    assert cfg.mode is EvaluationMode.ENDPOINT_2X
    assert cfg.preset is PRESETS["fast"]
    assert cfg.reference_video == "s.mp4"
    assert cfg.source_video == "s.mp4"
