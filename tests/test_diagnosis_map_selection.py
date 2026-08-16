"""Unit tests for the diagnosis map selection logic (USERPLAN §9).

The selection rule is: for a given issue type, pick the first map named in
``Rule.preferred_maps`` that is present in the window's ``error_maps`` dict;
if none of the preferred names are present, fall back to the first available
map (so box extraction still works for modes that did not produce the ideal
field).  With no maps at all, both ``select_map`` and ``select_map_name``
return ``None``.

These tests build small synthetic ``error_maps`` dicts (20x20 float32 arrays
with a bright rectangle at a known corner) so the chosen map can be
identified by the location of its hot region, independent of dict ordering.
"""

from __future__ import annotations

import numpy as np

from rr_vfiqa.diagnosis.rules import (
    RULES,
    Rule,
    preferred_maps_for,
    select_map,
    select_map_name,
)


# ----------------------------------------------------------------- helpers
def _hot_map(name: str, corner: str) -> np.ndarray:
    """Return a 20x20 float32 map with a bright 5x5 rectangle at ``corner``.

    ``corner`` is one of ``"tl"``, ``"br"`` (top-left / bottom-right).  The
    bright rectangle lets tests identify which map was returned by the
    location of its maximum, independent of the dict's key order.
    """
    m = np.zeros((20, 20), dtype=np.float32)
    if corner == "tl":
        m[0:5, 0:5] = 1.0
    elif corner == "br":
        m[15:20, 15:20] = 1.0
    else:
        raise ValueError(f"unknown corner {corner!r}")
    return m


def _max_corner(m: np.ndarray) -> str:
    """Return ``"tl"`` or ``"br"`` for the corner containing the max value."""
    r, c = np.unravel_index(np.argmax(m), m.shape)
    return "br" if r >= 15 and c >= 15 else "tl"


# -------------------------------------------------------------------- tests
def test_preferred_maps_for_duplicate_freeze():
    """duplicate_freeze lists duplicate_frame_indicator as its top preference."""
    prefs = preferred_maps_for("duplicate_freeze")
    assert isinstance(prefs, tuple)
    assert len(prefs) > 0, "duplicate_freeze must have at least one preferred map"
    assert prefs[0] == "duplicate_frame_indicator", (
        f"expected first preference 'duplicate_frame_indicator', got {prefs[0]!r}"
    )


def test_select_map_uses_preferred_over_available():
    """When a preferred map is present it wins over a non-preferred one.

    duplicate_freeze prefers ``duplicate_frame_indicator`` (hot bottom-right)
    over ``composition_error_map`` (hot top-left).  Both are in the dict, so
    the returned map must be the duplicate one.
    """
    error_maps = {
        "composition_error_map": _hot_map("composition_error_map", "tl"),
        "duplicate_frame_indicator": _hot_map("duplicate_frame_indicator", "br"),
    }
    chosen = select_map(error_maps, "duplicate_freeze")
    assert chosen is not None
    assert _max_corner(chosen) == "br", (
        "expected the preferred duplicate_frame_indicator map (hot bottom-right)"
    )


def test_select_map_name_returns_preferred_name():
    """select_map_name returns the preferred map's *name*, not the array."""
    error_maps = {
        "composition_error_map": _hot_map("composition_error_map", "tl"),
        "duplicate_frame_indicator": _hot_map("duplicate_frame_indicator", "br"),
    }
    name = select_map_name(error_maps, "duplicate_freeze")
    assert name == "duplicate_frame_indicator", (
        f"expected 'duplicate_frame_indicator', got {name!r}"
    )


def test_select_map_falls_back_when_preferred_absent():
    """If no preferred name is present, fall back to the first available map.

    duplicate_freeze's preferences are duplicate_frame_indicator /
    mct_residual_map / composition_error_map.  Supplying only an unrelated
    map forces the fallback path, which must still return a real map (never
    None) so box extraction can proceed.
    """
    error_maps = {
        "some_unrelated_map": _hot_map("some_unrelated_map", "tl"),
    }
    chosen = select_map(error_maps, "duplicate_freeze")
    assert chosen is not None, "fallback must return the available map, not None"
    # The only available map is the unrelated one (hot top-left).
    assert _max_corner(chosen) == "tl"


def test_select_map_returns_none_for_empty():
    """No maps at all -> None."""
    assert select_map({}, "duplicate_freeze") is None


def test_select_map_name_returns_none_for_empty():
    """No maps at all -> None for the name variant too."""
    assert select_map_name({}, "duplicate_freeze") is None


def test_preferred_maps_for_unknown_type():
    """An unknown issue type has no preferred maps (caller must fall back)."""
    assert preferred_maps_for("not_a_real_issue_type") == ()


def test_all_rules_have_non_empty_preferred_maps():
    """Every rule must name at least one preferred map.

    Box extraction and the rendered heatmap both source from the selected map
    (USERPLAN P0-6); a rule with an empty preferred_maps tuple would have no
    preferred source and could only rely on an arbitrary fallback, which is
    unsound for a named issue family.
    """
    assert len(RULES) > 0, "RULES table must not be empty"
    for rule in RULES:
        assert isinstance(rule, Rule)
        assert len(rule.preferred_maps) > 0, (
            f"rule '{rule.issue_type}' has empty preferred_maps; "
            "box extraction needs a named source map"
        )
        # Names must be non-empty strings, not whitespace placeholders.
        for name in rule.preferred_maps:
            assert isinstance(name, str) and name.strip(), (
                f"rule '{rule.issue_type}' has a blank preferred map name"
            )
