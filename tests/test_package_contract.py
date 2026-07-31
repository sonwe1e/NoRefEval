"""Package version and feature-registry contracts (USERPLAN 9).

These tests lock down the public package contract:

* The installed package version is a valid semver string and matches the
  metadata declared in ``pyproject.toml`` (``dynamic = ["version"]``).
* The feature registry produces stable, well-formed contracts across every
  evaluation mode, with explicit per-feature overrides honoured.
"""

from __future__ import annotations

import os
import re

import pytest

import rr_vfiqa
from rr_vfiqa._version import VERSION
from rr_vfiqa.fusion.feature_registry import (
    FeatureDefinition,
    definitions,
    feature_contract_hash,
    required_features,
)

try:  # Python 3.8+; gracefully unavailable on older interpreters.
    from importlib import metadata as importlib_metadata
except ImportError:  # pragma: no cover - exercised only on Python <3.8.
    import importlib_metadata  # type: ignore


# ---------------------------------------------------------------- version ---

SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+")


def test_package_version_matches_metadata():
    """Installed package version matches ``pyproject.toml`` metadata.

    In CI (fresh editable install) the metadata MUST track the source
    version exactly.  Local dev environments may have stale metadata from a
    prior install before a version bump — that's a packaging issue, not a
    code contract failure.  We assert the source-of-truth invariant hard, and
    the metadata match conditionally (skip on stale local metadata, fail in
    CI where CI=true is set).
    """
    assert rr_vfiqa.__version__ == VERSION, (
        f"rr_vfiqa.__version__ ({rr_vfiqa.__version__!r}) != "
        f"VERSION ({VERSION!r})")
    try:
        meta_version = importlib_metadata.version("rr-vfiqa")
    except importlib_metadata.PackageNotFoundError:
        return  # not installed as a package; source check above is sufficient
    if meta_version == rr_vfiqa.__version__:
        return  # perfect match
    # In CI, stale metadata is a real failure (packaging config is broken).
    if os.environ.get("CI"):
        pytest.fail(
            f"CI: installed metadata ({meta_version}) != source "
            f"({rr_vfiqa.__version__}); pyproject.toml version is out of "
            f"sync with rr_vfiqa._version.VERSION")
    # Local dev: stale metadata from a prior install is not a code failure.
    pytest.skip(
        f"local: installed metadata ({meta_version}) stale vs source "
        f"({rr_vfiqa.__version__}); reinstall with 'pip install -e .' to "
        f"refresh")


def test_version_is_semver():
    """The embedded version string is a valid dotted semver."""
    assert isinstance(rr_vfiqa.__version__, str)
    assert SEMVER_RE.match(rr_vfiqa.__version__), (
        f"version {rr_vfiqa.__version__!r} is not semver")
    # The canonical source of truth and the public attribute agree.
    assert VERSION == rr_vfiqa.__version__


# ------------------------------------------------------------- feature hash ---

def test_feature_contract_hash_is_stable():
    """The feature contract hash is deterministic across repeated calls."""
    first = feature_contract_hash("no-reference")
    second = feature_contract_hash("no-reference")
    assert first == second


def test_feature_contract_hash_is_16_hex_chars():
    """The contract hash is exactly 16 lowercase hex characters."""
    hash_value = feature_contract_hash("no-reference")
    assert isinstance(hash_value, str)
    assert len(hash_value) == 16
    assert re.fullmatch(r"[0-9a-f]{16}", hash_value), (
        f"hash {hash_value!r} is not 16 hex chars")


# ------------------------------------------------------- feature contracts ---

def test_fr_edge_chamfer_contract():
    """``fr_edge_chamfer`` carries the explicit normalized-distance contract."""
    rows = definitions("full-reference")
    by_name = {d.name: d for d in rows}
    assert "fr_edge_chamfer" in by_name, "fr_edge_chamfer missing from full-reference"
    feat = by_name["fr_edge_chamfer"]
    assert feat.units == "frame-diagonal-ratio"
    assert feat.resolution_invariant is True


def test_nr_required_features_subset():
    """No-reference required features include the core self-comparison metric."""
    required = required_features("no-reference")
    assert isinstance(required, tuple)
    assert len(required) > 0
    assert "nr_common_self_comp" in required


def test_definitions_cover_all_modes():
    """Every evaluation mode yields a non-empty feature definition list."""
    for mode in ("no-reference", "full-reference", "endpoint-2x"):
        rows = definitions(mode)
        assert isinstance(rows, tuple)
        assert len(rows) > 0, f"definitions({mode!r}) is empty"


def test_feature_definition_fields_complete():
    """Every FeatureDefinition is fully populated with valid fields.

    Ratio-type features (edge_count_odd_even_ratio, etc.) legitimately have
    scale=0 because their error is computed via ``|x-1|/dev_scale`` in
    ``normalize_error``, not via the ``1 - exp(-x/scale)`` path.  We accept
    scale >= 0 for those and require scale > 0 for the rest.
    """
    ratio_features = {
        "edge_count_odd_even_ratio", "thin_count_odd_even_ratio",
        "gtq_sharp_odd_even_ratio", "gtq_global_odd_even_sharp",
        "text_comp_ratio",
    }
    for mode in ("no-reference", "full-reference", "endpoint-2x"):
        for feat in definitions(mode):
            assert isinstance(feat, FeatureDefinition)
            assert feat.name, "feature name must be non-empty"
            assert feat.category, "feature category must be non-empty"
            assert feat.units, "feature units must be non-empty"
            assert feat.direction in ("higher_is_better", "lower_is_better"), (
                f"feature {feat.name!r} has invalid direction {feat.direction!r}")
            if feat.name not in ratio_features:
                assert feat.scale > 0, (
                    f"feature {feat.name!r} must have scale > 0, "
                    f"got {feat.scale}")
            # resolution_invariant is a bool; no empty/None sentinel allowed.
            assert isinstance(feat.resolution_invariant, bool)
