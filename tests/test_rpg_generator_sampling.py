"""Temporal sampling is by frame index, never an ffmpeg filter (PICPLAN §8)."""
from __future__ import annotations

import numpy as np
import pytest

from tools.rpg_validation_generator.timeline import gather, sample_indices


def test_sample_indices_ratios():
    assert sample_indices(120, 60, 480).tolist() == list(range(0, 480, 2))
    assert sample_indices(120, 30, 480).tolist() == list(range(0, 480, 4))
    assert sample_indices(60, 30, 240).tolist() == list(range(0, 240, 2))


def test_sample_indices_non_divisor_rejected():
    with pytest.raises(ValueError):
        sample_indices(120, 50, 480)


def test_gather_matches_indexing(rpg_masters):
    m = rpg_masters[0]
    idx = sample_indices(m.fps, m.fps // 2, m.n_frames)
    g = gather(m.rgb, idx)
    assert np.array_equal(g, m.rgb[idx])
    assert g.flags["C_CONTIGUOUS"]


def test_60_is_subset_of_120(rpg_masters):
    m = rpg_masters[0]
    full = np.asarray(m.rgb)
    half = gather(m.rgb, sample_indices(m.fps, m.fps // 2, m.n_frames))
    assert np.array_equal(half, full[0::2])
