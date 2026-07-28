"""Temporal non-maximum suppression for risk peaks (USERPLAN §10)."""

from __future__ import annotations

import numpy as np


def temporal_nms(values: np.ndarray, times: np.ndarray, top_k: int,
                 min_gap_seconds: float = 0.5) -> np.ndarray:
    """Indices (into `values`) of the top_k peaks at least min_gap apart."""
    order = np.argsort(-values)
    picked: list[int] = []
    picked_times: list[float] = []
    for idx in order:
        if len(picked) >= top_k:
            break
        if values[idx] <= 0:              # zero-risk frames never earn a slot
            break
        t = float(times[idx])
        if all(abs(t - pt) >= min_gap_seconds for pt in picked_times):
            picked.append(int(idx))
            picked_times.append(t)
    return np.asarray(sorted(picked), np.int64)
