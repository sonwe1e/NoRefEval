"""Window selection: uniform coverage + top-risk peaks (USERPLAN §10)."""

from __future__ import annotations

import numpy as np

from ..config import EvalConfig
from ..schema import Alignment, VideoMeta, Window
from .cheap_scan import CheapScan
from .risk_score import compute_risk
from .temporal_nms import temporal_nms


def select_windows(cfg: EvalConfig, cand_meta: VideoMeta, alignment: Alignment,
                   scan: CheapScan
                   ) -> tuple[list[Window], np.ndarray, dict[str, np.ndarray]]:
    """Build the evaluation windows.

    Returns (windows, per_frame_risk, risk_components).
    """
    preset = cfg.preset
    centers = alignment.generated_centers()          # candidate indices of M_i
    n_cand = cand_meta.n_frames

    # Drop centers whose source pair crosses a scene cut.
    cut_pairs = set(int(p) for p in alignment.scene_cuts)
    centers = np.asarray([c for c in centers
                          if int(alignment.pair_of_candidate[c]) not in cut_pairs],
                         np.int32)
    if len(centers) == 0:
        return [], np.zeros(n_cand), {}

    risk, comps = compute_risk(scan)
    times = cand_meta.pts_seconds

    half = preset.window_frames // 2
    usable = centers[(centers >= half) & (centers < n_cand - half)]
    if len(usable) == 0:
        usable = centers

    # Uniform sample across the timeline.
    n_uni = min(preset.uniform_windows, len(usable))
    uni_pos = np.linspace(0, len(usable) - 1, n_uni).astype(int)
    uni_centers = set(int(usable[p]) for p in uni_pos)

    # Risk sample with temporal NMS.
    risk_at_centers = np.asarray([risk[c] if c < len(risk) else 0.0 for c in usable])
    n_risk = min(preset.risk_windows, len(usable))
    picked = temporal_nms(risk_at_centers, times[usable], n_risk,
                          preset.temporal_nms_seconds)
    risk_centers = set(int(usable[p]) for p in picked)

    windows: list[Window] = []
    for c in sorted(uni_centers | risk_centers):
        src = "uniform" if c in uni_centers else "risk"
        if c in uni_centers and c in risk_centers:
            src = "uniform+risk"
        idx = np.arange(c - half, c + half + 1, dtype=np.int32)
        w = Window(
            center=int(c),
            indices=idx,
            pair=int(alignment.pair_of_candidate[c]),
            risk=float(risk[c]) if c < len(risk) else 0.0,
            source=src,
            risk_components={k: float(v[c]) for k, v in comps.items()}
                            if c < len(risk) else {},
        )
        windows.append(w)

    windows = windows[: preset.max_windows_hard_cap]
    return windows, risk, comps


def window_times(w: Window, cand_meta: VideoMeta) -> tuple[float, float]:
    i0, i1 = int(w.indices[0]), int(w.indices[-1])
    i0 = min(max(i0, 0), cand_meta.n_frames - 1)
    i1 = min(max(i1, 0), cand_meta.n_frames - 1)
    return float(cand_meta.pts_seconds[i0]), float(cand_meta.pts_seconds[i1])
