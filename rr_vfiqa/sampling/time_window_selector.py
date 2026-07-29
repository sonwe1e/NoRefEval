"""PTS-driven windows shared by no-reference and full-reference modes."""

from __future__ import annotations

import numpy as np

from ..config import EvalConfig
from ..schema import VideoMeta, Window
from .cheap_scan import CheapScan
from .risk_score import compute_risk
from .temporal_nms import temporal_nms


def _window_indices(meta: VideoMeta, center: int, half_span_seconds: float) -> np.ndarray:
    """All decoded samples inside a fixed time span around ``center``."""
    times = meta.pts_seconds
    tc = float(times[center])
    lo = int(np.searchsorted(times, tc - half_span_seconds, side="left"))
    hi = int(np.searchsorted(times, tc + half_span_seconds, side="right"))
    idx = np.arange(max(0, lo), min(meta.n_frames, hi), dtype=np.int32)
    return idx


def select_time_windows(
    cfg: EvalConfig,
    meta: VideoMeta,
    scan: CheapScan,
    *,
    eligible_centers: np.ndarray | None = None,
    excluded_indices: np.ndarray | None = None,
    external_risk: np.ndarray | None = None,
    half_span_seconds: float = 1.0 / 30.0,
    min_samples: int = 5,
) -> tuple[list[Window], np.ndarray, dict[str, np.ndarray]]:
    """Select uniform+risk windows with a fixed wall-clock duration.

    A ±33.3 ms window contains roughly five frames at 60 FPS and nine at
    120 FPS.  This preserves comparable temporal scales while retaining the
    extra high-frequency evidence available in a 120 FPS input.
    """
    n = meta.n_frames
    risk, components = compute_risk(scan)
    if external_risk is not None:
        ext = np.asarray(external_risk, np.float64)
        if len(ext) != len(risk):
            raise ValueError("external_risk must match the candidate timeline")
        finite = ext[np.isfinite(ext)]
        scale = float(np.percentile(finite, 90)) if len(finite) else 1.0
        normalized = np.clip(np.nan_to_num(ext) / max(scale, 1e-6), 0.0, 3.0)
        components["reference_risk"] = normalized.astype(np.float32)
        risk = risk + normalized
    centers = (
        np.asarray(eligible_centers, np.int32)
        if eligible_centers is not None
        else np.arange(n, dtype=np.int32)
    )
    centers = centers[(centers >= 0) & (centers < n)]
    excluded = set(int(i) for i in np.asarray(
        excluded_indices if excluded_indices is not None else [], np.int32))

    usable: list[int] = []
    indices_by_center: dict[int, np.ndarray] = {}
    for c in centers:
        idx = _window_indices(meta, int(c), half_span_seconds)
        if len(idx) < min_samples or any(int(i) in excluded for i in idx):
            continue
        usable.append(int(c))
        indices_by_center[int(c)] = idx
    if not usable:
        return [], risk, components

    usable_arr = np.asarray(usable, np.int32)
    p = cfg.preset
    n_uniform = min(p.uniform_windows, len(usable_arr))
    uniform_pos = np.linspace(0, len(usable_arr) - 1, n_uniform).astype(int)
    uniform = set(int(usable_arr[i]) for i in uniform_pos)

    n_risk = min(p.risk_windows, len(usable_arr))
    center_risk = np.asarray(
        [risk[c] if c < len(risk) else 0.0 for c in usable_arr], np.float64)
    chosen = temporal_nms(
        center_risk,
        meta.pts_seconds[usable_arr],
        n_risk,
        p.temporal_nms_seconds,
    )
    risky = set(int(usable_arr[i]) for i in chosen)

    windows: list[Window] = []
    for c in sorted(uniform | risky):
        source = "uniform+risk" if c in uniform and c in risky else (
            "uniform" if c in uniform else "risk")
        windows.append(Window(
            center=c,
            indices=indices_by_center[c],
            pair=-1,
            risk=float(risk[c]) if c < len(risk) else 0.0,
            source=source,
            risk_components={
                k: float(v[c]) for k, v in components.items() if c < len(v)
            },
        ))
    return windows[:p.max_windows_hard_cap], risk, components
