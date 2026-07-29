"""Phase-invariant temporal evidence for a single 60/120 FPS video.

These metrics estimate stability and artifact risk.  They deliberately do not
claim that an interpolated frame contains the true intermediate content.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..config import EvalConfig
from ..models.vqa_backend import VQABackend
from ..motion.flow_composition import composition_error
from ..schema import (
    FrameBundle,
    backward_warp,
    charbonnier,
    flow_magnitude,
    forward_splat,
    warp_flow,
)
from .global_technical_quality import compute_window as technical_window
from .window_flows import WindowFlows


def _luma(rgb: np.ndarray) -> np.ndarray:
    return (
        0.299 * rgb[..., 0].astype(np.float32)
        + 0.587 * rgb[..., 1].astype(np.float32)
        + 0.114 * rgb[..., 2].astype(np.float32)
    )


def _target_pairs(times: np.ndarray, target: float) -> list[tuple[int, int]]:
    if len(times) < 2:
        return []
    native = float(np.median(np.diff(times)))
    tolerance = max(0.35 * target, 0.35 * native)
    pairs: list[tuple[int, int]] = []
    for i in range(len(times) - 1):
        delta = times[i + 1:] - times[i]
        jrel = int(np.argmin(np.abs(delta - target)))
        if abs(float(delta[jrel]) - target) <= tolerance:
            pairs.append((i, i + 1 + jrel))
    return pairs


def _mc_residual(
    y: np.ndarray,
    flows: WindowFlows,
    pairs: list[tuple[int, int]],
    cfg: EvalConfig,
) -> tuple[float, float]:
    means: list[float] = []
    p90s: list[float] = []
    for a, b in pairs:
        f_ab, f_ba = flows.pair(a, b)
        warped = backward_warp(y[b], f_ab)
        cycle = flow_magnitude(warp_flow(f_ab, f_ba))
        visible = cycle < cfg.occlusion_cycle_threshold
        if visible.sum() < 64:
            continue
        residual = charbonnier(np.abs(warped - y[a]), cfg.charbonnier_tau)[visible]
        means.append(float(np.mean(residual)))
        p90s.append(float(np.percentile(residual, 90)))
    if not means:
        return float("nan"), float("nan")
    return float(np.mean(means)), float(np.mean(p90s))


def _reconstruct_mid(
    a: np.ndarray,
    b: np.ndarray,
    f_ab: np.ndarray,
    f_ba: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    wa, ca = forward_splat(a.astype(np.float32), 0.5 * f_ab)
    wb, cb = forward_splat(b.astype(np.float32), 0.5 * f_ba)
    denom = ca + cb + 1e-6
    reconstructed = (
        wa * ca[..., None] + wb * cb[..., None]
    ) / denom[..., None]
    return reconstructed, np.clip(0.5 * denom, 0.0, 1.0)


def compute_window(
    bundle: FrameBundle,
    flows: WindowFlows,
    cfg: EvalConfig,
    *,
    vqa_backend: VQABackend | None = None,
) -> dict[str, float]:
    """Compute two-phase self-reference and time-normalized generic evidence."""
    n = len(bundle.rgb)
    if n < 5:
        raise ValueError("no-reference windows require at least five samples")
    out = technical_window(bundle)
    y = bundle.y_channel()

    # Every adjacent triple is a virtual endpoint/middle/endpoint sample.
    # Even and odd triple starts are the two phase hypotheses; aggregation
    # never assumes either phase contains the original frames.
    comp_by_phase: list[list[float]] = [[], []]
    cycle_by_phase: list[list[float]] = [[], []]
    cycle_p90: list[float] = []
    for a in range(n - 2):
        m, b = a + 1, a + 2
        f_ab, f_ba = flows.pair(a, b)
        f_am, f_ma = flows.pair(a, m)
        f_mb, f_bm = flows.pair(m, b)
        forward = composition_error(
            f_ab, f_am, f_mb, tau_px=cfg.charbonnier_tau)
        backward = composition_error(
            f_ba, f_bm, f_ma, tau_px=cfg.charbonnier_tau)
        comp_by_phase[int(bundle.indices[a] % 2)].append(
            0.5 * (forward.weighted_mean + backward.weighted_mean))

        reconstructed, coverage = _reconstruct_mid(
            bundle.rgb[a], bundle.rgb[b], f_ab, f_ba)
        visible = coverage > 0.25
        if visible.sum() >= 64:
            residual = charbonnier(
                np.abs(_luma(reconstructed) - y[m]), cfg.charbonnier_tau)[visible]
            cycle_by_phase[int(bundle.indices[a] % 2)].append(float(np.mean(residual)))
            cycle_p90.append(float(np.percentile(residual, 90)))

    all_comp = comp_by_phase[0] + comp_by_phase[1]
    all_cycle = cycle_by_phase[0] + cycle_by_phase[1]
    out["nr_self_comp_mean"] = float(np.mean(all_comp)) if all_comp else float("nan")
    out["nr_self_comp_phase_max"] = float(max(
        np.mean(v) for v in comp_by_phase if v)) if all_comp else float("nan")
    out["nr_self_cycle_mean"] = float(np.mean(all_cycle)) if all_cycle else float("nan")
    out["nr_self_cycle_p90"] = float(np.mean(cycle_p90)) if cycle_p90 else float("nan")

    # Fixed wall-clock lags keep 60 and 120 FPS results on comparable scales.
    short_pairs = _target_pairs(bundle.times, 1.0 / 60.0)
    medium_pairs = _target_pairs(bundle.times, 1.0 / 30.0)
    short_mean, short_p90 = _mc_residual(y, flows, short_pairs, cfg)
    medium_mean, medium_p90 = _mc_residual(y, flows, medium_pairs, cfg)
    out.update({
        "nr_mct_short_mean": short_mean,
        "nr_mct_short_p90": short_p90,
        "nr_mct_medium_mean": medium_mean,
        "nr_mct_medium_p90": medium_p90,
    })

    # Velocity/acceleration/jerk use the 1/60 s samples even for 120 FPS.
    speeds: list[float] = []
    for a, b in short_pairs:
        dt = max(float(bundle.times[b] - bundle.times[a]), 1e-6)
        speeds.append(float(np.median(flow_magnitude(flows.forward(a, b))) / dt))
    if len(speeds) >= 2:
        speed = np.asarray(speeds, np.float64)
        dt = 1.0 / 60.0
        accel = np.abs(np.diff(speed)) / dt
        denom_accel = np.median(speed) / dt + 1e-6
        out["nr_flow_accel_ratio"] = float(np.median(accel) / denom_accel)
        if len(accel) >= 2:
            jerk = np.abs(np.diff(accel)) / dt
            denom_jerk = np.median(speed) / (dt * dt) + 1e-6
            out["nr_flow_jerk_ratio"] = float(np.median(jerk) / denom_jerk)
        else:
            out["nr_flow_jerk_ratio"] = 0.0
    else:
        out["nr_flow_accel_ratio"] = float("nan")
        out["nr_flow_jerk_ratio"] = float("nan")

    # Phase-invariant alternation: swapping phase labels leaves |gap| intact.
    sharp = np.asarray([
        cv2.Laplacian(y[i], cv2.CV_32F).var() for i in range(n)
    ], np.float64)
    edges = np.asarray([
        np.mean(cv2.Canny(y[i].astype(np.uint8), 60, 160) > 0) for i in range(n)
    ], np.float64)
    parity = bundle.indices.astype(np.int64) % 2

    def phase_gap(values: np.ndarray) -> float:
        a, b = values[parity == 0], values[parity == 1]
        if not len(a) or not len(b):
            return 0.0
        return float(abs(np.mean(a) - np.mean(b)) /
                     (np.mean(np.abs(values)) + 1e-6))

    out["nr_phase_sharp_gap"] = phase_gap(sharp)
    out["nr_phase_edge_gap"] = phase_gap(edges)

    # Duplicate/freeze evidence at a fixed 1/60 s lag.
    frame_diffs = [
        float(np.mean(np.abs(y[b] - y[a]))) for a, b in short_pairs
    ]
    if frame_diffs:
        diffs = np.asarray(frame_diffs)
        out["nr_duplicate_fraction"] = float(np.mean(diffs < 0.75))
        out["nr_freeze_fraction"] = float(np.mean(diffs < 1.5))
    else:
        out["nr_duplicate_fraction"] = float("nan")
        out["nr_freeze_fraction"] = float("nan")

    # Screen-coordinate edge stability in typical HUD bands. Camera motion
    # does not explain changes here; this remains an honest UI/text proxy.
    h = bundle.height
    band = np.zeros((h, bundle.width), bool)
    band[:max(1, h // 5)] = True
    band[-max(1, h // 5):] = True
    edge_maps = [
        cv2.Canny(y[i].astype(np.uint8), 60, 160) > 0 for i in range(n)
    ]
    ui_changes = [
        float(np.mean(np.logical_xor(edge_maps[a], edge_maps[b])[band]))
        for a, b in short_pairs
    ]
    out["nr_ui_edge_instability"] = (
        float(np.mean(ui_changes)) if ui_changes else float("nan"))

    # Learned backend is optional and weak. Missing means unavailable, not
    # perfect quality; the mode fusion simply omits this category.
    if vqa_backend is not None:
        quality = float(vqa_backend.technical_quality(bundle.rgb))
        out["nr_learned_vqa_error"] = float(np.clip(1.0 - quality, 0.0, 1.0))
    return out
