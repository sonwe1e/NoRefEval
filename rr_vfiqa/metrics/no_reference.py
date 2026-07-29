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
from ..motion.flow_geometry import geometry_stats
from ..motion.occlusion import cycle_occlusion
from ..models.tracker_backend import KLTTracker
from ..sampling.temporal_plan import TemporalLagPlan, TemporalTriplet
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


def _triplet_features(
    triplets: tuple[TemporalTriplet, ...],
    bundle: FrameBundle,
    flows: WindowFlows,
    cfg: EvalConfig,
) -> tuple[float, float, float, float]:
    composition: list[float] = []
    cycle: list[float] = []
    cycle_p90: list[float] = []
    disocclusion: list[float] = []
    y = bundle.y_channel()
    for triplet in triplets:
        a, m, b = triplet.left, triplet.middle, triplet.right
        f_ab, f_ba = flows.pair(a, b)
        f_am, f_ma = flows.pair(a, m)
        f_mb, f_bm = flows.pair(m, b)
        occ = cycle_occlusion(
            f_ab, f_ba, threshold_px=cfg.occlusion_cycle_threshold)
        weight_a = occ.conf_ab * (1.0 - occ.occ_ab.astype(np.float32))
        weight_b = occ.conf_ba * (1.0 - occ.occ_ba.astype(np.float32))
        forward = composition_error(
            f_ab, f_am, f_mb, weight=weight_a,
            tau_px=cfg.charbonnier_tau)
        backward = composition_error(
            f_ba, f_bm, f_ma, weight=weight_b,
            tau_px=cfg.charbonnier_tau)
        composition.append(
            0.5 * (forward.weighted_mean + backward.weighted_mean))
        disocclusion.append(float(
            0.5 * (np.mean(occ.occ_ab) + np.mean(occ.occ_ba))))

        reconstructed, coverage = _reconstruct_mid(
            bundle.rgb[a], bundle.rgb[b], f_ab, f_ba,
            weight_a=weight_a, weight_b=weight_b)
        visible = coverage > 0.25
        if visible.sum() >= 64:
            residual = charbonnier(
                np.abs(_luma(reconstructed) - y[m]), cfg.charbonnier_tau)[visible]
            cycle.append(float(np.mean(residual)))
            cycle_p90.append(float(np.percentile(residual, 90)))
    return (
        float(np.mean(composition)) if composition else float("nan"),
        float(np.mean(cycle)) if cycle else float("nan"),
        float(np.mean(cycle_p90)) if cycle_p90 else float("nan"),
        float(np.mean(disocclusion)) if disocclusion else float("nan"),
    )


def _reconstruct_mid(
    a: np.ndarray,
    b: np.ndarray,
    f_ab: np.ndarray,
    f_ba: np.ndarray,
    *,
    weight_a: np.ndarray | None = None,
    weight_b: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    wa, ca = forward_splat(a.astype(np.float32), 0.5 * f_ab)
    wb, cb = forward_splat(b.astype(np.float32), 0.5 * f_ba)
    if weight_a is not None:
        projected_a, _ = forward_splat(
            weight_a.astype(np.float32), 0.5 * f_ab)
        ca = ca * np.clip(projected_a, 0.0, 1.0)
    if weight_b is not None:
        projected_b, _ = forward_splat(
            weight_b.astype(np.float32), 0.5 * f_ba)
        cb = cb * np.clip(projected_b, 0.0, 1.0)
    denom = ca + cb + 1e-6
    reconstructed = (
        wa * ca[..., None] + wb * cb[..., None]
    ) / denom[..., None]
    return reconstructed, np.clip(0.5 * denom, 0.0, 1.0)


def _alternation_energy(values: np.ndarray, indices: np.ndarray) -> float:
    centered = values.astype(np.float64) - np.median(values)
    sign = (-1.0) ** (indices.astype(np.int64) % 2)
    return float(abs(np.sum(sign * centered)) /
                 (np.sum(np.abs(centered)) + 1e-6))


def _track_smoothness(bundle: FrameBundle) -> dict[str, float]:
    """Camera-relative sparse-track acceleration, jerk and direction change."""
    grays = [
        cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in bundle.rgb
    ]
    points = cv2.goodFeaturesToTrack(
        grays[0],
        maxCorners=180,
        qualityLevel=0.01,
        minDistance=7,
        blockSize=7,
    )
    if points is None or len(points) < 12:
        return {
            "nr_track_points": float(0 if points is None else len(points)),
            "nr_track_accel_p90": float("nan"),
            "nr_track_jerk_p90": float("nan"),
            "nr_track_turn_p90": float("nan"),
        }
    tracks, visible = KLTTracker().track(
        grays, points.reshape(-1, 2).astype(np.float32))
    good = np.all(visible, axis=0)
    if good.sum() < 8:
        return {
            "nr_track_points": float(good.sum()),
            "nr_track_accel_p90": float("nan"),
            "nr_track_jerk_p90": float("nan"),
            "nr_track_turn_p90": float("nan"),
        }

    positions = tracks[:, good].astype(np.float64)
    # Remove the median camera/body translation at every instant.  Remaining
    # local trajectories expose swords, thin contours and foreground jitter.
    positions -= np.median(positions, axis=1, keepdims=True)
    dt = np.maximum(np.diff(bundle.times), 1e-6)
    velocity = np.diff(positions, axis=0) / dt[:, None, None]
    speed = np.linalg.norm(velocity, axis=2)
    speed_ref = np.median(speed, axis=0) + 1.0

    accel_ratio = np.zeros((0, positions.shape[1]), np.float64)
    if len(velocity) >= 2:
        accel_dt = np.maximum(
            0.5 * (dt[:-1] + dt[1:]), 1e-6)
        accel = np.diff(velocity, axis=0) / accel_dt[:, None, None]
        accel_ratio = (
            np.linalg.norm(accel, axis=2)
            * accel_dt[:, None]
            / speed_ref[None, :]
        )
    jerk_ratio = np.zeros((0, positions.shape[1]), np.float64)
    if len(accel_ratio) >= 2:
        # Difference of already dimensionless acceleration ratios is a stable
        # FPS-normalized jerk proxy.
        jerk_ratio = np.abs(np.diff(accel_ratio, axis=0))

    turns: list[np.ndarray] = []
    for t in range(len(velocity) - 1):
        a, b = velocity[t], velocity[t + 1]
        na, nb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
        moving = (na > 8.0) & (nb > 8.0)
        if moving.any():
            cosine = np.einsum("ij,ij->i", a[moving], b[moving]) / (
                na[moving] * nb[moving] + 1e-6)
            turns.append(np.clip(1.0 - cosine, 0.0, 2.0))

    return {
        "nr_track_points": float(good.sum()),
        "nr_track_accel_p90": (
            float(np.percentile(accel_ratio, 90))
            if accel_ratio.size else float("nan")),
        "nr_track_jerk_p90": (
            float(np.percentile(jerk_ratio, 90))
            if jerk_ratio.size else float("nan")),
        "nr_track_turn_p90": (
            float(np.percentile(np.concatenate(turns), 90))
            if turns else float("nan")),
    }


def _tile_motion_dynamics(
    bundle: FrameBundle,
    flows: WindowFlows,
    pairs: tuple[tuple[int, int], ...],
    grid: int = 4,
) -> dict[str, float]:
    vectors: list[np.ndarray] = []
    centers: list[float] = []
    for a, b in pairs:
        dt = max(float(bundle.times[b] - bundle.times[a]), 1e-6)
        field = flows.forward(a, b)
        h, w = field.shape[:2]
        tiles = []
        for gy in range(grid):
            for gx in range(grid):
                tile = field[
                    gy * h // grid:(gy + 1) * h // grid,
                    gx * w // grid:(gx + 1) * w // grid,
                ]
                tiles.append(np.median(tile.reshape(-1, 2), axis=0) / dt)
        vectors.append(np.asarray(tiles, np.float64))
        centers.append(0.5 * float(bundle.times[a] + bundle.times[b]))
    if len(vectors) < 2:
        return {
            "nr_flow_accel_ratio": float("nan"),
            "nr_flow_jerk_ratio": float("nan"),
            "nr_tile_accel_p90": float("nan"),
            "nr_tile_jerk_p90": float("nan"),
            "nr_local_reversal_fraction": float("nan"),
        }
    velocity = np.stack(vectors)
    flow_times = np.asarray(centers, np.float64)
    acceleration = np.gradient(velocity, flow_times, axis=0)
    jerk = (
        np.gradient(acceleration, flow_times, axis=0)
        if len(velocity) >= 3 else np.zeros_like(acceleration))
    speed = np.linalg.norm(velocity, axis=2)
    dt_sample = max(float(np.median(np.diff(flow_times))), 1e-6)
    speed_ref = np.median(speed, axis=0) + 1.0
    accel_ratio = np.linalg.norm(acceleration, axis=2) * dt_sample / speed_ref
    jerk_ratio = (
        np.linalg.norm(jerk, axis=2) * dt_sample * dt_sample / speed_ref)
    a, b = velocity[:-1], velocity[1:]
    dot = np.sum(a * b, axis=2)
    moving = (np.linalg.norm(a, axis=2) > 8.0) & (
        np.linalg.norm(b, axis=2) > 8.0)
    reversal = float(np.mean(dot[moving] < 0.0)) if moving.any() else 0.0
    return {
        "nr_flow_accel_ratio": float(np.median(accel_ratio)),
        "nr_flow_jerk_ratio": float(np.median(jerk_ratio)),
        "nr_tile_accel_p90": float(np.percentile(accel_ratio, 90)),
        "nr_tile_jerk_p90": float(np.percentile(jerk_ratio, 90)),
        "nr_local_reversal_fraction": reversal,
    }


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

    lag_plan = TemporalLagPlan.build(bundle.times)
    native_comp, native_cycle, native_p90, native_occ = _triplet_features(
        lag_plan.native_triplets, bundle, flows, cfg)
    common_comp, common_cycle, common_p90, common_occ = _triplet_features(
        lag_plan.common_triplets, bundle, flows, cfg)
    out.update({
        "nr_native_self_comp": native_comp,
        "nr_native_self_cycle": native_cycle,
        "nr_native_self_cycle_p90": native_p90,
        "nr_native_disocclusion_risk": native_occ,
        "nr_common_self_comp": common_comp,
        "nr_common_self_cycle": common_cycle,
        "nr_common_self_cycle_p90": common_p90,
        "nr_common_disocclusion_risk": common_occ,
        # Compatibility aliases now explicitly point at the common time scale.
        "nr_self_comp_mean": common_comp,
        "nr_self_comp_phase_max": common_comp,
        "nr_self_cycle_mean": common_cycle,
        "nr_self_cycle_p90": common_p90,
    })

    native_pairs = lag_plan.native
    short_pairs = lag_plan.lag_1_60
    medium_pairs = lag_plan.lag_1_30
    native_mean, native_p90_mct = _mc_residual(
        y, flows, list(native_pairs), cfg)
    short_mean, short_p90 = _mc_residual(y, flows, short_pairs, cfg)
    medium_mean, medium_p90 = _mc_residual(y, flows, medium_pairs, cfg)
    out.update({
        "nr_mct_native_mean": native_mean,
        "nr_mct_native_p90": native_p90_mct,
        "nr_mct_1_60_mean": short_mean,
        "nr_mct_1_60_p90": short_p90,
        "nr_mct_1_30_mean": medium_mean,
        "nr_mct_1_30_p90": medium_p90,
        "nr_mct_short_mean": short_mean,
        "nr_mct_short_p90": short_p90,
        "nr_mct_medium_mean": medium_mean,
        "nr_mct_medium_p90": medium_p90,
    })
    out.update(_tile_motion_dynamics(bundle, flows, short_pairs))

    # Local flow geometry after removing the robust global translation proxy.
    # Negative/low Jacobian determinants expose folding and tearing without an
    # endpoint reference.
    geometry: list[dict[str, float]] = []
    for a, b in short_pairs:
        field = flows.forward(a, b)
        global_translation = np.median(field.reshape(-1, 2), axis=0)
        geometry.append(geometry_stats(field - global_translation))
    for source_key, output_key in (
        ("flow_fold_frac", "nr_flow_fold_fraction"),
        ("flow_jdet_low_frac", "nr_flow_jdet_low_fraction"),
        ("flow_div_std", "nr_flow_divergence_std"),
        ("flow_curl_std", "nr_flow_curl_std"),
    ):
        values = [
            item[source_key] for item in geometry
            if np.isfinite(item.get(source_key, float("nan")))
        ]
        out[output_key] = (
            float(np.mean(values)) if values else float("nan"))

    out.update(_track_smoothness(bundle))

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
    out["nr_phase_sharp_energy"] = _alternation_energy(
        sharp, bundle.indices)
    out["nr_phase_edge_energy"] = _alternation_energy(
        edges, bundle.indices)

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

    native_diffs = np.asarray([
        float(np.mean(np.abs(y[i + 1] - y[i]))) for i in range(n - 1)
    ], np.float64)
    out["nr_native_duplicate_fraction"] = (
        float(np.mean(native_diffs < 0.75))
        if len(native_diffs) else float("nan"))
    out["nr_native_freeze_fraction"] = (
        float(np.mean(native_diffs < 1.5))
        if len(native_diffs) else float("nan"))
    out["nr_duplicate_native"] = out["nr_native_duplicate_fraction"]
    out["nr_freeze_native"] = out["nr_native_freeze_fraction"]

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
    native_edge_changes = [
        float(np.mean(np.logical_xor(edge_maps[a], edge_maps[b])))
        for a, b in native_pairs
    ]
    common_edge_changes = [
        float(np.mean(np.logical_xor(edge_maps[a], edge_maps[b])))
        for a, b in short_pairs
    ]
    out["nr_edge_instability_native"] = (
        float(np.mean(native_edge_changes))
        if native_edge_changes else float("nan"))
    out["nr_edge_instability_1_60"] = (
        float(np.mean(common_edge_changes))
        if common_edge_changes else float("nan"))

    # Text-like screen-space regions: dense persistent strokes in HUD bands.
    edge_stack = np.stack(edge_maps).astype(np.float32)
    persistent = edge_stack.mean(0) >= 0.35
    density = cv2.blur(
        edge_stack[0], (15, 15), borderType=cv2.BORDER_REFLECT)
    text_roi = band & persistent & (density >= 0.10)
    text_changes = [
        float(np.mean(np.logical_xor(edge_maps[a], edge_maps[b])[text_roi]))
        for a, b in short_pairs
        if text_roi.sum() >= 32
    ]
    out["nr_text_stroke_instability"] = (
        float(np.mean(text_changes)) if text_changes else float("nan"))

    # Learned backend is optional and weak. Missing means unavailable, not
    # perfect quality; the mode fusion simply omits this category.
    if vqa_backend is not None:
        quality = float(vqa_backend.technical_quality(bundle.rgb))
        out["nr_learned_vqa_error"] = float(np.clip(1.0 - quality, 0.0, 1.0))
    return out
