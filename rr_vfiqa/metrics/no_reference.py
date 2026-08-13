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
from ..motion.flow_geometry import geometry_stats, jacobian_det
from ..motion.global_camera_motion import dense_affine_residual
from ..motion.occlusion import cycle_occlusion
from ..models.tracker_backend import KLTTracker
from ..sampling.cheap_scan import MOVING_PIXEL_DELTA
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
    from ..imutils import luma

    return luma(rgb)


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


def _camera_stabilize_tracks(positions: np.ndarray) -> np.ndarray:
    """Map tracks back to the first-frame grid using robust global motion."""
    reference = positions[0].astype(np.float32)
    stabilized = np.empty_like(positions, dtype=np.float64)
    stabilized[0] = reference
    for index in range(1, len(positions)):
        current = positions[index].astype(np.float32)
        translation = np.median(current - reference, axis=0)
        transform = np.asarray([
            [1.0, 0.0, translation[0]],
            [0.0, 1.0, translation[1]],
            [0.0, 0.0, 1.0],
        ], np.float64)
        projected = reference + translation
        best_residual = float(np.median(np.linalg.norm(
            current - projected, axis=1)))

        affine, inliers = cv2.estimateAffine2D(
            reference,
            current,
            method=cv2.RANSAC,
            ransacReprojThreshold=2.5,
            maxIters=1000,
            confidence=0.99,
            refineIters=10,
        )
        if affine is not None and inliers is not None and np.mean(inliers) >= 0.50:
            affine_h = np.vstack([affine, [0.0, 0.0, 1.0]])
            affine_projected = cv2.transform(
                reference[None], affine)[0]
            affine_residual = float(np.median(np.linalg.norm(
                current - affine_projected, axis=1)))
            if affine_residual < 0.90 * best_residual:
                transform = affine_h
                best_residual = affine_residual

        if len(reference) >= 12 and best_residual > 1.5:
            homography, inliers = cv2.findHomography(
                reference,
                current,
                method=cv2.RANSAC,
                ransacReprojThreshold=2.5,
                maxIters=1000,
                confidence=0.99,
            )
            if (homography is not None and inliers is not None
                    and np.mean(inliers) >= 0.55):
                homography_projected = cv2.perspectiveTransform(
                    reference[None], homography)[0]
                homography_residual = float(np.median(np.linalg.norm(
                    current - homography_projected, axis=1)))
                if homography_residual < 0.90 * best_residual:
                    transform = homography

        try:
            inverse = np.linalg.inv(transform)
            stabilized[index] = cv2.perspectiveTransform(
                current[None], inverse.astype(np.float64))[0]
        except np.linalg.LinAlgError:
            stabilized[index] = current - translation
    return stabilized


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
            "nr_track_initial_points": float(0 if points is None else len(points)),
            "nr_persistent_track_fraction": float("nan"),
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
            "nr_track_initial_points": float(len(points)),
            "nr_persistent_track_fraction": float(good.sum() / len(points)),
            "nr_track_accel_p90": float("nan"),
            "nr_track_jerk_p90": float("nan"),
            "nr_track_turn_p90": float("nan"),
        }

    positions = _camera_stabilize_tracks(
        tracks[:, good].astype(np.float64))
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
        "nr_track_initial_points": float(len(points)),
        "nr_persistent_track_fraction": float(good.sum() / len(points)),
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


# USERPLAN §7 flow-reliability thresholds.
FLOW_VALID_FRACTION_MIN = 0.25     # below this, motion evidence is not trusted
_PHOTOMETRIC_SUPPORT_THRESHOLD = 8.0   # luma delta explained by warping
_APPEARANCE_CHANGE_THRESHOLD = 12.0    # luma delta NOT explained by warping


def _visibility_masks(
    y: np.ndarray,
    flows: WindowFlows,
    pairs: list[tuple[int, int]],
    cfg: EvalConfig,
) -> list[np.ndarray]:
    """Per-pair forward-backward cycle-consistency visibility masks."""
    masks: list[np.ndarray] = []
    for a, b in pairs:
        f_ab, f_ba = flows.pair(a, b)
        cycle = flow_magnitude(warp_flow(f_ab, f_ba))
        masks.append(cycle < cfg.occlusion_cycle_threshold)
    return masks


def _flow_reliability(
    y: np.ndarray,
    flows: WindowFlows,
    pairs: list[tuple[int, int]],
    cfg: EvalConfig,
) -> dict[str, float]:
    """Flow validity / photometric support / effect-transience per window.

    USERPLAN §7: particles, explosions, flashes and occlusion make optical
    flow unreliable; those appearance changes must raise *uncertainty*, not
    motion error.  ``nr_effect_transient_fraction`` is the share of pixels
    whose change is NOT explained by the flow (new/vanishing content) — it is
    reported as uncertainty, never folded into accel/jerk/fold.
    """
    cycle_thr = float(cfg.occlusion_cycle_threshold)
    valid: list[float] = []
    fwbw: list[float] = []
    photo: list[float] = []
    trans: list[float] = []
    for a, b in pairs:
        f_ab, f_ba = flows.pair(a, b)
        warped = backward_warp(y[b], f_ab)
        cycle = flow_magnitude(warp_flow(f_ab, f_ba))
        vis = cycle < cycle_thr
        if vis.size == 0:
            continue
        valid.append(float(np.mean(vis)))
        fwbw.append(float(np.clip(
            1.0 - float(np.median(cycle)) / (cycle_thr + 1e-6), 0.0, 1.0)))
        res = np.abs(warped - y[a])
        photo.append(float(np.mean(res[vis] < _PHOTOMETRIC_SUPPORT_THRESHOLD)))
        trans.append(float(np.mean(res > _APPEARANCE_CHANGE_THRESHOLD)))

    def _m(vals: list[float]) -> float:
        return float(np.mean(vals)) if vals else float("nan")

    return {
        "nr_flow_valid_fraction": _m(valid),
        "nr_forward_backward_consistency": _m(fwbw),
        "nr_photometric_support_fraction": _m(photo),
        "nr_effect_transient_fraction": _m(trans),
    }


def _tile_motion_dynamics(
    bundle: FrameBundle,
    flows: WindowFlows,
    pairs: tuple[tuple[int, int], ...],
    grid: int = 4,
    visibility_masks: list[np.ndarray] | None = None,
) -> dict[str, float]:
    n_tiles = grid * grid
    vectors: list[np.ndarray] = []
    centers: list[float] = []
    for i, (a, b) in enumerate(pairs):
        dt = max(float(bundle.times[b] - bundle.times[a]), 1e-6)
        field = flows.forward(a, b)
        h, w = field.shape[:2]
        mask = (visibility_masks[i] if visibility_masks
                and i < len(visibility_masks) else None)
        # USERPLAN P2: remove affine camera motion (rotation/zoom/translation)
        # before measuring per-tile dynamics, so camera pans do not read as
        # local acceleration/jerk.
        residual = dense_affine_residual(field)
        tiles = np.full((n_tiles, 2), np.nan, np.float64)
        for ti, (gy, gx) in enumerate(
                (gy, gx) for gy in range(grid) for gx in range(grid)):
            tile = residual[
                gy * h // grid:(gy + 1) * h // grid,
                gx * w // grid:(gx + 1) * w // grid,
            ].reshape(-1, 2)
            if mask is not None:
                tm = mask[
                    gy * h // grid:(gy + 1) * h // grid,
                    gx * w // grid:(gx + 1) * w // grid,
                ].reshape(-1)
                if tm.sum() < 32:
                    continue          # unreliable tile -> no motion evidence
                tile = tile[tm]
            tiles[ti] = np.median(tile, axis=0) / dt
        vectors.append(tiles)
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
    # nan-aware: unreliable tiles (NaN) must not poison the aggregate.
    speed_ref = np.nanmedian(speed, axis=0) + 1.0
    accel_ratio = np.linalg.norm(acceleration, axis=2) * dt_sample / speed_ref
    jerk_ratio = (
        np.linalg.norm(jerk, axis=2) * dt_sample * dt_sample / speed_ref)
    a, b = velocity[:-1], velocity[1:]
    dot = np.sum(a * b, axis=2)
    moving = (np.linalg.norm(a, axis=2) > 8.0) & (
        np.linalg.norm(b, axis=2) > 8.0)
    reversal = (
        float(np.mean(dot[moving] < 0.0)) if moving.any() else 0.0)
    return {
        "nr_flow_accel_ratio": (
            float(np.nanmedian(accel_ratio))
            if np.isfinite(accel_ratio).any() else float("nan")),
        "nr_flow_jerk_ratio": (
            float(np.nanmedian(jerk_ratio))
            if np.isfinite(jerk_ratio).any() else float("nan")),
        "nr_tile_accel_p90": (
            float(np.nanpercentile(accel_ratio, 90))
            if np.isfinite(accel_ratio).any() else float("nan")),
        "nr_tile_jerk_p90": (
            float(np.nanpercentile(jerk_ratio, 90))
            if np.isfinite(jerk_ratio).any() else float("nan")),
        "nr_local_reversal_fraction": reversal,
    }


def _screen_static_ui_mask(
    edge_maps: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Find compact screen-static components across all four borders/screen.

    Large connected regions are rejected as likely static scene structure;
    compact persistent components are retained even near the screen center so
    floating labels and prompts are not excluded by a top/bottom-only prior.
    """
    edge_stack = np.stack(edge_maps).astype(np.float32)
    persistent = edge_stack.mean(0) >= 0.60
    h, w = persistent.shape
    border = np.zeros((h, w), bool)
    band_y = max(1, h // 5)
    band_x = max(1, w // 6)
    border[:band_y] = True
    border[-band_y:] = True
    border[:, :band_x] = True
    border[:, -band_x:] = True

    connected = cv2.dilate(
        persistent.astype(np.uint8), np.ones((5, 5), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        connected, connectivity=8)
    retained = np.zeros_like(connected)
    frame_area = h * w
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < 12 or area > 0.08 * frame_area:
            continue
        component = labels == label
        # Border HUD is expected; compact full-screen components cover
        # floating text/prompts without admitting broad static scenery.
        if np.any(component & border) or area <= 0.02 * frame_area:
            retained[component] = 1
    ui_mask = cv2.dilate(
        retained, np.ones((3, 3), np.uint8)).astype(bool)
    if ui_mask.sum() < 32:
        ui_mask = border & cv2.dilate(
            persistent.astype(np.uint8), np.ones((3, 3), np.uint8)
        ).astype(bool)
    return ui_mask, edge_stack


def _ui_reliability(
    edge_maps: list[np.ndarray],
    frame_shape: tuple[int, int],
) -> tuple[dict[str, float], np.ndarray]:
    """UI detection confidence / persistence / screen motion (USERPLAN §8).

    A real HUD persists at fixed screen coordinates with a compact area.  A
    flash or particle burst does not: its per-frame UI mask is erratic and
    moves around.  ``ui_screen_motion`` is the mean 1-IoU of consecutive
    per-frame UI masks; ``ui_detection_confidence`` is the fraction of frames
    showing a compact, non-trivial mask.  Both gate whether the UI instability
    features are trustworthy at all.
    """
    h, w = frame_shape
    frame_area = max(h * w, 1)
    per_frame = [_screen_static_ui_mask([em])[0] for em in edge_maps]
    areas = np.asarray([m.sum() for m in per_frame], np.float64)
    ious: list[float] = []
    for t in range(len(per_frame) - 1):
        inter = int((per_frame[t] & per_frame[t + 1]).sum())
        union = int((per_frame[t] | per_frame[t + 1]).sum())
        ious.append(inter / (union + 1e-6))
    screen_motion = 1.0 - (float(np.mean(ious)) if ious else 1.0)
    has_ui = areas >= max(32.0, 0.0005 * frame_area)
    persistence = (float(np.mean(has_ui)) if len(areas) else 0.0)
    median_area_ratio = float(np.median(areas)) / frame_area
    return {
        "ui_detection_confidence": round(persistence, 4),
        "ui_screen_motion": round(screen_motion, 4),
        "ui_component_area_ratio": round(median_area_ratio, 6),
    }, np.asarray(per_frame)


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
    # USERPLAN §7: flow reliability gate.  Particles, flashes, explosions and
    # occlusion make optical flow unreliable; those appearance changes raise
    # uncertainty, not motion error.  Reliable-region tile dynamics are
    # computed only over cycle-consistent pixels.
    vis_masks = _visibility_masks(y, flows, list(short_pairs), cfg)
    out.update(_flow_reliability(y, flows, list(short_pairs), cfg))
    out.update(_tile_motion_dynamics(
        bundle, flows, short_pairs, visibility_masks=vis_masks))
    out.update(_track_smoothness(bundle))

    # Local flow geometry after removing the affine camera motion (USERPLAN P2).
    # Negative/low Jacobian determinants expose folding and tearing without an
    # endpoint reference; using the affine residual stops camera rotation/zoom
    # from being mistaken for local folding.  Geometry is computed on reliable
    # pixels only (USERPLAN §7).
    geometry: list[dict[str, float]] = []
    for i, (a, b) in enumerate(short_pairs):
        field = flows.forward(a, b)
        residual = dense_affine_residual(field)
        mask = (vis_masks[i] if i < len(vis_masks) else None)
        geometry.append(geometry_stats(residual, mask=mask))
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

    # USERPLAN §7: separate trackable-motion error from appearance-change
    # uncertainty.  The flow-valid / photometric support fractions describe
    # how much of the frame's change is actually *motion*; the rest is
    # uncertainty that must NOT be scored as smoothness failure.
    vf = out.get("nr_flow_valid_fraction", float("nan"))
    if np.isfinite(vf) and vf < FLOW_VALID_FRACTION_MIN:
        # Motion evidence not trustworthy -> N/A (like phase gating).
        out["nr_trackable_motion_error"] = float("nan")
        out["nr_flow_reliable"] = 0.0
    else:
        out["nr_trackable_motion_error"] = float(np.nanmax([
            out.get(k, float("nan")) for k in (
                "nr_flow_accel_ratio", "nr_flow_jerk_ratio",
                "nr_flow_fold_fraction", "nr_flow_jdet_low_fraction",
                "nr_flow_divergence_std", "nr_flow_curl_std",
            ) if np.isfinite(out.get(k, float("nan")))]) or float("nan"))
        out["nr_flow_reliable"] = 1.0
    out["nr_appearance_change_uncertainty"] = out.get(
        "nr_effect_transient_fraction", float("nan"))

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

    # USERPLAN P0-R2: raw (non-motion-compensated) frame diff at 1/60s lag,
    # used by the cadence motion gate to distinguish real static scenes from
    # collapsed cadence. MCT residual is motion-compensated and can be low for
    # smooth motion even when real motion is large.
    if frame_diffs:
        out["nr_raw_diff_1_60"] = float(np.mean(diffs))
    else:
        out["nr_raw_diff_1_60"] = float("nan")

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

    # --- USERPLAN §4.1: FPS-adaptive native/parent lags ---------------------
    # For a 60 FPS input, native = 1/60 s and parent = 1/30 s (t -> t+2); for a
    # 120 FPS input, native = 1/120 s and parent = 1/60 s.  The cadence motion
    # gate MUST use the parent scale, never the fixed 1/60 s ``nr_raw_diff_1_60``
    # (which equals native for 60 FPS and is therefore not "longer-scale").
    if len(native_diffs):
        out["nr_raw_diff_native"] = float(np.mean(native_diffs))
    else:
        out["nr_raw_diff_native"] = float("nan")

    parent_pairs = getattr(lag_plan, "parent", ())
    parent_diffs = [
        float(np.mean(np.abs(y[b] - y[a]))) for a, b in parent_pairs
    ]
    parent_moving: list[float] = []
    for a, b in parent_pairs:
        parent_moving.append(float(np.mean(
            np.abs(y[b] - y[a]) > MOVING_PIXEL_DELTA)))
    if parent_diffs:
        pd = np.asarray(parent_diffs, np.float64)
        out["nr_raw_diff_parent"] = float(np.mean(pd))
        out["nr_parent_motion_p90"] = float(np.percentile(pd, 90))
    else:
        out["nr_raw_diff_parent"] = float("nan")
        out["nr_parent_motion_p90"] = float("nan")
    out["nr_parent_moving_pixel_fraction"] = (
        float(np.mean(parent_moving)) if parent_moving else float("nan"))

    # --- USERPLAN §4.2: parity asymmetry of native adjacent diffs -----------
    # d_t = mean |Y_{t+1} - Y_t| (in moving regions implicitly, since static
    # pixels contribute ~0).  Split by parity of t.  A 30->60 duplicate stream
    # has d alternating ~0 / real-motion, so |median(d_even) - median(d_odd)|
    # is large; a true 60 FPS combat clip has roughly uniform d, so the
    # asymmetry stays small.
    d_all = np.asarray(native_diffs, np.float64)
    k = bundle.indices.astype(np.int64)[: len(d_all)]
    even_sel = (k % 2 == 0)
    d_even = d_all[even_sel]
    d_odd = d_all[~even_sel]
    if len(d_even) and len(d_odd):
        med_e, med_o = float(np.median(d_even)), float(np.median(d_odd))
        denom = (med_e + med_o + 1e-6)
        out["nr_phase_asymmetry"] = float(abs(med_e - med_o) / denom)
        out["nr_phase_gap_signed"] = float((med_o - med_e) / denom)
        # +1 => even phase carries less new content; -1 => odd phase deficient;
        # 0 => no usable parity structure (numeric so scalars stay float).
        out["nr_phase_deficient_sign"] = (
            float(np.sign(med_e - med_o))
            if abs(med_e - med_o) > 1e-6 else 0.0)
    else:
        out["nr_phase_asymmetry"] = float("nan")
        out["nr_phase_gap_signed"] = float("nan")
        out["nr_phase_deficient_sign"] = 0.0
    # Alternation energy of the adjacent-diff sequence: the previously-referenced
    # but never-computed ``nr_native_motion_alternation`` signal used by the
    # cadence gate as parity-structure corroboration.
    out["nr_native_motion_alternation"] = (
        _alternation_energy(d_all, k) if len(d_all) else float("nan"))

    # Screen-coordinate edge stability in typical HUD bands. Camera motion
    # does not explain changes here; this remains an honest UI/text proxy.
    edge_maps = [
        cv2.Canny(y[i].astype(np.uint8), 60, 160) > 0 for i in range(n)
    ]
    # USERPLAN §8: only trust the UI instability signals when the "UI" is a
    # compact structure that persists at FIXED screen coordinates.  Combat
    # flashes and particles produce erratic, moving masks -> rejected.
    ui_rel, _ui_per_frame = _ui_reliability(edge_maps, (y.shape[1], y.shape[2]))
    native_dt = (float(np.median(np.diff(bundle.times)))
                 if len(bundle.times) > 1 else 1.0 / 60.0)
    ui_rel["ui_persistence_seconds"] = round(
        ui_rel["ui_detection_confidence"] * max(n - 1, 0) * native_dt, 4)
    out.update(ui_rel)
    # USERPLAN §8: reject what is clearly NOT a HUD.  A persistent compact
    # mask is trusted and scored (even one that drifts — a real UI defect must
    # be detected, not hidden); a mask that covers most of the frame (a
    # full-screen flash / particle burst) or is essentially absent is rejected.
    # ``ui_screen_motion`` stays in the report as diagnostic metadata only —
    # per-frame masks are too noisy to hard-gate on.
    ui_trustworthy = bool(
        ui_rel["ui_detection_confidence"] >= 0.5
        and 0.0002 <= ui_rel["ui_component_area_ratio"] <= 0.25)
    out["nr_ui_reliable"] = 1.0 if ui_trustworthy else 0.0

    ui_mask, edge_stack = _screen_static_ui_mask(edge_maps)
    ui_changes = [
        float(np.mean(np.logical_xor(
            edge_maps[a], edge_maps[b])[ui_mask]))
        for a, b in short_pairs
        if ui_mask.sum() >= 32
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
    persistent = edge_stack.mean(0) >= 0.35
    density = cv2.blur(
        edge_stack.mean(0), (15, 15), borderType=cv2.BORDER_REFLECT)
    text_roi = ui_mask & persistent & (density >= 0.10)
    text_changes = [
        float(np.mean(np.logical_xor(edge_maps[a], edge_maps[b])[text_roi]))
        for a, b in short_pairs
        if text_roi.sum() >= 32
    ]
    out["nr_text_stroke_instability"] = (
        float(np.mean(text_changes)) if text_changes else float("nan"))
    component_counts = []
    for edges in edge_maps:
        count, _, stats, _ = cv2.connectedComponentsWithStats(
            (edges & ui_mask).astype(np.uint8), connectivity=8)
        component_counts.append(sum(
            1 for label in range(1, count)
            if stats[label, cv2.CC_STAT_AREA] >= 3))
    counts = np.asarray(component_counts, np.float64)
    out["nr_ui_component_instability"] = float(
        np.std(counts) / (np.mean(counts) + 1.0))

    # USERPLAN §8: unreliable UI evidence is N/A, not a score.  This keeps
    # combat flashes / particles from producing a misleading UI instability.
    if not ui_trustworthy:
        for _key in ("nr_ui_edge_instability", "nr_text_stroke_instability",
                     "nr_ui_component_instability"):
            out[_key] = float("nan")

    # Learned backend is optional and weak. Missing means unavailable, not
    # perfect quality; the mode fusion simply omits this category.
    if vqa_backend is not None:
        quality = float(vqa_backend.technical_quality(bundle.rgb))
        out["nr_learned_vqa_error"] = float(np.clip(1.0 - quality, 0.0, 1.0))
    return out


# ---------------------------------------------------------------------------
# Dense error maps (USERPLAN §8).  Only computed for the highest-risk windows
# (the scalar ``compute_window`` runs for every window): these are what the
# report renders as heatmaps and what feeds issue-level spatial boxes.
# ---------------------------------------------------------------------------
def compute_window_maps(bundle: FrameBundle, flows: WindowFlows, cfg: EvalConfig,
                        ) -> dict[str, np.ndarray]:
    """Return named (H, W) diagnostic fields for one window.

    The set matches USERPLAN §8 for the no-reference mode:

    * ``mct_residual_map``     — luma motion-compensation residual (1/60 s lag);
    * ``composition_error_map`` — forward+backward composition error, max over
      the two directions, scaled to the working resolution;
    * ``self_cycle_residual_map`` — |warped_mid - true_mid| on the center frame;
    * ``flow_fold_map``        — ``max(0, 1 - jacobian_det)`` after removing the
                              global translation proxy (folding/tearing);
    * ``jacobian_determinant_map`` — raw ``det(I + ∇F)`` for the same field;
    * ``ui_edge_instability_map`` — per-pixel XOR of consecutive edge maps,
                              restricted to the screen-static UI mask;
    * ``phase_sharpness_map``   — per-pixel Laplacian variance, odd vs even;
    * ``duplicate_frame_indicator`` — per-pixel |Δluma| at the native lag.
    """
    n = len(bundle.rgb)
    if n < 5:
        return {}
    y = bundle.y_channel()
    h, w = y.shape[1], y.shape[2]
    out: dict[str, np.ndarray] = {}

    lag_plan = TemporalLagPlan.build(bundle.times)
    short_pairs = lag_plan.lag_1_60
    native_pairs = lag_plan.native

    # --- MCT residual map (1/60 s) ----------------------------------------
    res_fields: list[np.ndarray] = []
    for a, b in short_pairs:
        f_ab = flows.pair(a, b)[0]
        warped = backward_warp(y[b], f_ab)
        res_fields.append(charbonnier(np.abs(warped - y[a]),
                                      cfg.charbonnier_tau).astype(np.float32))
    out["mct_residual_map"] = _stack_mean(res_fields, h, w)

    # --- composition + self-cycle residual maps ---------------------------
    comp_fields: list[np.ndarray] = []
    cycle_fields: list[np.ndarray] = []
    for triplet in lag_plan.native_triplets:
        a, m, b = triplet.left, triplet.middle, triplet.right
        f_ab, f_ba = flows.pair(a, b)
        f_am, f_ma = flows.pair(a, m)
        f_mb, f_bm = flows.pair(m, b)
        occ = cycle_occlusion(f_ab, f_ba,
                              threshold_px=cfg.occlusion_cycle_threshold)
        weight_a = occ.conf_ab * (1.0 - occ.occ_ab.astype(np.float32))
        weight_b = occ.conf_ba * (1.0 - occ.occ_ba.astype(np.float32))
        fwd = composition_error(f_ab, f_am, f_mb, weight=weight_a,
                                tau_px=cfg.charbonnier_tau)
        bwd = composition_error(f_ba, f_bm, f_ma, weight=weight_b,
                                tau_px=cfg.charbonnier_tau)
        comp_fields.append(np.maximum(fwd.error_map, bwd.error_map))
        reconstructed, coverage = _reconstruct_mid(
            bundle.rgb[a], bundle.rgb[b], f_ab, f_ba,
            weight_a=weight_a, weight_b=weight_b)
        visible = coverage > 0.25
        field = np.zeros((h, w), np.float32)
        if visible.sum() >= 64:
            field[visible] = charbonnier(
                np.abs(_luma(reconstructed[visible]) - y[m][visible]),
                cfg.charbonnier_tau).astype(np.float32)
        cycle_fields.append(field)
    out["composition_error_map"] = _stack_mean(comp_fields, h, w)
    out["self_cycle_residual_map"] = _stack_mean(cycle_fields, h, w)

    # --- flow-fold / jacobian maps (1/60 s, affine camera residual) --------
    fold_fields: list[np.ndarray] = []
    jdet_fields: list[np.ndarray] = []
    for a, b in short_pairs:
        residual = dense_affine_residual(flows.forward(a, b))
        jdet = jacobian_det(residual)
        jdet_fields.append(jdet.astype(np.float32))
        fold_fields.append(np.clip(1.0 - jdet, 0.0, None).astype(np.float32))
    out["flow_fold_map"] = _stack_mean(fold_fields, h, w)
    out["jacobian_determinant_map"] = _stack_mean(jdet_fields, h, w)

    # --- UI edge instability map ------------------------------------------
    edge_maps = [cv2.Canny(y[i].astype(np.uint8), 60, 160) > 0 for i in range(n)]
    ui_mask, _ = _screen_static_ui_mask(edge_maps)
    ui_fields: list[np.ndarray] = []
    for a, b in short_pairs:
        field = np.logical_xor(edge_maps[a], edge_maps[b]).astype(np.float32)
        field[~ui_mask] = 0.0
        ui_fields.append(field)
    out["ui_edge_instability_map"] = _stack_mean(ui_fields, h, w)

    # --- phase sharpness map (per-pixel odd vs even Laplacian energy) -----
    # USERPLAN P1: a blur defect shows as lower sharpness on generated (odd)
    # frames.  Compute per-pixel Laplacian magnitude on each frame, then map
    # the odd/even difference so the heat concentrates on blurred regions.
    lap = np.stack([np.abs(cv2.Laplacian(y[i], cv2.CV_32F)) for i in range(n)])
    parity = bundle.indices.astype(np.int64) % 2
    even_mean = lap[parity == 0].mean(0)
    odd_mean = lap[parity == 1].mean(0)
    gap = np.abs(odd_mean - even_mean).astype(np.float32)
    gmax = max(float(np.percentile(gap, 99)), 1e-3)
    out["phase_sharpness_map"] = np.clip(gap / gmax, 0.0, 1.0).astype(np.float32)

    # --- duplicate frame indicator (native lag) ---------------------------
    # USERPLAN P1: high value = high duplication risk (hot = bad).  Raw frame
    # difference is near 0 for duplicates, so invert: risk = exp(-diff / sigma).
    dup_fields: list[np.ndarray] = []
    for a, b in native_pairs:
        diff = np.abs(y[b] - y[a]).astype(np.float32)
        dup_fields.append(np.exp(-diff / 8.0))
    out["duplicate_frame_indicator"] = _stack_mean(dup_fields, h, w)

    return out


def _stack_mean(fields: list[np.ndarray], h: int, w: int) -> np.ndarray:
    if not fields:
        return np.zeros((h, w), np.float32)
    # fields may be at a different resolution than (h, w) when flows run at the
    # working width; resize back to the luma grid so every map lines up.
    resized = []
    for f in fields:
        if f.shape[:2] != (h, w):
            f = cv2.resize(f, (w, h), interpolation=cv2.INTER_LINEAR)
        resized.append(f)
    return np.mean(resized, axis=0).astype(np.float32)
