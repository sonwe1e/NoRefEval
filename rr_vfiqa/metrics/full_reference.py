"""Spatial and temporal metrics for aligned same-rate full reference."""

from __future__ import annotations

import cv2
import numpy as np

from ..config import EvalConfig
from ..imutils import spatial_norm_factor
from ..schema import FrameBundle, backward_warp, flow_magnitude
from .window_flows import WindowFlows


def _global_ssim_proxy(a: np.ndarray, b: np.ndarray) -> float:
    """Whole-image statistics retained as a diagnostic proxy only."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    ma, mb = float(a.mean()), float(b.mean())
    va, vb = float(a.var()), float(b.var())
    cov = float(np.mean((a - ma) * (b - mb)))
    return float(((2 * ma * mb + c1) * (2 * cov + c2)) /
                 ((ma * ma + mb * mb + c1) * (va + vb + c2)))


def _local_ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Standard local SSIM with an 11x11 Gaussian window."""
    a, b = a.astype(np.float32), b.astype(np.float32)
    c1, c2 = (0.01 * 255.0) ** 2, (0.03 * 255.0) ** 2
    mu_a = cv2.GaussianBlur(a, (11, 11), 1.5)
    mu_b = cv2.GaussianBlur(b, (11, 11), 1.5)
    sigma_a = cv2.GaussianBlur(a * a, (11, 11), 1.5) - mu_a * mu_a
    sigma_b = cv2.GaussianBlur(b * b, (11, 11), 1.5) - mu_b * mu_b
    sigma_ab = cv2.GaussianBlur(a * b, (11, 11), 1.5) - mu_a * mu_b
    value = ((2 * mu_a * mu_b + c1) * (2 * sigma_ab + c2)) / (
        (mu_a * mu_a + mu_b * mu_b + c1)
        * (sigma_a + sigma_b + c2) + 1e-6)
    return float(np.clip(np.mean(value), -1.0, 1.0))


def _edge_metrics(
    reference: np.ndarray,
    candidate: np.ndarray,
    mask: np.ndarray | None = None,
) -> tuple[float, float, float, float]:
    er = cv2.Canny(reference.astype(np.uint8), 60, 160) > 0
    ec = cv2.Canny(candidate.astype(np.uint8), 60, 160) > 0
    if mask is not None:
        region = mask.astype(bool)
        er = er & region
        ec = ec & region
    if not er.any() and not ec.any():
        return 1.0, 1.0, 1.0, 0.0
    fallback_distance = float(np.hypot(*reference.shape[:2]))
    dist_c = (
        cv2.distanceTransform((~ec).astype(np.uint8), cv2.DIST_L2, 3)
        if ec.any()
        else np.full(reference.shape, fallback_distance, np.float32))
    dist_r = (
        cv2.distanceTransform((~er).astype(np.uint8), cv2.DIST_L2, 3)
        if er.any()
        else np.full(reference.shape, fallback_distance, np.float32))
    # USERPLAN §6.2: Chamfer distances + tolerance are resolution-normalized
    # (fraction of frame diagonal) so the same relative defect scores across
    # resolutions and presets.  Both the distance field AND the 2-pixel
    # tolerance are divided by the diagonal, so the comparison stays in the
    # same normalized unit (a 2px offset at any resolution is still "2px").
    norm = spatial_norm_factor(*reference.shape[:2])
    dist_c_n = dist_c / norm
    dist_r_n = dist_r / norm
    tol = 2.0 / norm
    recall = float(np.mean(dist_c_n[er] <= tol)) if er.any() else 1.0
    precision = float(np.mean(dist_r_n[ec] <= tol)) if ec.any() else 1.0
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-6)
    chamfer_parts = []
    if er.any():
        chamfer_parts.append(float(np.mean(dist_c_n[er])))
    if ec.any():
        chamfer_parts.append(float(np.mean(dist_r_n[ec])))
    return recall, precision, float(f1), float(np.mean(chamfer_parts))


def _multiscale_luma_error(a: np.ndarray, b: np.ndarray) -> float:
    values: list[float] = []
    aa, bb = a, b
    for _ in range(3):
        luma_error = float(np.mean(np.abs(aa - bb))) / 255.0
        ga_x = cv2.Sobel(aa, cv2.CV_32F, 1, 0, ksize=3)
        ga_y = cv2.Sobel(aa, cv2.CV_32F, 0, 1, ksize=3)
        gb_x = cv2.Sobel(bb, cv2.CV_32F, 1, 0, ksize=3)
        gb_y = cv2.Sobel(bb, cv2.CV_32F, 0, 1, ksize=3)
        gradient_error = float(np.mean(np.abs(
            np.hypot(ga_x, ga_y) - np.hypot(gb_x, gb_y)))) / 255.0
        values.append(0.7 * luma_error + 0.3 * gradient_error)
        if min(aa.shape[:2]) < 16:
            break
        aa = cv2.pyrDown(aa)
        bb = cv2.pyrDown(bb)
    return float(np.mean(values))


def _reference_rois(reference_y: np.ndarray) -> dict[str, np.ndarray]:
    """Short-window screen, text, salient and moving-region proxy masks."""
    n, h, w = reference_y.shape
    edges = np.stack([
        cv2.Canny(frame.astype(np.uint8), 60, 160) > 0
        for frame in reference_y
    ])
    band = np.zeros((h, w), bool)
    band[:max(1, h // 5)] = True
    band[-max(1, h // 5):] = True
    band[:, :max(1, w // 6)] = True      # USERPLAN P2: left/right HUD rails
    band[:, -max(1, w // 6):] = True
    persistent = edges.mean(0) >= 0.35
    ui = band & cv2.dilate(
        persistent.astype(np.uint8), np.ones((7, 7), np.uint8)
    ).astype(bool)
    density = cv2.blur(
        edges[0].astype(np.float32), (15, 15),
        borderType=cv2.BORDER_REFLECT)
    text = ui & (density >= 0.10)

    median_y = np.median(reference_y, axis=0).astype(np.float32)
    gx = cv2.Sobel(median_y, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(median_y, cv2.CV_32F, 0, 1, ksize=3)
    gradient = np.hypot(gx, gy)
    salient_threshold = float(np.percentile(gradient, 75))
    salient = gradient >= max(salient_threshold, 8.0)

    motion = np.zeros((h, w), np.float32)
    if n > 1:
        motion = np.mean(np.abs(np.diff(reference_y, axis=0)), axis=0)
    motion_threshold = float(np.percentile(motion, 75))
    center = np.zeros((h, w), bool)
    center[h // 5:4 * h // 5, w // 5:4 * w // 5] = True
    moving = center & (motion >= max(motion_threshold, 2.0))
    return {"ui": ui, "text": text, "salient": salient, "motion": moving}


def _masked_l1(
    reference: np.ndarray,
    candidate: np.ndarray,
    mask: np.ndarray,
) -> float:
    return (
        float(np.mean(np.abs(candidate[:, mask] - reference[:, mask])))
        if mask.sum() >= 32 else float("nan")
    )


def _camera_residual_flow(flow: np.ndarray) -> np.ndarray:
    """Dense flow minus its affine camera motion (USERPLAN P2, shared impl)."""
    from ..motion import dense_affine_residual
    return dense_affine_residual(flow)


def _tile_flow_errors(
    reference_flow: np.ndarray,
    candidate_flow: np.ndarray,
    grid: int = 4,
) -> list[float]:
    h, w = reference_flow.shape[:2]
    errors = []
    for gy in range(grid):
        for gx in range(grid):
            ys = slice(gy * h // grid, (gy + 1) * h // grid)
            xs = slice(gx * w // grid, (gx + 1) * w // grid)
            ref_vector = np.median(
                reference_flow[ys, xs].reshape(-1, 2), axis=0)
            cand_vector = np.median(
                candidate_flow[ys, xs].reshape(-1, 2), axis=0)
            errors.append(float(
                np.linalg.norm(cand_vector - ref_vector)
                / (np.linalg.norm(ref_vector) + 2.0)))
    return errors


def _frame_gaps(yr: np.ndarray, z_thresh: float = 6.0) -> list[bool]:
    """Detect scene cuts between consecutive frames (USERPLAN P2).

    Returns a list of length ``len(yr) - 1`` where ``True`` marks a *gap*
    between frame ``i`` and ``i + 1``.  Cuts are found from robust-z jumps in
    per-frame mean luma, so a global brightness drift does not fire while a
    genuine scene switch (large content change) does.
    """
    if yr.shape[0] < 2:
        return []
    mean_luma = np.mean(yr, axis=(1, 2))
    diffs = np.abs(np.diff(mean_luma))
    med = float(np.median(diffs))
    mad = float(np.median(np.abs(diffs - med)))
    if mad < 1e-6:
        return [False] * (yr.shape[0] - 1)
    z = (diffs - med) / (1.4826 * mad)
    return (z > z_thresh).tolist()


def compute_window(
    reference: FrameBundle,
    candidate: FrameBundle,
    reference_flows: WindowFlows,
    candidate_flows: WindowFlows,
    cfg: EvalConfig,
) -> dict[str, float]:
    if len(reference.rgb) != len(candidate.rgb):
        raise ValueError("full-reference bundles must contain the same number of frames")
    if reference.rgb.shape[1:] != candidate.rgb.shape[1:]:
        raise ValueError("full-reference bundles must have matching geometry")

    yr = reference.y_channel()
    yc = candidate.y_channel()
    l1: list[float] = []
    psnr: list[float] = []
    ssim: list[float] = []
    global_ssim_proxy: list[float] = []
    perceptual: list[float] = []
    rgb_l1: list[float] = []
    rgb_charbonnier: list[float] = []
    edge_recall: list[float] = []
    edge_precision: list[float] = []
    edge_f1: list[float] = []
    edge_chamfer: list[float] = []
    for index, (a, b) in enumerate(zip(yr, yc)):
        diff = a - b
        mse = float(np.mean(diff * diff))
        l1.append(float(np.mean(np.abs(diff))))
        rgb_diff = (
            candidate.rgb[index].astype(np.float32)
            - reference.rgb[index].astype(np.float32)
        )
        rgb_l1.append(float(np.mean(np.abs(rgb_diff))))
        rgb_charbonnier.append(float(np.mean(
            np.sqrt(rgb_diff * rgb_diff + cfg.charbonnier_tau ** 2)
            - cfg.charbonnier_tau)))
        psnr.append(100.0 if mse <= 1e-12
                    else float(20.0 * np.log10(255.0 / np.sqrt(mse))))
        ssim.append(_local_ssim(a, b))
        global_ssim_proxy.append(_global_ssim_proxy(a, b))
        perceptual.append(_multiscale_luma_error(a, b))
        er, ep, ef, ec = _edge_metrics(a, b)
        edge_recall.append(er)
        edge_precision.append(ep)
        edge_f1.append(ef)
        edge_chamfer.append(ec)

    rois = _reference_rois(yr)
    text_edge_f1: list[float] = []
    if rois["text"].sum() >= 32:
        for a, b in zip(yr, yc):
            _, _, f1, _ = _edge_metrics(a, b, rois["text"])
            text_edge_f1.append(f1)

    # --- gap detection (USERPLAN P2) -------------------------------------
    # A scene cut inside the window makes cross-frame chains (trajectory
    # cumsum, mcr_error, structure_persistence) meaningless across the cut.
    # Detect cuts from robust-z jumps in per-frame mean luma and reset state.
    gaps = _frame_gaps(yr)

    temporal_diff: list[float] = []
    flow_error: list[float] = []
    trajectory: list[float] = []
    mcr_error: list[float] = []
    structure_persistence: list[float] = []
    ref_vectors: list[np.ndarray] = []
    cand_vectors: list[np.ndarray] = []
    tile_flow_errors: list[float] = []
    camera_residual_errors: list[float] = []
    salient_flow_errors: list[float] = []
    # cumsum segments are restarted at each gap; we keep per-segment paths so
    # a cut corrupts only its own segment, not the whole window.
    seg_ref: list[list[np.ndarray]] = [[]]
    seg_cand: list[list[np.ndarray]] = [[]]
    for i in range(len(yr) - 1):
        dr = yr[i + 1] - yr[i]
        dc = yc[i + 1] - yc[i]
        temporal_diff.append(float(np.mean(np.abs(dc - dr))))

        fr = reference_flows.forward(i, i + 1)
        fc = candidate_flows.forward(i, i + 1)
        denom = flow_magnitude(fr) + 2.0
        flow_error.append(float(np.mean(flow_magnitude(fc - fr) / denom)))
        tile_flow_errors.extend(_tile_flow_errors(fr, fc))
        residual_ref = _camera_residual_flow(fr)
        residual_cand = _camera_residual_flow(fc)
        residual_denom = flow_magnitude(residual_ref) + 2.0
        camera_residual_errors.append(float(np.mean(
            flow_magnitude(residual_cand - residual_ref) / residual_denom)))
        if rois["salient"].sum() >= 32:
            salient_flow_errors.append(float(np.mean(
                (flow_magnitude(fc - fr) / denom)[rois["salient"]])))
        ref_vectors.append(np.median(fr.reshape(-1, 2), axis=0))
        cand_vectors.append(np.median(fc.reshape(-1, 2), axis=0))

        # Reset the trajectory chain at a gap; the mcr / structure-persistence
        # pair that straddles a cut is dropped (it compares unrelated frames).
        if gaps[i]:
            seg_ref.append([])
            seg_cand.append([])
            continue
        seg_ref[-1].append(ref_vectors[-1])
        seg_cand[-1].append(cand_vectors[-1])

        wr = backward_warp(yr[i + 1], fr)
        wc = backward_warp(yc[i + 1], fc)
        rr = np.abs(wr - yr[i])
        rc = np.abs(wc - yc[i])
        mcr_error.append(float(np.mean(np.abs(rc - rr))))

        er0 = cv2.Canny(yr[i].astype(np.uint8), 60, 160) > 0
        er1 = cv2.Canny(yr[i + 1].astype(np.uint8), 60, 160) > 0
        ec0 = cv2.Canny(yc[i].astype(np.uint8), 60, 160) > 0
        ec1 = cv2.Canny(yc[i + 1].astype(np.uint8), 60, 160) > 0
        structure_persistence.append(float(np.mean(
            np.logical_xor(np.logical_xor(er0, er1), np.logical_xor(ec0, ec1)))))

    # Trajectory = worst (longest) per-segment cumsum deviation (USERPLAN P2).
    # USERPLAN §6.2: normalize by frame diagonal so the metric is resolution-
    # independent (the per-frame flow vectors are in working-resolution px).
    norm = spatial_norm_factor(yr.shape[1], yr.shape[2])
    for seg_r, seg_c in zip(seg_ref, seg_cand):
        if len(seg_r) >= 2:
            ref_path = np.cumsum(np.asarray(seg_r), axis=0)
            cand_path = np.cumsum(np.asarray(seg_c), axis=0)
            trajectory.append(float(
                np.mean(np.linalg.norm(cand_path - ref_path, axis=1)) / norm))

    ref_flicker = float(np.std(np.diff(np.mean(yr, axis=(1, 2))))) if len(yr) > 1 else 0.0
    cand_flicker = float(np.std(np.diff(np.mean(yc, axis=(1, 2))))) if len(yc) > 1 else 0.0

    return {
        "fr_l1_y": float(np.mean(l1)),
        "fr_l1_rgb": float(np.mean(rgb_l1)),
        "fr_charbonnier_rgb": float(np.mean(rgb_charbonnier)),
        "fr_psnr": float(np.mean(psnr)),
        "fr_ssim": float(np.mean(ssim)),
        "fr_global_ssim_proxy": float(np.mean(global_ssim_proxy)),
        "fr_multiscale_luma_gradient_l1": float(np.mean(perceptual)),
        "fr_edge_recall": float(np.mean(edge_recall)),
        "fr_edge_precision": float(np.mean(edge_precision)),
        "fr_edge_f1": float(np.mean(edge_f1)),
        "fr_edge_chamfer": float(np.mean(edge_chamfer)),
        "fr_ui_roi_l1": _masked_l1(yr, yc, rois["ui"]),
        "fr_text_roi_edge_f1": (
            float(np.mean(text_edge_f1))
            if text_edge_f1 else float("nan")),
        "fr_salient_roi_l1": _masked_l1(yr, yc, rois["salient"]),
        "fr_motion_roi_l1": _masked_l1(yr, yc, rois["motion"]),
        "fr_temporal_diff_error": float(np.mean(temporal_diff)) if temporal_diff else float("nan"),
        "fr_flow_error": float(np.mean(flow_error)) if flow_error else float("nan"),
        "fr_tile_flow_error_p50": (
            float(np.percentile(tile_flow_errors, 50))
            if tile_flow_errors else float("nan")),
        "fr_tile_flow_error_p90": (
            float(np.percentile(tile_flow_errors, 90))
            if tile_flow_errors else float("nan")),
        "fr_tile_flow_error_p99": (
            float(np.percentile(tile_flow_errors, 99))
            if tile_flow_errors else float("nan")),
        "fr_camera_residual_flow_error": (
            float(np.mean(camera_residual_errors))
            if camera_residual_errors else float("nan")),
        "fr_salient_flow_error": (
            float(np.mean(salient_flow_errors))
            if salient_flow_errors else float("nan")),
        "fr_trajectory_deviation": float(np.mean(trajectory)) if trajectory else float("nan"),
        "fr_mcr_difference": float(np.mean(mcr_error)) if mcr_error else float("nan"),
        "fr_flicker_excess": float(max(cand_flicker - ref_flicker, 0.0)),
        "fr_structure_persistence_error": (
            float(np.mean(structure_persistence))
            if structure_persistence else float("nan")),
    }


# ---------------------------------------------------------------------------
# Dense error maps (USERPLAN P1).  Only computed for the highest-risk windows
# (the scalar ``compute_window`` runs for every window).
# ---------------------------------------------------------------------------
def compute_window_maps(
    reference: FrameBundle,
    candidate: FrameBundle,
    reference_flows: WindowFlows,
    candidate_flows: WindowFlows,
    cfg: EvalConfig,
) -> dict[str, np.ndarray]:
    """Return named (H, W) diagnostic fields for one FR window (USERPLAN §8).

    * ``luma_error_map``    — mean per-pixel |reference - candidate| luma;
    * ``edge_mismatch_map``  — per-pixel XOR of reference/candidate edges;
    * ``flow_error_map``     — mean per-pixel flow-error magnitude (normalized
      by reference flow magnitude);
    * ``mcr_difference_map`` — per-pixel |warp_residual_candidate -
      warp_residual_reference| (motion-compensation residual difference).
    """
    yr = reference.y_channel()
    yc = candidate.y_channel()
    h, w = yr.shape[1], yr.shape[2]
    out: dict[str, np.ndarray] = {}

    # --- luma error map (mean over frames) -------------------------------
    luma_fields: list[np.ndarray] = []
    for a, b in zip(yr, yc):
        luma_fields.append(np.abs(a - b).astype(np.float32))
    out["luma_error_map"] = _mean_with_resize(luma_fields, h, w)

    # --- edge mismatch map (mean over frames) ----------------------------
    edge_fields: list[np.ndarray] = []
    for a, b in zip(yr, yc):
        ea = cv2.Canny(a.astype(np.uint8), 60, 160) > 0
        eb = cv2.Canny(b.astype(np.uint8), 60, 160) > 0
        edge_fields.append(np.logical_xor(ea, eb).astype(np.float32))
    out["edge_mismatch_map"] = _mean_with_resize(edge_fields, h, w)

    # --- flow error map (mean over frame pairs) ---------------------------
    flow_fields: list[np.ndarray] = []
    for i in range(len(yr) - 1):
        fr = reference_flows.forward(i, i + 1)
        fc = candidate_flows.forward(i, i + 1)
        mag = flow_magnitude(fr) + 2.0
        flow_fields.append((flow_magnitude(fc - fr) / mag).astype(np.float32))
    out["flow_error_map"] = _mean_with_resize(flow_fields, h, w)

    # --- MCR difference map (mean over frame pairs) -----------------------
    mcr_fields: list[np.ndarray] = []
    for i in range(len(yr) - 1):
        fr = reference_flows.forward(i, i + 1)
        fc = candidate_flows.forward(i, i + 1)
        wr = backward_warp(yr[i + 1], fr)
        wc = backward_warp(yc[i + 1], fc)
        rr = np.abs(wr - yr[i])
        rc = np.abs(wc - yc[i])
        mcr_fields.append(np.abs(rc - rr).astype(np.float32))
    out["mcr_difference_map"] = _mean_with_resize(mcr_fields, h, w)

    return out


def _mean_with_resize(fields: list[np.ndarray], h: int, w: int) -> np.ndarray:
    if not fields:
        return np.zeros((h, w), np.float32)
    resized = []
    for f in fields:
        if f.shape[:2] != (h, w):
            f = cv2.resize(f, (w, h), interpolation=cv2.INTER_LINEAR)
        resized.append(f)
    return np.mean(resized, axis=0).astype(np.float32)
