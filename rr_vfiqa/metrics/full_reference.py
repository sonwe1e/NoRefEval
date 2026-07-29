"""Spatial and temporal metrics for aligned same-rate full reference."""

from __future__ import annotations

import cv2
import numpy as np

from ..config import EvalConfig
from ..schema import FrameBundle, backward_warp, flow_magnitude
from .window_flows import WindowFlows


def _ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Global SSIM on luma, dependency-free and deterministic."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    ma, mb = float(a.mean()), float(b.mean())
    va, vb = float(a.var()), float(b.var())
    cov = float(np.mean((a - ma) * (b - mb)))
    return float(((2 * ma * mb + c1) * (2 * cov + c2)) /
                 ((ma * ma + mb * mb + c1) * (va + vb + c2)))


def _edge_metrics(reference: np.ndarray, candidate: np.ndarray) -> tuple[float, float]:
    er = cv2.Canny(reference.astype(np.uint8), 60, 160) > 0
    ec = cv2.Canny(candidate.astype(np.uint8), 60, 160) > 0
    if not er.any() and not ec.any():
        return 1.0, 0.0
    dist_c = cv2.distanceTransform((~ec).astype(np.uint8), cv2.DIST_L2, 3)
    dist_r = cv2.distanceTransform((~er).astype(np.uint8), cv2.DIST_L2, 3)
    recall = float(np.mean(dist_c[er] <= 2.0)) if er.any() else 1.0
    precision = float(np.mean(dist_r[ec] <= 2.0)) if ec.any() else 1.0
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-6)
    chamfer_parts = []
    if er.any():
        chamfer_parts.append(float(np.mean(dist_c[er])))
    if ec.any():
        chamfer_parts.append(float(np.mean(dist_r[ec])))
    return float(f1), float(np.mean(chamfer_parts))


def _multiscale_luma_error(a: np.ndarray, b: np.ndarray) -> float:
    values: list[float] = []
    aa, bb = a, b
    for _ in range(3):
        values.append(float(np.mean(np.abs(aa - bb))) / 255.0)
        if min(aa.shape[:2]) < 16:
            break
        aa = cv2.pyrDown(aa)
        bb = cv2.pyrDown(bb)
    return float(np.mean(values))


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
    perceptual: list[float] = []
    edge_f1: list[float] = []
    edge_chamfer: list[float] = []
    for a, b in zip(yr, yc):
        diff = a - b
        mse = float(np.mean(diff * diff))
        l1.append(float(np.mean(np.abs(diff))))
        psnr.append(100.0 if mse <= 1e-12
                    else float(20.0 * np.log10(255.0 / np.sqrt(mse))))
        ssim.append(_ssim(a, b))
        perceptual.append(_multiscale_luma_error(a, b))
        ef, ec = _edge_metrics(a, b)
        edge_f1.append(ef)
        edge_chamfer.append(ec)

    temporal_diff: list[float] = []
    flow_error: list[float] = []
    trajectory: list[float] = []
    mcr_error: list[float] = []
    structure_persistence: list[float] = []
    ref_vectors: list[np.ndarray] = []
    cand_vectors: list[np.ndarray] = []
    for i in range(len(yr) - 1):
        dr = yr[i + 1] - yr[i]
        dc = yc[i + 1] - yc[i]
        temporal_diff.append(float(np.mean(np.abs(dc - dr))))

        fr = reference_flows.forward(i, i + 1)
        fc = candidate_flows.forward(i, i + 1)
        denom = flow_magnitude(fr) + 2.0
        flow_error.append(float(np.mean(flow_magnitude(fc - fr) / denom)))
        ref_vectors.append(np.median(fr.reshape(-1, 2), axis=0))
        cand_vectors.append(np.median(fc.reshape(-1, 2), axis=0))

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

    if ref_vectors:
        ref_path = np.cumsum(np.asarray(ref_vectors), axis=0)
        cand_path = np.cumsum(np.asarray(cand_vectors), axis=0)
        trajectory.append(float(np.mean(np.linalg.norm(cand_path - ref_path, axis=1))))

    ref_flicker = float(np.std(np.diff(np.mean(yr, axis=(1, 2))))) if len(yr) > 1 else 0.0
    cand_flicker = float(np.std(np.diff(np.mean(yc, axis=(1, 2))))) if len(yc) > 1 else 0.0

    return {
        "fr_l1_y": float(np.mean(l1)),
        "fr_psnr": float(np.mean(psnr)),
        "fr_ssim": float(np.mean(ssim)),
        "fr_multiscale_perceptual": float(np.mean(perceptual)),
        "fr_edge_f1": float(np.mean(edge_f1)),
        "fr_edge_chamfer": float(np.mean(edge_chamfer)),
        "fr_temporal_diff_error": float(np.mean(temporal_diff)) if temporal_diff else float("nan"),
        "fr_flow_error": float(np.mean(flow_error)) if flow_error else float("nan"),
        "fr_trajectory_deviation": float(np.mean(trajectory)) if trajectory else float("nan"),
        "fr_mcr_difference": float(np.mean(mcr_error)) if mcr_error else float("nan"),
        "fr_flicker_excess": float(max(cand_flicker - ref_flicker, 0.0)),
        "fr_structure_persistence_error": (
            float(np.mean(structure_persistence))
            if structure_persistence else float("nan")),
    }
