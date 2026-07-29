"""Full-timeline low-resolution reference evidence for FR sampling and fusion."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..io.video_reader import VideoReader
from ..schema import FullReferenceAlignment


@dataclass
class FullReferenceScan:
    per_frame_risk: np.ndarray
    y_l1: np.ndarray
    chroma_l1: np.ndarray
    gradient_l1: np.ndarray
    edge_mismatch: np.ndarray
    ssim_proxy_error: np.ndarray
    frame_diff_mismatch: np.ndarray

    def global_features(self) -> dict[str, float]:
        output = {}
        for name, values in (
            ("fr_scan_y_l1", self.y_l1),
            ("fr_scan_chroma_l1", self.chroma_l1),
            ("fr_scan_gradient_l1", self.gradient_l1),
            ("fr_scan_edge_mismatch", self.edge_mismatch),
            ("fr_scan_ssim_proxy_error", self.ssim_proxy_error),
            ("fr_scan_frame_diff_mismatch", self.frame_diff_mismatch),
        ):
            finite = values[np.isfinite(values)]
            if len(finite):
                output[name] = float(
                    0.5 * np.median(finite)
                    + 0.3 * np.percentile(finite, 90)
                    + 0.2 * np.percentile(finite, 99))
        return output


def _ssim_proxy(a: np.ndarray, b: np.ndarray) -> float:
    c1, c2 = (0.01 * 255.0) ** 2, (0.03 * 255.0) ** 2
    ma = cv2.GaussianBlur(a, (7, 7), 1.2)
    mb = cv2.GaussianBlur(b, (7, 7), 1.2)
    va = cv2.GaussianBlur(a * a, (7, 7), 1.2) - ma * ma
    vb = cv2.GaussianBlur(b * b, (7, 7), 1.2) - mb * mb
    cov = cv2.GaussianBlur(a * b, (7, 7), 1.2) - ma * mb
    value = ((2 * ma * mb + c1) * (2 * cov + c2)) / (
        (ma * ma + mb * mb + c1) * (va + vb + c2) + 1e-6)
    return float(np.clip(np.mean(value), -1.0, 1.0))


def scan_full_reference(
    reference: VideoReader,
    candidate: VideoReader,
    alignment: FullReferenceAlignment,
    *,
    width: int = 320,
) -> FullReferenceScan:
    ref_frames = reference.decode_all(width=width)
    cand_frames = candidate.decode_all(width=width)
    n = candidate.meta.n_frames
    arrays = [np.full(n, np.nan, np.float32) for _ in range(6)]
    y_l1, chroma_l1, gradient_l1, edge_mismatch, ssim_err, diff_err = arrays
    previous: tuple[np.ndarray, np.ndarray] | None = None
    for cand_index, ref_index in enumerate(alignment.reference_of_candidate):
        if ref_index < 0 or cand_index >= len(cand_frames) or ref_index >= len(ref_frames):
            previous = None
            continue
        ref = ref_frames[int(ref_index)].astype(np.float32)
        cand = cand_frames[cand_index].astype(np.float32)
        if ref.shape != cand.shape:
            cand = cv2.resize(cand, (ref.shape[1], ref.shape[0]),
                              interpolation=cv2.INTER_AREA)
        ref_yuv = cv2.cvtColor(ref.astype(np.uint8), cv2.COLOR_RGB2YCrCb).astype(np.float32)
        cand_yuv = cv2.cvtColor(cand.astype(np.uint8), cv2.COLOR_RGB2YCrCb).astype(np.float32)
        ry, cy = ref_yuv[..., 0], cand_yuv[..., 0]
        y_l1[cand_index] = np.mean(np.abs(ry - cy))
        chroma_l1[cand_index] = np.mean(np.abs(
            ref_yuv[..., 1:] - cand_yuv[..., 1:]))
        rg = cv2.Laplacian(ry, cv2.CV_32F)
        cg = cv2.Laplacian(cy, cv2.CV_32F)
        gradient_l1[cand_index] = np.mean(np.abs(rg - cg))
        re = cv2.Canny(ry.astype(np.uint8), 60, 160) > 0
        ce = cv2.Canny(cy.astype(np.uint8), 60, 160) > 0
        edge_mismatch[cand_index] = np.mean(np.logical_xor(re, ce))
        ssim_err[cand_index] = 1.0 - _ssim_proxy(ry, cy)
        if previous is not None:
            prev_r, prev_c = previous
            diff_err[cand_index] = np.mean(np.abs(
                (cy - prev_c) - (ry - prev_r)))
        previous = (ry, cy)

    normalized = []
    for values, scale in (
        (y_l1, 12.0), (chroma_l1, 10.0), (gradient_l1, 20.0),
        (edge_mismatch, 0.12), (ssim_err, 0.20), (diff_err, 10.0),
    ):
        normalized.append(np.nan_to_num(values / scale, nan=0.0))
    risk = np.mean(np.stack(normalized), axis=0).astype(np.float32)
    return FullReferenceScan(
        risk, y_l1, chroma_l1, gradient_l1, edge_mismatch, ssim_err, diff_err)
