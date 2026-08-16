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
    target_size: tuple[int, int] | None = None,
) -> FullReferenceScan:
    """Stream aligned frame pairs with O(one-frame) image memory.

    The scalar result arrays are timeline-sized, but decoded RGB frames are
    never accumulated. ``target_size`` is ``(width, height)`` and guarantees
    that non-strict geometry policies use one explicit comparison grid.
    """
    n = candidate.meta.n_frames
    arrays = [np.full(n, np.nan, np.float32) for _ in range(6)]
    y_l1, chroma_l1, gradient_l1, edge_mismatch, ssim_err, diff_err = arrays
    # USERPLAN P2: previous + the indices that produced it.  frame-diff
    # mismatch (diff_err) is only meaningful when BOTH the reference and the
    # candidate advance by exactly one frame since the last pair — an alignment
    # gap (ref 0→1→3) must reset state so we never compare a two-frame
    # reference delta against a one-frame candidate delta.
    previous: tuple[np.ndarray, np.ndarray] | None = None
    prev_ref_index = -2
    prev_cand_index = -2
    reference_frames = iter(reference.iter_frames(width=width))
    current_reference = next(reference_frames, None)
    for cand_index, cand_rgb in candidate.iter_frames(width=width):
        if cand_index >= len(alignment.reference_of_candidate):
            break
        ref_index = int(alignment.reference_of_candidate[cand_index])
        if ref_index < 0:
            previous = None
            continue
        while (current_reference is not None
               and current_reference[0] < ref_index):
            current_reference = next(reference_frames, None)
        if current_reference is None or current_reference[0] != ref_index:
            previous = None
            continue

        ref_rgb = current_reference[1]
        if target_size is not None:
            ref_rgb = _resize_frame(ref_rgb, target_size)
            cand_rgb = _resize_frame(cand_rgb, target_size)
        elif ref_rgb.shape != cand_rgb.shape:
            cand_rgb = _resize_frame(
                cand_rgb, (ref_rgb.shape[1], ref_rgb.shape[0]))
        ref = ref_rgb.astype(np.float32)
        cand = cand_rgb.astype(np.float32)
        ref_yuv = cv2.cvtColor(ref.astype(np.uint8), cv2.COLOR_RGB2YCrCb).astype(np.float32)
        cand_yuv = cv2.cvtColor(cand.astype(np.uint8), cv2.COLOR_RGB2YCrCb).astype(np.float32)
        ry, cy = ref_yuv[..., 0], cand_yuv[..., 0]
        y_l1[cand_index] = np.mean(np.abs(ry - cy))
        chroma_l1[cand_index] = np.mean(np.abs(
            ref_yuv[..., 1:] - cand_yuv[..., 1:]))
        rg = _gradient_magnitude(ry)
        cg = _gradient_magnitude(cy)
        gradient_l1[cand_index] = np.mean(np.abs(rg - cg))
        re = cv2.Canny(ry.astype(np.uint8), 60, 160) > 0
        ce = cv2.Canny(cy.astype(np.uint8), 60, 160) > 0
        edge_mismatch[cand_index] = np.mean(np.logical_xor(re, ce))
        ssim_err[cand_index] = 1.0 - _ssim_proxy(ry, cy)
        # Only compute frame-diff mismatch on consecutive (ref, cand) pairs:
        # a gap in either timeline makes the per-frame deltas incommensurable.
        if (previous is not None
                and ref_index == prev_ref_index + 1
                and cand_index == prev_cand_index + 1):
            prev_r, prev_c = previous
            diff_err[cand_index] = np.mean(np.abs(
                (cy - prev_c) - (ry - prev_r)))
        else:
            diff_err[cand_index] = float("nan")
        previous = (ry, cy)
        prev_ref_index = ref_index
        prev_cand_index = cand_index

    normalized = []
    for values, scale in (
        (y_l1, 12.0), (chroma_l1, 10.0), (gradient_l1, 20.0),
        (edge_mismatch, 0.12), (ssim_err, 0.20), (diff_err, 10.0),
    ):
        normalized.append(np.nan_to_num(values / scale, nan=0.0))
    risk = np.mean(np.stack(normalized), axis=0).astype(np.float32)
    return FullReferenceScan(
        risk, y_l1, chroma_l1, gradient_l1, edge_mismatch, ssim_err, diff_err)


def _resize_frame(
    frame: np.ndarray,
    target_size: tuple[int, int],
) -> np.ndarray:
    width, height = target_size
    if frame.shape[:2] == (height, width):
        return frame
    interpolation = (
        cv2.INTER_AREA
        if width <= frame.shape[1] and height <= frame.shape[0]
        else cv2.INTER_LINEAR)
    return cv2.resize(
        frame, (width, height), interpolation=interpolation)


def _gradient_magnitude(y: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(y, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(y, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)
