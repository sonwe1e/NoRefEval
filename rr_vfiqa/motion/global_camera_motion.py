"""Global camera motion estimation (USERPLAN.md §8.1).

Sparse LK features + RANSAC fitting, escalating identity → translation →
partial-affine → affine → homography only while the residual stays bad.
Foreground objects fall out as RANSAC outliers; an optional residual-flow
gate removes strongly-independent points before fitting.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..schema import CameraMotion


def _residual(pts0: np.ndarray, pts1: np.ndarray, model: str, M: np.ndarray) -> float:
    if pts0.size == 0:
        return float("inf")
    if model == "homography":
        p = cv2.perspectiveTransform(pts0, M)
    else:
        p = cv2.transform(pts0, M)
    return float(np.median(np.linalg.norm(p - pts1, axis=-1)))


def estimate_camera_motion(a_rgb: np.ndarray, b_rgb: np.ndarray,
                           max_features: int = 2000, down_width: int = 640
                           ) -> CameraMotion:
    ga = cv2.cvtColor(a_rgb, cv2.COLOR_RGB2GRAY)
    gb = cv2.cvtColor(b_rgb, cv2.COLOR_RGB2GRAY)
    scale = 1.0
    if ga.shape[1] > down_width:
        scale = down_width / ga.shape[1]
        h = max(2, int(round(ga.shape[0] * scale)) // 2 * 2)
        ga = cv2.resize(ga, (down_width, h), interpolation=cv2.INTER_AREA)
        gb = cv2.resize(gb, (down_width, h), interpolation=cv2.INTER_AREA)

    h, w = ga.shape
    pts0 = cv2.goodFeaturesToTrack(ga, maxCorners=max_features, qualityLevel=0.01,
                                   minDistance=6, blockSize=7)
    if pts0 is None or len(pts0) < 12:
        return CameraMotion("identity", np.float32([[1, 0, 0], [0, 1, 0]]), 0.0, 0.0)

    pts1, st, _ = cv2.calcOpticalFlowPyrLK(ga, gb, pts0, None,
                                           winSize=(25, 25), maxLevel=4,
                                           criteria=(cv2.TERM_CRITERIA_EPS |
                                                     cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    good = st.ravel() == 1
    p0, p1 = pts0[good].reshape(-1, 2), pts1[good].reshape(-1, 2)
    if len(p0) < 12:
        return CameraMotion("identity", np.float32([[1, 0, 0], [0, 1, 0]]), 0.0, 0.0)

    # Gate out foreground: drop points whose motion deviates strongly from the
    # median (camera) motion before fitting.
    med = np.median(p1 - p0, axis=0)
    dev = np.linalg.norm((p1 - p0) - med, axis=1)
    keep = dev < max(8.0, 3.0 * np.median(dev) + 1e-6)
    if keep.sum() >= 12:
        p0, p1 = p0[keep], p1[keep]

    p0f = p0.astype(np.float32).reshape(-1, 1, 2)
    p1f = p1.astype(np.float32).reshape(-1, 1, 2)
    tol = max(1.5, 0.004 * max(h, w))

    # Escalation ladder.
    tx, ty = med * 0.0
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    best = CameraMotion("identity", M, 1.0, _residual(p0f, p1f, "affine", M))
    if best.residual_px <= tol:
        return _upscale(best, scale)

    M, inl = cv2.estimateAffinePartial2D(p0f, p1f, method=cv2.RANSAC,
                                         ransacReprojThreshold=tol)
    if M is not None:
        r = _residual(p0f, p1f, "affine", M)
        inlier_ratio = float(np.mean(inl)) if inl is not None else 0.0
        best = CameraMotion("partial_affine", M, inlier_ratio, r)
        if r <= tol:
            return _upscale(best, scale)

    M, inl = cv2.estimateAffine2D(p0f, p1f, method=cv2.RANSAC,
                                  ransacReprojThreshold=tol)
    if M is not None:
        r = _residual(p0f, p1f, "affine", M)
        inlier_ratio = float(np.mean(inl)) if inl is not None else 0.0
        cand = CameraMotion("affine", M, inlier_ratio, r)
        if r < best.residual_px:
            best = cand
        if r <= tol:
            return _upscale(best, scale)

    M, inl = cv2.findHomography(p0f, p1f, method=cv2.RANSAC, ransacReprojThreshold=tol)
    if M is not None:
        r = _residual(p0f, p1f, "homography", M)
        inlier_ratio = float(np.mean(inl)) if inl is not None else 0.0
        cand = CameraMotion("homography", M, inlier_ratio, r)
        if r < best.residual_px:
            best = cand

    return _upscale(best, scale)


def _upscale(motion: CameraMotion, scale: float) -> CameraMotion:
    """Convert a motion estimated at reduced resolution back to native pixels."""
    if scale == 1.0:
        return motion
    M = motion.matrix.copy().astype(np.float32)
    s = 1.0 / scale
    if motion.model == "homography":
        T = np.float32([[s, 0, 0], [0, s, 0], [0, 0, 1]])
        Ti = np.float32([[scale, 0, 0], [0, scale, 0], [0, 0, 1]])
        M = T @ M @ Ti
    else:
        M[0, 2] *= s
        M[1, 2] *= s
    return CameraMotion(motion.model, M, motion.inlier_ratio,
                        motion.residual_px * s)
