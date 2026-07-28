"""Thin-object / motion-layer-attribution branch (USERPLAN §8.3).

Poles, railings and streetlights failing to carry their own motion is a layer
attribution error: the object's motion relative to its background collapses
toward zero in the generated frame.

Instances are LSD line segments filtered to genuinely *thin* structures
(width gate removes thick building/road edges), matched across BOTH endpoints
(persistent instances only), checked on BOTH half legs (0→m and m→1), and
gated on having relative motion at all — an object moving with the background
cannot have an attribution error.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..cache.source_cache import SourcePairData
from ..config import EvalConfig
from ..metrics.window_flows import WindowFlows
from ..schema import FrameBundle, forward_splat, resize_flow

_MAX_INSTANCES = 24
_MAX_WIDTH_PX = 5.0           # LSD width gate: thick segments are architecture
_MIN_REL_MOTION = 1.0         # px; below this, attribution error is undefined
_MATCH_DIST = 14.0            # px midpoint distance for endpoint association
_MATCH_ANGLE_DEG = 15.0


def _detect_thin_lines(gray: np.ndarray
                       ) -> tuple[np.ndarray, np.ndarray]:
    """(K,4) segments x1,y1,x2,y2 and (K,) widths — thin ones only."""
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    segs, widths, _prec, _nfa = lsd.detect(gray)
    if segs is None:
        return np.zeros((0, 4), np.float32), np.zeros(0, np.float32)
    segs = segs.reshape(-1, 4)
    widths = widths.reshape(-1)
    length = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
    keep = (length > 0.04 * max(gray.shape)) & (widths <= _MAX_WIDTH_PX)
    return segs[keep].astype(np.float32), widths[keep].astype(np.float32)


def _seg_angle(seg: np.ndarray) -> float:
    return float(np.degrees(np.arctan2(seg[3] - seg[1], seg[2] - seg[0])) % 180.0)


def _match_segments(s0: np.ndarray, s1: np.ndarray) -> list[tuple[int, int]]:
    """Associate endpoint instances by midpoint proximity + parallel angle."""
    if len(s0) == 0 or len(s1) == 0:
        return []
    m0 = np.stack([(s0[:, 0] + s0[:, 2]) / 2, (s0[:, 1] + s0[:, 3]) / 2], -1)
    m1 = np.stack([(s1[:, 0] + s1[:, 2]) / 2, (s1[:, 1] + s1[:, 3]) / 2], -1)
    a0 = np.array([_seg_angle(s) for s in s0])
    a1 = np.array([_seg_angle(s) for s in s1])
    pairs = []
    used = set()
    for i in range(len(s0)):
        d = np.linalg.norm(m1 - m0[i], axis=1)
        da = np.minimum(np.abs(a1 - a0[i]), 180 - np.abs(a1 - a0[i]))
        score = d + 0.5 * da
        score[np.array([j in used for j in range(len(s1))])] = np.inf
        j = int(np.argmin(score))
        if d[j] < _MATCH_DIST and da[j] < _MATCH_ANGLE_DEG:
            pairs.append((i, j))
            used.add(j)
    return pairs


def _line_mask(seg: np.ndarray, h: int, w: int, thickness: int = 3) -> np.ndarray:
    m = np.zeros((h, w), np.uint8)
    cv2.line(m, (int(seg[0]), int(seg[1])), (int(seg[2]), int(seg[3])), 1, thickness)
    return m


def _med_flow(flow: np.ndarray, mask: np.ndarray) -> np.ndarray:
    pts = flow[mask.astype(bool)]
    if len(pts) < 4:
        return np.full(2, np.nan, np.float32)
    return np.median(pts, axis=0)


def compute_window(bundle: FrameBundle, flow: WindowFlows, pair: SourcePairData,
                   cfg: EvalConfig) -> dict[str, float]:
    h, w = flow.height(), flow.width()
    g0 = cv2.cvtColor(bundle.rgb[1], cv2.COLOR_RGB2GRAY)   # X_i
    g1 = cv2.cvtColor(bundle.rgb[3], cv2.COLOR_RGB2GRAY)   # X_{i+1}

    segs0, _w0 = _detect_thin_lines(g0)
    segs1, _w1 = _detect_thin_lines(g1)
    matches = _match_segments(segs0, segs1)
    out: dict[str, float] = {"thin_count": float(len(segs0)),
                             "thin_matched_count": float(len(matches))}
    if not matches:
        out.update({"thin_layer_err": float("nan"),
                    "thin_inst_err_p90": float("nan"),
                    "thin_attrib_fail_frac": float("nan")})
        return out

    f_01 = resize_flow(pair.f_01, h, w)
    f_0m = flow.forward(1, 2)       # X_i → M_i
    f_m1 = flow.forward(2, 3)       # M_i → X_{i+1}
    kern_out = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    kern_in = np.ones((5, 5), np.uint8)

    layer_errs = []
    instances = []
    for i, j in matches[:_MAX_INSTANCES]:
        seg = segs0[i]
        m_in = _line_mask(seg, h, w, 3).astype(bool)
        m_dil = cv2.dilate(m_in.astype(np.uint8), kern_out).astype(bool)
        m_bg = m_dil & ~cv2.dilate(m_in.astype(np.uint8), kern_in).astype(bool)
        if m_in.sum() < 8 or m_bg.sum() < 32:
            continue

        # Full interval: object vs background motion.
        v_obj_01 = _med_flow(f_01, m_in)
        v_bg_01 = _med_flow(f_01, m_bg)
        if np.isnan(v_obj_01).any() or np.isnan(v_bg_01).any():
            continue
        d01 = v_obj_01 - v_bg_01
        denom = np.linalg.norm(d01) + 1.5
        if np.linalg.norm(d01) < _MIN_REL_MOTION:
            continue                        # no relative motion → nothing to misattribute

        # Front half leg on the X_i grid.
        v_obj_0m = _med_flow(f_0m, m_in)
        v_bg_0m = _med_flow(f_0m, m_bg)
        err_f = float("nan")
        if not (np.isnan(v_obj_0m).any() or np.isnan(v_bg_0m).any()):
            d0m = v_obj_0m - v_bg_0m
            err_f = float(np.linalg.norm(2 * d0m - d01) / denom)

        # Back half leg: carry the instance mask to the M_i grid by forward
        # splat, then measure object vs background flow M_i → X_{i+1}.
        m_m, cov = forward_splat(m_in.astype(np.float32), f_0m)
        m_in_m = (cov > 0.25) & (m_m > 0.5)
        m_dil_m = cv2.dilate(m_in_m.astype(np.uint8), kern_out).astype(bool)
        m_bg_m = m_dil_m & ~cv2.dilate(m_in_m.astype(np.uint8), kern_in).astype(bool)
        err_b = float("nan")
        if m_in_m.sum() >= 8 and m_bg_m.sum() >= 32:
            v_obj_m1 = _med_flow(f_m1, m_in_m)
            v_bg_m1 = _med_flow(f_m1, m_bg_m)
            if not (np.isnan(v_obj_m1).any() or np.isnan(v_bg_m1).any()):
                dm1 = v_obj_m1 - v_bg_m1
                err_b = float(np.linalg.norm(2 * dm1 - d01) / denom)

        errs = [e for e in (err_f, err_b) if e == e]
        if not errs:
            continue
        e = max(errs)
        layer_errs.append(e)
        instances.append({
            "box": [int(min(seg[0], seg[2])), int(min(seg[1], seg[3])),
                    int(max(seg[0], seg[2])), int(max(seg[1], seg[3]))],
            "layer_err": float(e),
            "err_front": float(err_f) if err_f == err_f else None,
            "err_back": float(err_b) if err_b == err_b else None,
        })

    if layer_errs:
        arr = np.asarray(layer_errs)
        out["thin_layer_err"] = float(arr.mean())
        out["thin_inst_err_p90"] = float(np.percentile(arr, 90)) if len(arr) > 2 \
            else float(arr.max())
        out["thin_attrib_fail_frac"] = float(np.mean(arr > 0.6))
    else:
        out["thin_layer_err"] = float("nan")
        out["thin_inst_err_p90"] = float("nan")
        out["thin_attrib_fail_frac"] = float("nan")

    # Whole-window thin-structure alternation (flicker proxy, kept separate
    # from the per-instance attribution signal).
    counts = [len(_detect_thin_lines(cv2.cvtColor(bundle.rgb[t],
                                                  cv2.COLOR_RGB2GRAY))[0])
              for t in range(bundle.rgb.shape[0])]
    c = np.asarray(counts, np.float64) + 1.0
    idx = bundle.indices.astype(np.int64)
    out["thin_count_odd_even_ratio"] = float(
        c[idx % 2 == 1].mean() / max(c[idx % 2 == 0].mean(), 1e-6))
    out["_thin_instances"] = sorted(instances, key=lambda d: -d["layer_err"])[:5]
    return out
