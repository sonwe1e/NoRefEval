"""Thin-object / motion-layer-attribution branch (USERPLAN §8.3).

Poles, railings and streetlights failing to carry their own motion is a layer
attribution error: the object's motion relative to its background collapses
toward zero in the generated frame. Detection runs LSD on endpoint frames and
compares in-mask vs surround flow on both the full source interval and the
half leg into M_i.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..cache.source_cache import SourcePairData
from ..config import EvalConfig
from ..metrics.window_flows import WindowFlows
from ..schema import FrameBundle, resize_flow

_MAX_INSTANCES = 24


def _detect_lines(gray: np.ndarray) -> np.ndarray:
    """(K, 4) float32 line segments x1,y1,x2,y2 via LSD."""
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    segs, *_ = lsd.detect(gray)
    if segs is None:
        return np.zeros((0, 4), np.float32)
    segs = segs.reshape(-1, 4)
    # Keep long, fairly straight segments.
    length = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
    keep = length > 0.04 * max(gray.shape)
    return segs[keep].astype(np.float32)


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
    g0 = cv2.cvtColor(bundle.rgb[1], cv2.COLOR_RGB2GRAY)   # X_i at flow res

    segs = _detect_lines(g0)
    out: dict[str, float] = {"thin_count": float(len(segs))}
    if len(segs) == 0:
        out.update({"thin_layer_err": float("nan"), "thin_inst_err_p90": float("nan")})
        return out

    f_01 = resize_flow(pair.f_01, h, w)
    f_0m = flow.forward(1, 2)

    kern_out = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    kern_in = np.ones((5, 5), np.uint8)

    layer_errs = []
    instances = []
    for seg in segs[:_MAX_INSTANCES]:
        m_in = _line_mask(seg, h, w, 3).astype(bool)
        m_dil = cv2.dilate(m_in.astype(np.uint8), kern_out).astype(bool)
        m_bg = m_dil & ~cv2.dilate(m_in.astype(np.uint8), kern_in).astype(bool)
        if m_in.sum() < 8 or m_bg.sum() < 32:
            continue
        v_obj_01 = _med_flow(f_01, m_in)
        v_bg_01 = _med_flow(f_01, m_bg)
        v_obj_0m = _med_flow(f_0m, m_in)
        v_bg_0m = _med_flow(f_0m, m_bg)
        if np.isnan(v_obj_01).any() or np.isnan(v_bg_0m).any():
            continue
        d01 = v_obj_01 - v_bg_01
        d0m = v_obj_0m - v_bg_0m
        denom = np.linalg.norm(d01) + 1.5
        # §8.3 E_layer on the first half leg (×2 maps half interval to full).
        e = np.linalg.norm(2 * d0m - d01) / denom
        layer_errs.append(float(e))
        instances.append({"box": [int(seg[0]), int(seg[1]), int(seg[2]), int(seg[3])],
                          "layer_err": float(e)})

    if layer_errs:
        arr = np.asarray(layer_errs)
        out["thin_layer_err"] = float(arr.mean())
        out["thin_inst_err_p90"] = float(np.percentile(arr, 90)) if len(arr) > 2 else float(arr.max())
        out["thin_attrib_fail_frac"] = float(np.mean(arr > 0.6))
    else:
        out["thin_layer_err"] = float("nan")
        out["thin_inst_err_p90"] = float("nan")
        out["thin_attrib_fail_frac"] = float("nan")

    # Line-length alternation across the window (sword-tip / pole flicker).
    counts = []
    for t in range(bundle.rgb.shape[0]):
        gt = cv2.cvtColor(bundle.rgb[t], cv2.COLOR_RGB2GRAY)
        counts.append(len(_detect_lines(gt)))
    c = np.asarray(counts, np.float64) + 1.0
    idx = bundle.indices.astype(np.int64)
    out["thin_count_odd_even_ratio"] = float(
        c[idx % 2 == 1].mean() / max(c[idx % 2 == 0].mean(), 1e-6))
    out["_thin_instances"] = instances     # carried for badcase export
    return out
