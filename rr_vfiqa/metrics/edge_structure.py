"""Edge / contour / topology integrity (USERPLAN.md §7).

Endpoint edges are half-warped to the mid instant to form a soft support map;
the generated frame's edges are scored against it. Aggregation is per
connected edge instance so tiny swords and UI rings are not drowned by
background pixels (§7.2).
"""

from __future__ import annotations

import cv2
import numpy as np

from ..cache.source_cache import SourceCache, SourcePairData
from ..config import EvalConfig
from ..schema import FrameBundle, resize_flow, warp_image
from .window_flows import WindowFlows

_TEXTURE_GRAD = 10.0          # gradient-energy floor for "textured" pixels
_MAX_INSTANCES = 64


def _multiscale_edges(gray: np.ndarray) -> np.ndarray:
    e1 = cv2.Canny(gray, 60, 160) > 0
    blur = cv2.GaussianBlur(gray, (0, 0), 1.0)
    e2 = cv2.Canny(blur.astype(np.uint8), 30, 100) > 0
    return (e1 | e2).astype(np.uint8)


def compute(bundle: FrameBundle, flow: WindowFlows, cache: SourceCache,
            pair: SourcePairData, cfg: EvalConfig) -> dict[str, float]:
    h, w = flow.height(), flow.width()
    out: dict[str, float] = {}

    # --- support map from endpoint edges ------------------------------------
    e0 = cache.get_edges(pair.pair).edges
    e1 = cache.get_edges(pair.pair + 1).edges
    g0 = cache.get_edges(pair.pair).grad_energy
    g1 = cache.get_edges(pair.pair + 1).grad_energy
    f_01 = resize_flow(pair.f_01, h, w)
    f_10 = resize_flow(pair.f_10, h, w)

    s0 = warp_image(e0.astype(np.float32), 0.5 * f_01) > 0.5
    s1 = warp_image(e1.astype(np.float32), -0.5 * f_10) > 0.5
    support = (s0 | s1).astype(np.uint8)

    tex = ((g0 + g1) * 0.5 > _TEXTURE_GRAD).astype(bool)

    gray_m = cv2.cvtColor(bundle.rgb[2], cv2.COLOR_RGB2GRAY)
    em = _multiscale_edges(gray_m)

    sup_tex = (support > 0) & tex
    em_tex = (em > 0) & tex
    n_sup = max(sup_tex.sum(), 1)
    n_em = max(em_tex.sum(), 1)

    # Half-warped support edges land at subpixel positions, so recall and
    # precision must be Chamfer-tolerant (±2 px), not exact-intersection.
    dist_to_em = cv2.distanceTransform((em == 0).astype(np.uint8), cv2.DIST_L2, 3)
    dist_to_sup = cv2.distanceTransform((support == 0).astype(np.uint8),
                                        cv2.DIST_L2, 3)
    out["edge_recall"] = float((dist_to_em[sup_tex] <= 2.0).mean())
    out["edge_precision"] = float((dist_to_sup[em_tex] <= 2.0).mean())
    # Ghosting / double contour: candidate edges 2–4 px away from support —
    # a parallel second silhouette, not a match and not unrelated structure.
    out["edge_ghost_frac"] = float(
        ((dist_to_sup[em_tex] > 2.0) & (dist_to_sup[em_tex] <= 4.0)).mean())

    # --- chamfer distances -----------------------------------------------------
    out["edge_chamfer_sup_to_em"] = float(dist_to_em[sup_tex].mean()) if sup_tex.any() else float("nan")
    out["edge_chamfer_em_to_sup"] = float(dist_to_sup[em_tex].mean()) if em_tex.any() else float("nan")

    # --- per-instance aggregation (§7.2) ---------------------------------------
    n_lab, labels, stats, _ = cv2.connectedComponentsWithStats(support, connectivity=8)
    recalls = []
    for lab in range(1, n_lab):
        area = stats[lab, cv2.CC_STAT_AREA]
        if area < 12:
            continue
        comp = labels == lab
        comp_tex = comp & tex
        if comp_tex.sum() < 8:
            continue
        r = ((em > 0) & comp_tex).sum() / max(comp_tex.sum(), 1)
        recalls.append(float(r))
    recalls = recalls[:_MAX_INSTANCES]
    if recalls:
        out["edge_inst_recall_med"] = float(np.median(recalls))
        out["edge_inst_recall_p10"] = float(np.percentile(recalls, 10))
        out["edge_inst_count"] = float(len(recalls))
    else:
        out["edge_inst_recall_med"] = float("nan")
        out["edge_inst_recall_p10"] = float("nan")
        out["edge_inst_count"] = 0.0

    # --- skeleton-length alternation across the window (flicker of thin
    # structures, §7): edge-pixel count inside the support region per frame,
    # compare generated vs anchor frames.
    counts = []
    region = cv2.dilate(support, np.ones((7, 7), np.uint8)) > 0
    for t in range(bundle.rgb.shape[0]):
        gt = cv2.cvtColor(bundle.rgb[t], cv2.COLOR_RGB2GRAY)
        et = _multiscale_edges(gt)
        counts.append(float(((et > 0) & region).sum()))
    c = np.asarray(counts)
    idx = bundle.indices.astype(np.int64)
    even = c[idx % 2 == 0]
    odd = c[idx % 2 == 1]
    out["edge_count_odd_even_ratio"] = float(odd.mean() / max(even.mean(), 1e-6)) \
        if len(even) and len(odd) and even.mean() > 16 else 1.0
    return out
