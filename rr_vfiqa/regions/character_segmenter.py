"""Character integrity branch (USERPLAN §8.2).

Endpoint character-proxy masks are flow-projected to the mid instant and
compared with the generated frame's own mask: missing area, extra area,
contour Chamfer, component changes, and endpoint-leak (ghosting evidence).
"""

from __future__ import annotations

import cv2
import numpy as np

from ..cache.source_cache import SourcePairData
from ..config import EvalConfig
from ..imutils import spatial_norm_factor
from ..metrics.window_flows import WindowFlows
from ..models.segmentation_backend import SegmentationBackend, get_segmentation_backend
from ..schema import FrameBundle, flow_magnitude, forward_splat, resize_flow
from ._common import bbox_of, clean_mask, luma, mask_chamfer


def _splat_mask(mask: np.ndarray, source_to_target_flow: np.ndarray) -> np.ndarray:
    """Carry a binary mask along a FORWARD flow.

    Thresholds the splatted value (fraction of splat mass coming from mask
    pixels): background zeros flow too, so raw coverage cannot distinguish
    "mask arrived" from "background arrived".
    """
    out, _ = forward_splat(mask.astype(np.float32), source_to_target_flow)
    return (out > 0.25).astype(np.uint8)


def compute_window(bundle: FrameBundle, flow: WindowFlows, pair: SourcePairData,
                   cfg: EvalConfig,
                   segmenter: SegmentationBackend | None = None
                   ) -> dict[str, float]:
    segmenter = segmenter or get_segmentation_backend("auto")
    h, w = flow.height(), flow.width()
    out: dict[str, float] = {}

    # Residual flow hints (dense minus camera) drive the classic segmenter.
    cam_flow = pair.camera.warp_flow(h, w) * pair.scale
    f_01 = resize_flow(pair.f_01, h, w)
    f_10 = resize_flow(pair.f_10, h, w)
    res_0 = f_01 - cam_flow
    res_1 = -f_10 + cam_flow  # residual seen from X_{i+1}

    s0 = segmenter.segment(bundle.rgb[1], {"residual_flow": res_0})
    s1 = segmenter.segment(bundle.rgb[3], {"residual_flow": res_1})
    sm = segmenter.segment(bundle.rgb[2], {"residual_flow": flow.forward(1, 2) - 0.5 * cam_flow})

    # Expected mid mask: forward-splat both endpoint masks halfway (f_01 is
    # X_i→X_{i+1}, f_10 is X_{i+1}→X_i) and fuse by union — the character may
    # grow into disoccluded areas.
    hat = np.maximum(_splat_mask(s0, 0.5 * f_01), _splat_mask(s1, 0.5 * f_10))
    hat = clean_mask(hat, min_area=48, close_k=5)

    out["char_expected_frac"] = float(hat.mean())
    # Character mask on the X_i grid for downstream branches (weapon tracker
    # point selection). "_"-prefixed keys are stripped from feature export.
    out["_char_mask"] = s0
    if hat.mean() < 0.002:
        out.update({"char_missing_frac": float("nan"), "char_extra_frac": float("nan"),
                    "char_chamfer": float("nan")})
        return out

    hb, mb = hat.astype(bool), sm.astype(bool)
    # Motion-threshold segmenters jitter by several pixels frame to frame;
    # compare with a dilation tolerance so jitter is not read as missing area.
    k7 = np.ones((7, 7), np.uint8)
    sm_tol = cv2.dilate(sm, k7).astype(bool)
    hat_tol = cv2.dilate(hat, k7).astype(bool)
    out["char_missing_frac"] = float(np.mean(hb & ~sm_tol))
    out["char_extra_frac"] = float(np.mean(mb & ~hat_tol))
    # USERPLAN §6.2: Chamfer distance normalized by frame diagonal.
    out["char_chamfer"] = mask_chamfer(hat, sm) / spatial_norm_factor(h, w)
    n_hat = cv2.connectedComponents(hat)[0] - 1
    n_sm = cv2.connectedComponents(sm)[0] - 1
    out["char_components_delta"] = float(abs(n_hat - n_sm))

    # Endpoint leak (§8.2 E_leak): how much M_i looks like a plain copy of one
    # endpoint inside the character region — combined with double edges this
    # is strong ghosting evidence. Endpoint luma is forward-splatted onto the
    # M_i grid; unsupported pixels are excluded from the statistic.
    f_0m = flow.forward(1, 2)
    f_1m = flow.forward(3, 2)
    xi_to_m, ci = forward_splat(luma(bundle.rgb[1]), f_0m)
    xj_to_m, cj = forward_splat(luma(bundle.rgb[3]), f_1m)
    ym = luma(bundle.rgb[2])
    supported = (ci > 0.5) | (cj > 0.5)
    leak = np.minimum(np.abs(ym - xi_to_m), np.abs(ym - xj_to_m))
    m = hb & supported
    # A SMALL leak means M_i resembles a plain copy of one endpoint — the bad
    # case — so fusion normalizes this inverted. Without local motion a copy
    # is legitimate (static character), hence the motion gate.
    if m.any() and hb.any() and float(np.median(flow_magnitude(f_01)[hb])) > 2.0:
        out["char_leak_mean"] = float(leak[m].mean())
    else:
        out["char_leak_mean"] = float("nan")

    # Double-contour ratio around the character (halo / second silhouette).
    band = cv2.dilate(hat, np.ones((7, 7), np.uint8)) > 0
    ring = band & ~hb
    em = cv2.Canny(cv2.cvtColor(bundle.rgb[2], cv2.COLOR_RGB2GRAY), 60, 160) > 0
    out["char_ring_edge_frac"] = float(em[ring].mean()) if ring.any() else float("nan")

    box = bbox_of(hat)
    if box:
        out["char_box"] = 1.0
        out["_char_bbox"] = box  # carried for badcase export (stripped later)
    return out
