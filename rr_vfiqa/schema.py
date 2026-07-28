"""Shared data contracts for rr_vfiqa.

Notation (see USERPLAN.md §1):
    source video  X_0 .. X_N        (60 FPS anchors)
    candidate     Y_{2i} = X_i      (even frames should be the original anchors)
                  Y_{2i+1} = M_i    (odd frames are the interpolated frames)

All spatial arrays are stored at a documented resolution. Flow arrays are
(H, W, 2) float32 with (dx, dy) in pixel units *at that resolution*.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Video / alignment metadata
# ---------------------------------------------------------------------------

@dataclass
class VideoMeta:
    path: str
    width: int
    height: int
    n_frames: int
    fps: float
    pts_seconds: np.ndarray          # (n_frames,) presentation timestamps
    content_hash: str                # hash of path + size + mtime for caching
    codec: str = ""
    pix_fmt: str = ""

    def time_to_index(self, t: float) -> int:
        return int(np.clip(np.searchsorted(self.pts_seconds, t), 0, self.n_frames - 1))


@dataclass
class Alignment:
    """How candidate frames map onto source frames.

    anchor_of_candidate[k] = source index i  if candidate frame k is an anchor
                             (expected for even k), else -1.
    pair_of_candidate[k]   = source pair index i such that M_i interpolates
                             (X_i, X_{i+1}) — defined for odd k, else -1.
    """

    anchor_of_candidate: np.ndarray     # (n_candidate_frames,) int32
    pair_of_candidate: np.ndarray       # (n_candidate_frames,) int32
    fps_ratio: float                    # candidate_fps / source_fps, ~2.0
    first_anchor_offset: int = 0        # detected start offset in candidate frames
    anchor_error: float = 0.0           # measured anchor mismatch baseline (Y L1)
    scene_cuts: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int32))
    warnings: list[str] = field(default_factory=list)

    def generated_centers(self) -> np.ndarray:
        """Candidate indices of generated frames M_i (odd positions)."""
        return np.nonzero(self.pair_of_candidate >= 0)[0].astype(np.int32)


# ---------------------------------------------------------------------------
# Frames and flow
# ---------------------------------------------------------------------------

@dataclass
class FrameBundle:
    """A small set of decoded candidate frames."""

    indices: np.ndarray       # (T,) int32 absolute candidate frame indices
    times: np.ndarray         # (T,) float64 seconds
    rgb: np.ndarray           # (T, H, W, 3) uint8, RGB order
    width: int
    height: int

    def y_channel(self) -> np.ndarray:
        """(T, H, W) float32 luma in 0..255, BT.601 weights."""
        r = self.rgb[..., 0].astype(np.float32)
        g = self.rgb[..., 1].astype(np.float32)
        b = self.rgb[..., 2].astype(np.float32)
        return 0.299 * r + 0.587 * g + 0.114 * b

    def at(self, k: int) -> np.ndarray:
        return self.rgb[np.searchsorted(self.indices, k)]


@dataclass
class FlowPair:
    """Bidirectional flow between two frames at a single working resolution."""

    flow_ab: np.ndarray       # (H, W, 2) float32, a -> b
    flow_ba: np.ndarray       # (H, W, 2) float32, b -> a
    height: int
    width: int
    scale: float              # working_res / native_res (uniform)

    def to_native_scale(self) -> float:
        return 1.0 / self.scale


@dataclass
class OcclusionMasks:
    """Forward/backward occlusion derived from flow cycle consistency.

    Values are uint8 probabilities/flags in {0,1} (1 = occluded / unreliable).
    """

    occ_ab: np.ndarray        # (H, W) uint8 — pixels of a occluded in b
    occ_ba: np.ndarray        # (H, W) uint8 — pixels of b occluded in a
    conf_ab: np.ndarray       # (H, W) float32 in [0,1] — matching confidence
    conf_ba: np.ndarray       # (H, W) float32 in [0,1]


@dataclass
class CameraMotion:
    """Global camera motion estimate between two frames."""

    model: str                # "identity" | "translation" | "affine" | "homography"
    matrix: np.ndarray        # (2,3) for affine/translation, (3,3) for homography
    inlier_ratio: float
    residual_px: float

    def warp_flow(self, h: int, w: int) -> np.ndarray:
        """Dense (H, W, 2) flow implied by the global model at size (h, w)."""
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        pts = np.stack([xx.ravel(), yy.ravel(), np.ones(h * w, np.float32)], 0)
        if self.model == "homography":
            p = self.matrix @ pts
            p = p[:2] / np.clip(p[2:3], 1e-6, None)
        else:
            p = self.matrix @ pts
        return np.stack([p[0] - xx.ravel(), p[1] - yy.ravel()], -1).reshape(h, w, 2)


# ---------------------------------------------------------------------------
# Windows and features
# ---------------------------------------------------------------------------

@dataclass
class Window:
    """A 5-frame evaluation window on the candidate timeline."""

    center: int                     # candidate frame index (usually odd = generated)
    indices: np.ndarray             # (T,) int32 candidate frame indices
    pair: int                       # source pair index i for the center M_i, -1 if n/a
    risk: float = 0.0
    source: str = "uniform"         # "uniform" | "risk"
    risk_components: dict[str, float] = field(default_factory=dict)

    @property
    def start_time(self) -> float:
        return self.indices[0] / 120.0  # replaced with real pts by selector

    def __repr__(self) -> str:
        return f"Window(center={self.center}, pair={self.pair}, src={self.source}, risk={self.risk:.2f})"


@dataclass
class WindowFeatures:
    """All features computed for one window, keyed by metric family."""

    window: Window
    scalars: dict[str, float] = field(default_factory=dict)
    instances: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    error_maps: dict[str, np.ndarray] = field(default_factory=dict)
    labels: dict[str, Any] = field(default_factory=dict)

    def add(self, key: str, value: float) -> None:
        self.scalars[key] = float(value)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

SUBSCORE_KEYS = [
    "motion_consistency",
    "temporal_stability",
    "structural_integrity",
    "character_integrity",
    "thin_object_weapon",
    "ui_text",
    "transition_quality",
    "global_technical_quality",
]


@dataclass
class WorstWindow:
    start: float
    end: float
    center_index: int
    types: list[str]
    severity: float
    confidence: float
    boxes: list[list[int]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "center_index": int(self.center_index),
            "type": self.types,
            "severity": round(float(self.severity), 4),
            "confidence": round(float(self.confidence), 4),
            "boxes": self.boxes,
        }


@dataclass
class Report:
    overall_score: float
    confidence: float
    scores: dict[str, float]
    event_ambiguity: float
    worst_windows: list[WorstWindow]
    features: dict[str, Any]
    meta: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall_score": round(float(self.overall_score), 2),
            "confidence": round(float(self.confidence), 4),
            "event_ambiguity": round(float(self.event_ambiguity), 4),
            "scores": {k: round(float(v), 2) for k, v in self.scores.items()},
            "worst_windows": [w.to_dict() for w in self.worst_windows],
            "features": self.features,
            "meta": self.meta,
        }


# ---------------------------------------------------------------------------
# Small numerics helpers shared across modules
# ---------------------------------------------------------------------------

def charbonnier(x: np.ndarray, tau: float = 4.0) -> np.ndarray:
    """sqrt(x^2 + tau^2) - tau, ~|x| for large x, smooth near 0."""
    return np.sqrt(x * x + tau * tau) - tau


def robust_z(values: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """(x - median) / (1.4826 * MAD + eps)."""
    med = np.median(values)
    mad = np.median(np.abs(values - med))
    return (values - med) / (1.4826 * mad + eps)


def percentiles(values: np.ndarray, ps: tuple[float, ...] = (50, 90, 99)) -> dict[str, float]:
    if values.size == 0:
        return {f"p{p}": float("nan") for p in ps}
    qs = np.percentile(values, list(ps))
    return {f"p{p}": float(q) for p, q in zip(ps, qs)}


def warp_image(img: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """Backward-warp `img` by `flow`: out(x) = img(x + flow(x)).

    img: (H, W) or (H, W, C) float32. flow: (H, W, 2) in pixels (dx, dy).
    """
    import cv2

    h, w = flow.shape[:2]
    base_x, base_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x = base_x + flow[..., 0]
    map_y = base_y + flow[..., 1]
    return cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REFLECT_101)


def warp_flow(f_ab: np.ndarray, f_bc: np.ndarray) -> np.ndarray:
    """Compose flows: f_ac(x) = f_ab(x) + f_bc(x + f_ab(x))."""
    warped = warp_image(f_bc, f_ab)
    return f_ab + warped


def resize_flow(flow: np.ndarray, h: int, w: int) -> np.ndarray:
    """Resize a flow field to (h, w), rescaling magnitudes accordingly."""
    import cv2

    oh, ow = flow.shape[:2]
    if (oh, ow) == (h, w):
        return flow
    sx, sy = w / ow, h / oh
    out = cv2.resize(flow, (w, h), interpolation=cv2.INTER_LINEAR)
    out[..., 0] *= sx
    out[..., 1] *= sy
    return out


def flow_magnitude(flow: np.ndarray) -> np.ndarray:
    return np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)


def downscale_frames(frames: np.ndarray, width: int) -> tuple[np.ndarray, float]:
    """Downscale (T, H, W, ...) frames to target width, keep aspect. Returns (out, scale)."""
    import cv2

    _, h, w = frames.shape[:3]
    if width >= w:
        return frames, 1.0
    scale = width / w
    new_h = max(2, int(round(h * scale)) // 2 * 2)
    out = np.empty((frames.shape[0], new_h, width) + frames.shape[3:], dtype=frames.dtype)
    for t in range(frames.shape[0]):
        out[t] = cv2.resize(frames[t], (width, new_h), interpolation=cv2.INTER_AREA)
    return out, scale
