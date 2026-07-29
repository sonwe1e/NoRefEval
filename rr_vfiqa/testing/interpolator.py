"""Reference interpolation models for calibration corpora (USERPLAN §12.1).

Hand-crafted defects are not models — their artifacts are authored. This
module provides actual interpolators whose failure modes are *emergent*:

* ``linear_blend``  — the naive 50/50 crossfade (worst-case baseline);
* ``ReferenceInterpolator`` — forward-splat half-flow VFI on top of any
  FlowBackend. Its quality scales with the flow backend's working
  resolution, so running it at several widths yields an organic
  quality ladder against the pseudo-GT truth frames.

Both produce interleaved 120 FPS candidates from 60 FPS source frames,
exactly like the real models the evaluator will eventually rank.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..motion.flow_estimator import FlowBackend, get_flow_backend
from ..motion.occlusion import cycle_occlusion
from ..schema import downscale_frames, forward_splat


def linear_blend_candidate(source_frames: np.ndarray) -> np.ndarray:
    """Interleave source with naive 50/50 endpoint blends."""
    n = len(source_frames)
    mids = np.clip(
        0.5 * source_frames[:-1].astype(np.float32)
        + 0.5 * source_frames[1:].astype(np.float32), 0, 255).astype(np.uint8)
    last = source_frames[-1:]
    mids = np.concatenate([mids, last], 0)
    out = np.empty((2 * n,) + source_frames.shape[1:], np.uint8)
    out[0::2] = source_frames
    out[1::2] = mids
    return out


class ReferenceInterpolator:
    """Visibility-aware forward-splat half-flow interpolation."""

    def __init__(self, backend: FlowBackend, width: int | None = None):
        self.backend = backend
        self.width = width

    def interpolate_pair(self, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
        native = x0.shape[:2]
        if self.width and x0.shape[1] > self.width:
            pair, scale = downscale_frames(np.stack([x0, x1]), self.width)
            a, b = pair[0], pair[1]
        else:
            a, b = x0, x1

        flows = self.backend.flow_many([(a, b), (b, a)])
        f01, f10 = flows[0], flows[1]
        occ = cycle_occlusion(f01, f10)

        # Carry endpoint content — and its visibility weight — to the mid.
        vis0 = occ.conf_ab * (1.0 - occ.occ_ab.astype(np.float32))
        vis1 = occ.conf_ba * (1.0 - occ.occ_ba.astype(np.float32))
        s0, c0 = forward_splat(a.astype(np.float32), 0.5 * f01)
        s1, c1 = forward_splat(b.astype(np.float32), 0.5 * f10)
        v0, _ = forward_splat(vis0, 0.5 * f01)
        v1, _ = forward_splat(vis1, 0.5 * f10)
        w0 = np.clip(c0, 0, 1) * np.clip(v0, 0, 1)
        w1 = np.clip(c1, 0, 1) * np.clip(v1, 0, 1)
        wsum = w0 + w1
        blend = np.zeros_like(s0)
        ok = wsum > 0.05
        blend[ok] = (s0[ok] * w0[ok, None] + s1[ok] * w1[ok, None]) / wsum[ok, None]
        if not ok.all():
            # Disocclusion holes: inpaint from the blended surroundings.
            hole = (~ok).astype(np.uint8)
            blend_u8 = np.clip(blend, 0, 255).astype(np.uint8)
            blend_u8 = cv2.inpaint(blend_u8, hole, 5, cv2.INPAINT_TELEA)
            blend = blend_u8.astype(np.float32)

        out = np.clip(blend, 0, 255).astype(np.uint8)
        if out.shape[:2] != native:
            out = cv2.resize(out, (native[1], native[0]),
                             interpolation=cv2.INTER_LINEAR)
        return out

    def candidate(self, source_frames: np.ndarray,
                  progress=None) -> np.ndarray:
        """Full interleaved candidate from 60 FPS source frames."""
        n = len(source_frames)
        mids = []
        for i in range(n - 1):
            mids.append(self.interpolate_pair(source_frames[i], source_frames[i + 1]))
            if progress and i % max(1, n // 10) == 0:
                progress(i, n - 1)
        mids.append(source_frames[-1])
        mids = np.stack(mids, 0)
        out = np.empty((2 * n,) + source_frames.shape[1:], np.uint8)
        out[0::2] = source_frames
        out[1::2] = mids
        return out


def make_ladder(source_frames: np.ndarray, truth_mids: np.ndarray,
                flow_backend: str = "auto", device: str = "cuda",
                widths: tuple[int | None, ...] = (320, None)
                ) -> dict[str, np.ndarray]:
    """Ordered quality ladder of candidates against the same pseudo-GT.

    ``perfect`` (truth interleave) > flow-based at descending internal
    resolutions > ``linear`` blend. Keys: perfect, flow_w<w|native>, linear.
    """
    backend = get_flow_backend(flow_backend, device)
    ladder = {"perfect": _interleave_truth(source_frames, truth_mids),
              "linear": linear_blend_candidate(source_frames)}
    for width in widths:
        interp = ReferenceInterpolator(backend, width=width)
        key = f"flow_w{width if width else 'native'}"
        ladder[key] = interp.candidate(source_frames)
    return ladder


def _interleave_truth(source_frames: np.ndarray,
                      truth_mids: np.ndarray) -> np.ndarray:
    n = len(source_frames)
    out = np.empty((2 * n,) + source_frames.shape[1:], np.uint8)
    out[0::2] = source_frames
    out[1::2] = np.concatenate([truth_mids, source_frames[-1:]], 0)[:n]
    return out
