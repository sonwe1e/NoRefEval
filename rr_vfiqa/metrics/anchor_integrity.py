"""Anchor integrity checks (USERPLAN.md §2.2).

Even candidate frames should *be* the source frames. We measure how far they
are (encoding baseline), whether a one-frame shift fits better (misalignment),
and whether color/gamma drifted.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..config import EvalConfig
from ..io.color_normalization import ColorTransform, estimate_from_anchor_pairs
from ..io.video_reader import VideoReader
from ..schema import Alignment, percentiles

_EVAL_WIDTH = 640
_MAX_ANCHORS = 12        # probed anchors (was 24 — decode-bound at 1080p, §5)
_SHIFT_PROBES = 4        # of those, how many also test ±1-frame misalignment


def _channel_errors(src: np.ndarray, cand: np.ndarray) -> dict[str, float]:
    from ..imutils import luma

    s = src.astype(np.float32)
    c = cand.astype(np.float32)
    y_s = luma(s)
    y_c = luma(c)
    g_s = cv2.Sobel(y_s, cv2.CV_32F, 1, 0) ** 2 + cv2.Sobel(y_s, cv2.CV_32F, 0, 1) ** 2
    g_c = cv2.Sobel(y_c, cv2.CV_32F, 1, 0) ** 2 + cv2.Sobel(y_c, cv2.CV_32F, 0, 1) ** 2
    chroma_s = np.stack([s[..., 0] - y_s, s[..., 2] - y_s], -1)
    chroma_c = np.stack([c[..., 0] - y_c, c[..., 2] - y_c], -1)
    return {
        "y_l1": float(np.abs(y_s - y_c).mean()),
        "chroma_l1": float(np.abs(chroma_s - chroma_c).mean()),
        "grad_l1": float(np.abs(np.sqrt(g_s) - np.sqrt(g_c)).mean()),
    }


def evaluate_anchors(cfg: EvalConfig, source: VideoReader, candidate: VideoReader,
                     alignment: Alignment
                     ) -> tuple[dict[str, float], ColorTransform, list[str]]:
    """Returns (feature dict, fitted color transform, warnings)."""
    warnings: list[str] = []
    anchors = np.nonzero(alignment.anchor_of_candidate >= 0)[0]
    if len(anchors) == 0:
        return {"anchor_y_l1_p50": float("nan")}, ColorTransform.identity(), \
               ["no anchors mapped"]

    pick = anchors[np.linspace(0, len(anchors) - 1,
                               min(_MAX_ANCHORS, len(anchors))).astype(int)]

    # First pass: color transform from a few pairs.  Decoded pairs are kept
    # so the second pass does not re-read (and re-seek) the same anchors.
    pairs: list[tuple[int, np.ndarray, np.ndarray]] = []
    for k in pick[:8]:
        s = source.read_one(int(alignment.anchor_of_candidate[k]), width=_EVAL_WIDTH)
        c = candidate.read_one(int(k), width=_EVAL_WIDTH)
        pairs.append((int(k), s, c))
    color = estimate_from_anchor_pairs([(s, c) for _, s, c in pairs])
    if not color.is_trivial():
        warnings.append(f"global color mismatch fitted: gain={np.round(color.gain, 3)} "
                        f"offset={np.round(color.offset, 2)} (residual "
                        f"{color.residual:.2f})")

    errs = {k: [] for k in ("y_l1", "chroma_l1", "grad_l1")}
    errs_shift = []
    shift_better = 0
    reused = {int(k): (s, c) for k, s, c in pairs}
    for n_probe, k in enumerate(pick):
        i = int(alignment.anchor_of_candidate[k])
        if int(k) in reused:
            s, c = reused[int(k)]
        else:
            s = source.read_one(i, width=_EVAL_WIDTH)
            c = candidate.read_one(int(k), width=_EVAL_WIDTH)
        c_norm = color.apply(c)
        e = _channel_errors(s, c_norm)
        for kk, v in e.items():
            errs[kk].append(v)
        # Would the neighbouring candidate frame match better? => misalignment.
        # Only the first few anchors pay for the extra ±1 reads.
        if n_probe < _SHIFT_PROBES:
            best_shift = e["y_l1"]
            for dk in (-1, 1):
                k2 = int(k) + dk
                if 0 <= k2 < candidate.meta.n_frames:
                    c2 = candidate.read_one(k2, width=_EVAL_WIDTH)
                    e2 = _channel_errors(s, color.apply(c2))
                    best_shift = min(best_shift, e2["y_l1"])
            errs_shift.append(best_shift)
            if best_shift < 0.6 * e["y_l1"] and e["y_l1"] > 3.0:
                shift_better += 1

    feats: dict[str, float] = {}
    for kk, vals in errs.items():
        feats.update({f"anchor_{kk}_{p}": v
                      for p, v in percentiles(np.asarray(vals), (50, 90)).items()})
    shift_gain = (1.0 - (np.median(errs_shift) / max(np.median(errs["y_l1"]), 1e-6))
                  if errs_shift else 0.0)
    feats["anchor_shift_gain"] = float(np.clip(shift_gain, 0.0, 1.0))
    feats["anchor_shift_better_frac"] = float(shift_better / max(len(pick), 1))
    feats["anchor_color_residual"] = float(color.residual)

    if feats["anchor_shift_better_frac"] > 0.3:
        warnings.append("candidate frames align better at shifted positions — "
                        "even/odd mapping is likely wrong")
    if feats["anchor_y_l1_p90"] > 12.0 and feats["anchor_color_residual"] < 4.0:
        warnings.append("anchors differ from source beyond encoding noise even after "
                        "color fit — the interpolator may modify original frames")
    return feats, color, warnings
