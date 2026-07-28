"""Source ↔ candidate alignment by PTS, anchor verification, scene-cut marking.

USERPLAN.md §2: never assume decoded frame 2i is source frame i. Detect the
offset from timestamps, verify anchors in pixel space, and flag trouble.
"""

from __future__ import annotations

import numpy as np

from ..config import EvalConfig
from ..schema import Alignment, VideoMeta, robust_z
from .video_reader import VideoReader

_VERIFY_WIDTH = 320
_N_VERIFY = 12


def _y_l1(a: np.ndarray, b: np.ndarray) -> float:
    """Mean absolute luma difference between two RGB uint8 frames."""
    if a.shape != b.shape:
        import cv2
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
    ya = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
    yb = 0.299 * b[..., 0] + 0.587 * b[..., 1] + 0.114 * b[..., 2]
    return float(np.mean(np.abs(ya.astype(np.float32) - yb.astype(np.float32))))


def detect_offset(source_meta: VideoMeta, cand_meta: VideoMeta) -> tuple[int, float]:
    """Find the candidate-frame offset k0 so that Y_{2i + k0} ↔ X_i by PTS.

    Returns (offset, mean_abs_pts_error_seconds).
    """
    ratio = cand_meta.fps / max(source_meta.fps, 1e-6)
    best_off, best_err = 0, float("inf")
    n_pairs = min(source_meta.n_frames, cand_meta.n_frames // 2) - 1
    if n_pairs < 4:
        return 0, float("inf")
    sample = np.linspace(0, n_pairs - 1, min(64, n_pairs)).astype(int)

    for off in (-3, -2, -1, 0, 1, 2, 3):
        cand_idx = 2 * sample + off
        ok = (cand_idx >= 0) & (cand_idx < cand_meta.n_frames)
        if ok.sum() < 4:
            continue
        cand_pts = cand_meta.pts_seconds[cand_idx[ok]]
        src_pts = source_meta.pts_seconds[sample[ok]]
        # Scale source pts into the candidate clock if clocks differ slightly.
        err = float(np.mean(np.abs(cand_pts - src_pts)))
        if err < best_err:
            best_err, best_off = err, off
    return best_off, best_err


def detect_scene_cuts(reader: VideoReader, width: int = 256,
                      hash_z: float = 6.0, hash_floor: float = 8.0,
                      hist_z: float = 8.0, hist_floor: float = 0.35) -> np.ndarray:
    """Frame indices where a hard cut is likely (standalone fallback when no
    tier-1 scan exists).

    Structure-based triggers only:
    * perceptual-hash hamming: robust structural change signal;
    * color-histogram Bhattacharyya: motion preserves the global palette,
      a cut destroys it — the decisive signal under constant camera motion.
    A pure luma-diff trigger is deliberately absent: it fires on abrupt
    defects inside a scene and is diluted by constant motion.
    """
    import cv2

    from ..imutils import hamming64, phash64

    def hist(img_rgb: np.ndarray) -> np.ndarray:
        h = cv2.calcHist([img_rgb], [0, 1, 2], None, [16, 16, 16], [0, 256] * 3)
        return cv2.normalize(h, None).flatten()

    prev_gray = None
    prev_hash = None
    prev_hist = None
    hdist: list[float] = []
    bhatt: list[float] = []
    idxs: list[int] = []
    for idx, img in reader.iter_frames(width=width):
        gray = np.mean(img.astype(np.float32), axis=-1).astype(np.float32)
        h = phash64(gray.astype(np.uint8))
        hg = hist(img)
        if prev_gray is not None and gray.shape == prev_gray.shape:
            hdist.append(float(hamming64(h, prev_hash)))
            bhatt.append(float(cv2.compareHist(prev_hist, hg,
                                               cv2.HISTCMP_BHATTACHARYYA)))
            idxs.append(idx)
        prev_gray = gray
        prev_hash = h
        prev_hist = hg
    if len(hdist) < 8:
        return np.zeros(0, np.int32)
    hd = np.asarray(hdist)
    bh = np.asarray(bhatt)
    zh, zb = robust_z(hd), robust_z(bh)
    cuts = [idxs[i] for i in range(len(hdist))
            if ((zh[i] > hash_z and hd[i] > hash_floor)
                or (zb[i] > hist_z and bh[i] > hist_floor))]
    return np.asarray(cuts, np.int32)


def build_alignment(cfg: EvalConfig, source: VideoReader, candidate: VideoReader,
                    scene_cuts_cand: np.ndarray | None = None) -> Alignment:
    sm, cm = source.meta, candidate.meta
    warnings: list[str] = []

    ratio = cm.fps / max(sm.fps, 1e-6)
    if not (1.9 <= ratio <= 2.1):
        warnings.append(
            f"fps ratio {ratio:.3f} is not ~2.0 (source {sm.fps:.2f}, "
            f"candidate {cm.fps:.2f}); alignment may be unreliable")

    offset, pts_err = detect_offset(sm, cm)
    if pts_err > 0.5 / max(sm.fps, 1e-6):
        warnings.append(f"pts alignment error {pts_err * 1000:.1f} ms exceeds half "
                        f"a source frame; check timestamps/VFR")

    n_cand = cm.n_frames
    n_src = sm.n_frames
    anchor_of = np.full(n_cand, -1, np.int32)
    pair_of = np.full(n_cand, -1, np.int32)

    # Candidate frame 2i+offset -> source i ; odd neighbour -> pair i.
    i_max = min(n_src - 1, (n_cand - 1 - offset) // 2)
    for i in range(max(0, i_max + 1)):
        k = 2 * i + offset
        if 0 <= k < n_cand:
            anchor_of[k] = i
        if 0 <= k + 1 < n_cand and i + 1 < n_src:
            pair_of[k + 1] = i

    if offset < 0:
        warnings.append(f"detected negative first-anchor offset {offset}: candidate "
                        f"starts before source frame 0")

    # --- pixel verification of anchors (also yields the encoding baseline) ---
    anchors = np.nonzero(anchor_of >= 0)[0]
    anchor_err = 0.0
    if len(anchors) >= 2:
        pick = anchors[np.linspace(0, len(anchors) - 1, min(_N_VERIFY, len(anchors))).astype(int)]
        errs0, errs_shift = [], []
        for k in pick:
            i = int(anchor_of[k])
            a = candidate.read_one(int(k), width=_VERIFY_WIDTH)
            b = source.read_one(i, width=_VERIFY_WIDTH)
            errs0.append(_y_l1(a, b))
            if 0 < k + 1 < n_cand:
                a2 = candidate.read_one(int(k) + 1, width=_VERIFY_WIDTH)
                errs_shift.append(_y_l1(a2, b))
        anchor_err = float(np.median(errs0))
        if errs_shift and np.median(errs_shift) < 0.5 * anchor_err and anchor_err > 2.0:
            warnings.append("anchors verify better at shifted positions: frames may "
                            "be misaligned by one")
        if anchor_err > 20.0:
            warnings.append(f"anchor mismatch Y-L1={anchor_err:.1f}: candidate even "
                            f"frames differ strongly from source (modified frames or "
                            f"color range issue)")

    # --- scene cuts, mapped to source pair indices ---
    # Reuse the tier-1 scan's statistics when available (the pipeline passes
    # them); only pay for a dedicated decode pass when called standalone.
    cuts_cand = (np.asarray(scene_cuts_cand, np.int32) if scene_cuts_cand is not None
                 else detect_scene_cuts(candidate,
                                        width=max(192, cfg.preset.scan_width // 2)))
    pair_cuts = sorted({int(pair_of[k]) for k in cuts_cand
                        if 0 <= k < n_cand and pair_of[k] >= 0})
    # A cut AT an anchor means pair (i-1) spans the cut as well.
    for k in cuts_cand:
        if 0 <= k < n_cand and anchor_of[k] >= 0:
            i = int(anchor_of[k])
            if i - 1 >= 0:
                pair_cuts.append(i - 1)
    scene_pairs = np.unique(np.asarray(sorted(set(pair_cuts)), np.int32))

    return Alignment(
        anchor_of_candidate=anchor_of,
        pair_of_candidate=pair_of,
        fps_ratio=float(ratio),
        first_anchor_offset=int(offset),
        anchor_error=anchor_err,
        scene_cuts=scene_pairs,
        warnings=warnings,
    )
