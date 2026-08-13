"""Tier-1: low-resolution full-frame scan of the candidate (USERPLAN §10).

Every frame gets a handful of scalar statistics at ~320-480 px width. This is
what guarantees short bad cases are never *completely* missed even though the
expensive core metrics only touch a few percent of frames.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from ..io.video_reader import VideoReader

# Luma delta (0-255) above which a pixel counts as "moving" for the scan-level
# cadence motion gate.  Kept deliberately low so genuine content motion is not
# dismissed; a collapsed cadence still shows a high parent-lag moving fraction.
MOVING_PIXEL_DELTA = 3.0


@dataclass
class CheapScan:
    width: int
    indices: np.ndarray             # (N,) int decode/frame index
    luma_mean: np.ndarray           # (N,)
    luma_std: np.ndarray
    sharpness: np.ndarray           # variance of Laplacian
    grad_energy: np.ndarray         # mean Sobel magnitude
    edge_frac: np.ndarray           # Canny edge pixel fraction
    hash64: np.ndarray              # (N,) uint64 perceptual hashes
    hash_dist: np.ndarray           # (N,) hamming distance to previous frame
    frame_diff: np.ndarray          # (N,) mean |Y_k - Y_{k-1}| at scan res
    hist_dist: np.ndarray           # (N,) per-channel Bhattacharyya to previous
    # USERPLAN §4.1/§4.4: parent-lag (2× native) differences and moving-pixel
    # fractions maintained by the Tier-1 scan so the *full-film* cadence is
    # estimated from uniform per-scene statistics, never from risk-selected
    # windows. ``moving_frac_*`` is the fraction of pixels whose luma moved by
    # more than ``MOVING_PIXEL_DELTA`` at that lag.
    frame_diff_parent: np.ndarray   # (N,) mean |Y_k - Y_{k-2}| at scan res
    moving_frac_native: np.ndarray  # (N,) moving-pixel fraction at lag-1
    moving_frac_parent: np.ndarray  # (N,) moving-pixel fraction at lag-2
    # (N, 9, 16) float32 luma descriptors (16x9 AREA downscale of the scan
    # gray), reused by alignment instead of a second full decode; empirically
    # identical alignment outcomes to the 96px descriptor pass.
    descriptors: np.ndarray | None = None

    def parity_sharpness_gap(self) -> np.ndarray:
        """|sharpness_even - local trend| — systematic odd-frame blur shows up
        as a one-frame-alternating signal (USERPLAN §6.1)."""
        q = self.sharpness.astype(np.float64)
        n = len(q)
        trend = np.empty_like(q)
        trend[1:-1] = 0.5 * (q[:-2] + q[2:])
        trend[0], trend[-1] = q[0], q[-1]
        gap = q - trend                       # >0 sharper than neighbours
        # one-frame alternation: multiply by (-1)^k and smooth
        k = self.indices.astype(np.int64)
        signed = gap * ((-1.0) ** (k % 2))
        kern = np.ones(9) / 9.0
        return np.abs(np.convolve(signed, kern, mode="same"))[:n]

    def edge_flicker(self) -> np.ndarray:
        """Lag-1 change of edge fraction — contours appearing/disappearing."""
        e = self.edge_frac.astype(np.float64)
        out = np.zeros_like(e)
        out[1:] = np.abs(np.diff(e))
        return out

    def scene_cut_prob(self) -> np.ndarray:
        z_hash = self.hash_dist.astype(np.float64)
        z_diff = self.frame_diff.astype(np.float64)
        med_h, mad_h = np.median(z_hash), np.median(np.abs(z_hash - np.median(z_hash)))
        med_d, mad_d = np.median(z_diff), np.median(np.abs(z_diff - np.median(z_diff)))
        zh = (z_hash - med_h) / (1.4826 * mad_h + 1e-6)
        zd = (z_diff - med_d) / (1.4826 * mad_d + 1e-6)
        return np.maximum(zh, zd)


def scene_cuts_from_scan(scan: "CheapScan", hash_z: float = 6.0,
                         hash_floor: float = 8.0, hist_z: float = 8.0,
                         hist_floor: float = 0.20) -> np.ndarray:
    """Cut frames from scan statistics, reusing the scan's decode pass
    (§5: the candidate must not be decoded twice).

    Structure-based triggers only (hash / color histogram): a luma-diff
    trigger would fire on abrupt *defects* inside one scene (ghost segment
    onsets, tears), whereas real cuts change structure and palette.
    """
    from ..schema import robust_z

    hd = scan.hash_dist.astype(np.float64)
    hg = scan.hist_dist.astype(np.float64)
    zh, zhg = robust_z(hd), robust_z(hg)
    cut = ((zh > hash_z) & (hd > hash_floor)) | ((zhg > hist_z) & (hg > hist_floor))
    return scan.indices[cut]


def _phash64(gray: np.ndarray) -> np.int64:
    from ..imutils import phash64
    return phash64(gray)


def _hamming(a: np.int64, b: np.int64) -> int:
    from ..imutils import hamming64
    return hamming64(a, b)


def scan_candidate(reader: VideoReader, width: int = 384) -> CheapScan:
    idxs, lmean, lstd, sharp, grad, edgef, hashes = [], [], [], [], [], [], []
    prev_gray = None
    prev2_gray = None
    prev_hist = None
    hdist, fdiff, hgdist = [0.0], [0.0], [0.0]
    fdiff_parent, mfrac_native, mfrac_parent = [0.0], [0.0], [0.0]
    descriptors: list[np.ndarray] = []

    for idx, img in reader.iter_frames(width=width):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        descriptors.append(cv2.resize(gray, (16, 9),
                                      interpolation=cv2.INTER_AREA)
                           .astype(np.float32))
        gf = gray.astype(np.float32)
        lap = cv2.Laplacian(gray, cv2.CV_32F)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        edges = cv2.Canny(gray, 60, 160)
        hist = [_norm_hist(cv2.calcHist([img], [c], None, [32], [0, 256]))
                for c in range(3)]
        idxs.append(idx)
        lmean.append(float(gf.mean()))
        lstd.append(float(gf.std()))
        sharp.append(float(lap.var()))
        grad.append(float(np.sqrt(gx * gx + gy * gy).mean()))
        edgef.append(float(edges.mean() / 255.0))
        hashes.append(_phash64(gray))
        if prev_gray is not None and prev_gray.shape == gray.shape:
            hdist.append(float(_hamming(hashes[-1], hashes[-2])))
            diff_native = np.abs(gf - prev_gray.astype(np.float32))
            fdiff.append(float(diff_native.mean()))
            mfrac_native.append(
                float(np.mean(diff_native > MOVING_PIXEL_DELTA)))
            hgdist.append(float(np.mean([
                cv2.compareHist(p, h, cv2.HISTCMP_BHATTACHARYYA)
                for p, h in zip(prev_hist, hist)])))
            if prev2_gray is not None and prev2_gray.shape == gray.shape:
                diff_parent = np.abs(gf - prev2_gray.astype(np.float32))
                fdiff_parent.append(float(diff_parent.mean()))
                mfrac_parent.append(
                    float(np.mean(diff_parent > MOVING_PIXEL_DELTA)))
            else:
                fdiff_parent.append(0.0)
                mfrac_parent.append(0.0)
        prev2_gray = prev_gray
        prev_gray = gray
        prev_hist = hist

    n = len(idxs)
    return CheapScan(
        width=width,
        indices=np.asarray(idxs, np.int32),
        luma_mean=np.asarray(lmean, np.float32),
        luma_std=np.asarray(lstd, np.float32),
        sharpness=np.asarray(sharp, np.float32),
        grad_energy=np.asarray(grad, np.float32),
        edge_frac=np.asarray(edgef, np.float32),
        hash64=np.asarray(hashes, np.int64),
        hash_dist=np.asarray(hdist + [0.0] * (n - len(hdist)), np.float32)[:n],
        frame_diff=np.asarray(fdiff + [0.0] * (n - len(fdiff)), np.float32)[:n],
        hist_dist=np.asarray(hgdist + [0.0] * (n - len(hgdist)), np.float32)[:n],
        frame_diff_parent=np.asarray(
            fdiff_parent + [0.0] * (n - len(fdiff_parent)), np.float32)[:n],
        moving_frac_native=np.asarray(
            mfrac_native + [0.0] * (n - len(mfrac_native)), np.float32)[:n],
        moving_frac_parent=np.asarray(
            mfrac_parent + [0.0] * (n - len(mfrac_parent)), np.float32)[:n],
        descriptors=(np.stack(descriptors) if descriptors
                     else np.zeros((0, 9, 16), np.float32)),
    )


def _norm_hist(h: np.ndarray) -> np.ndarray:
    return cv2.normalize(h, None, alpha=1.0, norm_type=cv2.NORM_L1)
