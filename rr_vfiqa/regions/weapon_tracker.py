"""Weapon / appendage jitter branch (USERPLAN §8.5).

Point tracks are made body-relative (median point = body proxy) and the
generated center frame is checked against the anchor-interpolated trajectory:
second-difference spikes at odd positions are the classic sword-tip jitter.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..config import EvalConfig
from ..metrics.window_flows import WindowFlows
from ..models.tracker_backend import TrackerBackend, get_tracker_backend
from ..schema import FrameBundle

_MAX_POINTS = 220


def _corner_points(gray: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    pts = cv2.goodFeaturesToTrack(gray, maxCorners=_MAX_POINTS, qualityLevel=0.01,
                                  minDistance=8, mask=mask, blockSize=7)
    if pts is None:
        return np.zeros((0, 2), np.float32)
    return pts.reshape(-1, 2).astype(np.float32)


def compute_window(bundle: FrameBundle, flow: WindowFlows, cfg: EvalConfig,
                   tracker: TrackerBackend | None = None,
                   roi_mask: np.ndarray | None = None) -> dict[str, float]:
    tracker = tracker or get_tracker_backend("auto")
    grays = [cv2.cvtColor(bundle.rgb[t], cv2.COLOR_RGB2GRAY)
             for t in range(bundle.rgb.shape[0])]
    h, w = grays[0].shape

    # Prefer corners inside the character region (passed from the character
    # branch — without it this measures generic central corners, §3.7); fall
    # back to the central frame area.
    if roi_mask is not None and roi_mask.sum() > 100:
        mask_u8 = cv2.dilate(roi_mask.astype(np.uint8), np.ones((11, 11), np.uint8))
    else:
        mask_u8 = np.zeros((h, w), np.uint8)
        mask_u8[h // 4: 3 * h // 4, w // 4: 3 * w // 4] = 1
    pts = _corner_points(grays[1], mask_u8)
    out: dict[str, float] = {"weapon_n_points": float(len(pts)),
                             "weapon_has_char_roi": float(roi_mask is not None
                                                          and roi_mask.sum() > 100)}
    if len(pts) < 12:
        out.update({"weapon_dev_p90": float("nan"), "weapon_dir_change_p90": float("nan")})
        return out

    # Points live on frame 1 (X_i): track forward through 2,3,4 and backward
    # to 0, then stitch. (The old code handed frame-1 points to a tracker
    # that assumed frame 0 — the trajectory started in the wrong frame.)
    tracks = np.full((5, len(pts), 2), np.nan, np.float32)
    vis = np.zeros((5, len(pts)), bool)
    fwd_t, fwd_v = tracker.track(grays[1:], pts)
    tracks[1:], vis[1:] = fwd_t, fwd_v
    back_t, back_v = tracker.track(grays[1::-1], pts)
    tracks[0], vis[0] = back_t[1], back_v[1]
    # Body-relative positions: subtract the per-frame median of visible points.
    rel = np.full_like(tracks, np.nan, dtype=np.float64)
    for t in range(tracks.shape[0]):
        v = vis[t]
        if v.sum() >= 6:
            med = np.median(tracks[t, v], axis=0)
            rel[t] = tracks[t] - med

    # Points visible at both anchors and the center.
    good = vis[1] & vis[2] & vis[3]
    if good.sum() < 8:
        out.update({"weapon_dev_p90": float("nan"), "weapon_dir_change_p90": float("nan")})
        return out
    r1, r2, r3 = rel[1, good], rel[2, good], rel[3, good]
    expected = 0.5 * (r1 + r3)
    span = np.linalg.norm(r3 - r1, axis=1) + 4.0
    dev = np.linalg.norm(r2 - expected, axis=1) / span
    out["weapon_dev_mean"] = float(dev.mean())
    out["weapon_dev_p90"] = float(np.percentile(dev, 90))

    # Direction reversal at the generated center (1-cos of the turn angle).
    a = r2 - r1
    b = r3 - r2
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    moving = (na > 1.5) & (nb > 1.5)
    if moving.sum() >= 6:
        cos = np.einsum("ij,ij->i", a[moving], b[moving]) / (na[moving] * nb[moving])
        out["weapon_dir_change_p90"] = float(np.percentile(1.0 - cos, 90))
    else:
        out["weapon_dir_change_p90"] = float("nan")

    # Odd-vs-even trajectory variance using the outer generated frames too.
    if vis[0].sum() >= 6 and vis[4].sum() >= 6:
        g = good & vis[0] & vis[4]
        if g.sum() >= 6:
            odd = np.stack([rel[0, g], rel[2, g], rel[4, g]])   # generated
            odd_var = np.linalg.norm(odd.std(0), axis=1)
            out["weapon_odd_spread"] = float(np.median(odd_var))
    return out
