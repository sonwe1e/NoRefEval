"""Synthetic 120 FPS game-like scene generator for tests and calibration.

Renders a deterministic scene with camera motion (translation + rotation),
a textured background, a world-fixed thin pole, an independently moving
character, and screen-static UI. From the 120 FPS render:

* source  = every 2nd frame (the 60 FPS anchors)
* truth   = the removed odd frames (pseudo ground truth, USERPLAN §12.1)
* good    = perfect interleaving
* bad     = interleaving with segment-local degradations (blur / ghost /
            freeze) that mimic real interpolation failures

Also usable to build calibration corpora: compute full-reference metrics on
`truth` vs endpoint-reference features on `bad` and learn the mapping.
"""

from __future__ import annotations

from pathlib import Path

import av
import cv2
import numpy as np


def _bg_canvas(rng: np.random.Generator, w2: int, h2: int) -> np.ndarray:
    canvas = np.zeros((h2, w2, 3), np.uint8)
    # smooth gradient base
    grad = np.linspace(40, 110, h2, dtype=np.uint8)
    canvas[..., :] = grad[:, None, None]
    # random structures for parallax texture
    for _ in range(500):
        x, y = rng.integers(0, w2 - 40), rng.integers(0, h2 - 40)
        bw, bh = rng.integers(12, 64), rng.integers(12, 64)
        color = tuple(int(c) for c in rng.integers(30, 230, 3))
        cv2.rectangle(canvas, (x, y), (x + bw, y + bh), color, -1)
    # fine checker texture so flow always locks
    yy, xx = np.mgrid[0:h2, 0:w2]
    checker = (((xx // 8) + (yy // 8)) % 2).astype(np.int16) * 18
    canvas = np.clip(canvas.astype(np.int16) + checker[..., None], 0, 255).astype(np.uint8)
    return canvas


def _draw_pole(frame: np.ndarray, mat: np.ndarray, w2: int, h2: int) -> None:
    """World-fixed vertical pole, projected through the camera affine."""
    x = int(w2 * 0.64)
    pts = np.float32([[x, 0], [x, h2]]).reshape(-1, 1, 2)
    proj = cv2.transform(pts, mat).reshape(-1, 2)
    cv2.line(frame, tuple(proj[0].astype(int)), tuple(proj[1].astype(int)),
             (25, 20, 20), 5, cv2.LINE_AA)
    cv2.line(frame, tuple(proj[0].astype(int)), tuple(proj[1].astype(int)),
             (70, 65, 60), 2, cv2.LINE_AA)


def _draw_character(frame: np.ndarray, cx: float, cy: float) -> None:
    cxi, cyi = int(cx), int(cy)
    # body
    cv2.rectangle(frame, (cxi - 18, cyi - 10), (cxi + 18, cyi + 46), (200, 60, 60), -1)
    for s in range(-10, 46, 8):   # stripes → trackable internal texture
        cv2.line(frame, (cxi - 18, cyi + s), (cxi + 18, cyi + s), (240, 220, 90), 2)
    # head
    cv2.circle(frame, (cxi, cyi - 26), 16, (235, 200, 170), -1)
    # sword — a thin moving appendage
    cv2.line(frame, (cxi + 14, cyi), (cxi + 52, cyi - 40), (210, 210, 220), 3,
             cv2.LINE_AA)


def _draw_ui(frame: np.ndarray, t: int, w: int, h: int) -> None:
    # top-left HP panel (slowly draining bar)
    cv2.rectangle(frame, (14, 12), (150, 44), (20, 20, 25), -1)
    cv2.rectangle(frame, (16, 14), (148, 42), (70, 70, 80), 1)
    frac = max(0.15, 0.9 - 0.12 * (t // 120))
    cv2.rectangle(frame, (18, 20), (int(18 + 128 * frac), 36), (60, 210, 90), -1)
    # bottom-right skill ring (screen-static geometry)
    cc = (w - 52, h - 52)
    cv2.circle(frame, cc, 34, (30, 30, 35), -1)
    cv2.circle(frame, cc, 34, (220, 220, 230), 2)
    for ang in range(0, 360, 45):
        a = np.deg2rad(ang)
        p2 = (int(cc[0] + 26 * np.cos(a)), int(cc[1] + 26 * np.sin(a)))
        cv2.line(frame, cc, p2, (160, 160, 170), 2)
    # text-like strokes top-right
    for i, x0 in enumerate(range(w - 170, w - 30, 16)):
        cv2.rectangle(frame, (x0, 16), (x0 + 9, 30), (235, 235, 235), -1)
        cv2.rectangle(frame, (x0 + 2, 20), (x0 + 7, 26), (20, 20, 25), -1)


def render_scene(n_frames: int = 480, w: int = 640, h: int = 360, fps: int = 120,
                 seed: int = 7) -> np.ndarray:
    """(N, H, W, 3) uint8 RGB at the requested fps."""
    rng = np.random.default_rng(seed)
    # Canvas oversized to cover the camera excursion without border artifacts.
    w2, h2 = int(w * 2.8), int(h * 2.4)
    canvas = _bg_canvas(rng, w2, h2)
    frames = np.empty((n_frames, h, w, 3), np.uint8)
    cx0, cy0 = w * 0.30, h * 0.52

    pan_x = 1.4 * w        # total camera pan over the clip (≈0.9 px/frame @640w)
    amp_y = 0.35 * h
    for t in range(n_frames):
        ph = t / n_frames
        # View center traverses the canvas; margins guarantee the window
        # never samples outside it (window half-size ≤ distance to border).
        ang = 9.0 * np.sin(2 * np.pi * ph)
        center = (w2 / 2.0, h2 / 2.0)
        rot = cv2.getRotationMatrix2D(center, ang, 1.0)
        rot[0, 2] += (w2 / 2.0 - w / 2.0) + (ph - 0.5) * pan_x
        rot[1, 2] += (h2 / 2.0 - h / 2.0) + amp_y * np.sin(2 * np.pi * ph)
        frame = cv2.warpAffine(canvas, rot, (w, h), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REFLECT_101)
        _draw_pole(frame, rot, w2, h2)
        cx = cx0 + (w * 0.42) * ph + 14 * np.sin(4 * np.pi * ph)
        cy = cy0 + 12 * np.sin(6 * np.pi * ph)
        _draw_character(frame, cx, cy)
        _draw_ui(frame, t, w, h)
        frames[t] = frame
    return frames


# ---------------------------------------------------------------------------
# video IO
# ---------------------------------------------------------------------------

def write_video(path: str | Path, frames: np.ndarray, fps: float,
                crf: int = 18) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames.shape[1:3]
    with av.open(str(path), "w") as out:
        st = out.add_stream("h264", rate=int(round(fps)))
        st.width = w
        st.height = h
        st.pix_fmt = "yuv420p"
        st.options = {"crf": str(crf), "preset": "medium"}
        for f in frames:
            frame = av.VideoFrame.from_ndarray(f, format="rgb24")
            for packet in st.encode(frame):
                out.mux(packet)
        for packet in st.encode():
            out.mux(packet)
    return path


# ---------------------------------------------------------------------------
# candidates
# ---------------------------------------------------------------------------

def make_source_and_truth(frames120: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(source_60 = even frames, truth_mids = odd frames)."""
    return frames120[0::2].copy(), frames120[1::2].copy()


def interleave(source: np.ndarray, mids: np.ndarray) -> np.ndarray:
    n = source.shape[0] + mids.shape[0]
    out = np.empty((n,) + source.shape[1:], source.dtype)
    out[0::2] = source[: (n + 1) // 2]
    out[1::2] = mids[: n // 2]
    return out


def make_bad_mids(source: np.ndarray, truth: np.ndarray
                  ) -> tuple[np.ndarray, dict[str, tuple[int, int]]]:
    """Apply segment-local degradations to the mid frames.

    Segments (in mid-frame indices): blur, crossfade ghost, freeze-copy.
    Even frames (anchors) are never touched.
    """
    n = len(truth)
    mids = truth.copy()
    segs = {"blur": (n // 8, n // 8 + n // 12),
            "ghost": (3 * n // 8, 3 * n // 8 + n // 12),
            "freeze": (6 * n // 8, 6 * n // 8 + n // 12)}

    a, b = segs["blur"]
    for i in range(a, min(b, n)):
        mids[i] = cv2.GaussianBlur(mids[i], (0, 0), 3.0)

    a, b = segs["ghost"]
    for i in range(a, min(b, n)):
        xi = source[i].astype(np.float32)
        xj = source[min(i + 1, len(source) - 1)].astype(np.float32)
        mids[i] = np.clip(0.5 * xi + 0.5 * xj, 0, 255).astype(np.uint8)

    a, b = segs["freeze"]
    for i in range(a, min(b, n)):
        mids[i] = source[i]

    return mids, segs


def build_test_set(out_dir: str | Path, n_frames: int = 480, w: int = 640,
                   h: int = 360, seed: int = 7) -> dict[str, Path]:
    """Render + encode the full synthetic test corpus. Returns paths."""
    out_dir = Path(out_dir)
    frames = render_scene(n_frames=n_frames, w=w, h=h, seed=seed)
    source, truth = make_source_and_truth(frames)
    good = frames
    bad_mids, _ = make_bad_mids(source, truth)
    bad = interleave(source, bad_mids)

    return {
        "source": write_video(out_dir / "source_60.mp4", source, 60),
        "truth": write_video(out_dir / "truth_mids_60.mp4", truth, 60),
        "good": write_video(out_dir / "candidate_good_120.mp4", good, 120),
        "bad": write_video(out_dir / "candidate_bad_120.mp4", bad, 120),
    }
