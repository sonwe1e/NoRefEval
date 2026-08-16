"""Synthetic 120 FPS game-like scene generator for tests and calibration.

Renders a deterministic scene with camera motion (translation + rotation),
a textured background, a world-fixed thin pole, an independently moving
striped character with a sword, and screen-static UI (HP panel, skill ring,
text blocks). From the 120 FPS render:

* source  = every 2nd frame (the 60 FPS anchors)
* truth   = the removed odd frames (pseudo ground truth, USERPLAN §12.1)
* good    = perfect interleaving
* bad     = interleaving with segment-local degradations

Defect library (USERPLAN §6/§12.3): blur, crossfade ghost, freeze, rotation
tear, head erasure, pole motion-attribution error, sword flicker, UI subpixel
drift, text-stroke merging, shop state double-exposure, disocclusion fill,
plus scene-cut splice, frame-offset and dropped-frame corpus variants.

Also usable to build calibration corpora: compute full-reference metrics on
`truth` vs endpoint-reference features on `bad` and learn the mapping.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import av
import cv2
import numpy as np


@dataclass
class SceneMeta:
    """Per-120FPS-frame ground truth used to craft precise defects."""

    camera: np.ndarray      # (N, 2, 3) canvas→frame affine matrices
    char_pos: np.ndarray    # (N, 2) character body center (cx, cy)
    pole_pts: np.ndarray    # (N, 2, 2) pole line endpoints in frame coords
    card: np.ndarray        # (N, 4) card cx, cy, half_width, face_sign(±1)
    w: int
    h: int

    @property
    def n(self) -> int:
        return len(self.char_pos)


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

def _bg_canvas(rng: np.random.Generator, w2: int, h2: int) -> np.ndarray:
    canvas = np.zeros((h2, w2, 3), np.uint8)
    grad = np.linspace(40, 110, h2, dtype=np.uint8)
    canvas[..., :] = grad[:, None, None]
    for _ in range(500):
        x, y = rng.integers(0, w2 - 40), rng.integers(0, h2 - 40)
        bw, bh = rng.integers(12, 64), rng.integers(12, 64)
        color = tuple(int(c) for c in rng.integers(30, 230, 3))
        cv2.rectangle(canvas, (x, y), (x + bw, y + bh), color, -1)
    yy, xx = np.mgrid[0:h2, 0:w2]
    checker = (((xx // 8) + (yy // 8)) % 2).astype(np.int16) * 18
    canvas = np.clip(canvas.astype(np.int16) + checker[..., None], 0, 255).astype(np.uint8)
    return canvas


def _draw_pole(frame: np.ndarray, mat: np.ndarray, w2: int, h2: int) -> np.ndarray:
    """World-fixed vertical pole; returns its projected endpoints."""
    x = int(w2 * 0.64)
    pts = np.float32([[x, 0], [x, h2]]).reshape(-1, 1, 2)
    proj = cv2.transform(pts, mat).reshape(-1, 2)
    p0, p1 = tuple(proj[0].astype(int)), tuple(proj[1].astype(int))
    cv2.line(frame, p0, p1, (25, 20, 20), 5, cv2.LINE_AA)
    cv2.line(frame, p0, p1, (70, 65, 60), 2, cv2.LINE_AA)
    return proj


def _char_center(t: int, n_frames: int, w: int, h: int) -> tuple[float, float]:
    ph = t / n_frames
    cx = w * 0.30 + (w * 0.42) * ph + 14 * np.sin(4 * np.pi * ph)
    cy = h * 0.52 + 12 * np.sin(6 * np.pi * ph)
    return cx, cy


def _draw_character(frame: np.ndarray, cx: float, cy: float,
                    draw_sword: bool = True) -> None:
    cxi, cyi = int(cx), int(cy)
    cv2.rectangle(frame, (cxi - 18, cyi - 10), (cxi + 18, cyi + 46), (200, 60, 60), -1)
    for s in range(-10, 46, 8):
        cv2.line(frame, (cxi - 18, cyi + s), (cxi + 18, cyi + s), (240, 220, 90), 2)
    cv2.circle(frame, (cxi, cyi - 26), 16, (235, 200, 170), -1)
    if draw_sword:
        cv2.line(frame, (cxi + 14, cyi), (cxi + 52, cyi - 40), (210, 210, 220), 3,
                 cv2.LINE_AA)


def _card_state(t: int, n_frames: int, w: int, h: int) -> tuple[float, float, float, int]:
    """Flipping card: width ∝ |cos| simulates a 3D flip, 2 flips per clip."""
    ph = t / n_frames
    c = np.cos(2 * np.pi * ph * 2)
    cx, cy = w * 0.52, h * 0.80
    half_w = max(4.0, 27.0 * abs(c))
    face = 1 if c >= 0 else -1
    return cx, cy, half_w, face


def _draw_card(frame: np.ndarray, cx: float, cy: float, half_w: float,
               face: int) -> None:
    color = (230, 215, 80) if face > 0 else (70, 110, 220)
    x0, x1 = int(cx - half_w), int(cx + half_w)
    y0, y1 = int(cy - 20), int(cy + 20)
    cv2.rectangle(frame, (x0, y0), (x1, y1), color, -1)
    cv2.rectangle(frame, (x0, y0), (x1, y1), (30, 30, 35), 2)
    if half_w > 12:
        cc = (int(cx), int(cy))
        cv2.circle(frame, cc, 7, (30, 30, 35), -1)


def _draw_ui(frame: np.ndarray, t: int, w: int, h: int) -> None:
    cv2.rectangle(frame, (14, 12), (150, 44), (20, 20, 25), -1)
    cv2.rectangle(frame, (16, 14), (148, 42), (70, 70, 80), 1)
    frac = max(0.15, 0.9 - 0.12 * (t // 120))
    cv2.rectangle(frame, (18, 20), (int(18 + 128 * frac), 36), (60, 210, 90), -1)
    cc = (w - 52, h - 52)
    cv2.circle(frame, cc, 34, (30, 30, 35), -1)
    cv2.circle(frame, cc, 34, (220, 220, 230), 2)
    for ang in range(0, 360, 45):
        a = np.deg2rad(ang)
        p2 = (int(cc[0] + 26 * np.cos(a)), int(cc[1] + 26 * np.sin(a)))
        cv2.line(frame, cc, p2, (160, 160, 170), 2)
    for x0 in range(w - 170, w - 30, 16):
        cv2.rectangle(frame, (x0, 16), (x0 + 9, 30), (235, 235, 235), -1)
        cv2.rectangle(frame, (x0 + 2, 20), (x0 + 7, 26), (20, 20, 25), -1)


def _iter_frames(n_frames: int, w: int, h: int, seed: int):
    """Yield (t, frame, camera_mat, char_xy, pole_pts, card_state) per frame.

    Generator core shared by in-memory and streaming renderers — no full-video
    buffer, so 60 s / 1080p corpora can be written frame by frame.
    """
    rng = np.random.default_rng(seed)
    w2, h2 = int(w * 2.8), int(h * 2.4)
    canvas = _bg_canvas(rng, w2, h2)
    pan_x = 1.4 * w
    amp_y = 0.35 * h
    for t in range(n_frames):
        ph = t / n_frames
        ang = 9.0 * np.sin(2 * np.pi * ph)
        center = (w2 / 2.0, h2 / 2.0)
        rot = cv2.getRotationMatrix2D(center, ang, 1.0)
        rot[0, 2] += (w2 / 2.0 - w / 2.0) + (ph - 0.5) * pan_x
        rot[1, 2] += (h2 / 2.0 - h / 2.0) + amp_y * np.sin(2 * np.pi * ph)
        frame = cv2.warpAffine(canvas, rot, (w, h), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REFLECT_101)
        pole = _draw_pole(frame, rot, w2, h2)
        cx, cy = _char_center(t, n_frames, w, h)
        _draw_character(frame, cx, cy)
        ccx, ccy, chw, cface = _card_state(t, n_frames, w, h)
        _draw_card(frame, ccx, ccy, chw, cface)
        _draw_ui(frame, t, w, h)
        yield t, frame, rot, (cx, cy), pole, (ccx, ccy, chw, cface)


def render_scene(n_frames: int = 480, w: int = 640, h: int = 360, fps: int = 120,
                 seed: int = 7, return_meta: bool = False):
    """(N, H, W, 3) uint8 RGB at the requested fps; optionally (+SceneMeta)."""
    frames = np.empty((n_frames, h, w, 3), np.uint8)
    cam = np.empty((n_frames, 2, 3), np.float64)
    chars = np.empty((n_frames, 2), np.float64)
    poles = np.empty((n_frames, 2, 2), np.float64)
    cards = np.empty((n_frames, 4), np.float64)
    for t, frame, rot, cxy, pole, card in _iter_frames(n_frames, w, h, seed):
        frames[t] = frame
        cam[t] = rot
        chars[t] = cxy
        poles[t] = pole
        cards[t] = card
    meta = SceneMeta(camera=cam, char_pos=chars, pole_pts=poles, card=cards,
                     w=w, h=h)
    return (frames, meta) if return_meta else frames


def _mux(container, stream, frame_rgb: np.ndarray) -> None:
    frame = av.VideoFrame.from_ndarray(frame_rgb, format="rgb24")
    for packet in stream.encode(frame):
        container.mux(packet)


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
# candidates and defects
# ---------------------------------------------------------------------------

def make_source_and_truth(frames120: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return frames120[0::2].copy(), frames120[1::2].copy()


def interleave(source: np.ndarray, mids: np.ndarray) -> np.ndarray:
    n = source.shape[0] + mids.shape[0]
    out = np.empty((n,) + source.shape[1:], source.dtype)
    out[0::2] = source[: (n + 1) // 2]
    out[1::2] = mids[: n // 2]
    return out


def _segments(n_mids: int, count: int) -> list[tuple[int, int]]:
    """Evenly spaced non-overlapping defect segments (mid-frame indices)."""
    step = n_mids // (count + 1)
    length = max(4, step // 3)
    return [(k * step, k * step + length) for k in range(1, count + 1)]


def _inpaint_circle(frame: np.ndarray, cx: int, cy: int, r: int) -> None:
    h, w = frame.shape[:2]
    m = np.zeros((h, w), np.uint8)
    cv2.circle(m, (int(cx), int(cy)), int(r), 1, -1)
    frame[:] = cv2.inpaint(frame, m, 5, cv2.INPAINT_TELEA)


def _inpaint_line(frame: np.ndarray, p0, p1, thickness: int) -> None:
    m = np.zeros(frame.shape[:2], np.uint8)
    cv2.line(m, tuple(np.asarray(p0, int)), tuple(np.asarray(p1, int)), 1, thickness)
    frame[:] = cv2.inpaint(frame, m, 5, cv2.INPAINT_TELEA)


DEFECTS = ("blur", "ghost", "freeze", "rotation_tear", "head_erase",
           "pole_wrong_motion", "sword_flicker", "ui_drift", "text_merge",
           "shop_jump", "disocc_fill", "card_freeze")


def make_defective_mids(source: np.ndarray, truth: np.ndarray, meta: SceneMeta,
                        defects: list[str] | None = None, seed: int = 11
                        ) -> tuple[np.ndarray, dict[str, tuple[int, int]]]:
    """Apply one segment per requested defect to the mid frames.

    Segments and returned dict are in MID-frame indices (candidate index of
    mid i is 2i+1). Even frames (anchors) are never touched.
    """
    defects = list(defects) if defects is not None else ["blur", "ghost", "freeze"]
    for d in defects:
        if d not in DEFECTS:
            raise ValueError(f"unknown defect {d!r}; choose from {DEFECTS}")
    n = len(truth)
    mids = truth.copy()
    segs = dict(zip(defects, _segments(n, len(defects))))

    for name, (a, b) in segs.items():
        b = min(b, n)
        for i in range(a, b):
            m = 2 * i + 1                       # 120fps index of this mid
            xi, xj = source[i], source[min(i + 1, len(source) - 1)]
            f = mids[i]
            cx, cy = meta.char_pos[m]
            if name == "blur":
                mids[i] = cv2.GaussianBlur(f, (0, 0), 3.0)
            elif name == "ghost":
                mids[i] = np.clip(0.5 * xi.astype(np.float32)
                                  + 0.5 * xj.astype(np.float32), 0, 255).astype(np.uint8)
            elif name == "freeze":
                mids[i] = xi.copy()
            elif name == "rotation_tear":
                # Left half frozen at X_i, right half from X_{i+1}: under
                # camera rotation the halves are mutually inconsistent — the
                # seam plus opposing rotations is a background tear.
                torn = xi.copy()
                torn[:, f.shape[1] // 2:] = xj[:, f.shape[1] // 2:]
                mids[i] = torn
            elif name == "head_erase":
                _inpaint_circle(f, cx, cy - 26, 18)
            elif name == "pole_wrong_motion":
                # Erase the pole where it should be, redraw it where X_i had
                # it: the pole no longer carries its own motion (§8.3).
                _inpaint_line(f, meta.pole_pts[m, 0], meta.pole_pts[m, 1], 9)
                p0, p1 = meta.pole_pts[2 * i]
                cv2.line(f, tuple(p0.astype(int)), tuple(p1.astype(int)),
                         (45, 40, 38), 5, cv2.LINE_AA)
            elif name == "sword_flicker":
                if i % 2 == 0:                  # sword gone on alternating mids
                    cxi, cyi = int(cx), int(cy)
                    _inpaint_line(f, (cxi + 12, cyi + 2), (cxi + 56, cyi - 44), 8)
            elif name == "ui_drift":
                # 2 px screen-space shift of both HUD elements.
                for (x0, y0, x1, y1) in ((10, 8, 154, 48),
                                         (meta.w - 94, meta.h - 94, meta.w - 8, meta.h - 8)):
                    patch = f[y0:y1, x0:x1].copy()
                    pw = x1 - x0
                    f[y0:y1, x0:x1] = 0
                    f[y0:y1, x0 + 2:x1] = patch[:, :pw - 2]
            elif name == "text_merge":
                x0, y0, x1, y1 = meta.w - 174, 12, meta.w - 26, 34
                patch = cv2.cvtColor(f[y0:y1, x0:x1], cv2.COLOR_RGB2GRAY)
                dark = (patch < 128).astype(np.uint8)
                dark = cv2.dilate(dark, np.ones((3, 3), np.uint8))
                rgb = f[y0:y1, x0:x1]
                rgb[dark > 0] = (rgb[dark > 0].astype(np.float32) * 0.35
                                 + 15).astype(np.uint8)
            elif name == "shop_jump":
                # Discrete UI state switch shown as a double exposure: the
                # panel shows a mixed state with both bars visible (§8.6).
                cv2.rectangle(f, (18, 20), (146, 36), (210, 70, 70), -1)
                panel = f[8:50, 10:156].astype(np.float32)
                prev = xi[8:50, 10:156].astype(np.float32)
                f[8:50, 10:156] = np.clip(0.5 * panel + 0.5 * prev, 0, 255).astype(np.uint8)
            elif name == "disocc_fill":
                # Newly exposed band (camera pans content leftward → exposure
                # on the right) filled with flat color: structure vanishes.
                band = 26
                fill = np.median(f[:, -band - 8:-band].reshape(-1, 3), 0)
                f[:, -band:] = fill.astype(np.uint8)
            elif name == "card_freeze":
                # The flip is frozen at the previous state: erase the real
                # (mid-flip) card and redraw the full-width front face.
                ccx, ccy = meta.card[m, 0], meta.card[m, 1]
                x0, x1 = int(ccx - 32), int(ccx + 32)
                y0, y1 = int(ccy - 24), int(ccy + 24)
                mask = np.zeros(f.shape[:2], np.uint8)
                mask[y0:y1, x0:x1] = 1
                f[:] = cv2.inpaint(f, mask, 5, cv2.INPAINT_TELEA)
                _draw_card(f, ccx, ccy, 27.0, 1)
    return mids, segs


# ---------------------------------------------------------------------------
# corpus variants
# ---------------------------------------------------------------------------

def build_test_set(out_dir: str | Path, n_frames: int = 480, w: int = 640,
                   h: int = 360, seed: int = 7,
                   defects: list[str] | None = None) -> dict[str, Path]:
    """Render + encode the synthetic corpus. Returns paths."""
    out_dir = Path(out_dir)
    frames, meta = render_scene(n_frames=n_frames, w=w, h=h, seed=seed,
                                return_meta=True)
    source, truth = make_source_and_truth(frames)
    good = frames
    bad_mids, _ = make_defective_mids(source, truth, meta, defects)
    bad = interleave(source, bad_mids)

    return {
        "source": write_video(out_dir / "source_60.mp4", source, 60),
        "truth": write_video(out_dir / "truth_mids_60.mp4", truth, 60),
        "good": write_video(out_dir / "candidate_good_120.mp4", good, 120),
        "bad": write_video(out_dir / "candidate_bad_120.mp4", bad, 120),
    }


def build_scene_cut_set(out_dir: str | Path, n_frames: int = 480, w: int = 640,
                        h: int = 360) -> dict[str, Path]:
    """Two distinct scenes spliced at the midpoint; the candidate carries a
    2-frame crossfade at the cut (a common interpolation failure there)."""
    out_dir = Path(out_dir)
    half = n_frames // 2
    a = render_scene(n_frames=half, w=w, h=h, seed=7)
    b = render_scene(n_frames=n_frames - half, w=w, h=h, seed=99)
    frames120 = np.concatenate([a, b], 0)
    frames120[half] = np.clip(0.5 * a[-1].astype(np.float32)
                              + 0.5 * b[0].astype(np.float32), 0, 255).astype(np.uint8)
    source, _ = make_source_and_truth(frames120)
    return {
        "source": write_video(out_dir / "cut_source_60.mp4", source, 60),
        "candidate": write_video(out_dir / "cut_candidate_120.mp4", frames120, 120),
        "cut_candidate_frame": half,
    }


def build_offset_candidate(source_path: str | Path, good_frames: np.ndarray,
                           out_dir: str | Path) -> Path:
    """Candidate shifted by one frame: anchors land on odd positions, so the
    alignment layer must detect the offset (§2.1)."""
    shifted = np.concatenate([good_frames[1:2], good_frames[:-1]], 0)
    return write_video(Path(out_dir) / "candidate_offset_120.mp4", shifted, 120)


def build_dropped_candidate(good_frames: np.ndarray, out_dir: str | Path,
                            drop_fractions: tuple[float, ...] = (0.25, 0.55, 0.79)
                            ) -> Path:
    """Candidate with frames deleted mid-stream (dropped-frame corruption)."""
    n = len(good_frames)
    keep = np.ones(n, bool)
    keep[[min(n - 1, int(f * n)) for f in drop_fractions]] = False
    return write_video(Path(out_dir) / "candidate_dropped_120.mp4",
                       good_frames[keep], 120)


def build_vfr_candidate(good_frames: np.ndarray, out_dir: str | Path,
                        jitter_hz: tuple[float, ...] = (112.0, 130.0)) -> Path:
    """Candidate with variable frame durations alternating between jitter_hz
    rates (VFR corruption, §2.1) — mean rate stays ~120 FPS."""
    path = Path(out_dir) / "candidate_vfr_120.mp4"
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = good_frames.shape[1:3]
    ticks = 10000                     # stream timebase: 0.1 ms units
    with av.open(str(path), "w") as out:
        st = out.add_stream("h264", rate=120)
        st.width, st.height, st.pix_fmt = w, h, "yuv420p"
        st.options = {"crf": "18", "preset": "medium"}
        st.time_base = Fraction(1, ticks)
        pts = 0
        for i, f in enumerate(good_frames):
            dur = ticks / jitter_hz[i % len(jitter_hz)]
            frame = av.VideoFrame.from_ndarray(f, format="rgb24")
            frame.pts = int(round(pts))
            frame.time_base = st.time_base
            pts += dur
            for packet in st.encode(frame):
                out.mux(packet)
        for packet in st.encode():
            out.mux(packet)
    return path
