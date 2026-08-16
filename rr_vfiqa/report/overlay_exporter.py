"""Overlay badcase clips (USERPLAN §10, item 2).

Re-encodes each issue window with the diagnostic evidence burned in:

* a severity-coloured top bar carrying the issue title + band;
* a running ``mm:ss.fff`` timecode;
* a translucent *motion-proxy* heat overlay (|frame - prev frame|) so the
  viewer sees where the temporal energy concentrates.  When a real error map
  is supplied it is used instead — the proxy is only a fallback and is
  labelled as such in the bar.

This is intentionally lightweight (OpenCV + PyAV, no learned saliency).  It
produces the ``*_overlay.mp4`` companion to the plain ``*_original.mp4`` cut.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import av
import cv2
import numpy as np


_PAD_SECONDS = 0.5
_MAX_CLIPS = 12
_BAND_BGR = {
    "severe": (43, 57, 192), "high": (34, 126, 230),
    "medium": (40, 196, 241), "low": (165, 165, 150),
}


def export_overlay_clips(candidate_video: str, issues: list[dict],
                         out_dir: str | Path, out_fps: float = 60.0,
                         error_maps: dict[int, np.ndarray] | None = None,
                         ) -> list[Path]:
    """One overlay clip per issue (capped), named ``issue_NNN_overlay.mp4``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for n, issue in enumerate(issues[:_MAX_CLIPS]):
        t0 = max(0.0, float(issue.get("start_time") or 0) - _PAD_SECONDS)
        t1 = float(issue.get("end_time") or 0) + _PAD_SECONDS
        out_path = out_dir / f"issue_{n:03d}_overlay.mp4"
        _overlay_cut(candidate_video, t0, t1, out_path, out_fps, issue,
                     error_maps.get(n) if error_maps else None)
        paths.append(out_path)
    return paths


def _overlay_cut(src: str, t0: float, t1: float, dst: Path, fps: float,
                 issue: dict, error_map: np.ndarray | None) -> None:
    band = issue.get("severity_band", "low")
    bgr = _BAND_BGR.get(band, (120, 120, 120))
    title = issue.get("title", issue.get("issue_type", "issue"))
    has_map = error_map is not None
    bar_text = f"{title}  [{issue.get('severity_label', band)}]"
    note = "error-map" if has_map else "motion-proxy"

    with av.open(src) as inp:
        vs = inp.streams.video[0]
        w, h = vs.codec_context.width, vs.codec_context.height
        with av.open(str(dst), "w") as out:
            os_ = out.add_stream("h264", rate=int(round(fps)))
            os_.width, os_.height = w, h
            os_.pix_fmt = "yuv420p"
            os_.options = {"crf": "18", "preset": "veryfast"}
            inp.seek(int(t0 * av.time_base), any_frame=False, backward=True)
            prev = None
            out_idx = 0
            for frame in inp.decode(vs):
                t = float(frame.pts * vs.time_base) if frame.pts is not None else 0.0
                if t > t1:
                    break
                if t < t0 - 1e-3:
                    prev = frame.to_ndarray(format="bgr24")
                    continue
                bgr_frame = frame.to_ndarray(format="bgr24")
                if has_map:
                    # Map may be at working resolution (e.g. 960px); resize to frame.
                    heat_norm = _normalize_map(error_map)
                    heat = (heat_norm if heat_norm.shape[:2] == (h, w)
                            else cv2.resize(heat_norm, (w, h),
                                            interpolation=cv2.INTER_LINEAR))
                else:
                    heat = _motion_proxy(bgr_frame, prev)
                annotated = _annotate(bgr_frame, heat, bgr, bar_text, note, t)
                of = av.VideoFrame.from_ndarray(annotated, format="bgr24")
                of.pts = out_idx
                of.time_base = Fraction(1, int(round(fps)))
                out_idx += 1
                prev = bgr_frame
                for packet in os_.encode(of):
                    out.mux(packet)
            for packet in os_.encode():
                out.mux(packet)


def _normalize_map(array: np.ndarray) -> np.ndarray:
    """Normalize an error map to 0..1 using P99.5 (robust to outliers)."""
    m = np.nan_to_num(array.astype(np.float32), nan=0.0)
    vmax = max(float(np.percentile(m, 99.5)), 1e-6)
    return np.clip(m / vmax, 0.0, 1.0)


def _motion_proxy(frame: np.ndarray, prev: np.ndarray | None) -> np.ndarray:
    """|frame - prev| luma, normalised to 0..1 (0 where no previous frame)."""
    if prev is None:
        return np.zeros(frame.shape[:2], dtype=np.float32)
    diff = np.abs(frame.astype(np.int16) - prev.astype(np.int16)).mean(axis=2)
    d = diff.astype(np.float32) / 255.0
    return np.clip(d / max(float(np.percentile(d, 95)), 1e-3), 0.0, 1.0)


def _heat_bgr(x: np.ndarray) -> np.ndarray:
    """Map 0..1 to a blue->red BGR ramp (no matplotlib dependency)."""
    x = np.clip(x, 0.0, 1.0)
    r = np.clip(1.5 * x, 0, 1)
    g = np.clip(1.5 * (1 - np.abs(x - 0.5) * 2), 0, 1) * 0.7
    b = np.clip(1.5 * (1 - x), 0, 1)
    return np.stack([b, g, r], axis=-1)


def _annotate(frame: np.ndarray, heat: np.ndarray, bgr, bar_text: str,
              note: str, t: float) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    # translucent heat where energy is high
    mask = heat > 0.25
    if mask.any():
        ramp = (_heat_bgr(heat) * 255).astype(np.uint8)
        alpha = (heat * 0.5)[..., None]
        blended = (out.astype(np.float32) * (1 - alpha)
                   + ramp.astype(np.float32) * alpha)
        out = np.where(mask[..., None], blended.astype(np.uint8), out)
    # top bar
    bar_h = max(22, h // 14)
    cv2.rectangle(out, (0, 0), (w, bar_h), bgr, -1)
    cv2.putText(out, _trim(bar_text, w), (8, bar_h - 7),
                cv2.FONT_HERSHEY_SIMPLEX, max(0.4, bar_h / 40.0),
                (255, 255, 255), 1, cv2.LINE_AA)
    # timecode bottom-right
    mm, ss = divmod(t, 60)
    tc = f"{int(mm):02d}:{ss:06.3f}  ({note})"
    cv2.putText(out, tc, (w - 8 - len(tc) * 8, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    # severity frame border
    cv2.rectangle(out, (0, 0), (w - 1, h - 1), bgr, 2)
    return out


def _trim(text: str, width: int) -> str:
    # crude width guard for Hershey at the chosen scale
    max_chars = max(8, width // 9)
    return text if len(text) <= max_chars else text[:max_chars - 1] + "…"
