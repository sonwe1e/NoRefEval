"""Diagnostic compare clips (USERPLAN §10, item 3).

For each issue we produce a side-by-side comparison video that makes the
defect *visually* verifiable, not just a heat blob:

* NR:  ``candidate | motion-compensated prediction | residual heat``
* Endpoint: ``prev endpoint | candidate middle | next endpoint | composition map``
* FR:  ``reference | candidate | error heatmap``

Each panel is labelled; a severity-coloured bar and running timecode are
burned in, matching the overlay exporter's look.  The result is the
``issue_NNN_compare.mp4`` companion to the original/overlay cuts.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from typing import Sequence

import av
import cv2
import numpy as np


_PAD_SECONDS = 0.5
_MAX_CLIPS = 12
# Per-panel width cap (USERPLAN P0-7): prevents 4K sources producing 8K+
# output videos. Sources narrower than this are never upscaled.
_PANEL_TARGET_W = 560
_BAND_BGR = {
    "severe": (43, 57, 192), "high": (34, 126, 230),
    "medium": (40, 196, 241), "low": (165, 165, 150),
}
_LABEL_FG = (255, 255, 255)
_LABEL_BG = (20, 20, 20)


def _heat_bgr(x: np.ndarray) -> np.ndarray:
    """0..1 -> blue->red BGR ramp."""
    x = np.clip(x, 0.0, 1.0)
    r = np.clip(1.5 * x, 0, 1)
    g = np.clip(1.5 * (1 - np.abs(x - 0.5) * 2), 0, 1) * 0.7
    b = np.clip(1.5 * (1 - x), 0, 1)
    return np.stack([b, g, r], axis=-1)


def _normalize_map_for_display(array: np.ndarray) -> np.ndarray:
    """Normalize an error map to 0-1 for display using P99.5 percentile.

    FR error maps (luma_error_map, mcr_difference_map) arrive in range 0-255,
    while NR maps are already ~0-1.  Naively clipping either to 0-1 loses all
    detail in the 0-255 case.  Scaling by the P99.5 percentile (floored to a
    small epsilon) maps the meaningful dynamic range into 0-1 for both.
    """
    m = np.nan_to_num(array.astype(np.float32), nan=0.0)
    vmax = max(float(np.percentile(m, 99.5)), 1e-6)
    return np.clip(m / vmax, 0.0, 1.0)


def _label(frame: np.ndarray, text: str, scale: float = 0.6) -> np.ndarray:
    """Draw a dark label box at the top-left of ``frame``."""
    out = frame.copy()
    h = max(20, int(round(frame.shape[0] * 0.08)))
    cv2.rectangle(out, (0, 0), (frame.shape[1], h), _LABEL_BG, -1)
    cv2.putText(out, text, (8, h - 6), cv2.FONT_HERSHEY_SIMPLEX, scale,
                _LABEL_FG, 1, cv2.LINE_AA)
    return out


def _panels_to_frame(panels: list[np.ndarray],
                     labels: list[str]) -> np.ndarray:
    """Horizontally stack equally-sized labelled panels into one frame."""
    if not panels:
        return np.zeros((10, 10, 3), np.uint8)
    h, w = panels[0].shape[:2]
    labelled = []
    for p, lab in zip(panels, labels):
        if p.shape[:2] != (h, w):
            p = cv2.resize(p, (w, h), interpolation=cv2.INTER_LINEAR)
        if p.ndim == 2:
            p = cv2.cvtColor(p, cv2.COLOR_GRAY2BGR)
        elif p.dtype != np.uint8:
            p = np.clip(p * 255, 0, 255).astype(np.uint8)
        labelled.append(_label(p, lab))
    return np.concatenate(labelled, axis=1)


def _stack_vertical(top: np.ndarray, bottom: np.ndarray) -> np.ndarray:
    """Stack two frames vertically, matching widths."""
    if top.shape[1] != bottom.shape[1]:
        bottom = cv2.resize(bottom, (top.shape[1], bottom.shape[0]))
    return np.concatenate([top, bottom], axis=0)


def export_compare_clips(
    issues: list[dict],
    out_dir: str | Path,
    *,
    out_fps: float = 30.0,
    panels: Sequence[Sequence[tuple[np.ndarray, str]]] | None = None,
    candidate_video: str | None = None,
    reference_video: str | None = None,
    error_maps: Sequence[np.ndarray | None] | dict[int, np.ndarray] | None = None,
    mode: str = "NR",
) -> list[Path]:
    """Write ``issue_NNN_compare.mp4`` per issue.

    ``panels`` is a per-issue list of (frame_bgr, label) pairs captured at
    the issue's centre time; used as a fallback when ``candidate_video`` is not
    given or when decoding the issue window fails.

    ``error_maps`` is a per-issue error map (aligned with ``issues`` by index,
    or a dict ``{issue_index: array}``).  When present for an issue it is
    rendered as a jet heat panel alongside the candidate; the temporal-heat
    proxy is only the fallback when no real map exists.

    When ``candidate_video`` is supplied, the exporter decodes the actual
    frames in the issue's time window (with ``_PAD_SECONDS`` of padding) and
    writes a *real* video sequence: each output frame is a multi-panel
    composition of the candidate frame alongside the diagnostic heat, so
    flicker / freeze / UI drift / temporal ghosting become visible.  If the
    decode fails or the window is too short, we fall back to holding the
    supplied ``panels`` for ~2 seconds (the old static behaviour).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for n, issue in enumerate(issues[:_MAX_CLIPS]):
        issue_panels = panels[n] if panels and n < len(panels) else None
        if isinstance(error_maps, dict):
            issue_emap = error_maps.get(n)
        elif error_maps is not None and n < len(error_maps):
            issue_emap = error_maps[n]
        else:
            issue_emap = None
        out_path = out_dir / f"issue_{n:03d}_compare.mp4"
        _write_compare_clip(issue, issue_panels, out_path, out_fps,
                            candidate_video, reference_video, issue_emap,
                            mode=mode)
        paths.append(out_path)
    return paths


def _decode_window_frames(video_path: str, t0: float, t1: float,
                          target_h: int, target_w: int,
                          ) -> list[tuple[np.ndarray, float]]:
    """Decode BGR frames from ``video_path`` in ``[t0, t1]`` resized to target.

    Best-effort: returns an empty list on any failure.  Frames are resized to
    exactly ``(target_h, target_w)`` so they can be composed into panels.
    Each element is ``(bgr_frame, pts_seconds)`` so callers can resample to
    the output FPS by real timestamps instead of assuming a constant rate.
    """
    try:
        frames: list[tuple[np.ndarray, float]] = []
        with av.open(video_path) as inp:
            vs = inp.streams.video[0]
            inp.seek(int(t0 * av.time_base), any_frame=False, backward=True)
            for frame in inp.decode(vs):
                t = float(frame.pts * vs.time_base) if frame.pts is not None else 0.0
                if t > t1 + 1e-3:
                    break
                if t < t0 - 1e-3:
                    continue
                bgr = frame.to_ndarray(format="bgr24")
                if bgr.shape[:2] != (target_h, target_w):
                    bgr = cv2.resize(bgr, (target_w, target_h),
                                     interpolation=cv2.INTER_LINEAR)
                frames.append((bgr, t))
        return frames
    except Exception:  # noqa: BLE001
        return []


def _temporal_heat(cur: np.ndarray, prev: np.ndarray | None) -> np.ndarray:
    """|cur - prev| luma normalised to 0..1 (0 where no previous frame)."""
    if prev is None:
        return np.zeros(cur.shape[:2], dtype=np.float32)
    diff = np.abs(cur.astype(np.int16) - prev.astype(np.int16)).mean(axis=2)
    d = diff.astype(np.float32) / 255.0
    return np.clip(d / max(float(np.percentile(d, 95)), 1e-3), 0.0, 1.0)


def _panel_size(nat_w: int, nat_h: int) -> tuple[int, int]:
    """Cap panel width at ``_PANEL_TARGET_W`` (never upscale narrow sources).

    Width and height are forced even so the composed yuv420p frame meets H.264's
    requirement (an odd height makes ``avcodec_open2`` fail to open libx264).
    """
    if nat_w <= _PANEL_TARGET_W:
        return nat_w & ~1, nat_h & ~1
    w = _PANEL_TARGET_W & ~1
    h = max(2, int(round(nat_h * w / nat_w)) & ~1)
    return w, h


def _render_heat(emap: np.ndarray, h: int, w: int) -> np.ndarray:
    """Jet heat ``emap`` (0..1) resized to ``(h, w)`` as uint8 BGR."""
    heat = _heat_bgr(_normalize_map_for_display(emap))
    heat_u8 = (heat * 255).astype(np.uint8)
    if heat_u8.shape[:2] != (h, w):
        heat_u8 = cv2.resize(heat_u8, (w, h), interpolation=cv2.INTER_LINEAR)
    return heat_u8


def _write_compare_clip(issue: dict,
                        panels: Sequence[tuple[np.ndarray, str]] | None,
                        dst: Path, fps: float,
                        candidate_video: str | None = None,
                        reference_video: str | None = None,
                        error_map: np.ndarray | None = None,
                        mode: str = "NR") -> None:
    band = issue.get("severity_band", "low")
    bgr = _BAND_BGR.get(band, (120, 120, 120))
    title = issue.get("title", issue.get("issue_type", "issue"))

    # Try to build a real sequence from the candidate video first.
    t0 = max(0.0, float(issue.get("start_time") or 0) - _PAD_SECONDS)
    t1 = float(issue.get("end_time") or 0) + _PAD_SECONDS
    sequence: list[np.ndarray] = []
    if candidate_video and t1 > t0 + 0.05:
        # Pick a target size: prefer the first supplied panel's size, then
        # the candidate video's native size, then a sane default.  Panel width
        # is capped so 4K sources do not blow the output resolution up.
        if panels:
            ph, pw = panels[0][0].shape[:2]
        else:
            try:
                with av.open(candidate_video) as inp:
                    vs = inp.streams.video[0]
                    pw, ph = vs.codec_context.width, vs.codec_context.height
            except Exception:  # noqa: BLE001
                ph, pw = 360, 640
        pw, ph = _panel_size(pw, ph)
        decoded = _decode_window_frames(candidate_video, t0, t1, ph, pw)
        # Decode the reference stream only for the two reference-bearing modes
        # so we can show reference | candidate | heat.  The panel layout and
        # the reference label are driven by the explicit ``mode`` (see P0-5):
        #   NR             -> candidate | heat
        #   endpoint-2x    -> source    | candidate | heat
        #   full-reference -> reference | candidate | heat
        show_ref = mode in ("endpoint-2x", "full-reference")
        ref_decoded = (_decode_window_frames(reference_video, t0, t1, ph, pw)
                       if show_ref and reference_video else [])
        if len(decoded) >= 2:
            t_start = decoded[0][1]
            t_end = decoded[-1][1]
            duration = max(t_end - t_start, 1e-3)
            # Resample decoded frames to the output FPS by real PTS so a
            # 60/120 FPS source plays back at real speed in a 30 FPS file
            # instead of slow motion.
            output_times = np.arange(0.0, duration, 1.0 / fps)
            use_emap = error_map is not None
            prev: np.ndarray | None = None
            for out_t in output_times:
                target_pts = t_start + out_t
                # Nearest decoded frame by PTS (candidate and ref share PTS).
                cur, _ = min(decoded, key=lambda x: abs(x[1] - target_pts))
                panel_frames: list[np.ndarray] = []
                panel_labels: list[str] = []
                if ref_decoded:
                    ref_frame, _ = min(ref_decoded,
                                       key=lambda x: abs(x[1] - target_pts))
                    panel_frames.append(ref_frame)
                    # Endpoint uses the reference as the interpolation source.
                    panel_labels.append(
                        "source" if mode == "endpoint-2x" else "reference")
                panel_frames.append(cur)
                panel_labels.append(f"{mode}/candidate  t={out_t:.3f}")
                if use_emap:
                    panel_frames.append(
                        _render_heat(error_map, ph, pw))
                    panel_labels.append("error map")
                else:
                    heat = _temporal_heat(cur, prev)
                    heat_u8 = (_heat_bgr(heat) * 255).astype(np.uint8)
                    panel_frames.append(heat_u8)
                    panel_labels.append("temporal heat")
                sequence.append(_panels_to_frame(panel_frames, panel_labels))
                prev = cur

    # Fall back to the static held-frame behaviour if we have no sequence.
    if not sequence:
        if not panels:
            return
        frames = [p.astype(np.uint8) if p.dtype == np.uint8 else
                  np.clip(p, 0, 255).astype(np.uint8) for p, _ in panels]
        labels = [lab for _, lab in panels]
        sequence = [_panels_to_frame(frames, labels)] * max(1, int(round(fps * 2.0)))

    h, w = sequence[0].shape[:2]
    with av.open(str(dst), "w") as out:
        os_ = out.add_stream("h264", rate=int(round(fps)))
        os_.width, os_.height = w, h
        os_.pix_fmt = "yuv420p"
        os_.options = {"crf": "18", "preset": "veryfast"}
        for idx, composed in enumerate(sequence):
            frame = composed.copy()
            # Severity border + title bar
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), bgr, 2)
            bar_h = max(22, h // 14)
            cv2.rectangle(frame, (0, 0), (w, bar_h), bgr, -1)
            cv2.putText(frame, f"{title}  [compare]", (8, bar_h - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, max(0.4, bar_h / 40.0),
                        (255, 255, 255), 1, cv2.LINE_AA)
            of = av.VideoFrame.from_ndarray(frame, format="bgr24")
            of.pts = idx
            of.time_base = Fraction(1, int(round(fps)))
            for packet in os_.encode(of):
                out.mux(packet)
        for packet in os_.encode():
            out.mux(packet)


def extract_compare_panels(candidate_video: str, issue: dict,
                           error_map: np.ndarray | None = None,
                           reference_video: str | None = None,
                           ) -> list[tuple[np.ndarray, str]]:
    """Capture the centre-frame panels for an issue (USERPLAN §10).

    Best-effort: decodes the candidate (and reference if given) at the issue
    centre time and builds the panel list.  Returns an empty list on any
    failure so callers can skip the clip gracefully.
    """
    t_center = 0.5 * (float(issue.get("start_time", 0)) +
                       float(issue.get("end_time", 0)))
    try:
        panels: list[tuple[np.ndarray, str]] = []
        with av.open(candidate_video) as inp:
            vs = inp.streams.video[0]
            w = vs.codec_context.width
            h = vs.codec_context.height
            inp.seek(int(t_center * av.time_base), any_frame=False,
                     backward=True)
            cand_frame = None
            for frame in inp.decode(vs):
                t = float(frame.pts * vs.time_base) if frame.pts is not None else 0.0
                if t >= t_center - 1e-3:
                    cand_frame = frame.to_ndarray(format="bgr24")
                    break
            if cand_frame is None:
                return []
            panels.append((cand_frame, "candidate"))

            if reference_video is not None:
                with av.open(reference_video) as refp:
                    rvs = refp.streams.video[0]
                    refp.seek(int(t_center * av.time_base), any_frame=False,
                              backward=True)
                    for rframe in refp.decode(rvs):
                        rt = float(rframe.pts * rvs.time_base) \
                            if rframe.pts is not None else 0.0
                        if rt >= t_center - 1e-3:
                            ref = rframe.to_ndarray(format="bgr24")
                            # Match sizes for clean stacking.
                            if ref.shape[:2] != (h, w):
                                ref = cv2.resize(ref, (w, h))
                            panels.append((ref, "reference"))
                            break

            if error_map is not None:
                heat = _heat_bgr(_normalize_map_for_display(error_map))[..., ::-1]
                heat_u8 = np.clip(heat * 255, 0, 255).astype(np.uint8)
                if heat_u8.shape[:2] != (h, w):
                    heat_u8 = cv2.resize(heat_u8, (w, h))
                panels.append((heat_u8, "error map"))
        return panels
    except Exception:  # noqa: BLE001
        return []
