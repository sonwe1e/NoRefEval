"""Export worst windows as short mp4 clips (PyAV re-encode)."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import av

from ..schema import WorstWindow

_PAD_SECONDS = 0.35
_MAX_CLIPS = 12


def export_badcase_clips(candidate_video: str, worst: list[WorstWindow],
                         out_dir: str | Path, out_fps: float = 60.0
                         ) -> list[Path]:
    """Re-encode each worst window ± padding into its own clip."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for n, w in enumerate(worst[:_MAX_CLIPS]):
        t0 = max(0.0, w.start - _PAD_SECONDS)
        t1 = w.end + _PAD_SECONDS
        tag = "+".join(w.types[:2]) if w.types else "badcase"
        out_path = out_dir / f"badcase_{n:02d}_{t0:07.2f}s_{tag}.mp4"
        _cut(candidate_video, t0, t1, out_path, out_fps)
        paths.append(out_path)
    return paths


def _cut(src: str, t0: float, t1: float, dst: Path, fps: float) -> None:
    with av.open(src) as inp:
        vs = inp.streams.video[0]
        w, h = vs.codec_context.width, vs.codec_context.height
        with av.open(str(dst), "w") as out:
            os_ = out.add_stream("h264", rate=int(round(fps)))
            os_.width = w
            os_.height = h
            os_.pix_fmt = "yuv420p"
            os_.options = {"crf": "18", "preset": "veryfast"}
            inp.seek(int(t0 * av.time_base), any_frame=False, backward=True)
            frame_idx = 0
            for frame in inp.decode(vs):
                t = float(frame.pts * vs.time_base) if frame.pts is not None else 0.0
                if t > t1:
                    break
                if t < t0 - 1e-3:
                    continue
                # Monotonic pts in the output stream's own timebase — leaving
                # the decoder's timebase in place truncates pts to duplicates.
                frame.pts = frame_idx
                frame.time_base = Fraction(1, int(round(fps)))
                frame_idx += 1
                for packet in os_.encode(frame):
                    out.mux(packet)
            for packet in os_.encode():
                out.mux(packet)
