"""Video encoding and stream inspection (PICPLAN §14).

Official candidates: MP4 / H.264 / yuv420p / fixed fps / no audio.
Oracles additionally get a lossless FFV1 MKV so the exact RGB can be
recovered without compression error.
"""
from __future__ import annotations

from pathlib import Path
from fractions import Fraction

import av
import numpy as np


def encode_rgb(frames: np.ndarray, fps: int, path: str | Path,
               codec: str = "libx264", crf: int = 10,
               pix_fmt: str = "yuv420p", preset: str = "medium") -> Path:
    """Encode a (n, h, w, 3) uint8 RGB array to ``path``.

    No audio stream is ever added.  Frames are fed one at a time so large
    masters never need a second in-memory copy.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n, h, w = frames.shape[0], frames.shape[1], frames.shape[2]
    container = av.open(str(path), mode="w",
                        options={"movflags": "+faststart"} if path.suffix == ".mp4" else {})
    try:
        if codec == "ffv1":
            stream = container.add_stream("ffv1", rate=fps)
            stream.pix_fmt = pix_fmt if pix_fmt != "yuv420p" else "bgr0"
            try:
                stream.options = {"coder": "1", "context": "0", "level": "3"}
            except Exception:
                pass
        else:
            stream = container.add_stream(codec, rate=fps)
            stream.pix_fmt = pix_fmt
            stream.options = {"crf": str(crf), "preset": preset}
        stream.width = w
        stream.height = h
        for i in range(n):
            frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(frames[i]),
                                               format="rgb24")
            frame.pts = i
            frame.time_base = Fraction(1, fps)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()
    return path


def probe(path: str | Path) -> dict:
    """Return fps, frame count, resolution and stream info of a video."""
    path = Path(path)
    with av.open(str(path)) as container:
        vstreams = [s for s in container.streams if s.type == "video"]
        astreams = [s for s in container.streams if s.type == "audio"]
        if not vstreams:
            raise ValueError(f"{path}: no video stream")
        s = vstreams[0]
        fps = float(s.average_rate) if s.average_rate else float(s.guessed_rate)
        n = s.frames
        if not n:
            n = 0
            for _ in container.decode(video=0):
                n += 1
        return {
            "fps": fps,
            "frames": int(n),
            "width": s.codec_context.width,
            "height": s.codec_context.height,
            "codec": s.codec_context.name,
            "audio_streams": len(astreams),
        }


def decode_all(path: str | Path, max_frames: int | None = None) -> np.ndarray:
    """Decode a video to (n, h, w, 3) uint8 RGB."""
    out = []
    with av.open(str(Path(path))) as container:
        for i, frame in enumerate(container.decode(video=0)):
            if max_frames is not None and i >= max_frames:
                break
            out.append(frame.to_ndarray(format="rgb24"))
    return np.stack(out)


def decode_frames(path: str | Path, indices) -> np.ndarray:
    """Decode only selected frame indices (sequential pass, no seeks)."""
    want = set(int(i) for i in indices)
    if not want:
        return np.empty((0,), dtype=np.uint8)
    out = {}
    with av.open(str(Path(path))) as container:
        for i, frame in enumerate(container.decode(video=0)):
            if i in want:
                out[i] = frame.to_ndarray(format="rgb24")
            if len(out) == len(want):
                break
    return np.stack([out[i] for i in sorted(want)])
