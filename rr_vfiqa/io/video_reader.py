"""PyAV-based video reading with presentation timestamps.

OpenCV is intentionally NOT used for demuxing: its frame-index access cannot
be trusted for VFR streams, dropped frames or B-frame reordering (USERPLAN §2.1).
"""

from __future__ import annotations

import hashlib
import os
from typing import Iterator

import av
import cv2
import numpy as np

from ..schema import FrameBundle, VideoMeta


def _content_hash(path: str) -> str:
    st = os.stat(path)
    key = f"{os.path.abspath(path)}|{st.st_size}|{int(st.st_mtime_ns)}"
    return hashlib.sha1(key.encode()).hexdigest()[:16]


# Process-level memo of per-frame 16x9 luma descriptors, keyed like the
# content hash (path | size | mtime_ns).  Alignment is built more than once
# per video in one process — mode routing probes it and the pipeline builds
# it again — and every pass decodes the FULL video even though only the tiny
# descriptors are kept.  Bounded; cleared wholesale when full.
_DESCRIPTOR_CACHE: dict[tuple[str, int, int], np.ndarray] = {}
_DESCRIPTOR_CACHE_MAX = 8


def frame_descriptors(reader: "VideoReader", width: int = 96) -> np.ndarray:
    """One-pass (N, 9, 16) float32 luma descriptors, memoized per file.

    Identical to decoding with reader.iter_frames(width=width) and
    downscaling each gray frame to 16x9 (INTER_AREA).  Only file-backed paths
    are cached; anything else (pipes/urls) is decoded without memoization.
    """
    try:
        st = os.stat(reader.path)
        is_file = os.path.isfile(reader.path)
    except OSError:
        is_file = False
        st = None
    key = None
    if is_file and st is not None:
        key = (os.path.abspath(reader.path), st.st_size, int(st.st_mtime_ns))
        hit = _DESCRIPTOR_CACHE.get(key)
        if hit is not None:
            return hit

    desc: list[np.ndarray] = []
    for _, rgb in reader.iter_frames(width=width):
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        desc.append(cv2.resize(gray, (16, 9), interpolation=cv2.INTER_AREA)
                    .astype(np.float32))
    out = np.stack(desc) if desc else np.zeros((0, 9, 16), np.float32)
    if key is not None:
        if len(_DESCRIPTOR_CACHE) >= _DESCRIPTOR_CACHE_MAX:
            _DESCRIPTOR_CACHE.clear()
        _DESCRIPTOR_CACHE[key] = out
    return out


class VideoReader:
    """Random + sequential frame access with PTS. Frames are RGB uint8."""

    def __init__(self, path: str, threads: int = 2):
        self.path = path
        self._threads = threads
        container = av.open(path)
        stream = container.streams.video[0]
        self.codec = stream.codec_context.name if stream.codec_context else ""
        self.pix_fmt = stream.codec_context.pix_fmt if stream.codec_context else ""

        # Build the PTS table by demuxing packets first (cheap), falling back
        # to a full decode index only when the container lacks per-frame pts.
        fps = float(stream.average_rate or stream.guessed_rate or 0.0) or 30.0
        time_base = float(stream.time_base) if stream.time_base else 1.0 / fps

        pts_list: list[float] = []
        try:
            for packet in container.demux(stream):
                if packet.pts is not None:
                    pts_list.append(packet.pts * time_base)
        except (av.error.InvalidDataError, av.error.ArgumentError, OSError):
            # Broken/VFR containers can fail mid-demux (EINVAL on some mp4
            # layouts); fall back to decode-counted synthetic pts.
            pts_list = []

        container.close()

        if not pts_list:
            # VFR / broken pts: decode once to count frames and synthesize pts.
            n = self._count_by_decode()
            pts = np.arange(n, dtype=np.float64) / fps
        else:
            pts = np.sort(np.asarray(pts_list, dtype=np.float64))
            # Deduplicate pts caused by B-frame repeats of the same instant.
            pts = np.unique(np.round(pts, 9))
            if fps <= 0 and len(pts) > 1:
                fps = 1.0 / float(np.median(np.diff(pts)))

        self.meta = VideoMeta(
            path=path,
            width=int(stream.codec_context.width) if stream.codec_context else 0,
            height=int(stream.codec_context.height) if stream.codec_context else 0,
            n_frames=int(len(pts)),
            fps=fps,
            pts_seconds=pts,
            content_hash=_content_hash(path),
            codec=self.codec,
            pix_fmt=self.pix_fmt,
        )
        if self.meta.width == 0:
            self._probe_dimensions()

    def _probe_dimensions(self) -> None:
        with av.open(self.path) as c:
            s = c.streams.video[0]
            self.meta.width = s.codec_context.width
            self.meta.height = s.codec_context.height

    def _count_by_decode(self) -> int:
        n = 0
        with av.open(self.path) as c:
            c.streams.video[0].thread_type = "AUTO"
            for _ in c.decode(video=0):
                n += 1
        return n

    # -- sequential --------------------------------------------------------

    def iter_frames(self, step: int = 1, width: int | None = None
                    ) -> Iterator[tuple[int, np.ndarray]]:
        """Yield (decode_index, rgb_uint8) for every step-th frame in order."""
        with av.open(self.path) as c:
            stream = c.streams.video[0]
            stream.thread_type = "AUTO"
            idx = 0
            for frame in c.decode(stream):
                if idx % step == 0:
                    img = frame.to_ndarray(format="rgb24")
                    if width is not None and img.shape[1] > width:
                        h = max(2, int(round(img.shape[0] * width / img.shape[1])) // 2 * 2)
                        img = cv2.resize(img, (width, h), interpolation=cv2.INTER_AREA)
                    yield idx, img
                idx += 1

    def decode_all(self, width: int | None = None) -> np.ndarray:
        """Decode the full video as (N, H, W, 3) uint8. Only for short clips."""
        out = [img for _, img in self.iter_frames(width=width)]
        return np.stack(out) if out else np.zeros((0, 0, 0, 3), np.uint8)

    # -- random access -------------------------------------------------------

    def read_frames(self, indices: np.ndarray | list[int], width: int | None = None
                    ) -> FrameBundle:
        """Decode specific frame indices, seeking per contiguous run.

        Seeks land on the keyframe before each run's start time; frames are
        then matched by timestamp (robust to VFR / B-frame reordering), so a
        late-window read costs one GOP decode, not a whole-video decode.
        Any frames the seek path misses fall back to a full counted pass.
        """
        indices = np.asarray(sorted(set(int(i) for i in indices)), dtype=np.int64)
        n = self.meta.n_frames
        indices = indices[(indices >= 0) & (indices < n)]

        def post(img: np.ndarray) -> np.ndarray:
            if width is not None and img.shape[1] > width:
                h = max(2, int(round(img.shape[0] * width / img.shape[1])) // 2 * 2)
                img = cv2.resize(img, (width, h), interpolation=cv2.INTER_AREA)
            return img

        decoded: dict[int, np.ndarray] = {}
        frame_dur = 1.0 / max(self.meta.fps, 1e-6)
        pts_table = self.meta.pts_seconds
        with av.open(self.path) as c:
            stream = c.streams.video[0]
            stream.thread_type = "AUTO"
            tb = float(stream.time_base) if stream.time_base else frame_dur

            for run in _split_runs(indices, gap=32):
                need = [int(i) for i in run if i not in decoded]
                if not need:
                    continue
                want_times = [float(pts_table[i]) for i in need]
                target = max(0.0, want_times[0] - frame_dur)
                ptr = 0
                tol = frame_dur * 0.6
                limit = want_times[-1] + tol
                try:
                    c.seek(int(target / tb), stream=stream, backward=True,
                           any_frame=False)
                    for frame in c.decode(stream):
                        if frame.pts is None:
                            continue
                        t = frame.pts * tb
                        if t > limit:
                            break
                        while ptr < len(need) and t > want_times[ptr] + tol:
                            ptr += 1                  # missed — fallback later
                        if ptr < len(need) and abs(t - want_times[ptr]) <= tol:
                            decoded[need[ptr]] = post(frame.to_ndarray(format="rgb24"))
                            ptr += 1
                        if ptr >= len(need):
                            break
                except (av.error.ArgumentError, av.error.ValueError,
                        av.error.InvalidDataError, OSError):
                    # Seek or decode failed for this run (EINVAL on some mp4
                    # layouts after seeking); the counted fallback below
                    # recovers the missing frames.
                    pass

            # Fallback: counted full pass for anything the seeks missed.
            missing = {int(i) for i in indices} - set(decoded)
            if missing:
                last = max(missing)
                idx = 0
                try:
                    c.seek(0, stream=stream)
                except (av.error.ValueError, OSError):
                    pass
                for frame in c.decode(stream):
                    if idx in missing:
                        decoded[idx] = post(frame.to_ndarray(format="rgb24"))
                        missing.discard(idx)
                        if not missing:
                            break
                    if idx >= last:
                        break
                    idx += 1

        keep = [i for i in indices if i in decoded]
        if not keep:
            raise RuntimeError(f"no frames decoded from {self.path} for {indices[:8]}...")
        rgb = np.stack([decoded[i] for i in keep])
        times = self.meta.pts_seconds[keep]
        return FrameBundle(
            indices=np.asarray(keep, np.int32),
            times=times,
            rgb=rgb,
            width=rgb.shape[2],
            height=rgb.shape[1],
        )

    def read_one(self, index: int, width: int | None = None) -> np.ndarray:
        return self.read_frames([index], width=width).rgb[0]


def _split_runs(indices: np.ndarray, gap: int) -> list[np.ndarray]:
    """Group sorted indices into runs; a new seek starts after gaps > `gap`."""
    if len(indices) == 0:
        return []
    cuts = np.nonzero(np.diff(indices) > gap)[0] + 1
    return list(np.split(indices, cuts))
