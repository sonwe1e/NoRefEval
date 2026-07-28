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
        except av.error.InvalidDataError:
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
        """Decode specific frame indices (seek-based, ascending order)."""
        indices = np.asarray(sorted(set(int(i) for i in indices)), dtype=np.int64)
        n = self.meta.n_frames
        indices = indices[(indices >= 0) & (indices < n)]

        decoded: dict[int, np.ndarray] = {}
        with av.open(self.path) as c:
            stream = c.streams.video[0]
            stream.thread_type = "AUTO"
            # Frame-exact seeking is unreliable across mp4 muxers/B-frame
            # patterns, so we do a single filtered decode pass and stop as soon
            # as every requested index is collected. Windows are small and we
            # decode at reduced width, so this is cheap in practice.
            idx = 0
            wanted = set(int(i) for i in indices)
            last = int(indices[-1]) if len(indices) else -1
            for frame in c.decode(stream):
                if idx in wanted:
                    img = frame.to_ndarray(format="rgb24")
                    if width is not None and img.shape[1] > width:
                        h = max(2, int(round(img.shape[0] * width / img.shape[1])) // 2 * 2)
                        img = cv2.resize(img, (width, h), interpolation=cv2.INTER_AREA)
                    decoded[idx] = img
                    if len(decoded) == len(wanted):
                        break
                if idx > last:
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
