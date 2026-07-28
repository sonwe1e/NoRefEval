"""Point-tracker backends (USERPLAN §8.5).

KLT for routine windows; CoTracker3 reserved for audit-tier complex cases
(stub raises so the pipeline degrades to KLT).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import cv2
import numpy as np


class TrackerBackend(ABC):
    name = "base"

    @abstractmethod
    def track(self, frames_gray: list[np.ndarray], points: np.ndarray
              ) -> tuple[np.ndarray, np.ndarray]:
        """Track points through a frame list.

        Returns (tracks (T, N, 2), visible (T, N) bool).
        """


class KLTTracker(TrackerBackend):
    name = "klt"

    def __init__(self, win_size: int = 25, max_level: int = 4):
        self._crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        self.win_size = win_size
        self.max_level = max_level

    def track(self, frames_gray: list[np.ndarray], points: np.ndarray
              ) -> tuple[np.ndarray, np.ndarray]:
        T, N = len(frames_gray), len(points)
        tracks = np.zeros((T, N, 2), np.float32)
        visible = np.zeros((T, N), bool)
        if N == 0:
            return tracks, visible
        tracks[0] = points.reshape(N, 2)
        visible[0] = True
        cur = points.reshape(N, 2).astype(np.float32)
        for t in range(1, T):
            nxt, st, _ = cv2.calcOpticalFlowPyrLK(
                frames_gray[t - 1], frames_gray[t], cur, None,
                winSize=(self.win_size, self.win_size), maxLevel=self.max_level,
                criteria=self._crit)
            good = st.ravel() == 1
            nxt[~good] = cur[~good]
            tracks[t] = nxt
            visible[t] = visible[t - 1] & good
            cur = nxt
        return tracks, visible


def get_tracker_backend(name: str = "auto", **kw) -> TrackerBackend:
    if name in ("auto", "klt"):
        return KLTTracker(**kw)
    if name == "cotracker":
        raise NotImplementedError("CoTracker3 backend not installed; install weights "
                                  "and register the backend for audit tier")
    raise ValueError(f"unknown tracker backend {name!r}")
