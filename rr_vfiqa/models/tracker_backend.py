"""Point-tracker backends (USERPLAN §8.5).

KLT for routine windows; CoTracker (optional dependency, ``pip install
-e ".[audit]"``) for audit-tier complex cases — long-horizon, occlusion and
fast-motion handling. Without the package installed the audit tier degrades
to KLT with a logged note.
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


class CoTrackerBackend(TrackerBackend):
    """CoTracker2 point tracker — optional dependency (§P2).

    Import and weight download are lazy: constructing this backend fails with
    a clear ImportError unless ``cotracker`` is installed.
    """

    name = "cotracker"

    def __init__(self, device: str = "cuda", checkpoint: str | None = None):
        import torch
        from cotracker.predictor import CoTrackerPredictor

        self._torch = torch
        self.device = device if torch.cuda.is_available() else "cpu"
        if checkpoint is None:
            # Fetch the default pretrained checkpoint on first use.
            checkpoint = str(
                torch.hub.load_state_dict_from_url(
                    "https://huggingface.co/facebook/cotracker2/resolve/main/"
                    "cotracker2.pth",
                    map_location=self.device,
                )
            )
        self._model = CoTrackerPredictor(checkpoint=checkpoint).to(self.device)

    def track(self, frames_gray: list[np.ndarray], points: np.ndarray
              ) -> tuple[np.ndarray, np.ndarray]:
        torch = self._torch
        T, N = len(frames_gray), len(points)
        if N == 0:
            return np.zeros((T, 0, 2), np.float32), np.zeros((T, 0), bool)
        video = torch.from_numpy(np.stack(frames_gray))[None, None]       # (1,1,T,H,W)
        video = video.repeat(1, 3, 1, 1, 1).float().to(self.device)
        queries = torch.zeros(1, N, 3, device=self.device)                # t,x,y
        queries[0, :, 1:] = torch.from_numpy(points.reshape(N, 2)).to(self.device)
        with torch.no_grad():
            pred_tracks, pred_vis = self._model(video, queries=queries)
        return (pred_tracks[0].cpu().numpy().astype(np.float32),
                pred_vis[0].cpu().numpy().astype(bool))


def get_tracker_backend(name: str = "auto", **kw) -> TrackerBackend:
    if name in ("auto", "klt"):
        return KLTTracker(**kw)
    if name == "cotracker":
        return CoTrackerBackend(**kw)
    raise ValueError(f"unknown tracker backend {name!r}")


def get_audit_tracker(device: str = "cuda") -> tuple[TrackerBackend, str]:
    """Best available audit-tier tracker: CoTracker if installed, else KLT.

    Returns (backend, note); the note belongs in the report meta so downgrades
    are never silent.
    """
    try:
        return CoTrackerBackend(device=device), "cotracker"
    except ImportError as exc:
        return KLTTracker(), f"cotracker unavailable ({exc.__class__.__name__}); " \
                             f"fell back to KLT — install with: pip install " \
                             f"\"rr-vfiqa[audit]\""
