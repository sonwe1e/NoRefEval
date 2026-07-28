"""Optical-flow backends behind one interface (USERPLAN.md §3, §13).

The evaluator must never consume a candidate model's internal flow as truth,
so this module provides independent estimators:

* ``raft``       — torchvision RAFT-small, accurate, GPU-friendly (default when
                   torch is importable);
* ``farneback``  — OpenCV Farneback, CPU fallback / cheap scan tier.
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod

import cv2
import numpy as np


class FlowBackend(ABC):
    name = "base"

    @abstractmethod
    def flow(self, a_rgb: np.ndarray, b_rgb: np.ndarray) -> np.ndarray:
        """(H, W, 2) float32 flow a -> b, pixel units at input resolution."""

    def flow_pair(self, a_rgb: np.ndarray, b_rgb: np.ndarray):
        """Bidirectional flows as a tuple (f_ab, f_ba)."""
        return self.flow(a_rgb, b_rgb), self.flow(b_rgb, a_rgb)

    def release(self) -> None:
        pass


class FarnebackBackend(FlowBackend):
    name = "farneback"

    def __init__(self, pyr_scale: float = 0.5, levels: int = 5, winsize: int = 21,
                 iterations: int = 5):
        self._kw = dict(pyr_scale=pyr_scale, levels=levels, winsize=winsize,
                        iterations=iterations, poly_n=7, poly_sigma=1.5, flags=0)

    def flow(self, a_rgb: np.ndarray, b_rgb: np.ndarray) -> np.ndarray:
        ga = cv2.cvtColor(a_rgb, cv2.COLOR_RGB2GRAY)
        gb = cv2.cvtColor(b_rgb, cv2.COLOR_RGB2GRAY)
        f = cv2.calcOpticalFlowFarneback(ga, gb, None, **self._kw)
        return f.astype(np.float32)


class RaftBackend(FlowBackend):
    name = "raft"

    def __init__(self, device: str = "cuda"):
        import torch
        from torchvision.models.optical_flow import raft_small, Raft_Small_Weights

        self._torch = torch
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu"
                                   else "cpu")
        weights = Raft_Small_Weights.DEFAULT
        self._transforms = weights.transforms()
        self._model = raft_small(weights=weights).to(self.device).eval()

    def flow(self, a_rgb: np.ndarray, b_rgb: np.ndarray) -> np.ndarray:
        torch = self._torch
        h, w = a_rgb.shape[:2]
        ph = (16 - h % 16) % 16
        pw = (16 - w % 16) % 16
        a = np.pad(a_rgb, ((0, ph), (0, pw), (0, 0)), mode="edge")
        b = np.pad(b_rgb, ((0, ph), (0, pw), (0, 0)), mode="edge")
        # transforms() takes CHW uint8 images to [-1, 1] float CHW tensors.
        ta = torch.from_numpy(a.transpose(2, 0, 1))
        tb = torch.from_numpy(b.transpose(2, 0, 1))
        ta, tb = self._transforms(ta, tb)
        with torch.no_grad():
            flow = self._model(ta[None].to(self.device),
                               tb[None].to(self.device))[-1][0].cpu().numpy()
        flow = flow[:, :h, :w]
        return np.transpose(flow, (1, 2, 0)).astype(np.float32)  # (H, W, 2)

    def release(self) -> None:
        del self._model
        if self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()


_backends: dict[str, FlowBackend] = {}
_lock = threading.Lock()


def get_flow_backend(name: str = "auto", device: str = "cuda") -> FlowBackend:
    """Return a cached backend instance. "auto" prefers RAFT when torch exists."""
    key = f"{name}:{device}"
    with _lock:
        if key in _backends:
            return _backends[key]
        if name in ("auto", "raft"):
            try:
                be: FlowBackend = RaftBackend(device=device)
            except Exception:
                if name == "raft":
                    raise
                be = FarnebackBackend()
        elif name == "farneback":
            be = FarnebackBackend()
        else:
            raise ValueError(f"unknown flow backend {name!r}")
        _backends[key] = be
        return be


def compute_pair_flow(backend: FlowBackend, a_rgb: np.ndarray, b_rgb: np.ndarray,
                      work_width: int | None = None):
    """Compute bidirectional flow, optionally at a reduced width bucket.

    Returns (f_ab, f_ba, scale) where flows are at the working resolution and
    scale = work_res / native_res.
    """
    from ..schema import downscale_frames

    scale = 1.0
    if work_width and a_rgb.shape[1] > work_width:
        stacked, scale = downscale_frames(np.stack([a_rgb, b_rgb]), work_width)
        a_rgb, b_rgb = stacked[0], stacked[1]
    f_ab, f_ba = backend.flow_pair(a_rgb, b_rgb)
    return f_ab, f_ba, scale
