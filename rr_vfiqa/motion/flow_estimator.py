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

    def flow_many(self, pairs: list[tuple[np.ndarray, np.ndarray]]) -> list[np.ndarray]:
        """Batch of directed flows. Base implementation: sequential."""
        return [self.flow(a, b) for a, b in pairs]

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
        return self.flow_many([(a_rgb, b_rgb)])[0]

    def flow_many(self, pairs: list[tuple[np.ndarray, np.ndarray]]) -> list[np.ndarray]:
        """One batched model call per (up to 16) directed pairs."""
        if not pairs:
            return []
        torch = self._torch
        h, w = pairs[0][0].shape[:2]
        ph = (16 - h % 16) % 16
        pw = (16 - w % 16) % 16
        out: list[np.ndarray] = []
        chunk = 16
        for start in range(0, len(pairs), chunk):
            batch = pairs[start:start + chunk]
            imgs_a, imgs_b = [], []
            for a, b in batch:
                imgs_a.append(np.pad(a, ((0, ph), (0, pw), (0, 0)), mode="edge")
                              .transpose(2, 0, 1))
                imgs_b.append(np.pad(b, ((0, ph), (0, pw), (0, 0)), mode="edge")
                              .transpose(2, 0, 1))
            ta = torch.from_numpy(np.stack(imgs_a))
            tb = torch.from_numpy(np.stack(imgs_b))
            # Same normalization as weights.transforms(): uint8 → [-1, 1].
            fa = ta.to(torch.float32).div(127.5).sub(1.0)
            fb = tb.to(torch.float32).div(127.5).sub(1.0)
            with torch.no_grad():
                flows = self._model(fa.to(self.device), fb.to(self.device))[-1]
            flows = flows[:, :, :h, :w].cpu().numpy()
            for i in range(len(batch)):
                out.append(np.transpose(flows[i], (1, 2, 0)).astype(np.float32))
        return out

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
