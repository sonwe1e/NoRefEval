"""Depth backends (USERPLAN §8.1 — high-parallax scenes, on demand only).

Video Depth Anything (or similar) runs only in audit tier on high-risk
windows. The stub keeps the interface so the pipeline can degrade cleanly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class DepthBackend(ABC):
    name = "base"

    @abstractmethod
    def depth(self, rgb: np.ndarray) -> np.ndarray:
        """(H, W) float32 relative inverse depth."""


def get_depth_backend(name: str = "auto", **kw) -> DepthBackend:
    if name in ("auto", "none"):
        raise NotImplementedError("depth backend not installed; depth layers are "
                                  "disabled — high-parallax scenes fall back to "
                                  "affine/homography camera models")
    raise ValueError(f"unknown depth backend {name!r}")
