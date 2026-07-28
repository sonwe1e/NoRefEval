"""Learned no-reference VQA backends — weak prior only (USERPLAN §9).

FAST-VQA / DOVER / VFIPQA may contribute ≤5–10 % of the fused score. The
interface returns a technical-quality scalar per clip; when unavailable the
fusion stage simply drops the prior.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class VQABackend(ABC):
    name = "base"

    @abstractmethod
    def technical_quality(self, frames_rgb: np.ndarray) -> float:
        """0..1 technical quality of a short clip (T, H, W, 3) uint8."""


def get_vqa_backend(name: str = "auto", **kw) -> VQABackend | None:
    if name in ("auto", "none"):
        return None
    raise ValueError(f"unknown VQA backend {name!r}; install DOVER/FAST-VQA and "
                     f"register it to enable the learned prior")
