"""Learned no-reference VQA backends — weak prior only (USERPLAN §9).

Learned VQA contributes at most 5 % of the no-reference fusion.  The built-in
adapter uses pyIQA's NIQE model when the optional dependency is installed.
When unavailable the fusion stage records that absence and drops the prior;
it must never substitute a perfect score.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class VQABackend(ABC):
    name = "base"

    @abstractmethod
    def technical_quality(self, frames_rgb: np.ndarray) -> float:
        """0..1 technical quality of a short clip (T, H, W, 3) uint8."""


class PyIQANIQEBackend(VQABackend):
    """Lazy pyIQA NIQE adapter, sampled across a short clip."""

    name = "pyiqa-niqe"

    def __init__(self, device: str = "cpu"):
        import pyiqa

        self.device = "cuda" if device == "cuda" else "cpu"
        self.metric = pyiqa.create_metric("niqe", device=self.device)

    def technical_quality(self, frames_rgb: np.ndarray) -> float:
        import torch

        if len(frames_rgb) == 0:
            return float("nan")
        positions = np.linspace(0, len(frames_rgb) - 1, min(4, len(frames_rgb))).astype(int)
        values: list[float] = []
        with torch.inference_mode():
            for pos in positions:
                tensor = torch.from_numpy(
                    frames_rgb[pos].copy()).permute(2, 0, 1).float().div_(255.0)
                raw = float(self.metric(tensor.unsqueeze(0).to(self.device)).item())
                # NIQE is lower-is-better and is unbounded.  This conservative
                # mapping only supplies a weak 0..1 prior to mode fusion.
                values.append(float(np.exp(-max(raw - 3.0, 0.0) / 10.0)))
        return float(np.clip(np.mean(values), 0.0, 1.0))


def get_vqa_backend(name: str = "auto", **kw) -> VQABackend | None:
    if name == "none":
        return None
    if name in ("auto", "pyiqa-niqe", "niqe"):
        try:
            return PyIQANIQEBackend(device=kw.get("device", "cpu"))
        except Exception as exc:
            if name == "auto":
                return None
            raise RuntimeError(
                "could not initialize pyIQA/NIQE; install rr-vfiqa[vqa] and "
                "verify the selected device") from exc
    raise ValueError(
        f"unknown VQA backend {name!r}; choose auto, none, or pyiqa-niqe")
