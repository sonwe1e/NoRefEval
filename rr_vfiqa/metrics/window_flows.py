"""On-demand candidate flows within an evaluation window.

Window layout (center = generated frame M_i, 5 frames):

    pos 0      pos 1     pos 2     pos 3       pos 4
    M_{i-1}    X_i       M_i       X_{i+1}     M_{i+1}
    (gen)      (anchor)  (gen)     (anchor)    (gen)

All flows are at the window bundle resolution.
"""

from __future__ import annotations

import numpy as np

from ..motion.flow_estimator import FlowBackend
from ..schema import FrameBundle


class WindowFlows:
    def __init__(self, bundle: FrameBundle, backend: FlowBackend):
        self.bundle = bundle
        self.backend = backend
        self._cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}

    def precompute(self, pairs: list[tuple[int, int]] | None = None) -> None:
        """Batch-compute both directions for every requested frame pair.

        One backend call (batched for RAFT) instead of ~14 sequential ones.
        """
        t = self.bundle.rgb.shape[0]
        if pairs is None:
            pairs = [(a, b) for a in range(t) for b in range(t)
                     if 0 < abs(a - b) <= 2]
        requested = sorted({
            (min(int(a), int(b)), max(int(a), int(b)))
            for a, b in pairs if a != b
        })
        todo: list[tuple[int, int, np.ndarray, np.ndarray]] = []
        for a, b in requested:
            cached = self._cache.get((a, b))
            if cached is not None and cached[0] is not None and cached[1] is not None:
                continue
            todo.append((a, b, self.bundle.rgb[a], self.bundle.rgb[b]))
            todo.append((b, a, self.bundle.rgb[b], self.bundle.rgb[a]))
        if not todo:
            return
        flows = self.backend.flow_many([(img_a, img_b) for _, _, img_a, img_b in todo])
        for (a, b, _, _), f in zip(todo, flows):
            key = (min(a, b), max(a, b))
            cur = self._cache.get(key)
            if key == (a, b):
                self._cache[key] = (f, cur[1] if cur else None)
            else:
                self._cache[key] = (cur[0] if cur else None, f)

    def pair(self, a_pos: int, b_pos: int) -> tuple[np.ndarray, np.ndarray]:
        """(f_ab, f_ba) between window positions a_pos and b_pos."""
        key = (min(a_pos, b_pos), max(a_pos, b_pos))
        cached = self._cache.get(key)
        if cached is not None and cached[0] is not None and cached[1] is not None:
            f_lo, f_hi = cached
        else:
            a = self.bundle.rgb[key[0]]
            b = self.bundle.rgb[key[1]]
            f_lo = self.backend.flow(a, b)
            f_hi = self.backend.flow(b, a)
            self._cache[key] = (f_lo, f_hi)
        f_lo, f_hi = self._cache[key]
        return (f_lo, f_hi) if a_pos <= b_pos else (f_hi, f_lo)

    def forward(self, a_pos: int, b_pos: int) -> np.ndarray:
        return self.pair(a_pos, b_pos)[0]

    def height(self) -> int:
        return self.bundle.height

    def width(self) -> int:
        return self.bundle.width
