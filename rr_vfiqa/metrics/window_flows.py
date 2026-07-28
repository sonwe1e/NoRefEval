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

    def pair(self, a_pos: int, b_pos: int) -> tuple[np.ndarray, np.ndarray]:
        """(f_ab, f_ba) between window positions a_pos and b_pos."""
        key = (a_pos, b_pos)
        if key not in self._cache:
            a = self.bundle.rgb[a_pos]
            b = self.bundle.rgb[b_pos]
            f_ab = self.backend.flow(a, b)
            f_ba = self.backend.flow(b, a)
            self._cache[key] = (f_ab, f_ba)
        return self._cache[key]

    def forward(self, a_pos: int, b_pos: int) -> np.ndarray:
        return self.pair(a_pos, b_pos)[0]

    def height(self) -> int:
        return self.bundle.height

    def width(self) -> int:
        return self.bundle.width
