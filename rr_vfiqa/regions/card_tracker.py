"""Card-flip continuity branch (USERPLAN §8.6 continuous physical motion).

A real 3D card flip is continuous: its projected area moves monotonically
from the X_i state toward the X_{i+1} state. A mid frame frozen at the
previous state breaks that progression. Detection is saturation-based
(game cards are bold flat-color rectangles), then area progression is
checked: with A_m ≈ A_i + (A_j − A_i)/2 expected,

    err = |2(A_m − A_i) − (A_j − A_i)| / |A_j − A_i|

is ≈ 0 for a correct flip and ≈ 1 for a freeze. Only windows where the card
is actually flipping (|A_j − A_i| significant) are evaluated — discrete
state swaps are the transition branch's job.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..config import EvalConfig
from ..schema import FrameBundle

_MIN_CARD_AREA = 250          # px² at flow resolution
_FLIP_THRESHOLD = 0.10        # |ΔA|/A_i above which a flip is in progress


def _card_area(rgb: np.ndarray) -> float:
    """Largest saturated flat-color region's area (0 if none)."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = ((hsv[..., 1] > 100) & (hsv[..., 2] > 90)).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return 0.0
    areas = stats[1:, cv2.CC_STAT_AREA]
    return float(areas.max())


def compute_window(bundle: FrameBundle, cfg: EvalConfig) -> dict[str, float]:
    a_i = _card_area(bundle.rgb[1])       # X_i
    out: dict[str, float] = {"card_found": float(a_i >= _MIN_CARD_AREA)}
    if a_i < _MIN_CARD_AREA:
        out["card_flip_err"] = float("nan")
        return out

    a_j = _card_area(bundle.rgb[3])       # X_{i+1}
    da = a_j - a_i
    out["card_flipping"] = float(abs(da) > _FLIP_THRESHOLD * a_i)
    if abs(da) <= _FLIP_THRESHOLD * a_i:
        out["card_flip_err"] = float("nan")     # not mid-flip: nothing to check
        return out

    a_m = _card_area(bundle.rgb[2])       # M_i
    out["card_flip_err"] = float(abs(2 * (a_m - a_i) - da) / (abs(da) + 1e-6))
    return out
