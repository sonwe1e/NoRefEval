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

_MIN_CARD_AREA = 150          # px² at flow resolution (narrow mid-flip cards)
_STATIC_THRESHOLD = 0.10      # below this |ΔA|/A_i AND consistent M → static card


def _card_area(rgb: np.ndarray) -> float:
    """Largest LANDSCAPE saturated region's area (0 if none).

    Landscape aspect separates cards from characters (tall) and skill rings
    (square); the top HUD strip is excluded by position. This is a declared
    heuristic proxy — see meta.proxy_branches.
    """
    h, w = rgb.shape[:2]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = ((hsv[..., 1] > 100) & (hsv[..., 2] > 90)).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    best = 0.0
    max_area = 0.04 * h * w
    for lab in range(1, n):
        area = stats[lab, cv2.CC_STAT_AREA]
        bw = stats[lab, cv2.CC_STAT_WIDTH]
        bh = stats[lab, cv2.CC_STAT_HEIGHT]
        cy = stats[lab, cv2.CC_STAT_TOP] + bh / 2.0
        if area < _MIN_CARD_AREA or area > max_area:
            continue
        if bw < bh * 1.15:                    # not landscape → not a card
            continue
        if cy < 0.15 * h:                     # top HUD strip
            continue
        best = max(best, float(area))
    return best


def compute_window(bundle: FrameBundle, cfg: EvalConfig) -> dict[str, float]:
    a_i = _card_area(bundle.rgb[1])       # X_i
    out: dict[str, float] = {"card_found": float(a_i >= _MIN_CARD_AREA)}
    if a_i < _MIN_CARD_AREA:
        out["card_flip_err"] = float("nan")
        return out

    a_j = _card_area(bundle.rgb[3])       # X_{i+1}
    a_m = _card_area(bundle.rgb[2])       # M_i
    da = a_j - a_i

    # Static, consistent card → nothing to evaluate. A frozen card can sit in
    # a window whose ENDPOINTS barely differ (both near the flip extreme) yet
    # M_i still deviates massively from both — that must not be skipped.
    if abs(da) <= _STATIC_THRESHOLD * a_i and abs(a_m - a_i) <= 0.15 * a_i:
        out["card_flipping"] = 0.0
        out["card_flip_err"] = float("nan")
        return out
    out["card_flipping"] = 1.0

    # Deviation of M_i from the expected half-progress, normalized by the
    # half-difference with a size-proportional floor (catches the frozen-wide
    # vs narrow-endpoints case where |ΔA| between endpoints is ~0).
    expected = a_i + 0.5 * da
    denom = 0.5 * abs(da) + 0.15 * 0.5 * (a_i + a_j) + 1e-6
    out["card_flip_err"] = float(min(abs(a_m - expected) / denom, 2.0))
    return out
