"""Drawing canvas with a parallel semantic-label buffer (PICPLAN §13).

Every draw call paints RGB and, when given a ``label`` class name, stamps
the same pixels into a uint8 label buffer.  Defect operators later use the
label buffer to restrict their effect to semantic regions (player, weapon,
UI, text, ...).  All calls wrap deterministic cv2 primitives.
"""
from __future__ import annotations

import cv2
import numpy as np

from ..schema import SEMANTIC_CLASSES


class Canvas:
    """RGB + label buffer pair of identical geometry."""

    def __init__(self, width: int, height: int, fill=(0, 0, 0)):
        self.width = width
        self.height = height
        self._fill = fill
        self.rgb = np.full((height, width, 3), fill, dtype=np.uint8)
        self.labels = np.zeros((height, width), dtype=np.uint8)

    def reset(self) -> None:
        """Reuse buffers for the next frame (avoids per-frame allocation)."""
        if self._fill == (0, 0, 0):
            self.rgb.fill(0)
        else:
            self.rgb[:] = self._fill
        self.labels.fill(0)

    # -- low level helpers -------------------------------------------------
    def _lid(self, label: str | None) -> int:
        return SEMANTIC_CLASSES[label] if label is not None else 0

    def _stamp(self, mask: np.ndarray, label: str | None) -> None:
        if label is not None:
            self.labels[mask > 0] = SEMANTIC_CLASSES[label]

    # -- primitives --------------------------------------------------------
    def rect(self, x, y, w, h, color, thickness=-1, label=None, angle=None):
        x, y, w, h = int(x), int(y), int(w), int(h)
        if angle:
            cx, cy = x + w / 2, y + h / 2
            box = cv2.boxPoints(((cx, cy), (w, h), angle))
            self.polygon(box, color, label=label)
            return
        cv2.rectangle(self.rgb, (x, y), (x + w, y + h), color, thickness)
        if label is not None and thickness < 0:
            self.labels[max(y, 0):max(y, 0) + max(h, 0),
                        max(x, 0):max(x, 0) + max(w, 0)] = self._lid(label)
        elif label is not None:
            m = np.zeros_like(self.labels)
            cv2.rectangle(m, (x, y), (x + w, y + h), 255, thickness)
            self._stamp(m, label)

    def circle(self, cx, cy, r, color, thickness=-1, label=None):
        cv2.circle(self.rgb, (int(cx), int(cy)), max(1, int(r)), color, thickness)
        if label is not None:
            m = np.zeros_like(self.labels)
            cv2.circle(m, (int(cx), int(cy)), max(1, int(r)), 255, thickness)
            self._stamp(m, label)

    def ellipse(self, cx, cy, ax, ay, angle, color, thickness=-1, label=None):
        cv2.ellipse(self.rgb, (int(cx), int(cy)), (max(1, int(ax)), max(1, int(ay))),
                    float(angle), 0, 360, color, thickness)
        if label is not None:
            m = np.zeros_like(self.labels)
            cv2.ellipse(m, (int(cx), int(cy)), (max(1, int(ax)), max(1, int(ay))),
                        float(angle), 0, 360, 255, thickness)
            self._stamp(m, label)

    def line(self, x0, y0, x1, y1, color, thickness=1, label=None):
        cv2.line(self.rgb, (int(x0), int(y0)), (int(x1), int(y1)), color,
                 max(1, int(thickness)), cv2.LINE_AA)
        if label is not None:
            m = np.zeros_like(self.labels)
            cv2.line(m, (int(x0), int(y0)), (int(x1), int(y1)), 255,
                     max(1, int(thickness)), cv2.LINE_AA)
            self._stamp(m, label)

    def polygon(self, pts, color, label=None):
        pts = np.asarray(pts, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(self.rgb, [pts], color, cv2.LINE_AA)
        if label is not None:
            m = np.zeros_like(self.labels)
            cv2.fillPoly(m, [pts], 255, cv2.LINE_AA)
            self._stamp(m, label)

    def polyline(self, pts, color, thickness=1, label=None, closed=False):
        pts = np.asarray(pts, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(self.rgb, [pts], closed, color, max(1, int(thickness)), cv2.LINE_AA)
        if label is not None:
            m = np.zeros_like(self.labels)
            cv2.polylines(m, [pts], closed, 255, max(1, int(thickness)), cv2.LINE_AA)
            self._stamp(m, label)

    def text(self, x, y, s, color, scale=0.6, thickness=1, label="text",
             font=cv2.FONT_HERSHEY_SIMPLEX, anchor="tl"):
        """Hershey-font text (no external font files, PICPLAN §4)."""
        (tw, th), baseline = cv2.getTextSize(str(s), font, scale, thickness)
        if anchor == "center":
            x, y = x - tw // 2, y + th // 2
        elif anchor == "tr":
            x = x - tw
        org = (int(x), int(y))
        cv2.putText(self.rgb, str(s), org, font, scale, color, thickness, cv2.LINE_AA)
        if label is not None:
            m = np.zeros_like(self.labels)
            cv2.putText(m, str(s), org, font, scale, 255, thickness, cv2.LINE_AA)
            self._stamp(m, label)

    def blend_masked(self, other_rgb, mask, alpha=1.0):
        """Alpha-composite ``other_rgb`` onto rgb where mask > 0."""
        m = (mask > 0)[:, :, None]
        if alpha >= 1.0:
            self.rgb = np.where(m, other_rgb, self.rgb).astype(np.uint8)
        else:
            blended = (alpha * other_rgb + (1 - alpha) * self.rgb)
            self.rgb = np.where(m, blended.astype(np.uint8), self.rgb)


def checker_fill(width, height, tile, c0, c1, ox=0, oy=0):
    """Deterministic checkerboard background (no RNG)."""
    xs = ((np.arange(width) + ox) // tile) & 1
    ys = ((np.arange(height) + oy) // tile) & 1
    pat = ys[:, None] ^ xs[None, :]
    out = np.empty((height, width, 3), dtype=np.uint8)
    out[:] = c0
    out[pat > 0] = c1
    return out
