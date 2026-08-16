"""Camera acting on world-space content only (PICPLAN §6.1).

The renderer draws world layers onto a large world canvas; the camera maps
that canvas into the viewport with an affine transform (translate + rotate +
zoom).  Screen-space UI is drawn afterwards, directly in viewport pixels,
and is never affected by the camera.
"""
from __future__ import annotations

import cv2
import numpy as np


class Camera:
    def __init__(self, center_x: float, center_y: float, zoom: float = 1.0,
                 rotation_deg: float = 0.0):
        self.cx = float(center_x)
        self.cy = float(center_y)
        self.zoom = float(zoom)
        self.rot = float(rotation_deg)

    def matrix(self, viewport_w: int, viewport_h: int) -> np.ndarray:
        """2x3 affine: world canvas pixels -> viewport pixels."""
        m = np.eye(3, dtype=np.float64)
        # world center -> origin
        t1 = np.array([[1, 0, -self.cx], [0, 1, -self.cy], [0, 0, 1]], dtype=np.float64)
        a = np.radians(self.rot)
        c, s = np.cos(a), np.sin(a)
        rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)
        z = np.array([[self.zoom, 0, 0], [0, self.zoom, 0], [0, 0, 1]], dtype=np.float64)
        t2 = np.array([[1, 0, viewport_w / 2], [0, 1, viewport_h / 2], [0, 0, 1]],
                      dtype=np.float64)
        m = t2 @ z @ rz @ t1
        return m[:2]

    def warp(self, world_rgb: np.ndarray, world_labels: np.ndarray,
             viewport_w: int, viewport_h: int):
        m = self.matrix(viewport_w, viewport_h)
        rgb = cv2.warpAffine(world_rgb, m, (viewport_w, viewport_h),
                             flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
        labels = cv2.warpAffine(world_labels, m, (viewport_w, viewport_h),
                                flags=cv2.INTER_NEAREST, borderValue=0)
        return rgb, labels
