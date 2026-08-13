"""Flow-direction colour wheel (USERPLAN §8 / §9).

Encodes a (H, W, 2) flow field as an HSV image: *hue* = motion direction,
*saturation* = 1, *value* = normalized magnitude.  This is the standard
Middlebury encoding and makes rotation / folding / divergence readable at a
glance — far better than arrow glyphs for dense fields.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np



def flow_to_bgr(flow: np.ndarray, *,
                magnitude_scale: float | None = None) -> np.ndarray:
    """Encode ``flow`` (H, W, 2) as a (H, W, 3) BGR colour wheel image.

    ``magnitude_scale`` fixes the magnitude that maps to full brightness; when
    ``None`` it is derived from the 99th percentile so a few fast pixels do not
    wash the rest out.
    """
    if flow.ndim != 3 or flow.shape[2] != 2:
        raise ValueError("flow must be (H, W, 2)")
    mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    ang = np.arctan2(flow[..., 1], flow[..., 0])   # -pi..pi
    hue = ((ang + np.pi) / (2.0 * np.pi) * 180.0).astype(np.float32)  # 0..180
    if magnitude_scale is None:
        magnitude_scale = max(float(np.percentile(mag, 99)), 1e-3)
    val = np.clip(mag / magnitude_scale, 0.0, 1.0).astype(np.float32)
    hsv = np.stack([hue, np.full_like(hue, 255, dtype=np.float32), val], -1)
    hsv_u8 = np.empty(hsv.shape, np.uint8)
    hsv_u8[..., 0] = hue.astype(np.uint8)
    hsv_u8[..., 1] = 255
    hsv_u8[..., 2] = (val * 255).astype(np.uint8)
    return cv2.cvtColor(hsv_u8, cv2.COLOR_HSV2BGR)


def flow_to_bgr_with_legend(flow: np.ndarray, path: str | Path,
                            *, magnitude_scale: float | None = None,
                            step: int = 24) -> Path:
    """Render the flow colour wheel with a sparse arrow legend, saved as PNG.

    The legend arrows on a grid show the direction each hue encodes, so the
    image is interpretable without a separate key.
    """
    bgr = flow_to_bgr(flow, magnitude_scale=magnitude_scale)
    h, w = flow.shape[:2]
    mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    ys = np.arange(step // 2, h, step)
    xs = np.arange(step // 2, w, step)
    for y in ys:
        for x in xs:
            dx, dy = float(flow[y, x, 0]), float(flow[y, x, 1])
            if mag[y, x] < 1e-3:
                continue
            ex = int(round(x + dx))
            ey = int(round(y + dy))
            cv2.arrowedLine(bgr, (int(x), int(y)), (ex, ey),
                            (255, 255, 255), 1, cv2.LINE_AA, tipLength=0.3)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), bgr)
    return path
