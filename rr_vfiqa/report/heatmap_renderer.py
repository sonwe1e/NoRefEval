"""Per-window error-map heatmaps (diagnostic PNGs)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def save_error_heatmap(error_map: np.ndarray, path: str | Path,
                       vmax: float | None = None, label: str = "") -> Path:
    """Render a (H, W) error field as a jet heatmap PNG."""
    m = np.nan_to_num(error_map.astype(np.float32), nan=0.0)
    vmax = vmax or max(float(np.percentile(m, 99.5)), 1e-6)
    norm = np.clip(m / vmax, 0, 1)
    heat = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    if label:
        cv2.putText(heat, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2, cv2.LINE_AA)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), heat)
    return path
