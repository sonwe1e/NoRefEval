"""Spatial diagnostic visualizations (USERPLAN §8 / §15).

The metrics emit per-window *scalars*; the pipelines can also emit dense
``error_maps`` (H, W) fields for the highest-risk windows.  This module turns
those fields into human-readable artefacts:

* ``maps.py`` — unified normalization (percentile clip, robust-z, 0..1) so
  different fields are comparable on a common scale;
* ``heatmaps.py`` — jet heatmap PNGs and translucent frame overlays;
* ``flow_overlay.py`` — HSV flow-direction colour wheel with magnitude
  brightness.

Nothing here depends on matplotlib: all rendering is OpenCV, so the same code
runs in the CLI and in CI.
"""

from .flow_overlay import flow_to_bgr, flow_to_bgr_with_legend
from .heatmaps import blend_heat, heat_bgr, heatmap_to_bgr, save_heatmap
from .maps import MapNormalization, fit_normalization, normalize_map, robust_z

__all__ = [
    "normalize_map",
    "robust_z",
    "fit_normalization",
    "MapNormalization",
    "heat_bgr",
    "heatmap_to_bgr",
    "blend_heat",
    "save_heatmap",
    "flow_to_bgr",
    "flow_to_bgr_with_legend",
]
