"""rr_vfiqa — Endpoint-Referenced / Reduced-Reference VFI Quality Assessment.

Quality evaluation for 60→120 FPS game frame interpolation: the original
60 FPS frames are anchors; generated frames are judged by endpoint flow
composition, motion-compensated temporal stability, reverse anchor cycle,
and local structural integrity (see USERPLAN.md).
"""

from .config import EvalConfig, PRESETS
from .pipeline import compare_models, evaluate_vfi
from .schema import Report

__version__ = "0.1.0"

__all__ = ["evaluate_vfi", "compare_models", "EvalConfig", "PRESETS", "Report",
           "__version__"]
