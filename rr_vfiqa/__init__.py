"""Mode-aware VFI quality assessment.

The public API separates single-video artifact risk, 2× endpoint-reference
evaluation, and same-rate full-reference fidelity.  Scores from different
modes intentionally use different schemas and must not be compared directly.
"""

from .config import EvalConfig, EvaluationMode, PRESETS
from .multimode import (
    compare,
    evaluate,
    evaluate_full_reference,
    evaluate_no_reference,
)
from .pipeline import compare_models, evaluate_endpoint_reference, evaluate_vfi
from .schema import Report

__version__ = "0.2.0"

__all__ = [
    "evaluate",
    "evaluate_no_reference",
    "evaluate_endpoint_reference",
    "evaluate_full_reference",
    "compare",
    "evaluate_vfi",
    "compare_models",
    "EvaluationMode",
    "EvalConfig",
    "PRESETS",
    "Report",
    "__version__",
]
