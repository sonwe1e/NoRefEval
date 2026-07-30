from .window_flows import WindowFlows
from . import anchor_integrity, flow_composition_metric, cycle_reconstruction
from . import temporal_compensation, parity_frequency, edge_structure
from . import global_technical_quality

__all__ = [
    "WindowFlows", "anchor_integrity", "flow_composition_metric",
    "cycle_reconstruction", "temporal_compensation", "parity_frequency",
    "edge_structure", "global_technical_quality", "full_reference",
    "no_reference",
]
from . import full_reference, no_reference  # noqa: E402,F811  (``no_reference`` re-exports)

# Dense error-map accessors (USERPLAN §8) — kept here so callers can do
# ``from rr_vfiqa.metrics import nr_window_maps`` without importing the
# heavy metric module directly.
from .no_reference import compute_window_maps as nr_window_maps  # noqa: E402
from .full_reference import compute_window_maps as fr_window_maps  # noqa: E402
