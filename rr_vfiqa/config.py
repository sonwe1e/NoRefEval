"""Evaluation presets (USERPLAN.md §10) and global configuration."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Preset:
    name: str

    # --- tier 1: cheap full-frame scan ---
    scan_width: int = 384                 # low-res width for the full scan

    # --- tier 2: core evaluation windows ---
    uniform_windows: int = 16
    risk_windows: int = 32
    window_frames: int = 5                # frames per window (odd number)
    temporal_nms_seconds: float = 0.5
    flow_width: int = 960                 # working resolution bucket for flow metrics
    run_region_branches: bool = True      # lightweight UI / character / thin branches

    # --- tier 3: audit diagnostics ---
    audit_top_fraction: float = 0.0       # fraction of tier-2 windows escalated
    audit_max_windows: int = 0
    full_res_edges: bool = False
    run_tracker: bool = False             # KLT/CoTracker weapon tracking
    run_depth: bool = False

    # --- fusion ---
    score_weights: dict[str, float] = field(default_factory=lambda: {
        "motion": 0.25,
        "temporal": 0.20,
        "structure": 0.20,
        "character_thin": 0.15,
        "ui": 0.10,
        "transition": 0.05,
        "global": 0.05,
    })
    calibrator_path: str | None = None    # trained LightGBM, else use the formula
    max_windows_hard_cap: int = 256


FAST = Preset(
    name="fast",
    scan_width=320,
    uniform_windows=8,
    risk_windows=8,
    flow_width=480,
    run_region_branches=False,
)

STANDARD = Preset(
    name="standard",
    scan_width=384,
    uniform_windows=16,
    risk_windows=32,
    flow_width=960,
    run_region_branches=True,
)

AUDIT = Preset(
    name="audit",
    scan_width=480,
    uniform_windows=24,
    risk_windows=48,
    flow_width=960,
    run_region_branches=True,
    audit_top_fraction=0.10,
    audit_max_windows=64,
    full_res_edges=True,
    run_tracker=True,
)

PRESETS: dict[str, Preset] = {"fast": FAST, "standard": STANDARD, "audit": AUDIT}


@dataclass
class EvalConfig:
    source_video: str
    candidate_video: str
    preset: Preset = STANDARD
    cache_dir: str = "./cache"
    device: str = "cuda"                  # "cuda" | "cpu" | "npu"
    flow_backend: str = "auto"            # "auto" | "raft" | "farneback"
    out_dir: str | None = None
    seed: int = 0

    # Numerical constants used across metrics (USERPLAN.md §3, §11)
    charbonnier_tau: float = 4.0          # px, flow error robustness
    occlusion_cycle_threshold: float = 3.0  # px, fwd-bwd cycle => occluded
    anchor_shift_search: int = 3          # candidate frames to try around 2i
    aggregation: tuple[float, float, float] = (0.4, 0.4, 0.2)  # P50/P90/P99

    @classmethod
    def build(cls, source_video: str, candidate_video: str, preset: str = "standard",
              **overrides) -> "EvalConfig":
        if preset not in PRESETS:
            raise ValueError(f"unknown preset {preset!r}; choose from {sorted(PRESETS)}")
        cfg = cls(source_video=source_video, candidate_video=candidate_video,
                  preset=PRESETS[preset])
        for k, v in overrides.items():
            if not hasattr(cfg, k):
                raise ValueError(f"unknown config override {k!r}")
            setattr(cfg, k, v)
        return cfg
