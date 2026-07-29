"""Evaluation presets and mode-aware global configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class EvaluationMode(str, Enum):
    """Mathematically distinct evaluation contracts.

    The string values are the public CLI/API spellings.  Keeping them in one
    enum prevents a silent fallback to endpoint assumptions when an input is
    actually no-reference or same-rate full-reference.
    """

    NO_REFERENCE = "no-reference"
    ENDPOINT_2X = "endpoint-2x"
    FULL_REFERENCE = "full-reference"


MODE_ALIASES = {
    "no_reference": EvaluationMode.NO_REFERENCE,
    "nr": EvaluationMode.NO_REFERENCE,
    "endpoint": EvaluationMode.ENDPOINT_2X,
    "endpoint_reference": EvaluationMode.ENDPOINT_2X,
    "endpoint-reference": EvaluationMode.ENDPOINT_2X,
    "full_reference": EvaluationMode.FULL_REFERENCE,
    "fr": EvaluationMode.FULL_REFERENCE,
}


def parse_mode(value: str | EvaluationMode) -> EvaluationMode:
    if isinstance(value, EvaluationMode):
        return value
    normalized = value.strip().lower()
    try:
        return EvaluationMode(normalized)
    except ValueError:
        try:
            return MODE_ALIASES[normalized]
        except KeyError as exc:
            choices = ", ".join(m.value for m in EvaluationMode)
            raise ValueError(f"unknown evaluation mode {value!r}; choose from {choices}") from exc


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
    source_video: str | None
    candidate_video: str
    mode: EvaluationMode = EvaluationMode.ENDPOINT_2X
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
        """Backward-compatible endpoint-2x configuration builder."""
        return cls.build_mode(
            candidate_video=candidate_video,
            reference_video=source_video,
            mode=EvaluationMode.ENDPOINT_2X,
            preset=preset,
            **overrides,
        )

    @classmethod
    def build_mode(
        cls,
        candidate_video: str,
        reference_video: str | None = None,
        mode: str | EvaluationMode = EvaluationMode.NO_REFERENCE,
        preset: str = "standard",
        **overrides,
    ) -> "EvalConfig":
        if preset not in PRESETS:
            raise ValueError(f"unknown preset {preset!r}; choose from {sorted(PRESETS)}")
        parsed_mode = parse_mode(mode)
        if parsed_mode is EvaluationMode.NO_REFERENCE:
            if reference_video is not None:
                raise ValueError("no-reference mode does not accept a reference video")
        elif not reference_video:
            raise ValueError(f"{parsed_mode.value} mode requires a reference video")
        cfg = cls(
            source_video=reference_video,
            candidate_video=candidate_video,
            mode=parsed_mode,
            preset=PRESETS[preset],
        )
        for k, v in overrides.items():
            if not hasattr(cfg, k):
                raise ValueError(f"unknown config override {k!r}")
            setattr(cfg, k, v)
        return cfg

    @property
    def reference_video(self) -> str | None:
        """Preferred public name; ``source_video`` remains for old callers."""
        return self.source_video
