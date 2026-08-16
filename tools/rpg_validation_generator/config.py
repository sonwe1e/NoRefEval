"""Fixed generation parameters (PICPLAN §3)."""
from __future__ import annotations

from dataclasses import dataclass

DEFAULT_SEED = 20260729


@dataclass(frozen=True)
class GeneratorConfig:
    """All geometry/timing parameters for one generation run.

    Production runs must use 1280x720 / 4 s / 120 FPS (PICPLAN §3); the
    draft preset (640x360 / 2 s) exists only for fast debugging.
    """

    width: int = 1280
    height: int = 720
    duration_seconds: float = 4.0
    master_fps: int = 120
    seed: int = DEFAULT_SEED
    # World canvas is larger than the viewport so the camera can pan
    # (PICPLAN §6.1).  Kept proportional for draft mode.
    world_width: int = 1600
    world_height: int = 1000

    @property
    def n_master_frames(self) -> int:
        return int(round(self.duration_seconds * self.master_fps))

    @property
    def resolution(self) -> tuple[int, int]:
        return (self.width, self.height)

    def scene_seed(self, scene_index: int) -> int:
        """Deterministic per-scene seed."""
        return self.seed + scene_index

    def case_seed(self, mode_id: int, case_index: int) -> int:
        """Deterministic per-case seed for defect injection."""
        return self.seed + 1000 * mode_id + case_index

    @classmethod
    def production(cls, seed: int = DEFAULT_SEED) -> "GeneratorConfig":
        return cls(seed=seed)

    @classmethod
    def draft(cls, seed: int = DEFAULT_SEED) -> "GeneratorConfig":
        return cls(
            width=640,
            height=360,
            duration_seconds=2.0,
            seed=seed,
            world_width=800,
            world_height=500,
        )
