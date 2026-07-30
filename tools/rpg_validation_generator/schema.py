"""Data contracts shared by renderer, defects, builders and manifests."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Semantic label ids used by the renderer's label buffer and consumed by
# defect operators / masks (PICPLAN §13).
SEMANTIC_CLASSES: dict[str, int] = {
    "background": 0,
    "character": 1,
    "enemy": 2,
    "weapon": 3,
    "projectile": 4,
    "particle": 5,
    "ui": 6,
    "text": 7,
    "foreground_occluder": 8,
}
CLASS_BY_ID: dict[int, str] = {v: k for k, v in SEMANTIC_CLASSES.items()}

MODES = ("nr", "endpoint", "fr")
MODE_DIR = {
    "nr": "nr_real",
    "endpoint": "endpoint_real",
    "fr": "fr_real",
}
MODE_ID = {"nr": 1, "endpoint": 2, "fr": 3}


@dataclass
class DefectResult:
    """Output contract of a DefectOperator (PICPLAN §12)."""

    frames: np.ndarray
    affected_indices: list[int]
    affected_interval_seconds: tuple[float, float]
    roi_boxes: list[list[int]]
    mask_paths: list[str]
    parameters: dict


@dataclass
class CaseRecord:
    """Everything a builder produces for one case, ready for manifesting."""

    case_id: str
    mode: str
    scene_id: str
    seed: int
    files: dict[str, str]          # role -> filename inside the case dir
    fps_by_role: dict[str, int]
    frame_count_by_role: dict[str, int]
    defects: list[dict]            # serializable defect descriptors
    raw_rgb_sha256: dict[str, str]  # role -> hash of raw (pre-encode) RGB
    scene_spec_sha256: str
    scene_spec: dict


@dataclass
class Manifest:
    schema_version: str
    case_id: str
    mode: str
    scene_id: str
    seed: int
    data_origin: str = "procedural_synthetic_rpg"
    real_capture: bool = False
    production_gate: bool = False
    resolution: tuple[int, int] = (1280, 720)
    duration_seconds: float = 4.0
    master_fps: int = 120
    files: dict[str, str] = field(default_factory=dict)
    fps_by_role: dict[str, int] = field(default_factory=dict)
    frame_count_by_role: dict[str, int] = field(default_factory=dict)
    defects: list[dict] = field(default_factory=list)
    generator: dict = field(default_factory=dict)
    hashes: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "case_id": self.case_id,
            "mode": self.mode,
            "scene_id": self.scene_id,
            "seed": self.seed,
            "data_origin": self.data_origin,
            "real_capture": self.real_capture,
            "production_gate": self.production_gate,
            "resolution": list(self.resolution),
            "duration_seconds": self.duration_seconds,
            "master_fps": self.master_fps,
            "files": self.files,
            "fps_by_role": self.fps_by_role,
            "frame_count_by_role": self.frame_count_by_role,
            "defects": self.defects,
            "generator": self.generator,
            "hashes": self.hashes,
        }
