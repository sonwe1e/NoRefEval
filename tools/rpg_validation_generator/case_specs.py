"""Single source of truth for the 15 cases (PICPLAN §9, §10, §11).

Each spec maps a mode + case index to its scene, defect operator, time
window, parameters, candidate FPS and the set of output roles it emits.
Builders and tests both import CASES so the contracts stay in lockstep.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CaseSpec:
    mode: str
    case_index: int            # 1-based, matches case_0X dirs
    scene_index: int           # 0-based into SCENES
    defect_type: str
    start_time: float
    end_time: float
    candidate_fps: int
    roles: tuple[str, ...]     # output video roles
    params: dict = field(default_factory=dict)
    semantic_targets: tuple[str, ...] = ()

    @property
    def case_id(self) -> str:
        return f"{self.mode}_case_{self.case_index:02d}"

    @property
    def case_dir_name(self) -> str:
        return f"case_{self.case_index:02d}"


def _sc(i):
    return i - 1  # scene_index from 1-based scene number in the plan


CASES: list[CaseSpec] = [
    # ---- No-Reference (PICPLAN §9) --------------------------------------
    CaseSpec("nr", 1, _sc(1), "freeze", 1.50, 2.20, 60,
             ("candidate", "oracle")),
    CaseSpec("nr", 2, _sc(2), "odd_frame_blur", 1.20, 2.80, 120,
             ("candidate", "oracle"), {"kernel": 15, "sigma": 3.0}),
    CaseSpec("nr", 3, _sc(3), "cadence_collapse", 1.40, 2.70, 120,
             ("candidate", "oracle")),
    CaseSpec("nr", 4, _sc(4), "ui_drift", 1.00, 3.10, 60,
             ("candidate", "oracle")),
    CaseSpec("nr", 5, _sc(5), "projectile_ghost_flicker", 1.30, 2.60, 120,
             ("candidate", "oracle")),
    # ---- Endpoint-2x (PICPLAN §10) --------------------------------------
    CaseSpec("endpoint", 1, _sc(1), "generated_motion_blur", 1.20, 2.50, 120,
             ("source", "candidate", "oracle"), {"kernel": 11, "sigma": 2.5},
             ("character", "enemy")),
    CaseSpec("endpoint", 2, _sc(2), "disocclusion_ghost", 1.30, 2.70, 120,
             ("source", "candidate", "oracle"), {},
             ("foreground_occluder", "character")),
    CaseSpec("endpoint", 3, _sc(3), "thin_weapon_wrong_motion", 1.10, 2.40, 120,
             ("source", "candidate", "oracle"),
             {"angle_error_degrees": 8, "tip_offset_pixels": 6}, ("weapon",)),
    CaseSpec("endpoint", 4, _sc(4), "ui_text_drift", 1.00, 3.00, 120,
             ("source", "candidate", "oracle")),
    CaseSpec("endpoint", 5, _sc(5), "generated_freeze_copy", 1.40, 2.50, 120,
             ("source", "candidate", "oracle"), {},
             ("particle", "projectile")),
    # ---- Full-Reference (PICPLAN §11) -----------------------------------
    CaseSpec("fr", 1, _sc(1), "global_blur", 1.20, 2.60, 60,
             ("source", "reference", "candidate"), {"kernel": 21, "sigma": 5.0}),
    CaseSpec("fr", 2, _sc(2), "local_spatial_shift", 1.30, 2.70, 60,
             ("source", "reference", "candidate"), {"dx": 4},
             ("character", "foreground_occluder")),
    CaseSpec("fr", 3, _sc(3), "thin_object_delete", 1.10, 2.40, 60,
             ("source", "reference", "candidate"), {}, ("weapon",)),
    CaseSpec("fr", 4, _sc(4), "ui_text_corruption", 1.00, 3.00, 60,
             ("source", "reference", "candidate")),
    CaseSpec("fr", 5, _sc(5), "temporal_freeze_flicker", 1.40, 2.60, 60,
             ("source", "reference", "candidate"),
             {"freeze_frames": 4, "luma_amp": 12},
             ("character", "enemy", "projectile", "particle")),
]


def cases_for_mode(mode: str) -> list[CaseSpec]:
    return [c for c in CASES if c.mode == mode]


def spec_by_id(case_id: str) -> CaseSpec:
    for c in CASES:
        if c.case_id == case_id:
            return c
    raise KeyError(case_id)
