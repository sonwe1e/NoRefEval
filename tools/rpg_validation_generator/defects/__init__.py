"""Defect operators (PICPLAN §12).  Each is applied to clean master frames
*after* rendering, never baked into scene code."""
from .base import DefectContext, DefectOperator
from .blur import GlobalBlur, GeneratedMotionBlur, OddFrameBlur
from .cadence import CadenceCollapse
from .delete import ThinObjectDelete
from .flicker import TemporalFreezeFlicker
from .freeze import Freeze, GeneratedFreezeCopy
from .ghost import DisocclusionGhost, ProjectileGhostFlicker
from .shift import LocalSpatialShift, ThinWeaponWrongMotion
from .ui import UIDrift, UITextCorruption, UITextDrift

REGISTRY: dict[str, type[DefectOperator]] = {
    "freeze": Freeze,
    "odd_frame_blur": OddFrameBlur,
    "cadence_collapse": CadenceCollapse,
    "ui_drift": UIDrift,
    "projectile_ghost_flicker": ProjectileGhostFlicker,
    "generated_motion_blur": GeneratedMotionBlur,
    "disocclusion_ghost": DisocclusionGhost,
    "thin_weapon_wrong_motion": ThinWeaponWrongMotion,
    "ui_text_drift": UITextDrift,
    "generated_freeze_copy": GeneratedFreezeCopy,
    "global_blur": GlobalBlur,
    "local_spatial_shift": LocalSpatialShift,
    "thin_object_delete": ThinObjectDelete,
    "ui_text_corruption": UITextCorruption,
    "temporal_freeze_flicker": TemporalFreezeFlicker,
}


def get_operator(defect_type: str) -> DefectOperator:
    return REGISTRY[defect_type]()


__all__ = ["DefectContext", "DefectOperator", "REGISTRY", "get_operator"]
