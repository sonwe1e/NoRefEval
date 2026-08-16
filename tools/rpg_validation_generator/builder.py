"""Per-case builder: assemble role videos, verify contracts *before*
encoding (PICPLAN §10 / §19), encode H.264, hash everything."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .case_specs import CaseSpec
from .config import GeneratorConfig
from .defects import get_operator
from .defects.base import DefectContext
from .encode import encode_rgb
from .hashing import file_sha256, json_sha256, raw_rgb_sha256, raw_rgb_sha256_subset
from .manifests import _role_fps, build_record
from .schema import CaseRecord
from .timeline import Master, gather, sample_indices

_ROLE_FILE = {
    "candidate": "candidate_{fps}.mp4",
    "oracle": "oracle_{tag}.mp4",
    "source": "source_{fps}.mp4",
    "reference": "reference_{fps}.mp4",
}
_ORACLE_TAG = {"nr": "clean", "endpoint": "gt"}


@dataclass
class _Contract:
    endpoint_even_exact: bool | None = None
    fr_outside_mask_exact: bool | None = None


def _role_array(master: Master, role: str, spec: CaseSpec,
                config: GeneratorConfig) -> np.ndarray:
    n = master.n_frames
    if spec.mode == "nr":
        idx = sample_indices(config.master_fps, spec.candidate_fps, n)
        return gather(master.rgb, idx)
    if spec.mode == "endpoint":
        if role == "source":
            return gather(master.rgb, sample_indices(config.master_fps, 60, n))
        return gather(master.rgb, np.arange(n))            # candidate / oracle = master
    # fr
    if role == "source":
        return gather(master.rgb, sample_indices(config.master_fps, 30, n))
    return gather(master.rgb, sample_indices(config.master_fps, 60, n))  # ref / cand


def _labels_for(master: Master, role: str, spec: CaseSpec,
                config: GeneratorConfig) -> np.ndarray:
    n = master.n_frames
    if spec.mode == "nr":
        idx = sample_indices(config.master_fps, spec.candidate_fps, n)
    elif spec.mode == "endpoint":
        idx = np.arange(n)
    elif role == "source":
        idx = sample_indices(config.master_fps, 30, n)
    else:
        idx = sample_indices(config.master_fps, 60, n)
    return np.ascontiguousarray(master.labels[idx])


def _world_for(master: Master, role: str, spec: CaseSpec,
               config: GeneratorConfig) -> np.ndarray:
    n = master.n_frames
    if spec.mode == "nr":
        idx = sample_indices(config.master_fps, spec.candidate_fps, n)
    elif spec.mode == "endpoint":
        idx = np.arange(n)
    elif role == "source":
        idx = sample_indices(config.master_fps, 30, n)
    else:
        idx = sample_indices(config.master_fps, 60, n)
    return np.ascontiguousarray(master.world_only[idx])


def _upscale_mask(png_path: Path, h: int, w: int) -> np.ndarray:
    small = cv2.imread(str(png_path), cv2.IMREAD_GRAYSCALE)
    full = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    return full > 127


def _verify_endpoint(candidate: np.ndarray, source: np.ndarray) -> bool:
    return bool(np.array_equal(candidate[0::2], source))


def _verify_fr(candidate: np.ndarray, reference: np.ndarray,
               defect_mask_paths: list[str], case_dir: Path) -> bool:
    h, w = candidate.shape[1:3]
    contract_dir = case_dir / "_contract_masks"
    for i in range(len(candidate)):
        png = contract_dir / f"{i:06d}.png"
        if png.exists():
            m = _upscale_mask(png, h, w)
            if not np.array_equal(candidate[i][~m], reference[i][~m]):
                return False
        else:
            if not np.array_equal(candidate[i], reference[i]):
                return False
    return True


def _encode_role(role: str, frames: np.ndarray, fps: int, case_dir: Path,
                 spec: CaseSpec) -> str:
    if role == "oracle":
        name = _ROLE_FILE[role].format(tag=_ORACLE_TAG[spec.mode])
    else:
        name = _ROLE_FILE[role].format(fps=fps)
    encode_rgb(frames, fps, case_dir / name)
    return name


def build_case(spec: CaseSpec, master: Master, config: GeneratorConfig,
               mode_dir: Path, case_seed: int,
               generation_command: str) -> tuple[CaseRecord, dict, _Contract]:
    """Build one case on disk; return (record, provenance, contract)."""
    case_dir = mode_dir / spec.case_dir_name
    case_dir.mkdir(parents=True, exist_ok=True)

    role_frames: dict[str, np.ndarray] = {
        r: _role_array(master, r, spec, config) for r in spec.roles
    }
    # defect is injected into the candidate buffer only
    cand = role_frames["candidate"]
    cand_labels = _labels_for(master, "candidate", spec, config)
    cand_world = _world_for(master, "candidate", spec, config)
    ctx = DefectContext(fps=spec.candidate_fps, n_frames=len(cand),
                        labels=cand_labels, world_only=cand_world,
                        config=config, rng=np.random.default_rng(case_seed),
                        case_dir=case_dir)
    result = get_operator(spec.defect_type).apply(
        cand, spec.candidate_fps, spec.start_time, spec.end_time, ctx,
        **spec.params)

    # ---- pre-encode contract verification (PICPLAN §10 / §19) ------------
    contract = _Contract()
    if spec.mode == "endpoint":
        contract.endpoint_even_exact = _verify_endpoint(cand, role_frames["source"])
        if not contract.endpoint_even_exact:
            raise RuntimeError(f"{spec.case_id}: even-frame contract violated")
    if spec.mode == "fr":
        contract.fr_outside_mask_exact = _verify_fr(
            cand, role_frames["reference"], result.mask_paths, case_dir)
        if not contract.fr_outside_mask_exact:
            raise RuntimeError(f"{spec.case_id}: FR outside-mask contract violated")

    # ---- encode each role -------------------------------------------------
    files: dict[str, str] = {}
    frame_counts: dict[str, int] = {}
    for role in spec.roles:
        fps = _role_fps(spec, role)
        files[role] = _encode_role(role, role_frames[role], fps, case_dir, spec)
        frame_counts[role] = int(role_frames[role].shape[0])

    # ---- hashes -----------------------------------------------------------
    raw_hashes = {role: raw_rgb_sha256(role_frames[role]) for role in spec.roles}
    if spec.mode == "endpoint":
        raw_hashes["candidate_even"] = raw_rgb_sha256_subset(
            role_frames["candidate"], np.arange(0, len(cand), 2))
    video_hashes = {role: file_sha256(case_dir / files[role]) for role in spec.roles}

    from .scenes import make_scene
    scene_spec = make_scene(spec.scene_index, config).spec()
    scene_spec_sha = json_sha256(scene_spec)

    defect_desc = [{
        "type": spec.defect_type,
        "start_time": result.affected_interval_seconds[0] if result.affected_indices
        else spec.start_time,
        "end_time": result.affected_interval_seconds[1] if result.affected_indices
        else spec.end_time,
        "start_frame": result.affected_indices[0] if result.affected_indices else None,
        "end_frame": (result.affected_indices[-1] + 1) if result.affected_indices else None,
        "semantic_targets": list(spec.semantic_targets),
        "roi_boxes": result.roi_boxes,
        "mask_directory": "defect_masks",
        "n_affected_frames": len(result.affected_indices),
        "parameters": result.parameters,
    }]

    from .provenance import build_provenance, write_provenance
    provenance = build_provenance(
        scene_seed=config.scene_seed(spec.scene_index),
        scene_spec_sha256=scene_spec_sha, raw_rgb_sha256=raw_hashes,
        video_sha256=video_hashes, generation_command=generation_command)
    provenance["contracts"] = {
        "endpoint_even_exact_pre_encode": contract.endpoint_even_exact,
        "fr_outside_mask_exact_pre_encode": contract.fr_outside_mask_exact,
    }
    write_provenance(case_dir, provenance)

    record = build_record(
        spec, config, scene_spec, scene_spec_sha, raw_hashes, files,
        frame_counts, defect_desc, provenance["generator_commit"])

    from .manifests import record_to_manifest, write_case_manifest
    write_case_manifest(case_dir, record_to_manifest(record, config, provenance))
    return record, provenance, contract
