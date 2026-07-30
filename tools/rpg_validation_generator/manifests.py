"""Manifest writers (PICPLAN §15).

Per-case ``manifest.json`` plus a per-mode ``manifest.jsonl`` summary (one
line per case).  Every manifest declares the synthetic origin so the data
can never be mistaken for a real capture.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import GENERATOR_NAME, SCHEMA_VERSION, __version__
from .case_specs import CaseSpec
from .config import GeneratorConfig
from .schema import CaseRecord


def _role_fps(spec: CaseSpec, role: str) -> int:
    if spec.mode == "nr":
        return spec.candidate_fps
    if spec.mode == "endpoint":
        return {"source": 60, "candidate": 120, "oracle": 120}[role]
    # fr
    return {"source": 30, "reference": 60, "candidate": 60}[role]


def build_record(spec: CaseSpec, config: GeneratorConfig,
                 scene_spec: dict, scene_spec_sha256: str,
                 raw_rgb_sha256: dict, files: dict[str, str],
                 frame_counts: dict[str, int], defects: list[dict],
                 commit: str) -> CaseRecord:
    fps_by_role = {r: _role_fps(spec, r) for r in spec.roles}
    return CaseRecord(
        case_id=spec.case_id, mode=spec.mode, scene_id=f"scene_{spec.scene_index + 1:02d}",
        seed=config.seed, files=files, fps_by_role=fps_by_role,
        frame_count_by_role=frame_counts, defects=defects,
        raw_rgb_sha256=raw_rgb_sha256, scene_spec_sha256=scene_spec_sha256,
        scene_spec=scene_spec,
    )


def record_to_manifest(rec: CaseRecord, config: GeneratorConfig,
                       provenance: dict) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "case_id": rec.case_id,
        "mode": rec.mode,
        "scene_id": rec.scene_id,
        "seed": rec.seed,
        "data_origin": "procedural_synthetic_rpg",
        "real_capture": False,
        "production_gate": False,
        "resolution": [config.width, config.height],
        "duration_seconds": config.duration_seconds,
        "master_fps": config.master_fps,
        "files": rec.files,
        "fps_by_role": rec.fps_by_role,
        "frame_count_by_role": rec.frame_count_by_role,
        "defects": rec.defects,
        "generator": {"name": GENERATOR_NAME, "version": __version__,
                      "commit": provenance.get("generator_commit", "unknown")},
        "hashes": {
            "scene_spec_sha256": rec.scene_spec_sha256,
            "raw_rgb_sha256": rec.raw_rgb_sha256,
            "video_sha256": provenance.get("video_sha256", {}),
        },
    }


def write_case_manifest(case_dir: Path, manifest: dict) -> Path:
    p = case_dir / "manifest.json"
    p.write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                 encoding="utf-8")
    return p


def write_mode_manifest(mode_dir: Path, manifests: list[dict]) -> Path:
    p = mode_dir / "manifest.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        for m in manifests:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    return p
