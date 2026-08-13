"""Top-level generation orchestration (PICPLAN §5, §21).

Renders each scene's master exactly once and reuses it for the three modes,
keeping RAM bounded by deleting the master memmaps between scenes.
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

from .case_specs import CASES
from .config import GeneratorConfig
from .manifests import record_to_manifest, write_case_manifest, write_mode_manifest
from .schema import MODES, MODE_DIR
from .scenes import make_scene
from .timeline import render_master

from .builder import build_case


def _modes_arg(mode: str) -> list[str]:
    return list(MODES) if mode == "all" else [mode]


def generate(output_root: str | Path, config: GeneratorConfig, mode: str = "all",
             overwrite: bool = False, workdir: str | Path | None = None,
             generation_command: str | None = None) -> dict:
    output_root = Path(output_root)
    modes = _modes_arg(mode)
    generation_command = generation_command or " ".join(sys.argv)

    if overwrite and output_root.exists():
        for m in modes:
            d = output_root / MODE_DIR[m]
            if d.exists():
                shutil.rmtree(d)

    work_root = Path(workdir) if workdir else Path(
        tempfile.mkdtemp(prefix="rpgvg_"))
    work_root.mkdir(parents=True, exist_ok=True)

    manifests_by_mode: dict[str, list[dict]] = {m: [] for m in modes}
    summary = {m: 0 for m in modes}
    provenance_all: list[dict] = []

    try:
        for scene_index in range(5):
            scene = make_scene(scene_index, config)
            master = render_master(scene, config, work_root / f"scene{scene_index}")
            scene_specs_for_index = [s for s in CASES
                                     if s.scene_index == scene_index
                                     and s.mode in modes]
            for spec in scene_specs_for_index:
                mode_dir = output_root / MODE_DIR[spec.mode]
                mode_dir.mkdir(parents=True, exist_ok=True)
                case_seed = config.case_seed({"nr": 1, "endpoint": 2, "fr": 3}[spec.mode],
                                             spec.case_index)
                record, provenance, _ = build_case(
                    spec, master, config, mode_dir, case_seed, generation_command)
                manifest = record_to_manifest(record, config, provenance)
                case_dir = mode_dir / spec.case_dir_name
                write_case_manifest(case_dir, manifest)
                manifests_by_mode[spec.mode].append(manifest)
                provenance_all.append(provenance)
                summary[spec.mode] += 1
            # free this scene's master before rendering the next
            del master, scene
    finally:
        if workdir is None:
            shutil.rmtree(work_root, ignore_errors=True)

    for m in modes:
        mode_dir = output_root / MODE_DIR[m]
        if manifests_by_mode[m]:
            manifests_by_mode[m].sort(key=lambda d: d["case_id"])
            write_mode_manifest(mode_dir, manifests_by_mode[m])

    return {"modes": modes, "cases_per_mode": summary,
            "output_root": str(output_root)}
