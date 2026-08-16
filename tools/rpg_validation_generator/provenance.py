"""Provenance records (PICPLAN §16).

Captures the exact software/commit/seed that produced a case so a result can
be reproduced or audited.  Git info is best-effort and never fabricated: if
the working tree is not a repo we record ``generator_commit = "unknown"``
and ``working_tree_dirty = null``.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from . import GENERATOR_NAME, __version__


def _git(repo: Path) -> tuple[str, bool | None]:
    try:
        commit = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10)
        if commit.returncode != 0:
            return "unknown", None
        dirty = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain"],
            capture_output=True, text=True, timeout=10)
        return commit.stdout.strip(), bool(dirty.stdout.strip())
    except Exception:
        return "unknown", None


def _ffmpeg_version() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        return "unknown"
    try:
        out = subprocess.run([exe, "-version"], capture_output=True, text=True,
                             timeout=10)
        return out.stdout.splitlines()[0] if out.stdout else "unknown"
    except Exception:
        return "unknown"


def _pyav_version() -> str:
    try:
        import av
        return av.__version__
    except Exception:
        return "unknown"


def build_provenance(scene_seed: int, scene_spec_sha256: str,
                     raw_rgb_sha256: dict, video_sha256: dict,
                     generation_command: str, repo: Path | None = None) -> dict:
    repo = repo or Path.cwd()
    commit, dirty = _git(repo)
    return {
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "opencv_version": cv2.__version__,
        "pyav_version": _pyav_version(),
        "ffmpeg_version": _ffmpeg_version(),
        "generator_name": GENERATOR_NAME,
        "generator_version": __version__,
        "generator_commit": commit,
        "working_tree_dirty": dirty,
        "scene_seed": scene_seed,
        "scene_spec_sha256": scene_spec_sha256,
        "raw_rgb_sha256": raw_rgb_sha256,
        "video_sha256": video_sha256,
        "generation_command": generation_command,
    }


def write_provenance(case_dir: Path, record: dict) -> Path:
    path = case_dir / "provenance.json"
    path.write_text(json.dumps(record, indent=2, sort_keys=True),
                    encoding="utf-8")
    return path
