"""Session-scoped fixtures for the metamorphic regression corpus (USERPLAN P4).

Generates a small RPG validation set (draft resolution, 2 s) once per test
session and provides the per-case file paths + defect ground truth to the
tests.  The generator is the single source of truth for defect intervals and
parameters (see ``tools/rpg_validation_generator/case_specs.py``).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

os.environ.setdefault("RR_VFIQA_TEST_FLOW", "farneback")


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


@pytest.fixture(scope="session")
def rpg_root(tmp_path_factory) -> Path:
    """Generate the RPG validation data once per session.

    Uses a reduced draft configuration (320x180, 4s) for speed.  The master
    FPS stays at 120 so the endpoint source/candidate FPS ratio (60/120) and
    the even-frame contract hold; only the spatial resolution is trimmed.
    The duration is 4s so that every defect window in case_specs.py (whose
    latest end_time is 3.10s) fits fully inside the video — a 2s video would
    truncate defects and leave no post-defect recovery interval (USERPLAN §2).
    """
    from tools.rpg_validation_generator.config import GeneratorConfig
    from tools.rpg_validation_generator.orchestration import generate
    from tools.rpg_validation_generator.case_specs import CASES

    duration_seconds = max(c.end_time for c in CASES) + 0.5
    assert duration_seconds <= 4.5, (
        f"defect windows require {duration_seconds}s; review case_specs.py")

    cfg = GeneratorConfig(
        width=320, height=180, duration_seconds=duration_seconds,
        master_fps=120, seed=20260729, world_width=400, world_height=250)
    root = tmp_path_factory.mktemp("rpg_corpus")
    generate(str(root), cfg, mode="all")
    return root


@pytest.fixture(scope="session")
def rpg_cases(rpg_root: Path) -> list[dict]:
    """Load every per-case manifest that has a candidate video.

    The "clean" baseline depends on the mode:

    * NR / Endpoint: ``oracle`` role (clean version of the same content).
    * FR: ``source`` role — the clean source that the defective ``candidate``
      should be compared against.  FR inspect runs candidate-vs-reference; here
      the reference is the same clean source (defects are injected only into
      the candidate).
    """
    cases = []
    for mode_dir in sorted(rpg_root.iterdir()):
        if not mode_dir.is_dir():
            continue
        for case_dir in sorted(mode_dir.iterdir()):
            if not case_dir.is_dir():
                continue
            manifest_path = case_dir / "manifest.json"
            if not manifest_path.exists():
                continue
            m = json.loads(manifest_path.read_text(encoding="utf-8"))
            files = m.get("files", {})
            if "candidate" not in files:
                continue
            mode = m["mode"]
            # Determine the clean baseline + the reference to pass to inspect.
            oracle_path = str(case_dir / files["oracle"]) \
                if "oracle" in files else None
            if oracle_path is None and "source" in files:
                oracle_path = str(case_dir / files["source"])
            # For FR, the reference passed to inspect is the clean source.
            reference_for_inspect = None
            if mode == "fr" and "source" in files:
                reference_for_inspect = str(case_dir / files["source"])
            cases.append({
                "case_id": m["case_id"],
                "mode": mode,
                "case_dir": case_dir,
                "candidate": str(case_dir / files["candidate"]),
                "oracle": oracle_path,
                "reference_for_inspect": reference_for_inspect,
                "defects": m.get("defects", []),
                "fps_by_role": m.get("fps_by_role", {}),
            })
    return cases
