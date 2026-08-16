"""Edge-case tests for Artifact Status (USERPLAN §10).

Verifies that the artifact pipeline degrades gracefully when individual
artifacts fail, and that the on-disk report always reflects the true state
even when the atomic final-write itself fails.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

pytest.importorskip("rr_vfiqa.testing.synth")

from rr_vfiqa.config import EvaluationMode
from rr_vfiqa.report.artifact_pipeline import (
    ArtifactContext,
    ArtifactStatus,
    build_report_artifacts,
)
from rr_vfiqa.schema import Report, Window, WindowFeatures


def _make_video(tmp_path: Path) -> str:
    """Create a tiny 16-frame 160x96 synthetic video."""
    from rr_vfiqa.testing.synth import render_scene, write_video

    frames = render_scene(n_frames=16, w=160, h=96, seed=42)
    return str(write_video(tmp_path / "cand_60.mp4", frames, 60))


def _make_report() -> Report:
    """A minimal Report with diagnostics.issues so the artifact pipeline has
    something to export."""
    meta = {
        "mode": "no-reference",
        "status": "ok",
        "score_schema": "nr-stability-risk-v4",
        "diagnostics": {
            "issues": [
                {
                    "issue_type": "duplicate_freeze",
                    "title": "freeze",
                    "severity": 0.8,
                    "severity_band": "high",
                    "severity_label": "明显问题",
                    "confidence": 0.7,
                    "start_time": 0.2,
                    "end_time": 0.5,
                    "track": "Temporal",
                    "maps": ["duplicate_frame_indicator"],
                    "center_index": 0,
                }
            ],
            "issue_count": 1,
        },
    }
    return Report(
        overall_score=75.0,
        confidence=0.6,
        scores={"temporal_stability": 70.0, "cadence_integrity": 95.0},
        event_ambiguity=0.1,
        worst_windows=[],
        features={"nr_native_duplicate_fraction": 0.2},
        meta=meta,
    )


def _make_ctx(tmp_path: Path, video: str) -> ArtifactContext:
    wf = WindowFeatures(
        Window(0, np.arange(5), 0),
        scalars={"nr_native_duplicate_fraction": 0.2},
        error_maps={"duplicate_frame_indicator": np.zeros((96, 160), np.float32)},
    )
    out = tmp_path / "out"
    out.mkdir(parents=True, exist_ok=True)
    return ArtifactContext(
        mode=EvaluationMode.NO_REFERENCE,
        candidate_video=video,
        reference_video=None,
        windows=[wf],
        issues=[
            {
                "issue_type": "duplicate_freeze",
                "title": "freeze",
                "start_time": 0.2,
                "end_time": 0.5,
                "severity_band": "high",
                "severity_label": "明显问题",
                "maps": ["duplicate_frame_indicator"],
                "center_index": 0,
            }
        ],
        output_dir=out,
        candidate_meta=SimpleMeta(fps=60.0, n_frames=16, width=160, height=96),
        export_clips=True,
    )


class SimpleMeta:
    def __init__(self, fps: float, n_frames: int, width: int, height: int):
        self.fps = fps
        self.n_frames = n_frames
        self.width = width
        self.height = height


def test_status_all_ok_initially():
    """A fresh ArtifactStatus reports all_ok=True."""
    status = ArtifactStatus()
    assert status.all_ok is True


def test_status_degraded_when_one_failed():
    """A single degraded sub-status makes all_ok=False."""
    status = ArtifactStatus()
    status.overlay_clips["status"] = "degraded"
    assert status.all_ok is False


def test_atomic_write_failure_persists_degraded_status(tmp_path):
    """USERPLAN §10: when os.replace fails, the on-disk report must still
    carry report_write=degraded via a best-effort plain overwrite."""
    video = _make_video(tmp_path)
    report = _make_report()
    ctx = _make_ctx(tmp_path, video)

    # First call build_report_artifacts with a working atomic write so the
    # base report.json exists.  Then simulate an atomic-write failure by
    # patching os.replace to raise on the *second* invocation (the atomic
    # final-write) but not the first (the base write in step 2).
    #
    # Simpler: patch os.replace to always raise — the base write in step 2
    # uses write_json_report directly (not os.replace), so only the atomic
    # final-write in step 7 fails.
    with patch("rr_vfiqa.report.artifact_pipeline.os.replace",
               side_effect=OSError("simulated atomic-write failure")):
        status = build_report_artifacts(
            ctx, report,
            render_timeline_md_fn=lambda *a: "",
            render_timeline_png_fn=lambda *a: None,
        )

    assert status.report_write["status"] == "degraded"
    # The score must remain valid — atomic write failure must not erase it.
    assert report.overall_score == 75.0

    # The on-disk report.json must exist and carry the degraded status.
    json_path = Path(ctx.output_dir) / "report.json"
    assert json_path.exists(), "report.json missing after atomic-write failure"
    on_disk = json.loads(json_path.read_text(encoding="utf-8"))
    assert on_disk["overall_score"] == 75.0
    artifact_status = on_disk.get("meta", {}).get("artifact_status", {})
    assert artifact_status.get("report_write", {}).get("status") == "degraded", (
        f"on-disk report does not reflect degraded report_write: "
        f"{artifact_status}")


def test_overlay_failure_does_not_affect_score(tmp_path):
    """A failed overlay export must not overturn a valid score."""
    video = _make_video(tmp_path)
    report = _make_report()
    ctx = _make_ctx(tmp_path, video)

    with patch(
        "rr_vfiqa.report.artifact_pipeline.export_overlay_clips",
        side_effect=RuntimeError("simulated overlay failure"),
    ):
        status = build_report_artifacts(
            ctx, report,
            render_timeline_md_fn=lambda *a: "",
            render_timeline_png_fn=lambda *a: None,
        )

    assert status.overlay_clips["status"] == "degraded"
    assert report.overall_score == 75.0
    json_path = Path(ctx.output_dir) / "report.json"
    assert json_path.exists()
    on_disk = json.loads(json_path.read_text(encoding="utf-8"))
    assert on_disk["overall_score"] == 75.0
