"""Unit tests for the unified artifact pipeline (USERPLAN §9, P0-2).

Failure isolation: every artifact is produced inside its own ``try/except`` in
``build_report_artifacts``, so a failed encoder or a bad write must never
overturn an already-valid score -- only the metric pipeline can do that.  These
tests verify that contract by monkeypatching each exporter in turn and asserting
that the overall score stays valid while only the targeted artifact is degraded.

The tests are self-contained: they build a minimal ``Report`` +
``ArtifactContext`` directly (including a real 16-frame 160x96 synthetic video
so the clip exporters run for real) and never depend on the full evaluate path.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from rr_vfiqa.config import EvaluationMode
from rr_vfiqa.report.artifact_pipeline import (
    ArtifactContext,
    ArtifactStatus,
    build_report_artifacts,
)
from rr_vfiqa.schema import Report, Window, WindowFeatures

pytest.importorskip("rr_vfiqa.testing.synth")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
class SimpleMeta:
    """VideoMeta-like stand-in carrying only what the pipeline reads (.fps)."""

    def __init__(self, fps: float, n_frames: int):
        self.fps = fps
        self.n_frames = n_frames


def _make_video(tmp_path: Path) -> str:
    """Create a real 16-frame 160x96 synthetic video at 30 fps."""
    from rr_vfiqa.testing.synth import render_scene, write_video

    frames = render_scene(n_frames=16, w=160, h=96, seed=42)
    return str(write_video(tmp_path / "candidate.mp4", frames, fps=30.0))


def _make_issue() -> dict:
    """One located issue carrying a map name + a matching center_index."""
    return {
        "issue_type": "duplicate_freeze",
        "title": "freeze",
        "severity": 0.8,
        "severity_band": "high",
        "severity_label": "obvious",
        "confidence": 0.7,
        "start_time": 0.05,
        "end_time": 0.2,
        "track": "Temporal",
        "maps": ["duplicate_frame_indicator"],
        "center_index": 0,
        "boxes": [],
    }


def _make_report(issue: dict) -> Report:
    return Report(
        overall_score=75.0,
        confidence=0.6,
        scores={"temporal_stability": 70.0, "cadence_integrity": 95.0},
        event_ambiguity=0.1,
        worst_windows=[],
        features={"nr_native_duplicate_fraction": 0.2},
        meta={
            "mode": "no-reference",
            "status": "ok",
            "score_schema": "nr-stability-risk-v4",
            "diagnostics": {"issues": [issue], "issue_count": 1},
        },
    )


def _make_ctx(tmp_path: Path, video: str, issue: dict) -> ArtifactContext:
    # center_index=0 matches the window's center so the heatmap renderer finds
    # the error map and actually invokes save_error_heatmap.
    wf = WindowFeatures(
        Window(0, np.arange(5), 0),
        scalars={"nr_native_duplicate_fraction": 0.2},
        error_maps={"duplicate_frame_indicator": np.zeros((96, 160), np.float32)},
    )
    return ArtifactContext(
        mode=EvaluationMode.NO_REFERENCE,
        candidate_video=video,
        reference_video=None,
        windows=[wf],
        issues=[issue],
        output_dir=tmp_path / "out",
        candidate_meta=SimpleMeta(fps=30.0, n_frames=16),
        export_clips=True,
    )


def _timeline_md(*_args) -> str:
    return "# timeline\n"


def _timeline_png(*_args) -> None:
    pass


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_overlay_export_failure_isolates_score(tmp_path: Path) -> None:
    """A failed overlay export must NOT overturn a valid score."""
    video = _make_video(tmp_path)
    issue = _make_issue()
    report = _make_report(issue)
    ctx = _make_ctx(tmp_path, video, issue)

    with patch(
        "rr_vfiqa.report.artifact_pipeline.export_overlay_clips",
        side_effect=RuntimeError("simulated overlay failure"),
    ):
        status = build_report_artifacts(
            ctx, report,
            render_timeline_md_fn=_timeline_md,
            render_timeline_png_fn=_timeline_png,
        )

    # Score stays valid; the report meta is not marked as failed.
    assert report.overall_score is not None
    assert report.meta.get("status") != "failed"

    # Only the overlay artifact is degraded.
    assert status.overlay_clips["status"] == "degraded"

    # report.json on disk persists the degraded overlay status.
    on_disk = json.loads((ctx.output_dir / "report.json").read_text(encoding="utf-8"))
    assert on_disk["overall_score"] == 75.0
    artifact_status = on_disk.get("meta", {}).get("artifact_status", {})
    assert artifact_status.get("overlay_clips", {}).get("status") == "degraded"


def test_compare_export_failure_isolates_score(tmp_path: Path) -> None:
    """A failed compare export must NOT overturn a valid score."""
    video = _make_video(tmp_path)
    issue = _make_issue()
    report = _make_report(issue)
    ctx = _make_ctx(tmp_path, video, issue)

    with patch(
        "rr_vfiqa.report.artifact_pipeline.export_compare_clips",
        side_effect=RuntimeError("simulated compare failure"),
    ):
        status = build_report_artifacts(
            ctx, report,
            render_timeline_md_fn=_timeline_md,
            render_timeline_png_fn=_timeline_png,
        )

    assert report.overall_score is not None
    assert report.meta.get("status") != "failed"
    assert status.compare_clips["status"] == "degraded"

    on_disk = json.loads((ctx.output_dir / "report.json").read_text(encoding="utf-8"))
    artifact_status = on_disk.get("meta", {}).get("artifact_status", {})
    assert artifact_status.get("compare_clips", {}).get("status") == "degraded"


def test_heatmap_render_failure_isolates_score(tmp_path: Path) -> None:
    """A failed heatmap render must NOT overturn a valid score."""
    video = _make_video(tmp_path)
    issue = _make_issue()
    report = _make_report(issue)
    ctx = _make_ctx(tmp_path, video, issue)

    with patch(
        "rr_vfiqa.report.artifact_pipeline.save_error_heatmap",
        side_effect=RuntimeError("simulated heatmap failure"),
    ):
        status = build_report_artifacts(
            ctx, report,
            render_timeline_md_fn=_timeline_md,
            render_timeline_png_fn=_timeline_png,
        )

    assert report.overall_score is not None
    assert status.heatmaps["status"] == "degraded"


def test_atomic_write_failure_marks_degraded(tmp_path: Path) -> None:
    """A failed atomic rename must mark report_write degraded while the base
    report.json (written before the atomic rewrite) persists on disk."""
    report = _make_report(_make_issue())
    ctx = ArtifactContext(
        mode=EvaluationMode.NO_REFERENCE,
        candidate_video="",
        reference_video=None,
        windows=[],
        issues=[],
        output_dir=tmp_path / "out",
        candidate_meta=SimpleMeta(fps=30.0, n_frames=0),
        export_clips=False,
    )

    with patch(
        "rr_vfiqa.report.artifact_pipeline.os.replace",
        side_effect=OSError("atomic rename blocked"),
    ):
        status = build_report_artifacts(
            ctx, report,
            render_timeline_md_fn=_timeline_md,
            render_timeline_png_fn=_timeline_png,
        )

    assert status.report_write["status"] == "degraded"
    # The pre-atomic base report.json still exists.
    assert (ctx.output_dir / "report.json").exists()


def test_all_artifacts_ok_when_nothing_fails(tmp_path: Path) -> None:
    """Happy path: status.all_ok is True and every sub-status is "ok"."""
    video = _make_video(tmp_path)
    issue = _make_issue()
    report = _make_report(issue)
    ctx = _make_ctx(tmp_path, video, issue)

    status = build_report_artifacts(
        ctx, report,
        render_timeline_md_fn=_timeline_md,
        render_timeline_png_fn=_timeline_png,
    )

    assert status.all_ok is True
    for key in (
        "heatmaps", "original_clips", "overlay_clips", "compare_clips",
        "keyframes", "timeline", "report_write",
    ):
        assert getattr(status, key)["status"] == "ok", (
            f"{key} not ok: {getattr(status, key)}")
    assert report.overall_score is not None
    assert (ctx.output_dir / "report.json").exists()
    assert (ctx.output_dir / "report.html").exists()
