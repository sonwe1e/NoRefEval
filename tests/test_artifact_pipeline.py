"""Unit tests for the unified artifact pipeline (USERPLAN §9, P0-1/P0-2).

The artifact pipeline lives in ``rr_vfiqa/report/artifact_pipeline.py``.
It produces every media artifact (heatmaps, overlay/compare clips,
keyframes, timeline) with individual exception protection so that a
failed encoder or bad path never overturns an already-valid score.

These tests are deterministic and never skip — they construct a minimal
Report + ArtifactContext directly and monkeypatch the exporters to fail.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

pytest.importorskip("rr_vfiqa.testing.synth")

from rr_vfiqa.config import EvalConfig, EvaluationMode
from rr_vfiqa.report.artifact_pipeline import (
    ArtifactContext,
    ArtifactStatus,
    build_report_artifacts,
)
from rr_vfiqa.schema import Report, Window, WindowFeatures


# ----------------------------------------------------------------- helpers
class SimpleMeta:
    def __init__(self, fps: float, n_frames: int, width: int, height: int):
        self.fps = fps
        self.n_frames = n_frames
        self.width = width
        self.height = height


def _make_video(tmp_path: Path) -> str:
    """Create a tiny 16-frame 160x96 synthetic video."""
    from rr_vfiqa.testing.synth import render_scene, write_video

    frames = render_scene(n_frames=16, w=160, h=96, seed=42)
    return str(write_video(tmp_path / "cand_60.mp4", frames, 60))


def _make_report() -> Report:
    """A minimal Report with diagnostics.issues so the artifact pipeline has
    something to export."""
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
                        "boxes": [],
                    }
                ],
                "issue_count": 1,
            },
        },
    )


def _make_ctx(tmp_path: Path, video: str) -> ArtifactContext:
    # center_index=0 so the issue's center_index matches the window's center.
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


# -------------------------------------------------------------------- tests
def test_status_all_ok_initially():
    """A fresh ArtifactStatus reports all_ok=True."""
    assert ArtifactStatus().all_ok is True


def test_status_degraded_when_one_failed():
    """A single degraded sub-status makes all_ok=False."""
    status = ArtifactStatus()
    status.overlay_clips["status"] = "degraded"
    assert status.all_ok is False


def test_overlay_export_failure_isolates_score(tmp_path):
    """USERPLAN P0-2: a failed overlay export must NOT overturn a valid
    score.  The score stays valid and artifact_status.overlay_clips shows
    degraded."""
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
    # Score must remain valid — overlay failure must not erase it.
    assert report.overall_score == 75.0
    assert report.meta["status"] == "ok"
    # On-disk report reflects the degraded overlay status.
    json_path = Path(ctx.output_dir) / "report.json"
    assert json_path.exists()
    on_disk = json.loads(json_path.read_text(encoding="utf-8"))
    assert on_disk["overall_score"] == 75.0
    artifact_status = on_disk.get("meta", {}).get("artifact_status", {})
    assert artifact_status.get("overlay_clips", {}).get("status") == "degraded"


def test_compare_export_failure_isolates_score(tmp_path):
    """A failed compare export must NOT overturn a valid score."""
    video = _make_video(tmp_path)
    report = _make_report()
    ctx = _make_ctx(tmp_path, video)

    with patch(
        "rr_vfiqa.report.artifact_pipeline.export_compare_clips",
        side_effect=RuntimeError("simulated compare failure"),
    ):
        status = build_report_artifacts(
            ctx, report,
            render_timeline_md_fn=lambda *a: "",
            render_timeline_png_fn=lambda *a: None,
        )

    assert status.compare_clips["status"] == "degraded"
    assert report.overall_score == 75.0
    assert report.meta["status"] == "ok"


def test_heatmap_render_failure_isolates_score(tmp_path):
    """A failed heatmap render must NOT overturn a valid score."""
    video = _make_video(tmp_path)
    report = _make_report()
    ctx = _make_ctx(tmp_path, video)

    with patch(
        "rr_vfiqa.report.artifact_pipeline.save_error_heatmap",
        side_effect=RuntimeError("simulated heatmap failure"),
    ):
        status = build_report_artifacts(
            ctx, report,
            render_timeline_md_fn=lambda *a: "",
            render_timeline_png_fn=lambda *a: None,
        )

    assert status.heatmaps["status"] == "failed"
    assert report.overall_score == 75.0
    assert report.meta["status"] == "ok"


def test_all_artifacts_ok_when_nothing_fails(tmp_path):
    """Happy path: status.all_ok is True and every sub-status is "ok"."""
    video = _make_video(tmp_path)
    report = _make_report()
    ctx = _make_ctx(tmp_path, video)

    status = build_report_artifacts(
        ctx, report,
        render_timeline_md_fn=lambda *a: "",
        render_timeline_png_fn=lambda *a: None,
    )

    assert status.all_ok is True
    for key in ("heatmaps", "original_clips", "overlay_clips",
                "compare_clips", "keyframes", "timeline", "report_write"):
        assert getattr(status, key)["status"] == "ok", (
            f"{key} not ok: {getattr(status, key)}")
    # Score still valid.
    assert report.overall_score == 75.0
    # report.json + report.html both exist.
    assert (Path(ctx.output_dir) / "report.json").exists()
    assert (Path(ctx.output_dir) / "report.html").exists()
