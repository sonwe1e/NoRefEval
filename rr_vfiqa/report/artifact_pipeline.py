"""Unified artifact pipeline (USERPLAN P0-1 / P0-2).

NR, Endpoint and FR all need the same post-evaluation media pipeline:

    resolve error-map ndarray
    → render heatmap
    → export original
    → export overlay
    → export compare
    → export thumbnail
    → update issue artifact manifest
    → atomic write JSON/HTML

Historically each mode maintained its own copy of this logic, which meant
Endpoint lagged behind NR/FR (no compare clip, no keyframe, no clip_paths
writeback, no error-map boxes).  :func:`build_report_artifacts` is the single
entry point all three modes now call.

Failure isolation (P0-2): every artifact is produced inside its own
``try/except`` and recorded in ``artifact_status``.  A failed H.264 encoder
or a bad output path must never overturn an already-valid score — only the
metric pipeline can do that.  The report is first written without media
links, then atomically replaced once all artifacts exist.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..config import EvaluationMode
from ..schema import Report, WindowFeatures, WorstWindow
from .compare_exporter import export_compare_clips
from .overlay_exporter import export_overlay_clips
from .badcase_exporter import export_badcase_clips
from .heatmap_renderer import save_error_heatmap
from .json_report import write_json_report
from .html_report import write_html_report


@dataclass
class ArtifactContext:
    """Everything the artifact pipeline needs to produce a report."""

    mode: EvaluationMode
    candidate_video: str
    reference_video: str | None
    windows: list[WindowFeatures]
    issues: list[dict]
    output_dir: Path
    candidate_meta: Any                          # VideoMeta (fps, n_frames)
    worst_windows: list[WorstWindow] | None = None
    export_clips: bool = True
    # Per-issue error maps keyed by issue index → (name, ndarray, map_h, map_w).
    error_maps: dict[int, np.ndarray] = field(default_factory=dict)
    map_dimensions: dict[int, tuple[int, int]] = field(default_factory=dict)


@dataclass
class ArtifactStatus:
    """Per-artifact success/failure log (USERPLAN P0-2)."""

    heatmaps: dict[str, Any] = field(default_factory=lambda: {"status": "ok", "errors": []})
    original_clips: dict[str, Any] = field(default_factory=lambda: {"status": "ok", "errors": []})
    overlay_clips: dict[str, Any] = field(default_factory=lambda: {"status": "ok", "errors": []})
    compare_clips: dict[str, Any] = field(default_factory=lambda: {"status": "ok", "errors": []})
    keyframes: dict[str, Any] = field(default_factory=lambda: {"status": "ok", "errors": []})
    timeline: dict[str, Any] = field(default_factory=lambda: {"status": "ok", "errors": []})
    report_write: dict[str, Any] = field(default_factory=lambda: {"status": "ok", "errors": []})

    def to_dict(self) -> dict[str, Any]:
        return {
            "heatmaps": dict(self.heatmaps),
            "original_clips": dict(self.original_clips),
            "overlay_clips": dict(self.overlay_clips),
            "compare_clips": dict(self.compare_clips),
            "keyframes": dict(self.keyframes),
            "timeline": dict(self.timeline),
            "report_write": dict(self.report_write),
        }

    @property
    def all_ok(self) -> bool:
        return all(
            getattr(self, k)["status"] == "ok"
            for k in ("heatmaps", "original_clips", "overlay_clips",
                      "compare_clips", "keyframes", "timeline", "report_write")
        )


def build_report_artifacts(
    ctx: ArtifactContext,
    report: Report,
    *,
    render_timeline_md_fn,
    render_timeline_png_fn,
) -> ArtifactStatus:
    """Produce every media artifact and write the final report (USERPLAN P0-1).

    The report is first written without media links, then each artifact is
    generated with individual exception protection (P0-2).  The final report
    is atomically replaced via temp-file + rename.
    """
    status = ArtifactStatus()
    out = Path(ctx.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # --- 1. Render heatmap PNGs for issues that have a map -----------------
    try:
        _render_issue_heatmaps(ctx, report)
    except Exception as exc:  # noqa: BLE001
        status.heatmaps["status"] = "degraded"
        status.heatmaps["errors"].append(repr(exc))
        if report.meta is not None:
            report.meta["heatmap_render_error"] = repr(exc)

    # --- 2. Write the base report (no media links yet) --------------------
    json_path = out / "report.json"
    html_path = out / "report.html"
    write_json_report(report, json_path)
    write_html_report(report, html_path)

    # --- 3. Original badcase clips ----------------------------------------
    if ctx.export_clips and ctx.worst_windows:
        try:
            export_badcase_clips(
                ctx.candidate_video, ctx.worst_windows,
                out / "badcases", out_fps=ctx.candidate_meta.fps)
        except Exception as exc:  # noqa: BLE001
            status.original_clips["status"] = "degraded"
            status.original_clips["errors"].append(repr(exc))

    # --- 4. Per-issue companion media (overlay / compare / thumbnail) ----
    if ctx.export_clips and ctx.issues:
        overlay_paths = _export_overlay(ctx, status)
        compare_paths = _export_compare(ctx, status)
        thumb_paths = _export_keyframes(ctx, status)

        # Write the paths back into the issues so the HTML report links them.
        for idx, issue in enumerate(ctx.issues):
            clips: dict[str, str] = {}
            op = overlay_paths[idx] if idx < len(overlay_paths) else None
            cp = compare_paths[idx] if idx < len(compare_paths) else None
            if op is not None and op.exists():
                clips["overlay"] = f"badcases/{op.name}"
            if cp is not None and cp.exists():
                clips["compare"] = f"badcases/{cp.name}"
            if clips:
                issue["clip_paths"] = clips
            tp = thumb_paths[idx] if idx < len(thumb_paths) else None
            if tp is not None:
                issue["thumbnail"] = f"badcases/{tp.name}"
    else:
        # No issues — nothing to export for overlay/compare/keyframes.
        status.overlay_clips["status"] = "skipped"
        status.compare_clips["status"] = "skipped"
        status.keyframes["status"] = "skipped"

    # --- 5. Timeline ------------------------------------------------------
    try:
        (out / "timeline.md").write_text(
            render_timeline_md_fn(ctx.windows, ctx.candidate_meta, report),
            encoding="utf-8")
        render_timeline_png_fn(ctx.windows, ctx.candidate_meta, report,
                               out / "timeline.png")
    except Exception as exc:  # noqa: BLE001
        status.timeline["status"] = "degraded"
        status.timeline["errors"].append(repr(exc))
        if report.meta is not None:
            report.meta["timeline_error"] = repr(exc)

    # --- 6. Attach artifact status to report meta -----------------------
    # Set BEFORE the atomic write so the on-disk report.json persists it.
    if report.meta is not None:
        report.meta["artifact_status"] = status.to_dict()

    # --- 7. Atomic re-write of the report with media links ---------------
    try:
        _atomic_write_report(report, json_path, html_path)
    except Exception as exc:  # noqa: BLE001
        status.report_write["status"] = "degraded"
        status.report_write["errors"].append(repr(exc))
        # USERPLAN §10: re-sync the (now degraded) status onto the report
        # meta BEFORE the best-effort write, so the on-disk JSON carries
        # report_write=degraded.  Step 6 snapshotted status.to_dict() before
        # the atomic write, so we must refresh it here.
        if report.meta is not None:
            report.meta["artifact_status"] = status.to_dict()
        # Best-effort plain overwrite so the on-disk JSON carries the
        # degraded status + whatever media links were produced.
        try:
            write_json_report(report, json_path)
        except Exception:  # noqa: BLE001
            pass  # base report is still readable; scores are valid

    return status


def _render_issue_heatmaps(ctx: ArtifactContext, report: Report) -> None:
    """Write ``heatmaps/issue_NNN_<map>.png`` for issues carrying a map name."""
    diag = ((report.meta or {}).get("diagnostics") or {})
    issues = diag.get("issues") or []
    if not issues or not ctx.windows:
        return
    by_center = {int(wf.window.center): wf for wf in ctx.windows}
    heat_dir = ctx.output_dir / "heatmaps"
    for i, issue in enumerate(issues):
        if not issue.get("maps"):
            continue
        wf = by_center.get(int(issue.get("center_index", -1)))
        if wf is None:
            continue
        original_keys = list(issue["maps"])
        rel_maps: list[str] = []
        for name in original_keys:
            arr = wf.error_maps.get(name) if wf else None
            if arr is None:
                continue
            label = f"{issue.get('title', name)}  #{i:03d}"
            png = save_error_heatmap(arr, heat_dir / f"issue_{i:03d}_{name}.png",
                               label=label)
            rel_maps.append(f"heatmaps/{png.name}")
        issue["map_keys"] = original_keys
        issue["maps"] = rel_maps


def _export_overlay(ctx: ArtifactContext, status: ArtifactStatus) -> list[Path | None]:
    """One overlay clip per issue (capped at 12) — batch call."""
    issues = ctx.issues[:12]
    if not issues:
        return []
    try:
        # Build error_maps dict keyed by LOCAL index for this batch
        error_maps = {}
        for i, issue in enumerate(issues):
            if i in ctx.error_maps:
                error_maps[i] = ctx.error_maps[i]
        results = export_overlay_clips(
            ctx.candidate_video, issues, ctx.output_dir / "badcases",
            out_fps=ctx.candidate_meta.fps,
            error_maps=error_maps if error_maps else None)
        # Pad to len(issues) if fewer results
        while len(results) < len(issues):
            results.append(None)
        return results[:len(issues)]
    except Exception as exc:  # noqa: BLE001
        status.overlay_clips["status"] = "degraded"
        status.overlay_clips["errors"].append(repr(exc))
        return [None] * len(issues)


def _export_compare(ctx: ArtifactContext, status: ArtifactStatus) -> list[Path | None]:
    """One compare clip per issue (capped at 12) — batch call."""
    issues = ctx.issues[:12]
    if not issues:
        return []
    try:
        # Build error_maps dict keyed by LOCAL index for this batch
        error_maps = {}
        for i, issue in enumerate(issues):
            if i in ctx.error_maps:
                error_maps[i] = ctx.error_maps[i]
        # Panel layout is driven by the explicit evaluation mode (P0-5): the
        # exporter can no longer guess from ``reference_video`` presence,
        # otherwise Endpoint (which has a reference) is mislabeled as FR.
        mode = {
            EvaluationMode.NO_REFERENCE: "NR",
            EvaluationMode.ENDPOINT_2X: "endpoint-2x",
            EvaluationMode.FULL_REFERENCE: "full-reference",
        }.get(ctx.mode, "NR")
        results = export_compare_clips(
            issues, ctx.output_dir / "badcases",
            out_fps=30.0,
            candidate_video=ctx.candidate_video,
            reference_video=ctx.reference_video,
            error_maps=error_maps if error_maps else None,
            mode=mode)
        # Pad to len(issues) if fewer results
        while len(results) < len(issues):
            results.append(None)
        return results[:len(issues)]
    except Exception as exc:  # noqa: BLE001
        status.compare_clips["status"] = "degraded"
        status.compare_clips["errors"].append(repr(exc))
        return [None] * len(issues)


def _export_keyframes(ctx: ArtifactContext, status: ArtifactStatus) -> list[Path | None]:
    """One keyframe per issue with defect boxes drawn (USERPLAN §10)."""
    paths: list[Path | None] = []
    for n, issue in enumerate(ctx.issues[:12]):
        try:
            t_center = 0.5 * (float(issue.get("start_time", 0)) +
                               float(issue.get("end_time", 0)))
            out_path = ctx.output_dir / "badcases" / f"issue_{n:03d}_keyframe.png"
            _render_single_keyframe(ctx, issue, t_center, out_path)
            paths.append(out_path if out_path.exists() else None)
        except Exception as exc:  # noqa: BLE001
            status.keyframes["status"] = "degraded"
            status.keyframes["errors"].append(repr(exc))
            paths.append(None)
    return paths


def _render_single_keyframe(ctx: ArtifactContext, issue: dict,
                            t_center: float, out_path: Path) -> None:
    """Extract one frame at ``t_center`` and draw defect boxes on it."""
    import cv2
    import av
    with av.open(ctx.candidate_video) as inp:
        vs = inp.streams.video[0]
        inp.seek(int(t_center * av.time_base), any_frame=False, backward=True)
        frame = None
        for f in inp.decode(vs):
            t = float(f.pts * vs.time_base) if f.pts is not None else 0.0
            if t >= t_center - 1e-3:
                frame = f.to_ndarray(format="bgr24")
                break
    if frame is None:
        return
    annotated = frame.copy()
    map_w = issue.get("_map_w")
    map_h = issue.get("_map_h")
    if map_w and map_h:
        scale_x = annotated.shape[1] / map_w
        scale_y = annotated.shape[0] / map_h
    else:
        scale_x = scale_y = 1.0
    for box in (issue.get("boxes") or []):
        if len(box) == 4:
            x, y, w, h = box
            x, w = int(x * scale_x), int(w * scale_x)
            y, h = int(y * scale_y), int(h * scale_y)
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 0, 255), 2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), annotated)


def _atomic_write_report(report: Report, json_path: Path, html_path: Path) -> None:
    """Write report to temp files then atomically rename (USERPLAN P0-2).

    On failure, the base report (written in step 2 of ``build_report_artifacts``)
    already exists on disk.  We log the error and re-raise so the caller can
    record ``report_write=degraded`` and do a best-effort plain overwrite
    (USERPLAN §10).
    """
    try:
        fd, tmp_json = tempfile.mkstemp(
            prefix="report.json.", dir=json_path.parent)
        os.close(fd)
        write_json_report(report, Path(tmp_json))
        os.replace(tmp_json, json_path)

        fd, tmp_html = tempfile.mkstemp(
            prefix="report.html.", dir=html_path.parent)
        os.close(fd)
        write_html_report(report, Path(tmp_html))
        os.replace(tmp_html, html_path)
    except Exception as exc:  # noqa: BLE001
        # Atomic write failed but the base report already exists — log and
        # re-raise so build_report_artifacts records report_write=degraded
        # and does a best-effort plain overwrite.
        if report.meta is not None:
            report.meta["atomic_write_error"] = repr(exc)
        raise
