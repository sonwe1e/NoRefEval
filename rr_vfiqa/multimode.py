"""Public multi-mode dispatch and the NR/FR execution pipelines."""

from __future__ import annotations

import collections
import time
from pathlib import Path
from typing import Callable

import numpy as np

from .config import EvalConfig, EvaluationMode, parse_mode
from .fusion.mode_score_schemas import (
    SCHEMA_IDS,
    compute_mode_confidence,
    compute_mode_scores,
    window_category_errors,
)
from .io.full_reference_alignment import build_full_reference_alignment
from .io.video_reader import VideoReader
from .metrics import full_reference, no_reference
from .metrics.window_flows import WindowFlows
from .models.vqa_backend import get_vqa_backend
from .motion.flow_estimator import get_flow_backend
from .pipeline import evaluate_endpoint_reference
from .report import (
    export_badcase_clips,
    render_timeline_md,
    render_timeline_png,
    write_json_report,
)
from .sampling.cheap_scan import scan_candidate, scene_cuts_from_scan
from .sampling.time_window_selector import select_time_windows
from .sampling.window_selector import window_times
from .schema import Report, WindowFeatures, WorstWindow

ProgressFn = Callable[[str], None]


def _feature_summary(windows: list[WindowFeatures]) -> dict[str, float]:
    output: dict[str, float] = {}
    keys = set().union(*(set(w.scalars) for w in windows)) if windows else set()
    for key in sorted(keys):
        if key.startswith("_"):
            continue
        values = [
            float(w.scalars[key])
            for w in windows
            if key in w.scalars and np.isfinite(w.scalars[key])
        ]
        if values:
            output[key] = round(float(np.median(values)), 5)
    return output


def _worst_windows(
    mode: EvaluationMode,
    windows: list[WindowFeatures],
    meta,
    confidence: float,
    temporal_nms_seconds: float,
) -> list[WorstWindow]:
    labels = {
        EvaluationMode.NO_REFERENCE: {
            "temporal": "temporal_instability",
            "motion": "motion_trajectory_risk",
            "phase": "phase_alternation",
            "ui_structure": "ui_text_instability",
            "technical": "technical_degradation",
        },
        EvaluationMode.FULL_REFERENCE: {
            "spatial": "spatial_reference_error",
            "structure": "structure_reference_error",
            "temporal": "temporal_reference_error",
            "motion": "motion_reference_error",
        },
    }[mode]
    candidates: list[WorstWindow] = []
    for wf in windows:
        errors = window_category_errors(mode, wf)
        firing = sorted(
            ((category, error) for category, error in errors.items() if error >= 0.35),
            key=lambda item: -item[1],
        )[:3]
        if not firing:
            continue
        t0, t1 = window_times(wf.window, meta)
        candidates.append(WorstWindow(
            start=t0,
            end=t1,
            center_index=wf.window.center,
            types=[labels[category] for category, _ in firing],
            severity=float(np.clip(max(error for _, error in firing), 0.0, 1.0)),
            confidence=confidence,
        ))
    candidates.sort(key=lambda item: -item.severity)
    kept: list[WorstWindow] = []
    for item in candidates:
        if all(abs(item.start - old.start) > temporal_nms_seconds for old in kept):
            kept.append(item)
        if len(kept) >= 12:
            break
    return kept


def _write_outputs(
    report: Report,
    windows: list[WindowFeatures],
    candidate: VideoReader,
    candidate_video: str,
    out_dir: str | None,
    export_clips: bool,
) -> None:
    if not out_dir:
        return
    out = Path(out_dir)
    write_json_report(report, out / "report.json")
    (out / "timeline.md").write_text(
        render_timeline_md(windows, candidate.meta, report),
        encoding="utf-8",
    )
    render_timeline_png(windows, candidate.meta, report, out / "timeline.png")
    if export_clips and report.worst_windows:
        export_badcase_clips(
            candidate_video,
            report.worst_windows,
            out / "badcases",
            out_fps=candidate.meta.fps,
        )


def evaluate_no_reference(
    candidate_video: str,
    preset: str = "standard",
    cache_dir: str = "./cache",
    device: str = "cuda",
    out_dir: str | None = None,
    flow_backend: str = "auto",
    vqa_backend: str = "auto",
    export_clips: bool = True,
    progress: ProgressFn | None = None,
) -> Report:
    """Evaluate temporal stability and artifact risk from one 60/120 FPS video."""
    started = time.perf_counter()
    say = progress or (lambda _: None)
    cfg = EvalConfig.build_mode(
        candidate_video=candidate_video,
        mode=EvaluationMode.NO_REFERENCE,
        preset=preset,
        cache_dir=cache_dir,
        device=device,
        flow_backend=flow_backend,
    )
    say("opening no-reference candidate")
    candidate = VideoReader(candidate_video)
    fps_supported = (
        abs(candidate.meta.fps - 60.0) <= 3.0
        or abs(candidate.meta.fps - 120.0) <= 6.0
    )
    warnings: list[str] = []
    if not fps_supported:
        warnings.append(
            f"no-reference calibration supports 60/120 FPS; got "
            f"{candidate.meta.fps:.3f} FPS")

    say("tier-1 candidate scan")
    scan = scan_candidate(candidate, width=cfg.preset.scan_width)
    cuts = scene_cuts_from_scan(scan)
    windows, _, _ = select_time_windows(
        cfg,
        candidate.meta,
        scan,
        excluded_indices=cuts,
        half_span_seconds=1.0 / 30.0,
        min_samples=5,
    )
    say(f"{len(windows)} time-normalized windows selected")
    backend = get_flow_backend(flow_backend, device)
    learned = get_vqa_backend(vqa_backend, device=device)

    stage_errors: collections.Counter[str] = collections.Counter()
    window_features: list[WindowFeatures] = []
    for index, window in enumerate(windows):
        if index % max(1, len(windows) // 10) == 0:
            say(f"no-reference window {index + 1}/{len(windows)}")
        wf = WindowFeatures(window=window)
        try:
            bundle = candidate.read_frames(
                window.indices, width=cfg.preset.flow_width)
            flows = WindowFlows(bundle, backend)
            flows.precompute()
            wf.scalars.update(no_reference.compute_window(
                bundle, flows, cfg, vqa_backend=learned))
        except Exception as exc:
            wf.labels["error_no_reference"] = repr(exc)
            stage_errors["no_reference"] += 1
        window_features.append(wf)

    required = (
        "nr_self_comp_mean",
        "nr_self_cycle_mean",
        "nr_mct_short_mean",
        "nr_phase_sharp_gap",
    )
    valid = [
        wf for wf in window_features
        if all(np.isfinite(wf.scalars.get(key, float("nan"))) for key in required)
    ]
    overall, subscores, category_errors = compute_mode_scores(
        EvaluationMode.NO_REFERENCE, valid)
    coverage = min(
        1.0,
        sum(len(wf.window.indices) for wf in valid)
        / max(candidate.meta.n_frames, 1),
    )
    confidence = compute_mode_confidence(
        valid_windows=len(valid),
        total_windows=len(window_features),
        coverage_fraction=coverage,
        warning_count=len(warnings),
        ceiling=0.75,  # intrinsic truth ambiguity in a single-video judgment
    )
    if not fps_supported:
        overall = float("nan")
        confidence = min(confidence, 0.01)
    event_ambiguity = float(np.clip(
        len(cuts) / max(candidate.meta.n_frames * 0.02, 1.0), 0.0, 1.0))
    status = (
        "failed" if not np.isfinite(overall)
        else "degraded" if stage_errors or len(valid) < len(window_features)
        else "ok"
    )
    worst = _worst_windows(
        EvaluationMode.NO_REFERENCE,
        valid,
        candidate.meta,
        confidence,
        cfg.preset.temporal_nms_seconds,
    )
    meta = {
        "mode": EvaluationMode.NO_REFERENCE.value,
        "status": status,
        "score_schema": SCHEMA_IDS[EvaluationMode.NO_REFERENCE],
        "score_semantics": "temporal stability and artifact risk; not interpolation truth",
        "limitations": [
            "Cannot prove true object trajectories or disoccluded content.",
            "A smooth, sharp hallucination may receive a low artifact-risk score.",
            "The heuristic formula requires real-data calibration before production gating.",
        ],
        "candidate": {
            "path": candidate.meta.path,
            "fps": candidate.meta.fps,
            "frames": candidate.meta.n_frames,
            "hash": candidate.meta.content_hash,
        },
        "reference": None,
        "time_scales_seconds": [round(1.0 / 60.0, 6), round(1.0 / 30.0, 6)],
        "phase_contract": "two-phase-self-reference",
        "flow_backend": getattr(backend, "name", flow_backend),
        "vqa_backend": getattr(learned, "name", None),
        "vqa_prior_available": learned is not None,
        "warnings": warnings,
        "scene_cut_frames": [int(i) for i in cuts],
        "valid_windows": len(valid),
        "windows_evaluated": len(window_features),
        "windows_selected": len(windows),
        "coverage_fraction": round(coverage, 5),
        "stage_errors": dict(stage_errors),
        "category_errors": {
            key: round(value, 4) if np.isfinite(value) else None
            for key, value in category_errors.items()
        },
        "calibrator": "nr-formula-v1",
        "elapsed_seconds": round(time.perf_counter() - started, 2),
    }
    report = Report(
        overall_score=overall,
        confidence=confidence,
        scores=subscores,
        event_ambiguity=event_ambiguity,
        worst_windows=worst,
        features=_feature_summary(valid),
        meta=meta,
    )
    _write_outputs(
        report, valid, candidate, candidate_video, out_dir, export_clips)
    return report


def evaluate_full_reference(
    reference_video: str,
    candidate_video: str,
    preset: str = "standard",
    cache_dir: str = "./cache",
    device: str = "cuda",
    out_dir: str | None = None,
    flow_backend: str = "auto",
    export_clips: bool = True,
    progress: ProgressFn | None = None,
) -> Report:
    """Evaluate an aligned same-rate candidate against frame ground truth."""
    started = time.perf_counter()
    say = progress or (lambda _: None)
    cfg = EvalConfig.build_mode(
        candidate_video=candidate_video,
        reference_video=reference_video,
        mode=EvaluationMode.FULL_REFERENCE,
        preset=preset,
        cache_dir=cache_dir,
        device=device,
        flow_backend=flow_backend,
    )
    say("opening full-reference videos")
    reference = VideoReader(reference_video)
    candidate = VideoReader(candidate_video)
    scan = scan_candidate(candidate, width=cfg.preset.scan_width)
    cuts = scene_cuts_from_scan(scan)
    say("building same-rate PTS/content alignment")
    alignment = build_full_reference_alignment(
        reference, candidate, scene_cuts_candidate=cuts)
    excluded = set(int(i) for i in cuts)
    for event in alignment.events:
        excluded.update(range(
            max(0, int(event["candidate_from"]) - 1),
            min(candidate.meta.n_frames, int(event["candidate_to"]) + 2),
        ))
    windows, _, _ = select_time_windows(
        cfg,
        candidate.meta,
        scan,
        eligible_centers=alignment.matched_centers(),
        excluded_indices=np.asarray(sorted(excluded), np.int32),
        half_span_seconds=1.0 / 30.0,
        min_samples=5,
    )
    say(f"{len(windows)} aligned windows selected")
    backend = get_flow_backend(flow_backend, device)

    stage_errors: collections.Counter[str] = collections.Counter()
    window_features: list[WindowFeatures] = []
    for index, window in enumerate(windows):
        if index % max(1, len(windows) // 10) == 0:
            say(f"full-reference window {index + 1}/{len(windows)}")
        wf = WindowFeatures(window=window)
        try:
            ref_indices = alignment.reference_of_candidate[window.indices]
            if np.any(ref_indices < 0) or np.any(np.diff(ref_indices) <= 0):
                raise ValueError("window crosses an unmatched alignment interval")
            cb = candidate.read_frames(
                window.indices, width=cfg.preset.flow_width)
            rb = reference.read_frames(
                ref_indices, width=cfg.preset.flow_width)
            cf = WindowFlows(cb, backend)
            rf = WindowFlows(rb, backend)
            pairs = [(i, i + 1) for i in range(len(cb.rgb) - 1)]
            cf.precompute(pairs)
            rf.precompute(pairs)
            wf.scalars.update(full_reference.compute_window(
                rb, cb, rf, cf, cfg))
        except Exception as exc:
            wf.labels["error_full_reference"] = repr(exc)
            stage_errors["full_reference"] += 1
        window_features.append(wf)

    required = (
        "fr_l1_y",
        "fr_ssim",
        "fr_temporal_diff_error",
        "fr_flow_error",
    )
    valid = [
        wf for wf in window_features
        if all(np.isfinite(wf.scalars.get(key, float("nan"))) for key in required)
    ]
    overall, subscores, category_errors = compute_mode_scores(
        EvaluationMode.FULL_REFERENCE, valid)
    coverage = min(
        1.0,
        sum(len(wf.window.indices) for wf in valid)
        / max(candidate.meta.n_frames, 1),
    )
    confidence = compute_mode_confidence(
        valid_windows=len(valid),
        total_windows=len(window_features),
        coverage_fraction=coverage,
        warning_count=len(alignment.warnings),
        alignment_fraction=alignment.matched_fraction,
        ceiling=0.98,
    )
    if not alignment.reliable:
        overall = float("nan")
        confidence = min(confidence, 0.01)
    status = (
        "failed" if not np.isfinite(overall)
        else "degraded" if stage_errors or len(valid) < len(window_features)
        else "ok"
    )
    event_ambiguity = float(np.clip(
        len(cuts) / max(candidate.meta.n_frames * 0.02, 1.0), 0.0, 1.0))
    worst = _worst_windows(
        EvaluationMode.FULL_REFERENCE,
        valid,
        candidate.meta,
        confidence,
        cfg.preset.temporal_nms_seconds,
    )
    meta = {
        "mode": EvaluationMode.FULL_REFERENCE.value,
        "status": status,
        "score_schema": SCHEMA_IDS[EvaluationMode.FULL_REFERENCE],
        "score_semantics": "same-rate full-reference spatial and temporal fidelity",
        "limitations": [
            "Requires the same capture, cadence, geometry, and frame correspondence.",
            "The formula score requires real-data calibration before production gating.",
        ],
        "reference": {
            "path": reference.meta.path,
            "fps": reference.meta.fps,
            "frames": reference.meta.n_frames,
            "hash": reference.meta.content_hash,
        },
        "candidate": {
            "path": candidate.meta.path,
            "fps": candidate.meta.fps,
            "frames": candidate.meta.n_frames,
            "hash": candidate.meta.content_hash,
        },
        "alignment": {
            "fps_ratio": alignment.fps_ratio,
            "matched_fraction": round(alignment.matched_fraction, 5),
            "image_error": round(alignment.image_error, 3),
            "reliable": alignment.reliable,
            "events": alignment.events,
            "warnings": alignment.warnings,
        },
        "flow_backend": getattr(backend, "name", flow_backend),
        "scene_cut_frames": [int(i) for i in cuts],
        "valid_windows": len(valid),
        "windows_evaluated": len(window_features),
        "windows_selected": len(windows),
        "coverage_fraction": round(coverage, 5),
        "stage_errors": dict(stage_errors),
        "category_errors": {
            key: round(value, 4) if np.isfinite(value) else None
            for key, value in category_errors.items()
        },
        "calibrator": "fr-formula-v1",
        "elapsed_seconds": round(time.perf_counter() - started, 2),
    }
    report = Report(
        overall_score=overall,
        confidence=confidence,
        scores=subscores,
        event_ambiguity=event_ambiguity,
        worst_windows=worst,
        features=_feature_summary(valid),
        meta=meta,
    )
    _write_outputs(
        report, valid, candidate, candidate_video, out_dir, export_clips)
    return report


def evaluate(
    candidate_video: str,
    reference_video: str | None = None,
    mode: str | EvaluationMode = EvaluationMode.NO_REFERENCE,
    **kwargs,
) -> Report:
    """Dispatch to one explicit evaluation contract."""
    parsed = parse_mode(mode)
    if parsed is EvaluationMode.NO_REFERENCE:
        if reference_video is not None:
            raise ValueError("no-reference mode does not accept reference_video")
        return evaluate_no_reference(candidate_video, **kwargs)
    if not reference_video:
        raise ValueError(f"{parsed.value} mode requires reference_video")
    if parsed is EvaluationMode.ENDPOINT_2X:
        return evaluate_endpoint_reference(reference_video, candidate_video, **kwargs)
    return evaluate_full_reference(reference_video, candidate_video, **kwargs)


def compare(
    candidate_videos: list[str],
    reference_video: str | None = None,
    mode: str | EvaluationMode = EvaluationMode.NO_REFERENCE,
    labels: list[str] | None = None,
    out_dir: str | None = None,
    **kwargs,
) -> list[dict]:
    """Rank candidates only within one shared mode/schema."""
    labels = labels or [Path(path).stem for path in candidate_videos]
    if len(labels) != len(candidate_videos):
        raise ValueError("labels must have the same length as candidate_videos")
    results: list[dict] = []
    for label, candidate in zip(labels, candidate_videos):
        report = evaluate(
            candidate,
            reference_video=reference_video,
            mode=mode,
            out_dir=str(Path(out_dir) / label) if out_dir else None,
            export_clips=False,
            **kwargs,
        )
        score = (
            float(report.overall_score)
            if np.isfinite(report.overall_score) else None
        )
        results.append({
            "model": label,
            "overall": score,
            "confidence": report.confidence,
            "scores": report.scores,
            "status": report.meta.get("status", "ok"),
            "mode": report.meta["mode"],
            "score_schema": report.meta["score_schema"],
            "relative_vs_mean": None,
        })
    results.sort(key=lambda item: (
        item["status"] == "failed",
        -(item["overall"] if item["overall"] is not None else float("-inf")),
    ))
    valid_scores = [
        item["overall"] for item in results if item["overall"] is not None
    ]
    mean = float(np.mean(valid_scores)) if valid_scores else None
    if mean is not None:
        for item in results:
            if item["overall"] is not None:
                item["relative_vs_mean"] = round(item["overall"] - mean, 2)
    if out_dir:
        import json

        output = Path(out_dir)
        output.mkdir(parents=True, exist_ok=True)
        (output / "ranking.json").write_text(
            json.dumps(results, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    return results
