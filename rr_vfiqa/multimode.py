"""Public multi-mode dispatch and the NR/FR execution pipelines."""

from __future__ import annotations

import collections
import time
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .config import EvalConfig, EvaluationMode, parse_mode
from .calibration.provenance import report_provenance
from .diagnosis import (
    affected_duration_fraction,
    build_diagnostics_block,
    cadence_integrity,
    diagnose_windows,
)
from .fusion.feature_registry import feature_contract_hash, required_features
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
    export_overlay_clips,
    render_timeline_md,
    render_timeline_png,
    write_html_report,
    write_json_report,
)
from .sampling.cheap_scan import scan_candidate, scene_cuts_from_scan
from .sampling.time_window_selector import select_time_windows
from .sampling.temporal_plan import FlowPairPlan, TemporalLagPlan
from .sampling.full_reference_scan import scan_full_reference
from .sampling.window_selector import window_times
from .schema import FrameBundle, MetricResult, Report, WindowFeatures, WorstWindow

ProgressFn = Callable[[str], None]


def _unique_coverage(
    windows: list[WindowFeatures],
    n_frames: int,
) -> float:
    if not windows:
        return 0.0
    covered = np.unique(np.concatenate([
        wf.window.indices for wf in windows
    ]))
    return float(len(covered) / max(n_frames, 1))


def _compute_nr_error_maps(candidate, cfg, backend, window_features, worst, *,
                           top_k: int = 8):
    """Compute dense error maps for the top-risk NR windows (USERPLAN §8).

    Only the windows that feed the worst-issue cards pay this cost; the maps
    are stored on ``wf.error_maps`` and later matched to ``DiagnosticIssue``
    objects by ``center_index``.  Returns ``{center_index: {name: array}}``.
    """
    from .metrics import nr_window_maps
    if not worst or not window_features:
        return {}
    # worst windows are already sorted worst-first by the caller.
    centers = [int(w.center_index) for w in worst[:top_k]]
    by_center = {int(wf.window.center): wf for wf in window_features}
    out: dict[int, dict[str, np.ndarray]] = {}
    for center in centers:
        wf = by_center.get(center)
        if wf is None:
            continue
        try:
            bundle = candidate.read_frames(
                wf.window.indices, width=cfg.preset.flow_width)
            flows = WindowFlows(bundle, backend)
            lag_plan = TemporalLagPlan.build(bundle.times)
            flow_plan = FlowPairPlan.for_no_reference(lag_plan)
            flows.precompute(flow_plan.unique_pairs())
            maps = nr_window_maps(bundle, flows, cfg)
            if maps:
                wf.error_maps.update(maps)
                out[center] = dict(maps)
        except Exception as exc:  # never let map generation fail the report
            wf.labels["error_map_failure"] = repr(exc)
    return out


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


def _metric_diagnostics(
    windows: list[WindowFeatures],
    *,
    limit: int = 20,
) -> dict:
    samples = []
    failed = 0
    warnings = 0
    exceptions = 0
    coverages = []
    confidences = []
    for wf in windows:
        status = wf.labels.get("metric_status")
        messages = list(wf.labels.get("metric_warnings") or [])
        error_labels = {
            key: value for key, value in wf.labels.items()
            if key.startswith("error_")
        }
        if error_labels:
            exceptions += 1
        if status == "failed" or error_labels:
            failed += 1
        warnings += len(messages)
        coverage = wf.labels.get("metric_coverage")
        confidence = wf.labels.get("metric_confidence")
        if coverage is not None and np.isfinite(coverage):
            coverages.append(float(coverage))
        if confidence is not None and np.isfinite(confidence):
            confidences.append(float(confidence))
        if (messages or error_labels) and len(samples) < limit:
            samples.append({
                "center_index": int(wf.window.center),
                "status": status or "exception",
                "warnings": messages,
                "errors": error_labels,
            })
    return {
        "failed_windows": failed,
        "exception_windows": exceptions,
        "warning_count": warnings,
        "mean_metric_coverage": (
            round(float(np.mean(coverages)), 5) if coverages else None),
        "mean_metric_confidence": (
            round(float(np.mean(confidences)), 5) if confidences else None),
        "samples": samples,
    }


def _resize_bundle(
    bundle: FrameBundle,
    *,
    width: int,
    height: int,
) -> FrameBundle:
    if (bundle.width, bundle.height) == (width, height):
        return bundle
    interpolation = (
        cv2.INTER_AREA
        if width <= bundle.width and height <= bundle.height
        else cv2.INTER_LINEAR)
    rgb = np.stack([
        cv2.resize(frame, (width, height), interpolation=interpolation)
        for frame in bundle.rgb
    ])
    return FrameBundle(
        indices=bundle.indices,
        times=bundle.times,
        rgb=rgb,
        width=width,
        height=height,
    )


def _fr_target_size(
    reference_meta,
    candidate_meta,
    *,
    max_width: int,
    geometry_policy: str,
) -> tuple[int, int]:
    """Return one explicit FR comparison grid for both video streams."""
    width_limit = min(max_width, reference_meta.width)
    if geometry_policy in ("strict", "common-resolution"):
        width_limit = min(width_limit, candidate_meta.width)
    width = max(2, int(width_limit))
    height = max(
        2,
        int(round(reference_meta.height * width / reference_meta.width)) // 2 * 2,
    )
    return width, height


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
    # --- USERPLAN §8: render heatmap PNGs for issues that have a map -----
    # Done before the JSON/HTML writers so both embed the final relative
    # paths.  Failures here must never abort the report, so the whole step is
    # guarded and the issue's ``maps`` list falls back to the raw map name.
    try:
        _render_issue_heatmaps(report, windows, out)
    except Exception as exc:  # noqa: BLE001
        if report.meta is not None:
            report.meta["heatmap_render_error"] = repr(exc)
    write_json_report(report, out / "report.json")
    write_html_report(report, out / "report.html")
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
    # USERPLAN §10: overlay companion clips (severity bar + timecode + heat).
    # USERPLAN P1-R1: pass the real per-issue error map to the exporter so the
    # heat overlay shows the actual diagnostic evidence, not the motion-proxy
    # fallback (the proxy is only used when an issue has no map).
    diag_issues = ((report.meta or {}).get("diagnostics") or {}).get("issues") or []
    if export_clips and diag_issues:
        overlay_maps = _issue_error_maps(diag_issues, windows)
        export_overlay_clips(
            candidate_video, diag_issues, out / "badcases",
            out_fps=candidate.meta.fps, error_maps=overlay_maps)


def _issue_error_maps(diag_issues: list[dict],
                      windows: list[WindowFeatures]) -> dict[int, np.ndarray]:
    """Collect the preferred error map for each issue (by center_index).

    Returns ``{issue_index: error_map}`` for the overlay exporter.  Issues
    without a map or without a matching window are simply omitted — the
    exporter falls back to its motion-proxy for those.
    """
    by_center = {int(wf.window.center): wf for wf in windows}
    out: dict[int, np.ndarray] = {}
    for i, issue in enumerate(diag_issues):
        if not issue.get("maps"):
            continue
        wf = by_center.get(int(issue.get("center_index", -1)))
        if wf is None:
            continue
        for name in issue["maps"]:
            arr = wf.error_maps.get(name)
            if arr is not None:
                out[i] = arr
                break
    return out


def _render_issue_heatmaps(report: Report, windows: list[WindowFeatures],
                           out: Path) -> None:
    """Write ``heatmaps/issue_NNN_<map>.png`` for issues carrying a map name.

    Each issue was associated with one preferred error map (see the
    ``_NR_ISSUE_MAP`` table in ``evaluate_no_reference``); here we look up that
    map on the matching window and save a jet PNG.  The issue's ``maps`` list
    is rewritten from the map-name to the relative path so the HTML report can
    link to it (USERPLAN §9 directory layout).
    """
    from rr_vfiqa.visualization import save_heatmap
    diag = ((report.meta or {}).get("diagnostics") or {})
    issues = diag.get("issues") or []
    if not issues or not windows:
        return
    by_center = {int(wf.window.center): wf for wf in windows}
    heat_dir = out / "heatmaps"
    for i, issue in enumerate(issues):
        if not issue.get("maps"):
            continue
        wf = by_center.get(int(issue.get("center_index", -1)))
        if wf is None:
            continue
        rel_maps: list[str] = []
        for name in issue["maps"]:
            arr = wf.error_maps.get(name) if wf else None
            if arr is None:
                continue
            label = f"{issue.get('title', name)}  #{i:03d}"
            png = save_heatmap(arr, heat_dir / f"issue_{i:03d}_{name}.png",
                               label=label)
            rel_maps.append(f"heatmaps/{png.name}")
        issue["maps"] = rel_maps


def evaluate_no_reference(
    candidate_video: str,
    preset: str = "standard",
    cache_dir: str = "./cache",
    device: str = "cuda",
    out_dir: str | None = None,
    flow_backend: str = "auto",
    vqa_backend: str = "none",
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
    if vqa_backend == "auto":
        raise ValueError(
            "vqa_backend='auto' is not reproducible; choose 'none' or "
            "'pyiqa-niqe' explicitly")
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
            lag_plan = TemporalLagPlan.build(bundle.times)
            flow_plan = FlowPairPlan.for_no_reference(lag_plan)
            flows.precompute(flow_plan.unique_pairs())
            metric = MetricResult.from_scalars(
                no_reference.compute_window(
                    bundle, flows, cfg, vqa_backend=learned),
                required=required_features(EvaluationMode.NO_REFERENCE),
            )
            wf.scalars.update(metric.scalars)
            wf.labels["metric_status"] = metric.status
            wf.labels["metric_warnings"] = metric.warnings
            wf.labels["metric_coverage"] = metric.coverage
            wf.labels["metric_confidence"] = metric.confidence
        except Exception as exc:
            wf.labels["error_no_reference"] = repr(exc)
            stage_errors["no_reference"] += 1
        window_features.append(wf)

    required = required_features(EvaluationMode.NO_REFERENCE)
    valid = [
        wf for wf in window_features
        if all(np.isfinite(wf.scalars.get(key, float("nan"))) for key in required)
    ]
    overall, subscores, category_errors = compute_mode_scores(
        EvaluationMode.NO_REFERENCE, valid)
    coverage = _unique_coverage(valid, candidate.meta.n_frames)
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

    # --- USERPLAN §5 cadence integrity + §7 diagnostic evidence ------------
    # The shared 1/60 s score must not hide a collapsed native cadence, so we
    # penalise the common-time overall multiplicatively and surface both the
    # common-time and cadence-integrity numbers.  Diagnostics are built from
    # every window (incl. ones that failed the required-feature gate) so a
    # localized defect still produces a located issue.
    # USERPLAN P0-R1: shared-scale (1/60 s) motion per window drives the
    # cadence motion gate — windows with no longer-scale motion and no
    # odd/even alternation are treated as genuinely static, not collapsed.
    cad_rep = cadence_integrity(
        [wf.scalars for wf in valid], overall,
        alternation_per_window=[
            float(wf.scalars.get("nr_native_motion_alternation", float("nan")))
            for wf in valid],
        common_time_motion_per_window=[
            float(wf.scalars.get("nr_mct_1_60_mean", float("nan")))
            for wf in valid],
    )
    if np.isfinite(overall):
        overall = cad_rep.overall
    subscores["common_time_quality"] = cad_rep.common_time_quality
    subscores["cadence_integrity"] = cad_rep.cadence_integrity
    diag_ws = [
        (window_times(wf.window, candidate.meta)[0],
         window_times(wf.window, candidate.meta)[1],
         int(wf.window.center), wf.scalars,
         float(wf.labels.get("metric_confidence", 1.0)))
        for wf in window_features
    ]
    nr_issues = diagnose_windows(diag_ws, cfg.preset.temporal_nms_seconds)

    # --- USERPLAN §8: dense error maps for the top-risk windows -----------
    # Scalar metrics run for every window; the heavier dense fields are only
    # computed for the windows that feed the worst-issue cards, so report-side
    # rendering stays bounded.  Maps are matched to issues by center_index.
    nr_error_maps = _compute_nr_error_maps(
        candidate, cfg, backend, window_features, worst)

    # Pick, for each issue, the single most diagnostic map to render: the
    # map whose family best explains the issue's evidence.  The mapping is by
    # issue_type -> preferred NR map name (USERPLAN §8).
    _NR_ISSUE_MAP = {
        "duplicate_freeze": "duplicate_frame_indicator",
        "generated_blur": "phase_sharpness_map",
        "ghost_double_exposure": "composition_error_map",
        "tearing_flow_folding": "flow_fold_map",
        "ui_text_instability": "ui_edge_instability_map",
    }
    if nr_error_maps:
        by_center = {int(wf.window.center): wf for wf in window_features}
        for issue in nr_issues:
            wf = by_center.get(int(issue.center_index))
            if wf is None:
                continue
            preferred = _NR_ISSUE_MAP.get(issue.issue_type)
            available = [preferred] if preferred and preferred in wf.error_maps \
                else list(wf.error_maps.keys())
            if available:
                # record the chosen map name; the heatmap PNG is written later
                # in _write_outputs, keyed by (issue_index, map_name).
                issue.maps = [available[0]]

    score_schema = SCHEMA_IDS[EvaluationMode.NO_REFERENCE] + (
        "+niqe" if learned is not None else "")
    meta = {
        "mode": EvaluationMode.NO_REFERENCE.value,
        "status": status,
        "score_schema": score_schema,
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
        "temporal_lag_contract": {
            "native": "input-cadence",
            "lag_1_60_seconds": round(1.0 / 60.0, 6),
            "lag_1_30_seconds": round(1.0 / 30.0, 6),
            "self_reference_half_span_seconds": round(1.0 / 60.0, 6),
        },
        "phase_contract": "two-phase-self-reference",
        "diagnostic_branches": {
            "flow_geometry": "global-translation-residual-jacobian",
            "track_smoothness": "camera-relative-klt",
            "ui_text": "screen-band-persistent-edge-proxy",
        },
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
        "metric_diagnostics": _metric_diagnostics(window_features),
        "category_errors": {
            key: round(value, 4) if np.isfinite(value) else None
            for key, value in category_errors.items()
        },
        "calibrator": "nr-common-time-formula-v3",
        "cadence": {
            "cadence_risk": round(cad_rep.cadence_risk, 4),
            "cadence_integrity": round(cad_rep.cadence_integrity, 2),
            "common_time_quality": round(cad_rep.common_time_quality, 2)
            if np.isfinite(cad_rep.common_time_quality) else None,
            "penalty_lambda": 2.2,
            "evidence": cad_rep.evidence,
        },
        "diagnostics": build_diagnostics_block(
            nr_issues, candidate.meta, confidence),
        "elapsed_seconds": round(time.perf_counter() - started, 2),
    }
    meta.update(report_provenance(
        mode=EvaluationMode.NO_REFERENCE.value,
        score_schema=score_schema,
        metric_contract="nr-metrics-v2",
        preset_contract=f"nr-{cfg.preset.name}-v1",
        feature_contract_hash=feature_contract_hash(
            EvaluationMode.NO_REFERENCE),
        backend_contract={
            "flow": backend.cache_identity(),
            "vqa": getattr(learned, "name", "none"),
        },
    ))
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
    geometry_policy: str = "strict",
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
        geometry_policy=geometry_policy,
    )
    say("opening full-reference videos")
    reference = VideoReader(reference_video)
    candidate = VideoReader(candidate_video)
    scan = scan_candidate(candidate, width=cfg.preset.scan_width)
    cuts = scene_cuts_from_scan(scan)
    say("building same-rate PTS/content alignment")
    alignment = build_full_reference_alignment(
        reference, candidate, scene_cuts_candidate=cuts,
        geometry_policy=geometry_policy)
    scan_size = _fr_target_size(
        reference.meta,
        candidate.meta,
        max_width=cfg.preset.scan_width,
        geometry_policy=geometry_policy,
    )
    say("full-reference low-resolution timeline scan")
    reference_scan = scan_full_reference(
        reference,
        candidate,
        alignment,
        width=scan_size[0],
        target_size=scan_size,
    )
    reference_globals = reference_scan.global_features()
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
        external_risk=reference_scan.per_frame_risk,
        half_span_seconds=1.0 / 30.0,
        min_samples=5,
    )
    say(f"{len(windows)} aligned windows selected")
    backend = get_flow_backend(flow_backend, device)
    working_size = _fr_target_size(
        reference.meta,
        candidate.meta,
        max_width=cfg.preset.flow_width,
        geometry_policy=geometry_policy,
    )
    working_width, working_height = working_size

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
            rb = reference.read_frames(
                ref_indices, width=working_width)
            cb = candidate.read_frames(
                window.indices, width=working_width)
            rb = _resize_bundle(
                rb, width=working_width, height=working_height)
            cb = _resize_bundle(
                cb, width=working_width, height=working_height)
            cf = WindowFlows(cb, backend)
            rf = WindowFlows(rb, backend)
            pairs = [(i, i + 1) for i in range(len(cb.rgb) - 1)]
            cf.precompute(pairs)
            rf.precompute(pairs)
            metric = MetricResult.from_scalars(
                full_reference.compute_window(rb, cb, rf, cf, cfg),
                required=required_features(EvaluationMode.FULL_REFERENCE),
            )
            wf.scalars.update(metric.scalars)
            wf.labels["metric_status"] = metric.status
            wf.labels["metric_warnings"] = metric.warnings
            wf.labels["metric_coverage"] = metric.coverage
            wf.labels["metric_confidence"] = metric.confidence
        except Exception as exc:
            wf.labels["error_full_reference"] = repr(exc)
            stage_errors["full_reference"] += 1
        window_features.append(wf)

    required = required_features(EvaluationMode.FULL_REFERENCE)
    valid = [
        wf for wf in window_features
        if all(np.isfinite(wf.scalars.get(key, float("nan"))) for key in required)
    ]
    overall, subscores, category_errors = compute_mode_scores(
        EvaluationMode.FULL_REFERENCE, valid, reference_globals)
    coverage = _unique_coverage(valid, candidate.meta.n_frames)
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
    # USERPLAN §7: structured diagnostic evidence (no cadence term for FR).
    fr_diag_ws = [
        (window_times(wf.window, candidate.meta)[0],
         window_times(wf.window, candidate.meta)[1],
         int(wf.window.center), wf.scalars,
         float(wf.labels.get("metric_confidence", 1.0)))
        for wf in window_features
    ]
    fr_issues = diagnose_windows(fr_diag_ws, cfg.preset.temporal_nms_seconds)
    fr_score_schema = SCHEMA_IDS[EvaluationMode.FULL_REFERENCE] + (
        "" if geometry_policy == "strict" else f"+{geometry_policy}")
    meta = {
        "mode": EvaluationMode.FULL_REFERENCE.value,
        "status": status,
        "score_schema": fr_score_schema,
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
            "geometry_policy": geometry_policy,
            "resize_transform": (
                None if geometry_policy == "strict" else {
                    "policy": geometry_policy,
                    "working_width": working_width,
                    "working_height": working_height,
                    "reference_native": [
                        reference.meta.width, reference.meta.height],
                    "candidate_native": [
                        candidate.meta.width, candidate.meta.height],
                }),
        },
        "flow_backend": getattr(backend, "name", flow_backend),
        "roi_branches": {
            "ui": "screen-band-persistent-edge-proxy",
            "text": "dense-stroke-proxy",
            "salient": "reference-gradient-proxy",
            "motion": "center-reference-motion-proxy",
        },
        "scene_cut_frames": [int(i) for i in cuts],
        "full_reference_scan": {
            key: round(value, 6)
            for key, value in reference_globals.items()
        },
        "full_reference_scan_contract": {
            "streaming": True,
            "target_width": scan_size[0],
            "target_height": scan_size[1],
            "gradient_operator": "sobel-magnitude-3x3",
        },
        "valid_windows": len(valid),
        "windows_evaluated": len(window_features),
        "windows_selected": len(windows),
        "coverage_fraction": round(coverage, 5),
        "stage_errors": dict(stage_errors),
        "metric_diagnostics": _metric_diagnostics(window_features),
        "category_errors": {
            key: round(value, 4) if np.isfinite(value) else None
            for key, value in category_errors.items()
        },
        "calibrator": "fr-formula-v2",
        "diagnostics": build_diagnostics_block(
            fr_issues, candidate.meta, confidence),
        "elapsed_seconds": round(time.perf_counter() - started, 2),
    }
    meta.update(report_provenance(
        mode=EvaluationMode.FULL_REFERENCE.value,
        score_schema=fr_score_schema,
        metric_contract="fr-metrics-v2",
        preset_contract=f"fr-{cfg.preset.name}-v1",
        feature_contract_hash=feature_contract_hash(
            EvaluationMode.FULL_REFERENCE),
        backend_contract={"flow": backend.cache_identity()},
    ))
    report = Report(
        overall_score=overall,
        confidence=confidence,
        scores=subscores,
        event_ambiguity=event_ambiguity,
        worst_windows=worst,
        features={**_feature_summary(valid), **reference_globals},
        meta=meta,
    )
    _write_outputs(
        report, valid, candidate, candidate_video, out_dir, export_clips)
    return report


def evaluate(
    candidate_video: str,
    reference_video: str | None = None,
    mode: str | EvaluationMode | None = None,
    **kwargs,
) -> Report:
    """Dispatch to one explicit evaluation contract."""
    if mode is None:
        raise ValueError("mode must be explicitly specified")
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
    mode: str | EvaluationMode | None = None,
    labels: list[str] | None = None,
    out_dir: str | None = None,
    allow_cross_content: bool = False,
    comparison_group_id: str | None = None,
    **kwargs,
) -> list[dict]:
    """Rank candidates only within one shared mode/schema."""
    if mode is None:
        raise ValueError("mode must be explicitly specified")
    parsed_mode = parse_mode(mode)
    labels = labels or [Path(path).stem for path in candidate_videos]
    if len(labels) != len(candidate_videos):
        raise ValueError("labels must have the same length as candidate_videos")
    comparison_issues: list[str] = []
    if parsed_mode is EvaluationMode.NO_REFERENCE and len(candidate_videos) > 1:
        comparison_issues = _nr_compare_issues(
            candidate_videos,
            trusted_content=bool(comparison_group_id),
        )
        if comparison_issues and not allow_cross_content:
            raise ValueError(
                "no-reference candidates are not safely comparable: "
                + "; ".join(comparison_issues)
                + ". Pass allow_cross_content=True only for independent reports.")
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
    comparable = not comparison_issues
    if comparable:
        results.sort(key=lambda item: (
            item["status"] == "failed",
            -(item["overall"] if item["overall"] is not None else float("-inf")),
        ))
    valid_scores = [
        item["overall"] for item in results if item["overall"] is not None
    ]
    mean = float(np.mean(valid_scores)) if valid_scores else None
    if mean is not None and comparable:
        for item in results:
            if item["overall"] is not None:
                item["relative_vs_mean"] = round(item["overall"] - mean, 2)
    for item in results:
        item["comparison_status"] = (
            "comparable" if comparable else "independent-report")
        item["comparison_warnings"] = comparison_issues
        item["comparison_group_id"] = comparison_group_id
    if out_dir:
        import json

        output = Path(out_dir)
        output.mkdir(parents=True, exist_ok=True)
        (output / "ranking.json").write_text(
            json.dumps(results, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    return results


def _nr_compare_issues(
    candidate_videos: list[str],
    *,
    trusted_content: bool = False,
) -> list[str]:
    import cv2

    from .imutils import hamming64, phash64

    readers = [VideoReader(path) for path in candidate_videos]
    base = readers[0].meta
    issues: list[str] = []
    base_bucket = 60 if abs(base.fps - 60.0) <= 3.0 else 120
    base_duration = float(base.pts_seconds[-1] - base.pts_seconds[0])
    for index, reader in enumerate(readers[1:], 1):
        meta = reader.meta
        bucket = 60 if abs(meta.fps - 60.0) <= 3.0 else 120
        if bucket != base_bucket:
            issues.append(f"candidate {index} FPS bucket differs")
        duration = float(meta.pts_seconds[-1] - meta.pts_seconds[0])
        if abs(duration - base_duration) / max(base_duration, 1e-6) > 0.05:
            issues.append(f"candidate {index} duration differs by more than 5%")
        if abs(meta.width / meta.height - base.width / base.height) > 0.005:
            issues.append(f"candidate {index} aspect ratio differs")
        elif (meta.width, meta.height) != (base.width, base.height):
            issues.append(f"candidate {index} resolution differs")
        pts_count = min(32, len(base.pts_seconds), len(meta.pts_seconds))
        if pts_count >= 2:
            base_pos = np.linspace(
                0, len(base.pts_seconds) - 1, pts_count).astype(int)
            cand_pos = np.linspace(
                0, len(meta.pts_seconds) - 1, pts_count).astype(int)
            base_pts = base.pts_seconds[base_pos] - base.pts_seconds[0]
            cand_pts = meta.pts_seconds[cand_pos] - meta.pts_seconds[0]
            pts_error = float(np.max(np.abs(base_pts - cand_pts)))
            if pts_error > max(0.25 / max(base.fps, 1e-6), 0.002):
                issues.append(f"candidate {index} PTS timeline differs")
        count = min(16, base.n_frames, meta.n_frames)
        if count and not trusted_content:
            a_pos = np.linspace(0, base.n_frames - 1, count).astype(int)
            b_pos = np.linspace(0, meta.n_frames - 1, count).astype(int)
            base_bundle = readers[0].read_frames(a_pos, width=96)
            candidate_bundle = reader.read_frames(b_pos, width=96)
            distances = []
            for base_frame, candidate_frame in zip(
                    base_bundle.rgb, candidate_bundle.rgb):
                base_gray = cv2.equalizeHist(cv2.cvtColor(
                    base_frame, cv2.COLOR_RGB2GRAY))
                candidate_gray = cv2.equalizeHist(cv2.cvtColor(
                    candidate_frame, cv2.COLOR_RGB2GRAY))
                distances.append(hamming64(
                    phash64(base_gray), phash64(candidate_gray)))
            if distances and float(np.median(distances)) > 12:
                issues.append(f"candidate {index} scene fingerprint differs")
    return list(dict.fromkeys(issues))
