"""Evaluation pipeline orchestration (USERPLAN.md §13 pseudocode).

    source cache → alignment → cheap scan → window selection →
    per-window core metrics → high-risk region diagnostics → fusion → report
"""

from __future__ import annotations

import collections
import time
from pathlib import Path
from typing import Callable

import numpy as np

from .cache.source_cache import SourceCache
from .calibration.provenance import (
    calibrator_provenance,
    report_provenance,
)
from .config import EvalConfig, EvaluationMode
from .diagnosis import (
    build_diagnostics_block,
    diagnose_windows,
    select_map_name,
)
from .fusion import (build_category_errors, compute_confidence, compute_scores,
                     maybe_load)
from .fusion.feature_registry import feature_contract_hash
from .fusion.feature_normalizer import normalize_error
from .fusion.score_schema import CATEGORY_FEATURES, category_window_valid
from .io.timestamp_alignment import build_alignment
from .io.video_reader import VideoReader
from .metrics import (anchor_integrity, cycle_reconstruction, edge_structure,
                      flow_composition_metric, global_technical_quality,
                      parity_frequency, temporal_compensation, WindowFlows)
from .models.tracker_backend import get_audit_tracker
from .motion.flow_estimator import get_flow_backend
from .regions import (card_tracker, character_segmenter, text_evaluator,
                      thin_object_detector, transition_evaluator, ui_detector,
                      weapon_tracker)
from .regions.ui_detector import UIDetector
from .report import (
    ArtifactContext,
    build_report_artifacts,
    render_timeline_md,
    render_timeline_png,
    save_error_heatmap,
)
from .sampling.cheap_scan import scan_candidate, scene_cuts_from_scan
from .sampling.window_selector import select_windows, window_times
from .schema import Report, WindowFeatures, WorstWindow

ProgressFn = Callable[[str], None]


# ---------------------------------------------------------------------------
# per-window classification
# ---------------------------------------------------------------------------

_TYPE_THRESHOLDS = (
    ("motion", 0.45), ("temporal", 0.40), ("structure", 0.45),
    ("character", 0.40), ("thin_weapon", 0.45), ("ui", 0.40),
    ("transition", 0.35),
)


def _window_category_errors(wf: WindowFeatures) -> dict[str, float]:
    """Trigger-oriented blend: mean pulled up toward the worst single feature,
    so one strongly-firing detector can label a window even if the rest of the
    category is quiet."""
    out = {}
    for cat, keys in CATEGORY_FEATURES.items():
        vals = []
        for k in keys:
            raw = wf.scalars.get(k)
            if raw is None:
                continue
            err = normalize_error(k, raw)
            if err is not None:
                vals.append(err)
        if not vals:
            out[cat] = 0.0
        else:
            out[cat] = float(max(np.mean(vals), 0.75 * max(vals)))
    return out


def _classify_window(wf: WindowFeatures, conf_base: float) -> WorstWindow | None:
    cat_err = _window_category_errors(wf)
    types: list[str] = []
    s = wf.scalars
    # High-confidence specific labels bypass the category cap — they are
    # precise signals that the top-3-by-error rule would otherwise crowd out.
    if (s.get("card_flip_err") or 0) > 0.5:
        types.append("card_flip_error")
    if (s.get("freeze_copy_score") or 0) > 0.5:
        types.append("freeze_copy")
    # Only the three loudest categories may label a window further.
    firing = sorted(((c, t) for c, t in _TYPE_THRESHOLDS
                     if cat_err.get(c, 0.0) >= t),
                    key=lambda ct: -cat_err[ct[0]])[:3]
    for cat, thr in firing:
        if cat == "motion":
            types.append("background_tear" if s.get("flow_fold_frac", 0) > 0.03
                         else "motion_inconsistent")
        elif cat == "temporal":
            # parity_flicker needs direct alternation evidence, else generic
            # (freeze_copy is handled as a high-confidence special label).
            par = normalize_error("parity_window_sharp_gap",
                                  s.get("parity_window_sharp_gap", float("nan"))) or 0
            types.append("parity_flicker" if par > 0.35 else "temporal_instability")
        elif cat == "structure":
            types.append("structure_loss")
        elif cat == "character":
            types.append("character_ghost"
                         if s.get("char_leak_mean", 99) < 6 and
                         s.get("char_ring_edge_frac", 0) > 0.18
                         else "character_missing")
        elif cat == "thin_weapon":
            types.append("weapon_jitter" if s.get("weapon_dev_p90", 0) > 0.35
                         else "thin_object_stick")
        elif cat == "ui":
            te_err = normalize_error("text_edge_f", s["text_edge_f"]) \
                if "text_edge_f" in s else None
            types.append("text_degraded" if (te_err or 0) > 0.5 else "ui_unstable")
        elif cat == "transition":
            if (s.get("event_ghost_frac") or 0) > 0.08 or not types:
                types.append("transition_ghost")
    if not types:
        return None
    types = list(dict.fromkeys(types))[:4]
    sev_special = max(s.get("card_flip_err") or 0.0, s.get("freeze_copy_score") or 0.0)
    severity = float(np.clip(max(max(cat_err.values()), sev_special) / 1.2, 0, 1))
    boxes = []
    for inst in (s.get("_thin_instances") or [])[:3]:
        boxes.append(inst["box"])
    if s.get("_char_bbox"):
        boxes.append(s["_char_bbox"])
    start = wf.window.indices[0] / 120.0     # refined by caller with real pts
    return WorstWindow(start=start, end=start, center_index=wf.window.center,
                       types=types, severity=severity, confidence=conf_base,
                       boxes=boxes)


# ---------------------------------------------------------------------------
# main entry
# ---------------------------------------------------------------------------

def evaluate_endpoint_reference(
    reference_video: str,
    candidate_video: str,
    preset: str = "standard",
    cache_dir: str = "./cache",
    device: str = "cuda",
    out_dir: str | None = None,
    flow_backend: str = "auto",
    export_clips: bool = True,
    calibrator_path: str | None = None,
    progress: ProgressFn | None = None,
) -> Report:
    """Evaluate a 2× candidate against its endpoint/reduced reference."""
    t_start = time.perf_counter()

    def say(msg: str) -> None:
        if progress:
            progress(msg)

    cfg = EvalConfig.build(reference_video, candidate_video, preset,
                           cache_dir=cache_dir, device=device,
                           flow_backend=flow_backend)
    p = cfg.preset

    say("opening videos")
    source = VideoReader(reference_video)
    candidate = VideoReader(candidate_video)

    say("tier-1 cheap full-frame scan")
    cheap = scan_candidate(candidate, width=p.scan_width)
    scene_cuts = scene_cuts_from_scan(cheap)

    say("aligning candidate to source (PTS + anchors)")
    alignment = build_alignment(cfg, source, candidate,
                                scene_cuts_cand=scene_cuts)

    say("anchor integrity + color baseline")
    anchor_feats, color_tf, anchor_warnings = anchor_integrity.evaluate_anchors(
        cfg, source, candidate, alignment)
    alignment.warnings.extend(anchor_warnings)

    say("selecting windows")
    windows, risk, _ = select_windows(cfg, candidate.meta, alignment, cheap)
    say(f"{len(windows)} windows selected")

    backend = get_flow_backend(cfg.flow_backend, cfg.device)
    pairs = sorted({w.pair for w in windows if w.pair >= 0})
    cache = SourceCache(source, cfg, backend=backend)
    say(f"caching {len(pairs)} source pairs (flows/occlusion/camera)")
    cache.ensure_pairs(pairs, skip=set(int(c) for c in alignment.scene_cuts),
                       progress=lambda n, tot: say(f"source cache {n}/{tot}")
                       if tot and n % max(1, tot // 10) == 0 else None)

    ui = UIDetector(source, cfg) if p.run_region_branches else None

    say("tier-2 core window metrics")
    wfs: list[WindowFeatures] = []
    stage_errors: dict[str, int] = collections.Counter()
    audit_tracker, tracker_note = (get_audit_tracker(cfg.device)
                                   if p.run_tracker else (None, None))
    cut_pair_set = set(int(c) for c in alignment.scene_cuts)
    for n, w in enumerate(windows):
        if n % max(1, len(windows) // 10) == 0:
            say(f"window {n + 1}/{len(windows)}")
        if w.pair < 0 or int(w.pair) in cut_pair_set:
            continue
        bundle = candidate.read_frames(w.indices, width=p.flow_width)
        if bundle.rgb.shape[0] < p.window_frames:
            continue
        wflows = WindowFlows(bundle, backend)
        # All flows this window's metrics need, in one batched backend call.
        wflows.precompute()
        pair = cache.get_pair(w.pair)
        wf = WindowFeatures(window=w)

        stages = [
            ("composition", lambda: flow_composition_metric.compute_with_maps(
                wflows, pair, cfg)),
            ("cycle", lambda: cycle_reconstruction.compute(bundle, wflows, cfg)),
            ("temporal", lambda: temporal_compensation.compute(bundle, wflows, cfg)),
            ("parity", lambda: parity_frequency.compute_window(bundle, wflows, cfg)),
            ("edges", lambda: edge_structure.compute(bundle, wflows, cache, pair, cfg)),
            ("gtq", lambda: global_technical_quality.compute_window(bundle)),
        ]
        if p.run_region_branches and ui is not None:
            stages += [
                ("ui", lambda: ui_detector.compute_window(bundle, ui, cfg)),
                ("text", lambda: text_evaluator.compute_window(bundle, ui, cfg)),
                ("transition", lambda: transition_evaluator.compute_window(bundle, wflows, cfg)),
                ("card", lambda: card_tracker.compute_window(bundle, cfg)),
                ("character", lambda: character_segmenter.compute_window(bundle, wflows, pair, cfg)),
                ("thin", lambda: thin_object_detector.compute_window(bundle, wflows, pair, cfg)),
            ]
        if p.run_tracker:                            # audit-tier point tracking
            stages.append(("weapon", lambda: weapon_tracker.compute_window(
                bundle, wflows, cfg, tracker=audit_tracker,
                roi_mask=wf.scalars.get("_char_mask"))))
        for name, fn in stages:
            try:                                    # one failing stage must not
                result = fn()                       # take down the whole window
                # USERPLAN P1: stages may return (scalars, maps) to populate
                # dense error maps alongside the per-window scalars.
                if isinstance(result, tuple):
                    scalars, maps = result
                    wf.scalars.update(scalars)
                    if maps:
                        wf.error_maps.update(maps)
                else:
                    wf.scalars.update(result)
            except Exception as exc:
                wf.labels[f"error_{name}"] = repr(exc)
                stage_errors[name] += 1

        wfs.append(wf)

    # --- tier 3: audit escalation (§10) -------------------------------------
    # The highest-risk windows get the core metrics recomputed at a higher
    # resolution (sharper flows/edges), capped so runtime and VRAM stay
    # bounded.  USERPLAN §3: ``auto_audit`` lets the *balanced* path escalate
    # high-risk windows automatically without the caller opting into an
    # "audit" preset.
    # USERPLAN §6 P0.2: Tier-3 is *not* native resolution by default — the
    # ``tier3_max_width`` cap keeps RAFT within a safe working grid on 16 GB
    # GPUs.  Native resolution is only used when the source is already below
    # the cap.
    n_audit = 0
    eff_frac = (p.audit_top_fraction if p.audit_top_fraction > 0
                else (p.auto_audit_fraction if p.auto_audit else 0.0))
    eff_max = (p.audit_max_windows if p.audit_max_windows > 0
               else (p.auto_audit_max if p.auto_audit else 0))
    # USERPLAN §6 P0.2: cap the Tier-3 working resolution.
    tier3_width = min(source.meta.width, p.tier3_max_width) \
        if p.tier3_max_width > 0 else source.meta.width
    if eff_frac > 0 and wfs:
        ranked = sorted(wfs, key=lambda wf: -max(
            _window_category_errors(wf).values(), default=0.0))
        n_audit = min(eff_max,
                      max(1, int(round(len(ranked) * eff_frac))))
        audit_wfs = ranked[:n_audit]
        audit_cache = SourceCache(
            source, cfg, flow_width=tier3_width, backend=backend)
        audit_pairs = {wf.window.pair for wf in audit_wfs if wf.window.pair >= 0}
        audit_cache.ensure_pairs(
            audit_pairs, skip=set(int(c) for c in alignment.scene_cuts))
        say(f"tier-3 audit: re-evaluating top {n_audit} windows at width {tier3_width}")
        for wf in audit_wfs:
            try:
                nb = candidate.read_frames(wf.window.indices, width=tier3_width)
                if nb.rgb.shape[0] < p.window_frames:
                    continue
                nw = WindowFlows(nb, backend)
                nw.precompute()
                npair = audit_cache.get_pair(wf.window.pair)
                for name, fn in (
                    ("composition", lambda: flow_composition_metric.compute(nw, npair, cfg)),
                    ("cycle", lambda: cycle_reconstruction.compute(nb, nw, cfg)),
                    ("temporal", lambda: temporal_compensation.compute(nb, nw, cfg)),
                    ("edges", lambda: edge_structure.compute(
                        nb, nw, audit_cache, npair, cfg)),
                ):
                    wf.scalars.update(fn())
                if p.run_tracker:
                    # Rebuild the proxy ROI at the audit resolution before passing
                    # it to OpenCV/CoTracker; never reuse the tier-2 960px mask.
                    wf.scalars.update(character_segmenter.compute_window(
                        nb, nw, npair, cfg))
                    wf.scalars.update(weapon_tracker.compute_window(
                        nb, nw, cfg, tracker=audit_tracker,
                        roi_mask=wf.scalars.get("_char_mask")))
                wf.labels["audited"] = True
            except Exception as exc:
                wf.labels["error_audit"] = repr(exc)

    audit_notes: list[str] = []
    if tracker_note is not None and "fell back" in tracker_note:
        audit_notes.append(tracker_note)
    if p.run_depth:
        try:
            from .models.depth_backend import get_depth_backend
            get_depth_backend("auto")
        except NotImplementedError as exc:
            audit_notes.append(f"depth disabled: {exc}")

    CORE_STAGES = {"composition", "cycle", "temporal", "parity", "edges", "gtq"}
    valid_wfs = [wf for wf in wfs
                 if not any(k.startswith("error_") and k[len("error_"):] in CORE_STAGES
                            for k in wf.labels)]

    say("global features + fusion")
    global_feats: dict[str, float] = {}
    global_feats.update(parity_frequency.compute_global(cheap))
    global_feats.update(global_technical_quality.compute_global(cheap))
    global_feats.update(anchor_feats)

    cat_errors = build_category_errors(wfs, global_feats)
    overall, subscores, A = compute_scores(cfg, cat_errors)
    conf, event_amb = compute_confidence(wfs, alignment, candidate.meta.n_frames,
                                         alignment.anchor_error,
                                         valid_windows=len(valid_wfs),
                                         total_windows=len(wfs))
    if not alignment.reliable:
        overall = float("nan")
        conf = min(conf, 0.01)

    # Optional trained calibrator (§11.4) — never overrides a fail-closed NaN.
    active_calibrator_path = calibrator_path or p.calibrator_path
    calibrator = maybe_load(active_calibrator_path)
    endpoint_feature_contract = feature_contract_hash("endpoint-2x")
    calibrator_contract = calibrator_provenance(
        active_calibrator_path,
        meta=(calibrator.meta if calibrator else None),
        expected_feature_contract_hash=endpoint_feature_contract,
    )
    if (calibrator_contract is not None
            and calibrator_contract["feature_contract_hash"] not in (
                "unverified", endpoint_feature_contract)):
        raise ValueError(
            "calibrator feature contract does not match the current "
            "endpoint feature contract")
    if calibrator is not None and overall == overall:
        feats = {f"A_{cat}": A.get(cat, float("nan")) for cat in CATEGORY_FEATURES}
        feats = {k: (v if v == v else 0.0) for k, v in feats.items()}
        feats["confidence"] = conf
        feats["event_ambiguity"] = event_amb
        overall = calibrator.predict_quality(feats)

    # Worst windows with real timestamps.
    worst: list[WorstWindow] = []
    for wf in wfs:
        ww = _classify_window(wf, conf)
        if ww is None:
            continue
        t0, t1 = window_times(wf.window, candidate.meta)
        ww.start, ww.end = t0, t1
        worst.append(ww)
    worst.sort(key=lambda x: -x.severity)
    # Temporal dedup of reported worst windows.
    kept: list[WorstWindow] = []
    for ww in worst:
        if all(abs(ww.start - k.start) > p.temporal_nms_seconds for k in kept):
            kept.append(ww)
        if len(kept) >= 12:
            break
    worst = kept

    # USERPLAN §7: structured multi-evidence diagnosis (endpoint scalars).
    # USERPLAN P0-1: build per-window dense error maps so issues can carry
    # spatial boxes extracted from the real diagnostic evidence.
    # USERPLAN P0-6: pass every dense field per window to diagnosis; the map
    # best matching each fired rule's issue type is chosen *inside*
    # diagnose_windows (preferred_maps_for), so the boxes come from the same
    # field the report renders.  No fixed-priority pre-selection here.
    by_center = {int(wf.window.center): wf for wf in wfs}
    ep_error_maps: dict[int, dict[str, np.ndarray]] = {}
    for wf in wfs:
        center = int(wf.window.center)
        if not wf.error_maps:
            continue
        ep_error_maps[center] = wf.error_maps

    ep_diag_ws = [
        (window_times(wf.window, candidate.meta)[0],
         window_times(wf.window, candidate.meta)[1],
         int(wf.window.center), wf.scalars,
         float(wf.labels.get("metric_confidence", conf)))
        for wf in wfs
    ]
    ep_issues = diagnose_windows(ep_diag_ws, p.temporal_nms_seconds,
                                 error_maps=ep_error_maps)

    # USERPLAN P0-6: record, for each endpoint issue, the map name used for
    # its boxes so the rendered heatmap matches the annotation.
    for issue in ep_issues:
        wf = by_center.get(int(issue.center_index))
        if wf is None:
            continue
        name = select_map_name(wf.error_maps, issue.issue_type)
        if name is not None:
            issue.maps = [name]

    # Feature export: robust per-key median across windows + globals.
    feat_out: dict[str, float] = {}
    if wfs:
        keys = set().union(*[set(wf.scalars) for wf in wfs])
        for k in sorted(keys):
            if k.startswith("_"):
                continue
            vals = [wf.scalars[k] for wf in wfs
                    if k in wf.scalars and wf.scalars[k] == wf.scalars[k]]
            if vals:
                feat_out[k] = round(float(np.median(vals)), 5)
    for k, v in global_feats.items():
        if v == v:
            feat_out[f"global_{k}"] = round(float(v), 5)

    status = ("failed" if overall != overall
              else "degraded" if stage_errors or len(valid_wfs) < len(wfs)
              else "ok")
    error_samples = {}
    for wf in wfs:
        for k, v in wf.labels.items():
            if k.startswith("error_") and k not in error_samples:
                error_samples[k] = v
    # Honest capability declaration (USERPLAN §P2): these branches are
    # classical heuristics until learned game-domain backends are registered;
    # their scores are proxy evidence, not semantic ground truth.
    proxy_branches = {
        "character": "classic_motion_segmenter",
        "thin_object": "lsd_line_heuristic",
        "weapon": "klt_local_corners",
        "ui": "static_mask_template",
        "text": "stroke_edge_topology",
        "card": "saturation_area_progression",
    }
    if p.run_tracker and tracker_note == "cotracker":
        proxy_branches["weapon"] = "cotracker2"   # learned tracker, not a proxy
    if not p.run_region_branches:
        proxy_branches = {k: v + " (disabled by preset)"
                          for k, v in proxy_branches.items()}

    meta = {
        "mode": "endpoint-2x",
        "score_schema": "endpoint-reduced-reference-v3",
        "score_semantics": "endpoint-referenced interpolation quality",
        "limitations": [
            "Source frames constrain endpoints but are not ground-truth intermediate frames.",
            "The formula score requires real-data calibration before production gating.",
        ],
        "status": status,
        "valid_windows": len(valid_wfs),
        "audited_windows": n_audit,
        "audit_notes": audit_notes,
        "proxy_branches": proxy_branches,
        "stage_errors": dict(sorted(stage_errors.items())),
        "stage_error_samples": error_samples,
        "preset": p.name,
        "flow_backend": getattr(backend, "name", cfg.flow_backend),
        "source_cache": {
            "path": str(cache.root),
            "contract": cache.contract,
        },
        "device": cfg.device,
        "source": {"path": source.meta.path, "fps": source.meta.fps,
                   "frames": source.meta.n_frames, "hash": source.meta.content_hash},
        "candidate": {"path": candidate.meta.path, "fps": candidate.meta.fps,
                      "frames": candidate.meta.n_frames,
                      "hash": candidate.meta.content_hash},
        "alignment": {"fps_ratio": alignment.fps_ratio,
                      "first_anchor_offset": alignment.first_anchor_offset,
                      "anchor_error": round(alignment.anchor_error, 3),
                      "reliable": alignment.reliable,
                      "events": alignment.events,
                      "scene_cut_pairs": [int(c) for c in alignment.scene_cuts],
                      "warnings": alignment.warnings},
        "windows_evaluated": len(wfs),
        "windows_selected": len(windows),
        "coverage_fraction": round(
            len({
                int(frame)
                for wf in valid_wfs
                for frame in wf.window.indices
            }) / max(candidate.meta.n_frames, 1),
            5,
        ),
        "category_errors": {k: round(v, 4) if v == v else None
                            for k, v in A.items()},
        "category_valid_windows": {
            category: sum(
                1 for wf in wfs if category_window_valid(wf, category))
            for category in CATEGORY_FEATURES
        },
        "calibrator": "lightgbm" if calibrator else "formula",
        "calibrator_contract": calibrator_contract,
        "diagnostics": build_diagnostics_block(
            ep_issues, candidate.meta, conf, overall=overall),
        "elapsed_seconds": round(time.perf_counter() - t_start, 2),
    }
    meta.update(report_provenance(
        mode="endpoint-2x",
        score_schema="endpoint-reduced-reference-v3",
        metric_contract="endpoint-metrics-v3",
        preset_contract=f"endpoint-{p.name}-v1",
        feature_contract_hash=endpoint_feature_contract,
        backend_contract={"flow": backend.cache_identity()},
        calibration_id=(
            f"sha256:{calibrator_contract['sha256'][:16]}"
            if calibrator_contract else None),
    ))

    # USERPLAN §6 P0.5: collect backend performance telemetry for the report.
    from .motion.flow_estimator import collect_backend_telemetry
    performance = collect_backend_telemetry(
        backend,
        tier2_flow_width=p.flow_width,
        tier3_flow_width=tier3_width,
    )

    report = Report(overall_score=overall,
                    confidence=conf, scores=subscores,
                    event_ambiguity=event_amb, worst_windows=worst,
                    features=feat_out, meta=meta,
                    performance=performance)

    if out_dir:
        say("writing reports")
        out = Path(out_dir)
        ep_diag_issues = (meta.get("diagnostics") or {}).get("issues") or []

        # Build per-issue error maps + dimensions for the unified pipeline.
        ep_overlay_maps: dict[int, np.ndarray] = {}
        ep_map_dims: dict[int, tuple[int, int]] = {}
        for ei, eiss in enumerate(ep_diag_issues):
            wf_e = by_center.get(int(eiss.get("center_index", -1)))
            if wf_e is None:
                continue
            for nm in (eiss.get("map_keys") or eiss.get("maps") or []):
                arr = wf_e.error_maps.get(nm)
                if arr is not None:
                    ep_overlay_maps[ei] = arr
                    ep_map_dims[ei] = (int(arr.shape[0]), int(arr.shape[1]))
                    eiss["_map_h"] = int(arr.shape[0])
                    eiss["_map_w"] = int(arr.shape[1])
                    break

        ctx = ArtifactContext(
            mode=EvaluationMode.ENDPOINT_2X,
            candidate_video=candidate_video,
            reference_video=reference_video,
            windows=wfs,
            issues=ep_diag_issues,
            output_dir=out,
            candidate_meta=candidate.meta,
            worst_windows=worst,
            export_clips=export_clips,
            error_maps=ep_overlay_maps,
            map_dimensions=ep_map_dims,
        )
        build_report_artifacts(
            ctx, report,
            render_timeline_md_fn=render_timeline_md,
            render_timeline_png_fn=render_timeline_png,
        )
        say(f"reports written to {out}")

    return report


def evaluate_vfi(
    source_video: str,
    candidate_video: str,
    preset: str = "standard",
    **kwargs,
) -> Report:
    """Backward-compatible name for :func:`evaluate_endpoint_reference`.

    New integrations should call ``evaluate(..., mode="endpoint-2x")`` or the
    explicit executor so the mathematical contract is visible at the callsite.
    """
    return evaluate_endpoint_reference(
        source_video,
        candidate_video,
        preset=preset,
        **kwargs,
    )


def compare_models(source_video: str, candidate_videos: list[str],
                   preset: str = "standard", cache_dir: str = "./cache",
                   device: str = "cuda", out_dir: str | None = None,
                   labels: list[str] | None = None,
                   flow_backend: str = "auto",
                   progress: ProgressFn | None = None) -> list[dict]:
    """Rank multiple interpolation models on the same source (§11.5).

    The source cache is shared automatically (keyed by content hash), so the
    cost is ~T_source + N · T_candidate.
    """
    labels = labels or [Path(c).stem for c in candidate_videos]
    if len(labels) != len(candidate_videos):
        raise ValueError("labels must have the same length as candidate_videos")
    results = []
    for lab, cand in zip(labels, candidate_videos):
        rep = evaluate_vfi(source_video, cand, preset=preset, cache_dir=cache_dir,
                           device=device, flow_backend=flow_backend,
                           out_dir=str(Path(out_dir) / lab) if out_dir else None,
                           export_clips=False, progress=progress)
        valid = bool(np.isfinite(rep.overall_score)) and \
            rep.meta.get("status") != "failed"
        result_status = rep.meta.get("status", "ok") if valid else "failed"
        results.append({"model": lab,
                        "overall": float(rep.overall_score) if valid else None,
                        "confidence": rep.confidence, "scores": rep.scores})
        results[-1]["status"] = result_status
        results[-1]["relative_vs_mean"] = None
    results.sort(key=lambda r: (r["status"] == "failed",
                                -(r["overall"] if r["overall"] is not None
                                  else float("-inf"))))
    valid_scores = [r["overall"] for r in results if r["status"] != "failed"]
    mean = float(np.mean(valid_scores)) if valid_scores else None
    for r in results:
        if r["status"] != "failed" and mean is not None:
            r["relative_vs_mean"] = round(r["overall"] - mean, 2)
    if out_dir:
        import json
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "ranking.json").write_text(
            json.dumps(results, indent=2), encoding="utf-8")
    return results
