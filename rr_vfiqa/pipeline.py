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
from .config import EvalConfig
from .fusion import (build_category_errors, compute_confidence, compute_scores,
                     maybe_load)
from .fusion.feature_normalizer import normalize_error
from .fusion.score_schema import CATEGORY_FEATURES
from .io.timestamp_alignment import build_alignment
from .io.video_reader import VideoReader
from .metrics import (anchor_integrity, cycle_reconstruction, edge_structure,
                      flow_composition_metric, global_technical_quality,
                      parity_frequency, temporal_compensation, WindowFlows)
from .motion.flow_estimator import get_flow_backend
from .regions import (character_segmenter, text_evaluator, thin_object_detector,
                      transition_evaluator, ui_detector, weapon_tracker)
from .regions.ui_detector import UIDetector
from .report import (export_badcase_clips, render_timeline_md, render_timeline_png,
                     save_error_heatmap, write_json_report)
from .sampling.cheap_scan import scan_candidate
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
    # Only the three loudest categories may label a window.
    firing = sorted(((c, t) for c, t in _TYPE_THRESHOLDS
                     if cat_err.get(c, 0.0) >= t),
                    key=lambda ct: -cat_err[ct[0]])[:3]
    for cat, thr in firing:
        if cat == "motion":
            types.append("background_tear" if s.get("flow_fold_frac", 0) > 0.03
                         else "motion_inconsistent")
        elif cat == "temporal":
            # parity_flicker needs direct alternation evidence, else generic
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
            types.append("transition_ghost")
    if not types:
        return None
    severity = float(np.clip(max(cat_err.values()) / 1.2, 0, 1))
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

def evaluate_vfi(source_video: str, candidate_video: str, preset: str = "standard",
                 cache_dir: str = "./cache", device: str = "cuda",
                 out_dir: str | None = None, flow_backend: str = "auto",
                 export_clips: bool = True, calibrator_path: str | None = None,
                 progress: ProgressFn | None = None) -> Report:
    """Full endpoint-referenced VFI quality evaluation (§13 interface)."""
    t_start = time.perf_counter()

    def say(msg: str) -> None:
        if progress:
            progress(msg)

    cfg = EvalConfig.build(source_video, candidate_video, preset,
                           cache_dir=cache_dir, device=device,
                           flow_backend=flow_backend)
    p = cfg.preset

    say("opening videos")
    source = VideoReader(source_video)
    candidate = VideoReader(candidate_video)

    say("aligning candidate to source (PTS + anchors)")
    alignment = build_alignment(cfg, source, candidate)

    say("anchor integrity + color baseline")
    anchor_feats, color_tf, anchor_warnings = anchor_integrity.evaluate_anchors(
        cfg, source, candidate, alignment)
    alignment.warnings.extend(anchor_warnings)

    say("tier-1 cheap full-frame scan")
    cheap = scan_candidate(candidate, width=p.scan_width)

    say("selecting windows")
    windows, risk, _ = select_windows(cfg, candidate.meta, alignment, cheap)
    say(f"{len(windows)} windows selected")

    pairs = sorted({w.pair for w in windows if w.pair >= 0})
    cache = SourceCache(source, cfg)
    say(f"caching {len(pairs)} source pairs (flows/occlusion/camera)")
    cache.ensure_pairs(pairs, skip=set(int(c) for c in alignment.scene_cuts),
                       progress=lambda n, tot: say(f"source cache {n}/{tot}")
                       if tot and n % max(1, tot // 10) == 0 else None)

    backend = get_flow_backend(cfg.flow_backend, cfg.device)
    ui = UIDetector(source, cfg) if p.run_region_branches else None

    say("tier-2 core window metrics")
    wfs: list[WindowFeatures] = []
    stage_errors: dict[str, int] = collections.Counter()
    for n, w in enumerate(windows):
        if n % max(1, len(windows) // 10) == 0:
            say(f"window {n + 1}/{len(windows)}")
        if w.pair < 0 or int(w.pair) in set(int(c) for c in alignment.scene_cuts):
            continue
        bundle = candidate.read_frames(w.indices, width=p.flow_width)
        if bundle.rgb.shape[0] < p.window_frames:
            continue
        wflows = WindowFlows(bundle, backend)
        pair = cache.get_pair(w.pair)
        wf = WindowFeatures(window=w)

        stages = [
            ("composition", lambda: flow_composition_metric.compute(wflows, pair, cfg)),
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
                ("character", lambda: character_segmenter.compute_window(bundle, wflows, pair, cfg)),
                ("thin", lambda: thin_object_detector.compute_window(bundle, wflows, pair, cfg)),
                ("weapon", lambda: weapon_tracker.compute_window(bundle, wflows, cfg)),
            ]
        for name, fn in stages:
            try:                                    # one failing stage must not
                wf.scalars.update(fn())             # take down the whole window
            except Exception as exc:
                wf.labels[f"error_{name}"] = repr(exc)
                stage_errors[name] += 1

        wfs.append(wf)

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

    # Optional trained calibrator (§11.4) — never overrides a fail-closed NaN.
    calibrator = maybe_load(calibrator_path or p.calibrator_path)
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
    meta = {
        "status": status,
        "valid_windows": len(valid_wfs),
        "stage_errors": dict(sorted(stage_errors.items())),
        "stage_error_samples": error_samples,
        "preset": p.name,
        "flow_backend": getattr(backend, "name", cfg.flow_backend),
        "device": cfg.device,
        "source": {"path": source.meta.path, "fps": source.meta.fps,
                   "frames": source.meta.n_frames, "hash": source.meta.content_hash},
        "candidate": {"path": candidate.meta.path, "fps": candidate.meta.fps,
                      "frames": candidate.meta.n_frames,
                      "hash": candidate.meta.content_hash},
        "alignment": {"fps_ratio": alignment.fps_ratio,
                      "first_anchor_offset": alignment.first_anchor_offset,
                      "anchor_error": round(alignment.anchor_error, 3),
                      "scene_cut_pairs": [int(c) for c in alignment.scene_cuts],
                      "warnings": alignment.warnings},
        "windows_evaluated": len(wfs),
        "windows_selected": len(windows),
        "coverage_fraction": round(len(wfs) * p.window_frames /
                                   max(candidate.meta.n_frames, 1), 5),
        "category_errors": {k: round(v, 4) if v == v else None
                            for k, v in A.items()},
        "calibrator": "lightgbm" if calibrator else "formula",
        "elapsed_seconds": round(time.perf_counter() - t_start, 2),
    }

    report = Report(overall_score=overall,
                    confidence=conf, scores=subscores,
                    event_ambiguity=event_amb, worst_windows=worst,
                    features=feat_out, meta=meta)

    if out_dir:
        say("writing reports")
        out = Path(out_dir)
        write_json_report(report, out / "report.json")
        (out / "timeline.md").write_text(
            render_timeline_md(wfs, candidate.meta, report), encoding="utf-8")
        render_timeline_png(wfs, candidate.meta, report, out / "timeline.png")
        if export_clips and worst:
            export_badcase_clips(candidate_video, worst, out / "badcases",
                                 out_fps=source.meta.fps * 2)
        say(f"reports written to {out}")

    return report


def compare_models(source_video: str, candidate_videos: list[str],
                   preset: str = "standard", cache_dir: str = "./cache",
                   device: str = "cuda", out_dir: str | None = None,
                   labels: list[str] | None = None,
                   progress: ProgressFn | None = None) -> list[dict]:
    """Rank multiple interpolation models on the same source (§11.5).

    The source cache is shared automatically (keyed by content hash), so the
    cost is ~T_source + N · T_candidate.
    """
    labels = labels or [Path(c).stem for c in candidate_videos]
    results = []
    for lab, cand in zip(labels, candidate_videos):
        rep = evaluate_vfi(source_video, cand, preset=preset, cache_dir=cache_dir,
                           device=device, flow_backend="auto",
                           out_dir=str(Path(out_dir) / lab) if out_dir else None,
                           export_clips=False, progress=progress)
        results.append({"model": lab, "overall": rep.overall_score,
                        "confidence": rep.confidence, "scores": rep.scores})
    results.sort(key=lambda r: -r["overall"])
    mean = float(np.mean([r["overall"] for r in results])) if results else 0.0
    for r in results:
        r["relative_vs_mean"] = round(r["overall"] - mean, 2)
    if out_dir:
        import json
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "ranking.json").write_text(
            json.dumps(results, indent=2), encoding="utf-8")
    return results
