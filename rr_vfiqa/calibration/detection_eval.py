"""Per-defect-category detection evaluation (USERPLAN §P3 acceptance:
各坏例类别 AP/F1 and worst-10% recall).

For each defect type a single-defect candidate is generated with a known
time segment; the evaluator's worst-window labels are then scored against
that ground truth:

* recall    — fraction of defects for which a correctly-typed window fired
              within the segment (± tolerance);
* precision — fraction of fired windows whose labels are admissible for the
              defect present in the video;
* F1        — their harmonic mean.

On synthetic content this validates *localization ability*; absolute numbers
on real game content require the §P3 corpora.
"""

from __future__ import annotations

from pathlib import Path

from ..pipeline import evaluate_vfi
from ..testing.synth import (interleave, make_defective_mids,
                             make_source_and_truth, render_scene, write_video)
from .provenance import calibration_provenance

# Admissible worst-window labels per defect type.
DEFECT_TO_TYPES: dict[str, set[str]] = {
    "blur": {"parity_flicker", "temporal_instability", "structure_loss"},
    "ghost": {"parity_flicker", "temporal_instability", "transition_ghost"},
    "freeze": {"parity_flicker", "temporal_instability", "freeze_copy"},
    "rotation_tear": {"background_tear", "motion_inconsistent", "structure_loss"},
    "head_erase": {"character_missing"},
    "pole_wrong_motion": {"thin_object_stick"},
    "sword_flicker": {"weapon_jitter", "temporal_instability"},
    "ui_drift": {"ui_unstable", "text_degraded"},
    "text_merge": {"text_degraded", "ui_unstable"},
    "shop_jump": {"transition_ghost"},
    "card_freeze": {"card_flip_error"},
    "disocc_fill": {"structure_loss", "motion_inconsistent"},
}

_DEFAULT_DEFECTS = tuple(DEFECT_TO_TYPES)


def _is_temporal_typed_hit(ww, t0: float, t1: float,
                           expected: set[str]) -> bool:
    return bool(ww.start <= t1 and ww.end >= t0
                and expected.intersection(ww.types))


def run_detection_eval(workdir: str | Path, defects: list[str] | None = None,
                       preset: str = "standard", flow_backend: str = "farneback",
                       device: str = "cuda", n_frames: int = 160,
                       w: int = 320, h: int = 192, tol_s: float = 0.15) -> dict:
    workdir = Path(workdir)
    defects = list(defects) if defects is not None else list(_DEFAULT_DEFECTS)
    unknown = sorted(set(defects) - set(DEFECT_TO_TYPES))
    if unknown:
        raise ValueError(f"unknown defects: {unknown}")
    frames, meta = render_scene(n_frames=n_frames, w=w, h=h, seed=7,
                                return_meta=True)
    source, truth = make_source_and_truth(frames)
    src_path = write_video(workdir / "source_60.mp4", source, 60)
    fps = 120.0

    per_defect = {
        d: {
            "status": "not_evaluated",
            "detected": None,
            "expected_types": sorted(DEFECT_TO_TYPES[d]),
        }
        for d in DEFECT_TO_TYPES
    }
    total_windows = labeled_admissible = 0
    for d in defects:
        mids, segs = make_defective_mids(source, truth, meta, [d], seed=11)
        cand = interleave(source, mids)
        cand_path = write_video(workdir / f"cand_{d}_120.mp4", cand, fps)
        rep = evaluate_vfi(str(src_path), str(cand_path), preset=preset,
                           cache_dir=str(workdir / "cache"), device=device,
                           flow_backend=flow_backend, out_dir=None,
                           export_clips=False)
        a, b = segs[d]
        duration_s = len(cand) / fps
        effective_tol = min(max(tol_s, 0.0), 0.10 * duration_s)
        t0 = (2 * a + 1) / fps - effective_tol
        t1 = (2 * b + 1) / fps + effective_tol
        expected = DEFECT_TO_TYPES[d]
        hits = [ww for ww in rep.worst_windows
                if _is_temporal_typed_hit(ww, t0, t1, expected)]
        per_defect[d] = {
            "status": "evaluated",
            "segment_s": [round((2 * a + 1) / fps, 3), round((2 * b + 1) / fps, 3)],
            "tolerance_s": round(effective_tol, 4),
            "detected": len(hits) > 0,
            "typed_windows_in_segment": len(hits),
            "expected_types": sorted(expected),
            "raw_fired_windows": [
                {
                    "start": round(float(ww.start), 4),
                    "end": round(float(ww.end), 4),
                    "types": list(ww.types),
                    "severity": round(float(ww.severity), 4),
                    "confidence": round(float(ww.confidence), 4),
                    "true_positive": _is_temporal_typed_hit(
                        ww, t0, t1, expected),
                }
                for ww in rep.worst_windows
            ],
        }
        for ww in rep.worst_windows:
            total_windows += 1
            if _is_temporal_typed_hit(ww, t0, t1, expected):
                labeled_admissible += 1

    evaluated = [per_defect[d] for d in defects]
    detected = sum(bool(v["detected"]) for v in evaluated)
    recall = detected / max(len(evaluated), 1)
    precision = labeled_admissible / max(total_windows, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return {
        "provenance": calibration_provenance(
            "synthetic_defect_localization", preset=preset,
            flow_backend=flow_backend, device=device,
            dataset={
                "defects": defects, "n_frames": n_frames,
                "width": w, "height": h, "seed": 7,
            }),
        "per_defect": per_defect,
        "recall": round(recall, 4),
        "precision": round(precision, 4),
        "f1": round(f1, 4),
        "fired_windows": total_windows,
    }


def main() -> int:
    import argparse
    import json

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workdir", default="./detection_eval")
    ap.add_argument("--preset", default="standard")
    ap.add_argument("--flow-backend", default="farneback")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--defects", nargs="*", default=None,
                    help=f"choices: {sorted(DEFECT_TO_TYPES)}")
    ap.add_argument("--output", default=None,
                    help="write the complete reproducible JSON artifact")
    args = ap.parse_args()
    out = run_detection_eval(args.workdir, defects=args.defects,
                             preset=args.preset, flow_backend=args.flow_backend,
                             device=args.device)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(
            json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nrecall={out['recall']}  precision={out['precision']}  f1={out['f1']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
