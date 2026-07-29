"""Model-ranking validation on the pseudo-GT interpolator ladder (USERPLAN
§12.1 / §P3).

Unlike the hand-crafted defect corpora, the candidates here are *model
outputs* — a naive linear blender and a flow-based interpolator run at
several internal resolutions — ranked against full-reference quality on the
truth mid frames, across several scene seeds. This validates the actual
product use case: same-source model ranking.

Reports:
* per-row FR PSNR vs endpoint-reference overall;
* SRCC / PLCC / pairwise accuracy of the evaluator ranking vs FR ranking;
* per-feature correlation strength with FR quality (|SRCC| table) — the
  feature-level acceptance view of §P3.

    python -m rr_vfiqa.calibration.model_validation --workdir mv_run
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..pipeline import evaluate_vfi
from ..testing.interpolator import make_ladder
from ..testing.synth import (make_source_and_truth, render_scene, write_video)
from .pseudo_gt import full_reference_quality
from .validate import pairwise_accuracy, plcc, srcc


def run_model_validation(workdir: str | Path, seeds: tuple[int, ...] = (7, 21, 43),
                         n_frames: int = 240, w: int = 640, h: int = 360,
                         interp_backend: str = "auto", device: str = "cuda",
                         widths: tuple[int | None, ...] = (320, None),
                         eval_preset: str = "fast",
                         eval_backend: str = "farneback") -> dict:
    workdir = Path(workdir)
    cache = str(workdir / "cache")
    rows = []
    feature_table: dict[str, list[tuple[float, float]]] = {}

    for seed in seeds:
        frames, _meta = render_scene(n_frames=n_frames, w=w, h=h, seed=seed,
                                     return_meta=True)
        source, truth = make_source_and_truth(frames)
        src_path = write_video(workdir / f"source_{seed}_60.mp4", source, 60)
        ladder = make_ladder(source, truth, flow_backend=interp_backend,
                             device=device, widths=widths)
        for model, cand_frames in ladder.items():
            fr = full_reference_quality(cand_frames, truth)
            cand_path = write_video(workdir / f"cand_{seed}_{model}_120.mp4",
                                    cand_frames, 120)
            rep = evaluate_vfi(str(src_path), str(cand_path), preset=eval_preset,
                               cache_dir=cache, device=device,
                               flow_backend=eval_backend, out_dir=None,
                               export_clips=False)
            overall = float(rep.overall_score)
            rows.append({"seed": seed, "model": model,
                         "fr_psnr": round(fr["psnr_db"], 2),
                         "overall": round(overall, 2)})
            for k, v in rep.features.items():
                if isinstance(v, (int, float)) and v == v and not k.startswith("global_"):
                    feature_table.setdefault(k, []).append((fr["psnr_db"], float(v)))

    psnr = np.array([r["fr_psnr"] for r in rows])
    overall = np.array([r["overall"] for r in rows])

    # Per-feature correlation strength with FR quality (direction-free: a
    # good error feature correlates negatively, a quality feature positively).
    # Need ≥6 candidates for meaningful SRCC, but never more than exist.
    feat_corr = {}
    min_pts = min(len(rows), 6)
    for k, pts in feature_table.items():
        if len(pts) < min_pts:
            continue
        x = np.array([p[0] for p in pts])
        y = np.array([p[1] for p in pts])
        if np.std(x) < 1e-9 or np.std(y) < 1e-12:
            continue
        feat_corr[k] = round(abs(float(srcc(x, y))), 3)
    top_features = sorted(feat_corr.items(), key=lambda kv: -kv[1])[:15]

    return {
        "rows": rows,
        "srcc": round(float(srcc(psnr, overall)), 4),
        "plcc": round(float(plcc(psnr, overall)), 4),
        "pairwise_accuracy": round(float(pairwise_accuracy(overall, psnr)), 4),
        "within_seed_pairwise": _within_seed_pairwise(rows),
        "top_feature_correlations": dict(top_features),
        "feature_correlations": feat_corr,   # full table, unsorted
    }


def _within_seed_pairwise(rows: list[dict]) -> float:
    """Same-source ranking accuracy — the primary product metric (§11.5)."""
    ok = tot = 0
    by_seed: dict[int, list[dict]] = {}
    for r in rows:
        by_seed.setdefault(r["seed"], []).append(r)
    for group in by_seed.values():
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a, b = group[i], group[j]
                if a["fr_psnr"] == b["fr_psnr"]:
                    continue
                tot += 1
                if np.sign(a["overall"] - b["overall"]) == \
                        np.sign(a["fr_psnr"] - b["fr_psnr"]):
                    ok += 1
    return round(ok / tot, 4) if tot else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workdir", default="./model_validation")
    ap.add_argument("--seeds", nargs="*", type=int, default=[7, 21, 43])
    ap.add_argument("--interp-backend", default="auto")
    ap.add_argument("--eval-backend", default="farneback")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    out = run_model_validation(args.workdir, seeds=tuple(args.seeds),
                               interp_backend=args.interp_backend,
                               eval_backend=args.eval_backend, device=args.device)
    print(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nSRCC={out['srcc']}  PLCC={out['plcc']}  "
          f"pairwise={out['pairwise_accuracy']}  "
          f"within-seed={out['within_seed_pairwise']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
