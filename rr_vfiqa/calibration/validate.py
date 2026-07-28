"""Validation metrics and synthetic calibration demo (USERPLAN §12.5 / §P3).

Computes how well the endpoint-reference score tracks full-reference quality
on a pseudo-GT corpus: SRCC, PLCC and A/B pairwise accuracy. Run the demo:

    python -m rr_vfiqa.calibration.validate --synthetic

Real acceptance (SRCC on human rankings, leave-one-game-out, etc.) requires
the real-data corpora from §P3 — the harness here validates the pipeline and
provides bootstrap numbers on synthetic content.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..pipeline import evaluate_vfi
from .pseudo_gt import build_severity_corpus


def srcc(x: np.ndarray, y: np.ndarray) -> float:
    from scipy.stats import spearmanr
    return float(spearmanr(x, y).statistic)


def plcc(x: np.ndarray, y: np.ndarray) -> float:
    from scipy.stats import pearsonr
    return float(pearsonr(x, y).statistic)


def pairwise_accuracy(scores: np.ndarray, quality: np.ndarray) -> float:
    """Fraction of pairs ranked consistently with ground-truth quality."""
    n = len(scores)
    ok = tot = 0
    for i in range(n):
        for j in range(i + 1, n):
            if quality[i] == quality[j]:
                continue
            tot += 1
            if np.sign(scores[i] - scores[j]) == np.sign(quality[i] - quality[j]):
                ok += 1
    return float(ok / tot) if tot else float("nan")


def run_synthetic_validation(workdir: str | Path, preset: str = "fast",
                             flow_backend: str = "farneback",
                             device: str = "cuda", levels: int = 6
                             ) -> dict:
    """Build the severity ladder, evaluate every level, correlate with FR."""
    workdir = Path(workdir)
    src_path, entries = build_severity_corpus(workdir / "corpus", levels=levels)
    cache = str(workdir / "cache")

    rows = []
    for e in entries:
        rep = evaluate_vfi(str(src_path), str(e.candidate_path), preset=preset,
                           cache_dir=cache, device=device,
                           flow_backend=flow_backend, out_dir=None,
                           export_clips=False)
        rows.append({
            "level": e.level,
            "defects": e.defects,
            "fr_psnr": round(e.fr_psnr, 2),
            "overall": round(float(rep.overall_score), 2),
            "confidence": round(float(rep.confidence), 3),
            "scores": {k: round(float(v), 1) if v == v else None
                       for k, v in rep.scores.items()},
            "elapsed_s": rep.meta.get("elapsed_seconds"),
        })

    scores = np.array([r["overall"] for r in rows])
    quality = np.array([r["fr_psnr"] for r in rows])
    summary = {
        "rows": rows,
        "srcc_overall_vs_psnr": round(srcc(scores, quality), 4),
        "plcc_overall_vs_psnr": round(plcc(scores, quality), 4),
        "pairwise_accuracy": round(pairwise_accuracy(scores, quality), 4),
    }
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--synthetic", action="store_true",
                    help="run the synthetic severity-ladder validation")
    ap.add_argument("--workdir", default="./calibration_run")
    ap.add_argument("--preset", default="fast")
    ap.add_argument("--flow-backend", default="farneback")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--levels", type=int, default=6)
    args = ap.parse_args()
    if not args.synthetic:
        ap.error("currently only --synthetic is implemented; real corpora "
                 "follow §P3")
    out = run_synthetic_validation(args.workdir, preset=args.preset,
                                   flow_backend=args.flow_backend,
                                   device=args.device, levels=args.levels)
    print(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nSRCC={out['srcc_overall_vs_psnr']}  PLCC={out['plcc_overall_vs_psnr']}"
          f"  pairwise={out['pairwise_accuracy']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
