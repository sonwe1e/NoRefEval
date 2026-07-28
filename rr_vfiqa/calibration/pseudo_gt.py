"""Pseudo-ground-truth corpora for calibrating endpoint-reference metrics.

Real 120/240 FPS captures downsampled to 60 FPS leave the removed frames as
ground truth (USERPLAN §12.1). This module provides:

* full-reference quality measurement of candidate mid frames vs truth;
* a synthetic severity ladder (more defect types per level → monotonically
  worse FR quality) for smoke-validating the evaluator without real data.

Real-data calibration still requires actual HFR captures and human A/B
rankings — this harness supplies the machinery, not the data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..testing.synth import (DEFECTS, interleave, make_defective_mids,
                             make_source_and_truth, render_scene, write_video)


@dataclass
class CorpusEntry:
    level: int
    candidate_path: Path
    fr_psnr: float                      # dB, over all mid frames vs truth
    fr_luma_err: float                  # mean Y L1
    defects: list[str] = field(default_factory=list)


def full_reference_quality(candidate_frames: np.ndarray, truth_mids: np.ndarray
                           ) -> dict[str, float]:
    """FR quality of a candidate's ODD frames against the true mid frames."""
    mids = candidate_frames[1::2][: len(truth_mids)].astype(np.float64)
    t = truth_mids.astype(np.float64)
    wy = np.array([0.299, 0.587, 0.114])
    y_m = (mids * wy).sum(-1)
    y_t = (t * wy).sum(-1)
    mse = float(np.mean((mids - t) ** 2))
    luma_err = float(np.mean(np.abs(y_m - y_t)))
    psnr = 10.0 * np.log10(255.0 ** 2 / max(mse, 1e-9))
    return {"psnr_db": psnr, "luma_err": luma_err}


_SEVERITY_AXIS = ["blur", "ghost", "freeze", "rotation_tear", "head_erase"]


def build_severity_corpus(out_dir: str | Path, levels: int = 6,
                          n_frames: int = 120, w: int = 320, h: int = 192,
                          seed: int = 7) -> tuple[Path, list[CorpusEntry]]:
    """Level k applies the first k defects of the severity axis.

    FR quality is monotonically non-increasing in k by construction (each
    level corrupts strictly more mid frames). Returns (out_dir, entries).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for d in _SEVERITY_AXIS:
        assert d in DEFECTS
    frames, meta = render_scene(n_frames=n_frames, w=w, h=h, seed=seed,
                                return_meta=True)
    source, truth = make_source_and_truth(frames)
    src_path = write_video(out_dir / "source_60.mp4", source, 60)

    entries = []
    for k in range(levels):
        defects = _SEVERITY_AXIS[:k]
        if defects:
            mids, _ = make_defective_mids(source, truth, meta, defects, seed=seed + k)
            cand = interleave(source, mids)
        else:
            cand = frames
        cand_path = write_video(out_dir / f"cand_level{k}.mp4", cand, 120)
        fr = full_reference_quality(cand, truth)
        entries.append(CorpusEntry(level=k, candidate_path=cand_path,
                                   fr_psnr=fr["psnr_db"],
                                   fr_luma_err=fr["luma_err"],
                                   defects=defects))
    return src_path, entries
