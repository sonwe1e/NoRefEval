"""Performance benchmark: per-candidate timing, cache speedup, peak VRAM.

    python -m rr_vfiqa.benchmark --source src_60.mp4 \
        --candidates a_120.mp4 b_120.mp4 c_120.mp4 --preset fast

Reports the §P3 performance numbers: single-candidate wall time, the speedup
of the 2nd/3rd/Nth candidate from source-cache reuse, and peak GPU memory
when a torch backend is active.
"""

from __future__ import annotations

import argparse
import json
import time

from .pipeline import evaluate_vfi


def _peak_vram_mb(reset: bool = False) -> float | None:
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        if reset:
            torch.cuda.reset_peak_memory_stats()
            return None
        return round(torch.cuda.max_memory_allocated() / 1e6, 1)
    except Exception:
        return None


def benchmark(source: str, candidates: list[str], preset: str = "fast",
              cache_dir: str = "./cache_bench", device: str = "cuda",
              flow_backend: str = "auto", labels: list[str] | None = None
              ) -> dict:
    labels = labels or [c for c in candidates]
    rows = []
    first_elapsed = None
    for lab, cand in zip(labels, candidates):
        _peak_vram_mb(reset=True)
        t0 = time.perf_counter()
        rep = evaluate_vfi(source, cand, preset=preset, cache_dir=cache_dir,
                           device=device, flow_backend=flow_backend,
                           out_dir=None, export_clips=False)
        elapsed = time.perf_counter() - t0
        if first_elapsed is None:
            first_elapsed = elapsed
        rows.append({
            "model": lab,
            "overall": round(float(rep.overall_score), 2),
            "wall_seconds": round(elapsed, 2),
            "speedup_vs_first": round(first_elapsed / elapsed, 2),
            "windows": rep.meta.get("windows_evaluated"),
            "peak_vram_mb": _peak_vram_mb(),
            "flow_backend": rep.meta.get("flow_backend"),
        })
    return {"preset": preset, "device": device, "rows": rows,
            "cache_dir": cache_dir}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True)
    ap.add_argument("--candidates", nargs="+", required=True)
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--preset", default="fast")
    ap.add_argument("--cache-dir", default="./cache_bench")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--flow-backend", default="auto")
    args = ap.parse_args()
    out = benchmark(args.source, args.candidates, preset=args.preset,
                    cache_dir=args.cache_dir, device=args.device,
                    flow_backend=args.flow_backend, labels=args.labels)
    print(f"| {'model':<24} | {'overall':>7} | {'wall s':>7} | {'speedup':>7} |"
          f" | {'VRAM MB':>8} |")
    print(f"|{'-' * 26}|{'-' * 9}|{'-' * 9}|{'-' * 9}|{'-' * 10}|")
    for r in out["rows"]:
        vram = r["peak_vram_mb"] if r["peak_vram_mb"] is not None else "-"
        print(f"| {r['model']:<24} | {r['overall']:>7.2f} | "
              f"{r['wall_seconds']:>7.2f} | {r['speedup_vs_first']:>7.2f} | "
              f"{str(vram):>8} |")
    print(json.dumps({"note": "2nd+ candidates reuse the source cache "
                              "(T_source + N*T_candidate)",
                      "flow_backend": out["rows"][0]["flow_backend"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
