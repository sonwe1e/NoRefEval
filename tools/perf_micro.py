"""Micro-benchmarks for rr_vfiqa hot-path primitives (CPU, farneback).

Reproducible evidence base for performance work: run with the canonical
interpreter from the repo root and compare outputs across changes.

    G:/ds-torch/Scripts/python.exe tools/perf_micro.py

Everything is synthetic (no repo caches are touched).  ~60 s total.
"""

from __future__ import annotations

import os
import sys
import tempfile
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _timeit(fn, n: int = 5):
    fn()  # warmup
    best = float("inf")
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def _row(name: str, seconds: float, note: str = "") -> None:
    print(f"{name:<42} {seconds * 1000:9.2f} ms  {note}")


def bench_arrays() -> None:
    from rr_vfiqa.motion.flow_geometry import geometry_stats
    from rr_vfiqa.schema import CameraMotion, forward_splat, warp_flow

    rng = np.random.default_rng(0)
    h, w = 540, 960
    img = rng.integers(0, 256, (h, w, 3)).astype(np.uint8)
    flow = rng.uniform(-6, 6, (h, w, 2)).astype(np.float32)
    _row("forward_splat RGB 960x540",
         _timeit(lambda: forward_splat(img, flow)))
    img_s = img[::2, ::2]
    flow_s = flow[::2, ::2] * 0.5
    _row("forward_splat RGB 480x270",
         _timeit(lambda: forward_splat(img_s, flow_s)))
    mask = rng.uniform(0, 1, (h, w)) > 0.2
    _row("geometry_stats 960x540",
         _timeit(lambda: geometry_stats(flow, mask)))
    cm = CameraMotion("affine", rng.uniform(-1, 1, (2, 3)).astype(np.float32),
                      0.9, 0.5)
    _row("CameraMotion.warp_flow (grid cached)",
         _timeit(lambda: cm.warp_flow(h, w)))
    _row("warp_flow compose (flow,flow)",
         _timeit(lambda: warp_flow(flow, flow)))


def bench_video() -> None:
    from rr_vfiqa.io.video_reader import VideoReader, frame_descriptors, _DESCRIPTOR_CACHE
    from rr_vfiqa.motion.flow_estimator import FarnebackBackend
    from rr_vfiqa.sampling.cheap_scan import scan_candidate
    from rr_vfiqa.testing.synth import render_scene, write_video

    td = tempfile.mkdtemp(prefix="perf_micro_")
    try:
        p = os.path.join(td, "clip.mp4")
        frames = render_scene(n_frames=600, w=1280, h=720, seed=3)
        write_video(p, frames, 60)
        r = VideoReader(p)
        _row("scan_candidate 720p/600f (incl. decode)",
             _timeit(lambda: scan_candidate(r, width=384), 3),
             "10 s clip")
        _DESCRIPTOR_CACHE.clear()
        # Measure with a cold cache on every iteration: warmup would memoize.
        best = float("inf")
        for _ in range(3):
            _DESCRIPTOR_CACHE.clear()
            t0 = time.perf_counter()
            frame_descriptors(r, width=96)
            best = min(best, time.perf_counter() - t0)
        _row("frame_descriptors first pass (full decode)", best)
        _row("frame_descriptors memo hit",
             _timeit(lambda: frame_descriptors(r, width=96), 10),
             "process-level memo")
        a = frames[100].copy()
        b = frames[110].copy()
        bk = FarnebackBackend()
        _row("farneback one direction 720p",
             _timeit(lambda: bk.flow(a, b), 3))
    finally:
        import shutil
        shutil.rmtree(td, ignore_errors=True)


def bench_cache_io() -> None:
    from rr_vfiqa.cache.feature_store import FeatureStore

    td = tempfile.mkdtemp(prefix="perf_micro_")
    try:
        fs = FeatureStore(os.path.join(td, "cache"))
        rng = np.random.default_rng(1)
        d = {
            "f_01": rng.uniform(-20, 20, (540, 960, 2)).astype(np.float16),
            "f_10": rng.uniform(-20, 20, (540, 960, 2)).astype(np.float16),
            "occ_ab": rng.integers(0, 2, (540, 960)).astype(np.uint8),
            "occ_ba": rng.integers(0, 2, (540, 960)).astype(np.uint8),
            "conf_ab": rng.uniform(0, 1, (540, 960)).astype(np.float16),
            "conf_ba": rng.uniform(0, 1, (540, 960)).astype(np.float16),
            "meta": np.array([540, 960, 1.0], np.float64),
        }
        _row("feature_store save_npz (pair payload)",
             _timeit(lambda: fs.save_npz("pair_000001", **d), 5))
        fs.save_npz("pair_000001", **d)
        _row("feature_store load_npz (pair payload)",
             _timeit(lambda: fs.load_npz("pair_000001"), 10))
    finally:
        import shutil
        shutil.rmtree(td, ignore_errors=True)


def main() -> None:
    print(f"rr_vfiqa micro-benchmarks — python {sys.version.split()[0]}")
    print("cwd=" + ROOT)
    print()
    print("== array primitives ==")
    bench_arrays()
    print()
    print("== video pipeline (decode-bound) ==")
    bench_video()
    print()
    print("== cache IO ==")
    bench_cache_io()


if __name__ == "__main__":
    main()
