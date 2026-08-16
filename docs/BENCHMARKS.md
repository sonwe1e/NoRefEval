# Benchmarks (USERPLAN §5 / §P3 performance acceptance)

Measured on RTX 4090, Ryzen workstation, NVMe storage; RAFT-small flow
backend; corpus rendered with `rr_vfiqa.testing.synth` (streaming renderer).

## 60 s @ 1920×1080, fast preset

Corpus: 3600 source frames (60 FPS) / 7200 candidate frames (120 FPS);
corpus render (one pass, three encodes) took 228 s.

| candidate | overall | wall time | vs first | peak VRAM |
|---|---:|---:|---:|---:|
| good #1 | 89.5 | 65.5 s | 1.00× | 907 MB |
| good #2 | 89.5 | 64.6 s | 1.01× | 907 MB |
| bad (blur+freeze segments) | 84.1 | 64.6 s | 1.01× | 907 MB |

Run-to-run variance < 2% with warm source cache and OS page cache. Cold
first-ever run on these files measured ~150 s (page-cache population
dominates the delta, not the source cache — at fast preset the source cache
itself builds in ~2 s because only the evaluated windows' source pairs are
computed, at 480 px flow width).

### Per-phase breakdown (warm)

| phase | time | note |
|---|---:|---|
| tier-1 cheap scan | ~35 s | full sequential decode at 320 px + per-frame stats; the floor — parity analysis needs every frame |
| alignment (incl. scene cuts) | ~0 s | no dedicated decode pass: 16x9 descriptors come from the tier-1 scan (CheapScan.descriptors) + a process-level memo shared with routing; scene cuts reuse scan statistics |
| anchor integrity | ~3.6 s | 12 probed anchors, ±1-frame misalignment probes on the first 4 |
| 16 evaluation windows | ~10 s | 1080p seek+decode + batched RAFT (10 directed flows per window — the 5 consumed pairs; (0,1)/(3,4) are no longer precomputed) |
| fusion + reports | <1 s | |

### What this does not cover

- **4K and longer videos**: unmeasured. Decode cost scales linearly with
  pixel count × duration; the window phase stays constant (window count is
  preset-bounded), so scan/decode share grows.
- **standard / audit presets**: region branches and (audit) native-resolution
  re-evaluation add cost per window; not benchmarked at 1080p yet.
- **Multi-candidate speedup at scale**: the source cache pays off when the
  cached feature set is large relative to candidate cost (many windows,
  high flow width). At fast preset/1080p it is ~2 s of a 65 s budget —
  the §P3 N-candidate speedup ratio should be re-measured at standard
  preset, where source-side flow/occlusion/edge caching is heavier.

### Reproduce

```bash
python scripts/bench_1080p.py          # renders the 60s corpus + benchmarks
python -m rr_vfiqa.benchmark --source ... --candidates a.mp4 b.mp4 \
    --preset fast --flow-backend raft
```

## CPU micro-benchmarks (farneback, 2026-08, ds-torch)

tools/perf_micro.py is the reproducible primitive-level baseline for CPU
perf work (synthetic inputs, ~60 s, no repo caches touched):

| primitive | time |
|---|---:|
| forward_splat RGB 960x540 | 138 ms (bincount accumulate; was 171 ms with add.at) |
| forward_splat RGB 480x270 | 27 ms |
| geometry_stats 960x540 | 24 ms (single gradient pass) |
| CameraMotion.warp_flow (grid cached) | 2.6 ms (was 11.6 ms) |
| scan_candidate 720p/600 frames (incl. decode) | 7.0 s |
| frame_descriptors first pass (full decode) | 4.05 s |
| frame_descriptors memo hit | 0.07 ms |
| farneback one direction 720p | 321 ms |
| feature_store save/load pair npz | 20 / 9 ms |

Full-run reference (10 s @ 120 FPS 640x360, NR fast, farneback): 97 s ->
14 s after the splat/bincount, eager error maps, shared reconstructions,
luma-only reconstruction, shared affine residual, the parallel window
loop and the parallel scan (overall score identical at 80.69, features
bit-identical). Endpoint fast: 28 s -> 16 s; FR fast: 44 s -> 25 s.
Workers default to min(8, cpu_count) for farneback; RR_VFIQA_WORKERS=1
forces sequential. Decode-pass inventory per inspect: endpoint 6 -> 3,
full-reference 7 -> 5, no-reference 1 (descriptor memo + scan reuse;
see PROJECT_STRUCTURE §9.11).
