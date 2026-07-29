# rr_vfiqa — Endpoint-Referenced VFI Quality Assessment

Quality evaluation for **60 → 120 FPS game frame interpolation**. True 120 FPS
ground truth does not exist, but the original 60 FPS frames are reliable
anchors, so this is *endpoint-referenced / reduced-reference* assessment
(see `USERPLAN.md` for the technical route and the acceptance review):

> A generated frame is judged by whether it can be explained by both endpoint
> frames, lies on a plausible motion trajectory, and introduces no flicker,
> tearing, ghosting or structural loss — not by guessing a unique ground truth.

**Status: research prototype / metric development framework.** The architecture,
coordinate math, fail-closed scoring, defect corpus, and calibration harness
are in place and tested; absolute score calibration requires the real-data
corpora from USERPLAN §P3 (real HFR pseudo-GT + human A/B rankings). Do not
use `overall_score` as a production gate until that calibration exists. The
report declares which branches are heuristic proxies (`meta.proxy_branches`).

## What it produces

- one overall quality score (0–100) + judgment confidence + event ambiguity;
- eight interpretable subscores: motion consistency, temporal stability,
  structural integrity, character integrity, thin-object/weapon, UI/text,
  transition quality, global technical quality;
- worst time segments with defect-type labels (`background_tear`,
  `parity_flicker`, `character_missing`, `thin_object_stick`, `weapon_jitter`,
  `ui_unstable`, `text_degraded`, `transition_ghost`, `card_flip_error`, …)
  and region boxes;
- **fail-closed**: core-metric failure yields `overall_score: null`,
  near-zero confidence and `meta.status: "failed"` — never a falsely good
  number; stage exceptions are recorded in `meta.stage_errors`;
- exploratory *same-source ranking* across multiple interpolation models;
  synthetic ladders validate the mechanism, while real-model ranking remains
  unvalidated. Source features are cached once (`T_source + N · T_candidate`).

## Core metrics

| Metric | Detects |
|---|---|
| Endpoint flow composition, bidirectional with grid-correct occlusion weights | tearing, layer misassignment, wrong motion paths |
| Reverse anchor cycle (independent forward-splat reconstructor) | inter-frame inconsistency, systematic odd-frame blur |
| Motion-compensated temporal residuals (5-frame windows, lag 1/2) | drag, shimmer, edge wobble |
| Odd/even parity & temporal frequency | generated-frame blur/sharpen/brightness alternation |
| Endpoint edge support (forward-splatted, per-instance aggregation) | missing structure, ghost contours |
| UI (static/dynamic, component-level, α-blend fitting), text stroke topology | HUD blur/drift, crossfade ghosts, stroke merging |
| Card flip area progression, transition state analysis | frozen flips, double-exposure switches |
| Thin-object layer attribution (bidirectional, width-gated instances) | poles/railings following the background |
| Character proxy (motion segmentation, mask → weapon KLT) | missing area, copy-like mids — *proxy, see meta* |

Warp semantics are explicit throughout: `backward_warp(img, target→source flow)`
vs `forward_splat(img, source→target flow)` — no ambiguous warp calls.

## Install

```bash
pip install -e .            # core (numpy, opencv-headless, scipy, PyAV)
pip install -e ".[torch]"   # RAFT flow backend (recommended, GPU)
pip install -e ".[fusion]"  # LightGBM monotonic calibrator
pip install -e ".[dev]"     # pytest
```

## Usage

```python
from rr_vfiqa import evaluate_vfi

report = evaluate_vfi(
    source_video="source_60fps.mp4",
    candidate_video="candidate_120fps.mp4",
    preset="standard",          # fast | standard | audit
    cache_dir="./cache",
    device="cuda",              # cuda | cpu
)
```

CLI:

```bash
rr-vfiqa evaluate --source src.mp4 --candidate cand.mp4 --preset standard --out run1/
rr-vfiqa compare  --source src.mp4 --candidates a.mp4 b.mp4 c.mp4 \
    --flow-backend raft --out cmp/
```

`compare` passes the requested flow backend to every candidate. Failed/NaN
candidates are marked `failed`, sorted after valid candidates, and excluded
from the mean and `relative_vs_mean`.

### Presets (three-tier cascade)

| | scan | windows | flow | region branches | tier-3 escalation |
|---|---|---|---|---|---|
| `fast` | 320 px | 8+8 | 480 | off | – |
| `standard` | 384 px | 16+32 | 960 | lightweight | – |
| `audit` | 480 px | 24+48 | 960 | full | top 10% windows (≤64) recomputed at native resolution + point tracking |

Tier 3 is real: `audit_top_fraction` / `audit_max_windows` select the
highest-risk windows after tier 2 and re-run core metrics at native
resolution. Candidate flow, source flow/occlusion/camera, endpoint edges and
the character ROI are all regenerated on the native grid; the selected
CoTracker/KLT backend is reused for weapon tracking. `meta.audited_windows`
reports the count.

Source caches are isolated by resolved backend (including `auto` fallback),
weight identifier/hash, evaluator schema, flow width, occlusion threshold,
camera algorithm and edge algorithm. The full contract and cache path are
recorded in `meta.source_cache`; Farneback and RAFT data cannot be mixed.

## Calibration & benchmarks

```bash
# FR-vs-endpoint-reference validation on the synthetic severity ladder
python -m rr_vfiqa.calibration.validate --synthetic --workdir cal_run
# → SRCC / PLCC / pairwise accuracy of overall_score vs full-reference PSNR

# model-ranking validation with ORGANIC outputs: reference interpolators
# (naive linear blend + forward-splat flow VFI at several resolutions)
# ranked against FR truth across scene seeds, + per-feature SRCC table
python -m rr_vfiqa.calibration.model_validation --workdir mv_run --interp-backend raft

# per-defect-category localization: recall / precision / F1 of worst-window
# labels against ground-truth defect segments
python -m rr_vfiqa.calibration.detection_eval --workdir det_run \
    --output det_run/calibration.json
python -m rr_vfiqa.calibration.detection_eval --defects blur ghost freeze \
    rotation_tear head_erase pole_wrong_motion sword_flicker ui_drift \
    text_merge shop_jump card_freeze disocc_fill

# timing, cache speedup, peak VRAM across candidates
python -m rr_vfiqa.benchmark --source src.mp4 --candidates a.mp4 b.mp4 --flow-backend raft
```

Bootstrap numbers (RTX 4090):

| validation | result |
|---|---|
| model ranking vs FR-PSNR (interpolators × 3 seeds) | SRCC 0.95, pairwise 0.93, **same-source ranking 1.00** |
| strongest FR correlates | motion-compensated temporals (|SRCC| 0.97–0.99), cycle (0.95), composition (0.83) |
| severity ladder vs FR-PSNR | PLCC 0.96, SRCC 0.7, pairwise 0.8 |
| defect localization (12 types, strict temporal overlap, standard/Farneback) | recall 0.667, precision 0.229, F1 0.340 — [raw JSON](docs/calibration/synthetic_detection_12class.json) |
| 60 s 1080p, fast/RAFT | 64.6–65.5 s/candidate, 907 MB VRAM — see `docs/BENCHMARKS.md` |

The strict localization result replaces the earlier inflated 6-type
recall 1.0 / F1 0.67 claim: the old ±0.6 s tolerance covered most of a
1.33 s clip, and precision did not require temporal overlap. The current
artifact evaluates all 12 classes, caps tolerance at 10% of clip duration,
and preserves every fired window with its true-positive decision. Its low
precision is a real remaining research gap, not a passing production metric.

The interpolator ladder (`rr_vfiqa.testing.interpolator`) closes the §12.1
pseudo-GT loop with emergent rather than authored artifacts: ranking
perfect > flow-native ≈ flow-320 > linear-blend matches FR ordering on
every same-source pair. Real acceptance targets (SRCC ≥ 0.8
leave-one-game-out, ≥ 80% A/B accuracy on real model outputs, worst-10%
recall ≥ 90%) still require the §P3 data:

1. real 120/240 FPS pseudo-GT (the harness in `rr_vfiqa.calibration`
   measures SRCC/PLCC/pairwise once you supply it);
2. real model outputs across games/motion;
3. human A/B rankings for the monotonic LightGBM fusion calibrator
   (`rr_vfiqa.fusion.monotonic_calibrator` — pairwise rank loss,
   monotone-constrained, replaces the bootstrap formula via `--calibrator`).

## Testing

53 tests (51 pass, 2 optional-environment skips in the CPU environment):

```bash
python -m pytest                            # Farneback (CPU-portable)
RR_VFIQA_TEST_FLOW=raft python -m pytest    # RAFT (GPU)
```

Coverage includes deterministic warp-convention math tests, backend/weight
cache isolation, compare failure quarantine, dynamic-UI mask containment,
local drop/duplicate recovery, CoTracker layout/fallback, per-defect
response tests over a 12-type synthetic defect corpus (blur, ghost, freeze,
rotation tear, head erasure, pole misattribution, sword flicker, UI drift,
text merging, shop double-exposure, disocclusion fill, card freeze), scene
cuts, frame-offset/dropped-frame robustness, fail-closed behavior, calibrator
monotonicity + pairwise ranking, calibration-harness self-tests, and a
per-defect localization recall test (freeze-copy and card-flip included).

```python
from rr_vfiqa.testing.synth import build_test_set, DEFECTS  # corpus generator
```

## Layout

```
rr_vfiqa/
├── io/          PyAV reader (seek-based random access), PTS alignment, color fit
├── cache/       per-source feature store (flows, occlusion, camera, edges)
├── sampling/    cheap scan, robust-z risk, temporal NMS, window selection
├── motion/      flow backends (batched RAFT / Farneback), occlusion, camera,
│                composition, flow geometry
├── metrics/     anchor integrity, cycle, temporal compensation, parity, edges, GTQ
├── regions/     UI, text, transition, card, character, thin-object, weapon
├── models/      unified backend interfaces (flow/segmentation/tracker/depth/VQA)
├── fusion/      normalization, aggregation, confidence, monotonic calibrator
├── report/      JSON, timeline, badcase clips, heatmaps
├── testing/     synthetic scene, 12-defect corpus, reference interpolators,
│                alignment stress variants (scene cut / offset / drops / VFR)
├── calibration/ pseudo-GT corpora, FR metrics, model-ranking validation,
│                per-defect localization, per-feature FR correlations
├── benchmark.py timing / VRAM / cache-speedup measurement
├── schema.py    contracts + explicit backward_warp / forward_splat
└── pipeline.py / cli.py
```

CI (`.github/workflows/ci.yml`): clean-checkout install + import + CPU suite
on every push; opt-in GPU/RAFT job via repository variables.
