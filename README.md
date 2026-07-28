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
- reliable *same-source ranking* across multiple interpolation models
  (source features cached once: cost is `T_source + N · T_candidate`).

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
rr-vfiqa compare  --source src.mp4 --candidates a.mp4 b.mp4 c.mp4 --out cmp/
```

### Presets (three-tier cascade)

| | scan | windows | flow | region branches | tier-3 escalation |
|---|---|---|---|---|---|
| `fast` | 320 px | 8+8 | 480 | off | – |
| `standard` | 384 px | 16+32 | 960 | lightweight | – |
| `audit` | 480 px | 24+48 | 960 | full | top 10% windows (≤64) recomputed at native resolution + point tracking |

Tier 3 is real: `audit_top_fraction` / `audit_max_windows` select the
highest-risk windows after tier 2 and re-run core metrics at native
resolution; `meta.audited_windows` reports the count.

## Calibration & benchmarks

```bash
# FR-vs-endpoint-reference validation on the synthetic severity ladder
python -m rr_vfiqa.calibration.validate --synthetic --workdir cal_run
# → SRCC / PLCC / pairwise accuracy of overall_score vs full-reference PSNR

# timing, cache speedup, peak VRAM across candidates
python -m rr_vfiqa.benchmark --source src.mp4 --candidates a.mp4 b.mp4 --flow-backend raft
```

Bootstrap numbers on synthetic content (4 s 640×360, fast/RAFT/RTX 4090):
~14 s per candidate, ~0.9 GB peak VRAM, PLCC ≈ 0.96 vs FR-PSNR on the
severity ladder. Real acceptance targets (SRCC ≥ 0.8 leave-one-game-out,
≥ 80% A/B accuracy, worst-10% recall ≥ 90%) require the §P3 data:

1. real 120/240 FPS pseudo-GT (the harness in `rr_vfiqa.calibration`
   measures SRCC/PLCC/pairwise once you supply it);
2. real model outputs across games/motion;
3. human A/B rankings for the monotonic LightGBM fusion calibrator
   (`rr_vfiqa.fusion.monotonic_calibrator` — pairwise rank loss,
   monotone-constrained, replaces the bootstrap formula via `--calibrator`).

## Testing

42 tests, green on both flow backends:

```bash
python -m pytest                            # Farneback (CPU-portable)
RR_VFIQA_TEST_FLOW=raft python -m pytest    # RAFT (GPU)
```

Coverage includes deterministic warp-convention math tests, per-defect
response tests over an 11-type synthetic defect corpus (blur, ghost, freeze,
rotation tear, head erasure, pole misattribution, sword flicker, UI drift,
text merging, shop double-exposure, disocclusion fill, card freeze), scene
cuts, frame-offset/dropped-frame robustness, fail-closed behavior, calibrator
monotonicity + pairwise ranking, and the calibration harness self-test.

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
├── testing/     synthetic scene + 11-defect corpus + alignment stress variants
├── calibration/ pseudo-GT corpora, FR metrics, SRCC/PLCC/pairwise validation
├── benchmark.py timing / VRAM / cache-speedup measurement
├── schema.py    contracts + explicit backward_warp / forward_splat
└── pipeline.py / cli.py
```

CI (`.github/workflows/ci.yml`): clean-checkout install + import + CPU suite
on every push; opt-in GPU/RAFT job via repository variables.
