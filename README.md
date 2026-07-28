# rr_vfiqa — Endpoint-Referenced VFI Quality Assessment

Quality evaluation for **60 → 120 FPS game frame interpolation**. True 120 FPS
ground truth does not exist, but the original 60 FPS frames are reliable
anchors, so this is *endpoint-referenced / reduced-reference* assessment
(see `USERPLAN.md` for the full technical route):

> A generated frame is judged by whether it can be explained by both endpoint
> frames, lies on a plausible motion trajectory, and introduces no flicker,
> tearing, ghosting or structural loss — not by guessing a unique ground truth.

## What it produces

- one overall quality score (0–100) and a judgment confidence;
- eight interpretable subscores: motion consistency, temporal stability,
  structural integrity, character integrity, thin-object/weapon, UI/text,
  transition quality, global technical quality;
- event ambiguity for discrete state switches whose timing is unknowable;
- worst time segments with defect-type labels and region boxes;
- reliable *same-source ranking* across multiple interpolation models.

## Core metrics

| Metric | Detects |
|---|---|
| Endpoint flow composition (bidirectional, occlusion-masked) | background tearing, layer misassignment, regional freezes, wrong motion paths |
| Reverse anchor cycle (independent half-warp reconstructor) | inter-frame inconsistency, systematic odd-frame blur, flicker |
| Motion-compensated temporal residuals (5-frame windows, lag 1/2) | drag, shimmer, edge wobble, clarity alternation |
| Odd/even parity & temporal frequency | systematic generated-frame blur/sharpen/brightness alternation |
| Endpoint edge support (per-instance aggregation) | missing heads/legs, broken poles, ghost contours |
| Region branches (UI, character, thin objects, weapon, text, transitions) | game-specific defects, run only on high-risk windows |
| Learned NR-VQA / depth / CoTracker backends | optional low-weight priors / audit-tier diagnostics |

All source-only computation (flows, occlusion, camera motion, endpoint edges)
is **cached once per source video**, so evaluating N candidate models costs
`T_source + N · T_candidate`.

## Install

```bash
pip install -e .            # core (numpy, opencv, scipy, PyAV)
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
print(report.overall_score, report.scores)
```

CLI:

```bash
rr-vfiqa evaluate --source src.mp4 --candidate cand.mp4 --preset standard \
    --out run1/                 # report.json, timeline.md/png, badcase clips
rr-vfiqa compare --source src.mp4 --candidates a.mp4 b.mp4 c.mp4 --out cmp/
```

### Presets (three-tier cascade, USERPLAN §10)

| | tier-1 scan | windows | flow width | region branches | audit tier |
|---|---|---|---|---|---|
| `fast` | 320 px | 8 uniform + 8 risk | 480 | off | – |
| `standard` | 384 px | 16 + 32 | 960 | lightweight | – |
| `audit` | 480 px | 24 + 48 | 960 | full | trackers, full-res edges (≤64 windows) |

Heavy models never run per-frame: cheap scan covers the whole video, core
metrics touch a few percent of frames, and heavy diagnostics are reserved for
the highest-risk windows.

## Report format

```json
{
  "overall_score": 79.07,
  "confidence": 0.95,
  "event_ambiguity": 0.58,
  "scores": {"motion_consistency": 80.6, "temporal_stability": 59.9, "...": "..."},
  "worst_windows": [
    {"start": 0.525, "end": 0.608, "center_index": 63,
     "type": ["parity_flicker", "structure_loss"], "severity": 0.94,
     "confidence": 0.95, "boxes": [[...]]}
  ],
  "features": {"comp_mean": 0.022, "edge_recall": 0.78, "...": "..."},
  "meta": {"preset": "standard", "alignment": {"anchor_error": 2.9, "...": "..."}}
}
```

## Calibration status

Score normalization scales are **bootstrap values** empirically grounded on
the built-in synthetic corpus (`rr_vfiqa.testing.synth` renders a deterministic
120 FPS game-like scene with camera motion, a character, a thin pole and
screen-static UI, plus degraded candidates: blur / ghost / freeze segments).
Per USERPLAN §12, final weights and thresholds must be re-calibrated on:

1. high-FPS pseudo ground truth (drop frames from real 120/240 FPS captures);
2. real 60→120 outputs with human A/B ranking;
3. synthetic defects for localization ability only.

A monotonic LightGBM calibrator (`rr_vfiqa.fusion.monotonic_calibrator`) with
monotone constraints + pairwise preference training support replaces the
exponential formula once labeled data exists; pass `--calibrator model.pkl`.

## Tests

```bash
python -m pytest                      # fast path (Farneback flow)
RR_VFIQA_TEST_FLOW=raft python -m pytest   # accurate path (needs GPU + torchvision)
```

## Layout

```
rr_vfiqa/
├── io/        PyAV reader with PTS, source↔candidate alignment, color fit
├── cache/     per-source feature store (flows, occlusion, camera, edges)
├── sampling/  cheap full-frame scan, robust-z risk, temporal NMS, windows
├── motion/    flow backends (RAFT/Farneback), occlusion, camera, composition, geometry
├── metrics/   anchor integrity, cycle, temporal compensation, parity, edges, GTQ
├── regions/   UI, character, thin objects, weapon, text, transition branches
├── models/    unified backend interfaces (flow/segmentation/tracker/depth/VQA)
├── fusion/    feature normalization, aggregation, confidence, monotonic calibrator
├── report/    JSON, timeline, badcase clips, heatmaps
├── testing/   synthetic scene + pseudo-GT corpus generator
└── pipeline.py / cli.py
```
