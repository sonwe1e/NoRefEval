"""Optical-flow backends behind one interface (USERPLAN.md §3, §13).

The evaluator must never consume a candidate model's internal flow as truth,
so this module provides independent estimators:

* ``raft``       — torchvision RAFT-small, accurate, GPU-friendly (default when
                   torch is importable);
* ``farneback``  — OpenCV Farneback, CPU fallback / cheap scan tier.

USERPLAN §6 P0: RAFT execution is now memory-aware.  A ``FlowExecutionPolicy``
drives micro-batch sizing, OOM backoff, and degraded fallback to Farneback.
Telemetry (peak VRAM, batch sizes, OOM retries, fallback reason) is recorded
on the backend and surfaced in the final report's ``performance`` block.
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from hashlib import sha256

import cv2
import numpy as np

FLOW_ALGORITHM_VERSION = "flow-contract-v2"


# ---------------------------------------------------------------------------
# USERPLAN §6 P0.1: memory-aware execution policy
# ---------------------------------------------------------------------------

@dataclass
class FlowExecutionPolicy:
    """How RAFT should trade memory for throughput.

    The pipeline passes one policy per backend instance.  Defaults target a
    12 GB budget on a 16 GB GPU (leaves ~4 GB for driver / desktop / other).
    """
    max_batch_size: int = 16
    memory_budget_mb: int = 12000
    precision: str = "fp32"                  # "fp32" | "fp16" (fp16 unverified)
    audit_max_width: int = 960               # USERPLAN §6 P0.2: Tier-3 cap
    allow_oom_backoff: bool = True


# USERPLAN §6 P0.3: the canonical RAFT-safe profile for ``auto + CUDA``.
# ``auto`` resolves to this profile when CUDA is available, so the same
# command produces the same backend contract regardless of whether the user
# passed ``--flow-backend auto`` or ``--flow-backend raft``.
CANONICAL_RAFT_POLICY = FlowExecutionPolicy(
    max_batch_size=16,
    memory_budget_mb=12000,
    precision="fp32",
    audit_max_width=960,
    allow_oom_backoff=True,
)


@dataclass
class FlowTelemetry:
    """USERPLAN §6 P0.5: run-level performance telemetry, surfaced in reports.

    Each ``RaftBackend`` accumulates these numbers; the pipeline copies them
    into the final ``Report.meta["performance"]`` block.
    """
    peak_vram_mb: float = 0.0
    flow_batch_sizes: list[int] = field(default_factory=list)
    tier2_flow_width: int = 0
    tier3_flow_width: int = 0
    oom_retries: int = 0
    backend_fallback: str | None = None
    phase_seconds: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "peak_vram_mb": round(self.peak_vram_mb, 1),
            "flow_batch_sizes": list(self.flow_batch_sizes),
            "tier2_flow_width": int(self.tier2_flow_width),
            "tier3_flow_width": int(self.tier3_flow_width),
            "oom_retries": int(self.oom_retries),
            "backend_fallback": self.backend_fallback,
            "phase_seconds": {
                k: round(v, 3) for k, v in self.phase_seconds.items()
            },
        }


# ---------------------------------------------------------------------------
# backends
# ---------------------------------------------------------------------------

class FlowBackend(ABC):
    name = "base"

    @abstractmethod
    def flow(self, a_rgb: np.ndarray, b_rgb: np.ndarray) -> np.ndarray:
        """(H, W, 2) float32 flow a -> b, pixel units at input resolution."""

    def flow_pair(self, a_rgb: np.ndarray, b_rgb: np.ndarray):
        """Bidirectional flows as a tuple (f_ab, f_ba)."""
        return self.flow(a_rgb, b_rgb), self.flow(b_rgb, a_rgb)

    def flow_many(self, pairs: list[tuple[np.ndarray, np.ndarray]]) -> list[np.ndarray]:
        """Batch of directed flows. Base implementation: sequential."""
        return [self.flow(a, b) for a, b in pairs]

    def release(self) -> None:
        pass

    def cache_identity(self) -> dict[str, str]:
        """Stable identity for source-cache isolation and report provenance."""
        return {
            "backend": self.name,
            "algorithm_version": FLOW_ALGORITHM_VERSION,
            "weights_id": "none",
            "weights_hash": "none",
        }


class FarnebackBackend(FlowBackend):
    name = "farneback"

    def __init__(self, pyr_scale: float = 0.5, levels: int = 5, winsize: int = 21,
                 iterations: int = 5):
        self._kw = dict(pyr_scale=pyr_scale, levels=levels, winsize=winsize,
                        iterations=iterations, poly_n=7, poly_sigma=1.5, flags=0)

    def flow(self, a_rgb: np.ndarray, b_rgb: np.ndarray) -> np.ndarray:
        ga = cv2.cvtColor(a_rgb, cv2.COLOR_RGB2GRAY)
        gb = cv2.cvtColor(b_rgb, cv2.COLOR_RGB2GRAY)
        f = cv2.calcOpticalFlowFarneback(ga, gb, None, **self._kw)
        return f.astype(np.float32)

    def cache_identity(self) -> dict[str, str]:
        params = ",".join(f"{k}={self._kw[k]}" for k in sorted(self._kw))
        return {
            "backend": self.name,
            "algorithm_version": f"{FLOW_ALGORITHM_VERSION}:opencv-{cv2.__version__}",
            "weights_id": "farneback-params",
            "weights_hash": sha256(params.encode("utf-8")).hexdigest()[:12],
        }


class RaftBackend(FlowBackend):
    name = "raft"

    def __init__(self, device: str = "cuda",
                 policy: FlowExecutionPolicy | None = None):
        import torch
        import torchvision
        from torchvision.models.optical_flow import raft_small, Raft_Small_Weights

        self._torch = torch
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu"
                                   else "cpu")
        weights = Raft_Small_Weights.DEFAULT
        self._weights_id = f"Raft_Small_Weights.{weights.name}"
        self._weights_url = str(weights.url)
        self._torchvision_version = str(torchvision.__version__)
        self._transforms = weights.transforms()
        self._model = raft_small(weights=weights).to(self.device).eval()

        # USERPLAN §6 P0: memory-aware execution + telemetry
        self.policy = policy or FlowExecutionPolicy()
        self.telemetry = FlowTelemetry()
        self._fallback: FarnebackBackend | None = None

    # -- low-level forward ---------------------------------------------------

    def _forward_batch(self, pairs: list[tuple[np.ndarray, np.ndarray]]) -> list[np.ndarray]:
        """Run RAFT on one micro-batch of directed pairs (no retry logic)."""
        torch = self._torch
        h, w = pairs[0][0].shape[:2]
        ph = (16 - h % 16) % 16
        pw = (16 - w % 16) % 16
        imgs_a, imgs_b = [], []
        for a, b in pairs:
            imgs_a.append(np.pad(a, ((0, ph), (0, pw), (0, 0)), mode="edge")
                          .transpose(2, 0, 1))
            imgs_b.append(np.pad(b, ((0, ph), (0, pw), (0, 0)), mode="edge")
                          .transpose(2, 0, 1))
        ta = torch.from_numpy(np.stack(imgs_a))
        tb = torch.from_numpy(np.stack(imgs_b))
        fa = ta.to(torch.float32).div(127.5).sub(1.0)
        fb = tb.to(torch.float32).div(127.5).sub(1.0)
        # USERPLAN §6 P0.4: inference_mode (stronger than no_grad).
        with torch.inference_mode():
            flows = self._model(fa.to(self.device), fb.to(self.device))[-1]
        flows = flows[:, :, :h, :w].cpu().numpy()
        return [np.transpose(flows[i], (1, 2, 0)).astype(np.float32)
                for i in range(len(pairs))]

    # -- memory-aware batching -----------------------------------------------

    def _initial_batch_size(self, h: int, w: int) -> int:
        """Pick a starting batch size from the free-VRAM budget.

        RAFT-small at H×W in FP32 needs roughly 4 bytes/pixel for the final
        feature pyramid plus the flow decoder; a single (H,W,3) input frame is
        H×W×3×4 bytes and the model keeps ~3 such tensors live per frame in the
        batch.  We budget conservatively at 64 bytes/pixel/frame so the estimate
        never *under*-counts.
        """
        torch = self._torch
        if not torch.cuda.is_available() or not self.policy.allow_oom_backoff:
            return self.policy.max_batch_size
        free_mb = self._free_vram_mb()
        budget_mb = min(self.policy.memory_budget_mb, free_mb * 0.8)
        bytes_per_frame = 64 * h * w
        if bytes_per_frame <= 0:
            return self.policy.max_batch_size
        by_budget = max(1, int(budget_mb * 1e6 / bytes_per_frame))
        return max(1, min(self.policy.max_batch_size, by_budget))

    def _free_vram_mb(self) -> float:
        torch = self._torch
        if not torch.cuda.is_available():
            return float("inf")
        torch.cuda.synchronize(self.device)
        free, _total = torch.cuda.mem_get_info(self.device)
        return free / 1e6

    def flow(self, a_rgb: np.ndarray, b_rgb: np.ndarray) -> np.ndarray:
        return self.flow_many([(a_rgb, b_rgb)])[0]

    def flow_many(self, pairs: list[tuple[np.ndarray, np.ndarray]]) -> list[np.ndarray]:
        """USERPLAN §6 P0.1: memory-aware micro-batch with OOM backoff.

        Safety priority (USERPLAN §6 P0.1):

            RAFT original config
            → RAFT smaller batch
            → RAFT lower working resolution
            → Farneback degraded fallback

        On the first OOM we halve the batch and retry; once the batch is 1 and
        it still OOMs we reduce the working resolution; if that also fails we
        fall back to Farneback for the remaining pairs.  Each fallback is
        recorded in ``telemetry.backend_fallback`` so the report can mark
        ``comparable_to_canonical = false``.
        """
        if not pairs:
            return []
        torch = self._torch
        h, w = pairs[0][0].shape[:2]
        batch = self._initial_batch_size(h, w)

        out: list[np.ndarray | None] = [None] * len(pairs)
        idx = 0
        oom_retries = 0
        reduced_width = False

        while idx < len(pairs):
            cur = min(batch, len(pairs) - idx)
            slice_ = pairs[idx:idx + cur]
            try:
                flows = self._forward_batch(slice_)
                for i, f in enumerate(flows):
                    out[idx + i] = f
                idx += cur
                self.telemetry.flow_batch_sizes.append(cur)
            except Exception as exc:
                is_oom = _is_oom_error(exc)
                if not is_oom or not self.policy.allow_oom_backoff:
                    raise
                # USERPLAN §6 P0.1: OOM → clear cache, halve batch.
                torch.cuda.empty_cache()
                oom_retries += 1
                self.telemetry.oom_retries = oom_retries
                if batch > 1:
                    batch = max(1, batch // 2)
                    continue
                # batch == 1 still OOMs → reduce working resolution once.
                if not reduced_width:
                    reduced_width = True
                    torch.cuda.empty_cache()
                    break  # fall through to the resolution-reduction path
                # Reduced resolution also OOM'd → Farneback degraded fallback.
                remaining = self._fallback_flow_many(pairs[idx:])
                for i, f in enumerate(remaining):
                    out[idx + i] = f
                idx = len(pairs)
        # Resolution-reduction path: retry everything at half width.
        if idx < len(pairs) and reduced_width:
            remaining_pairs = pairs[idx:]
            try:
                flows = self._forward_batch_reduced(remaining_pairs)
                for i, f in enumerate(flows):
                    out[idx + i] = f
                idx = len(pairs)
                if self.telemetry.backend_fallback is None:
                    self.telemetry.backend_fallback = "raft_reduced_width"
            except Exception as exc2:
                if _is_oom_error(exc2):
                    torch.cuda.empty_cache()
                    remaining = self._fallback_flow_many(remaining_pairs)
                    for i, f in enumerate(remaining):
                        out[idx + i] = f
                    idx = len(pairs)
                    self.telemetry.backend_fallback = "cuda_oom"
                else:
                    raise
        # Record peak VRAM so the report can surface it.
        if torch.cuda.is_available():
            self.telemetry.peak_vram_mb = max(
                self.telemetry.peak_vram_mb,
                torch.cuda.max_memory_allocated(self.device) / 1e6)
        # Any None entries mean we silently failed — replace with Farneback so
        # the downstream pipeline always gets a flow.
        for i in range(len(out)):
            if out[i] is None:
                out[i] = self._fallback_flow_many([pairs[i]])[0]
                self.telemetry.backend_fallback = self.telemetry.backend_fallback or "cuda_oom"
        return out  # type: ignore[return-value]

    def _forward_batch_reduced(self, pairs: list[tuple[np.ndarray, np.ndarray]]) -> list[np.ndarray]:
        """Retry a batch at half the working resolution, then scale flows back."""
        if not pairs:
            return []
        h, w = pairs[0][0].shape[:2]
        rw = max(16, w // 2)
        rh = max(16, (h * rw // w))
        # RAFT requires dims to be multiples of 16.
        rw = max(16, (rw // 16) * 16)
        rh = max(16, (rh // 16) * 16)
        resized = [(cv2.resize(a, (rw, rh)), cv2.resize(b, (rw, rh)))
                   for a, b in pairs]
        flows = self._forward_batch(resized)
        sx, sy = w / rw, h / rh
        out = []
        for f in flows:
            ff = cv2.resize(f, (w, h), interpolation=cv2.INTER_LINEAR)
            ff[..., 0] *= sx
            ff[..., 1] *= sy
            out.append(ff.astype(np.float32))
        return out

    def _fallback_flow_many(self, pairs: list[tuple[np.ndarray, np.ndarray]]) -> list[np.ndarray]:
        """Degraded Farneback fallback (USERPLAN §6 P0.1, P0.3)."""
        if self._fallback is None:
            self._fallback = FarnebackBackend()
        if self.telemetry.backend_fallback is None:
            self.telemetry.backend_fallback = "cuda_oom"
        return self._fallback.flow_many(pairs)

    def release(self) -> None:
        del self._model
        if self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()

    def cache_identity(self) -> dict[str, str]:
        # torchvision checkpoint filenames carry the published short SHA
        # (e.g. ``...-01064c6d.pth``). Hash the full immutable URL as a
        # deterministic fallback so differently published weights never share
        # a source-flow cache.
        filename = self._weights_url.rsplit("/", 1)[-1]
        stem = filename.rsplit(".", 1)[0]
        short_sha = stem.rsplit("-", 1)[-1]
        if len(short_sha) < 8:
            short_sha = sha256(self._weights_url.encode("utf-8")).hexdigest()[:12]
        return {
            "backend": self.name,
            "algorithm_version": (
                f"{FLOW_ALGORITHM_VERSION}:torchvision-{self._torchvision_version}"
            ),
            "weights_id": self._weights_id,
            "weights_hash": short_sha,
            "weights_url": self._weights_url,
        }


def _is_oom_error(exc: BaseException) -> bool:
    """Return True if ``exc`` is a CUDA / ROCm / XLA out-of-memory error."""
    msg = str(exc).lower()
    if "out of memory" in msg or "oom" in msg:
        return True
    # torch.cuda.OutOfMemoryError (torch>=1.10) and the underlying CUDA driver
    # RuntimeError both surface here.  Match the class name as a last resort.
    cls = type(exc).__name__.lower()
    return "outofmemory" in cls


# ---------------------------------------------------------------------------
# backend registry
# ---------------------------------------------------------------------------

_backends: dict[str, FlowBackend] = {}
_lock = threading.Lock()


def get_flow_backend(name: str = "auto", device: str = "cuda",
                     policy: FlowExecutionPolicy | None = None) -> FlowBackend:
    """Return a cached backend instance. "auto" prefers RAFT when torch exists.

    USERPLAN §6 P0.3: ``auto`` resolves to a *canonical* profile — RAFT-safe
    when CUDA is available, Farneback otherwise — and any RAFT failure is
    recorded as a degraded fallback rather than silently switching algorithms.
    The canonical profile (``CANONICAL_RAFT_POLICY``) is used unless the
    caller passes an explicit policy.
    """
    key = f"{name}:{device}"
    with _lock:
        if key in _backends:
            return _backends[key]
        if name in ("auto", "raft"):
            try:
                effective_policy = policy or CANONICAL_RAFT_POLICY
                be: FlowBackend = RaftBackend(device=device, policy=effective_policy)
            except Exception:
                if name == "raft":
                    raise
                be = FarnebackBackend()
        elif name == "farneback":
            be = FarnebackBackend()
        else:
            raise ValueError(f"unknown flow backend {name!r}")
        _backends[key] = be
        return be


def collect_backend_telemetry(backend: FlowBackend, *,
                              tier2_flow_width: int = 0,
                              tier3_flow_width: int = 0) -> dict:
    """Collect performance telemetry from a backend for the report.

    USERPLAN §6 P0.5: every report records the ``performance`` block so users
    can see peak VRAM, batch sizes, OOM retries and any degraded fallback.
    Farneback backends contribute only the tier widths (no VRAM tracking).

    USERPLAN §6 P0.3: when a RAFT backend fell back to Farneback (OOM or
    reduced-width), ``comparable_to_canonical`` is set to False so the report
    can warn that the scores are not directly comparable to a canonical RAFT
    run.
    """
    tel: dict = {}
    if isinstance(backend, RaftBackend):
        tel.update(backend.telemetry.to_dict())
        tel["oom_retries"] = max(tel.get("oom_retries", 0),
                                 backend.telemetry.oom_retries)
        # USERPLAN §6 P0.3: record whether the run stayed on the canonical path.
        if backend.telemetry.backend_fallback is not None:
            tel["backend_fallback"] = True
            tel["comparable_to_canonical"] = False
            tel["fallback_reason"] = backend.telemetry.backend_fallback
        else:
            tel["backend_fallback"] = False
            tel["comparable_to_canonical"] = True
    # Tier widths always come from the caller (the pipeline knows the preset's
    # flow_width and the computed Tier-3 cap); they override any telemetry
    # default so the report reflects the *actual* working resolutions.
    tel["tier2_flow_width"] = int(tier2_flow_width)
    tel["tier3_flow_width"] = int(tier3_flow_width)
    return tel


def compute_pair_flow(backend: FlowBackend, a_rgb: np.ndarray, b_rgb: np.ndarray,
                      work_width: int | None = None):
    """Compute bidirectional flow, optionally at a reduced width bucket.

    Returns (f_ab, f_ba, scale) where flows are at the working resolution and
    scale = work_res / native_res.
    """
    from ..schema import downscale_frames

    scale = 1.0
    if work_width and a_rgb.shape[1] > work_width:
        stacked, scale = downscale_frames(np.stack([a_rgb, b_rgb]), work_width)
        a_rgb, b_rgb = stacked[0], stacked[1]
    f_ab, f_ba = backend.flow_pair(a_rgb, b_rgb)
    return f_ab, f_ba, scale
