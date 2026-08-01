"""USERPLAN §6 P0: VRAM safety regression tests.

These tests cover the memory-aware RAFT execution policy, Tier-3 resolution
cap, VRAM telemetry recording, and the canonical backend selection policy.

They run on CPU / Farneback by default (no GPU required); the RAFT-specific
branches are exercised via a mock that simulates OOM behavior without needing
a real CUDA device.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from rr_vfiqa.config import AUDIT, BALANCED, FAST, STANDARD, Preset
from rr_vfiqa.motion.flow_estimator import (
    CANONICAL_RAFT_POLICY,
    FarnebackBackend,
    FlowBackend,
    FlowExecutionPolicy,
    FlowTelemetry,
    RaftBackend,
    collect_backend_telemetry,
    get_flow_backend,
    _is_oom_error,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _pair(h: int = 48, w: int = 48):
    """One directed flow pair of deterministic uint8 RGB frames."""
    rng = np.random.default_rng(0)
    a = rng.integers(0, 255, (h, w, 3)).astype(np.uint8)
    b = rng.integers(0, 255, (h, w, 3)).astype(np.uint8)
    return a, b


class _FakeTorch:
    """Minimal torch stub for testing RaftBackend without importing torch.

    Simulates a CUDA device with a configurable free-VRAM budget and optional
    OOM injection on the Nth ``_model`` call.
    """

    class _Device:
        def __init__(self, name):
            self.name = name

        def __str__(self):
            return self.name

    class _Cuda:
        def __init__(self, outer):
            self._outer = outer

        def is_available(self):
            return True

        def synchronize(self, device=None):
            pass

        def mem_get_info(self, device=None):
            # (free_bytes, total_bytes)
            free = int(self._outer._free_mb * 1e6)
            return free, free + int(8e9)

        def empty_cache(self):
            pass

        def max_memory_allocated(self, device=None):
            return int(self._outer._peak_mb * 1e6)

        def reset_peak_memory_stats(self, device=None):
            self._outer._peak_mb = 0.0

    def __init__(self, *, free_mb: float = 12000.0, oom_after: int = 0):
        self._free_mb = free_mb
        self._oom_after = oom_after
        self._calls = 0
        self._peak_mb = 0.0

    # USERPLAN §6 P0.4: the backend now uses inference_mode(); the stub must
    # provide it so the mock path exercises the real code.
    def inference_mode(self):
        return _FakeInferenceMode()

    @property
    def cuda(self):
        return self._Cuda(self)

    def device(self, name):
        return self._Device(name)

    def from_numpy(self, arr):
        return _FakeTensor(arr)

    @staticmethod
    def float32():
        return "float32"


class _FakeInferenceMode:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _FakeTensor:
    """Tensor-like wrapper around an ndarray for the stub."""

    def __init__(self, arr):
        self._arr = np.asarray(arr, dtype=np.float32)

    def to(self, device):
        return self

    def div(self, v):
        return _FakeTensor(self._arr / v)

    def sub(self, v):
        return _FakeTensor(self._arr - v)

    def cpu(self):
        return self

    def numpy(self):
        return self._arr

    @property
    def shape(self):
        return self._arr.shape

    def __getitem__(self, idx):
        return _FakeTensor(self._arr[idx])


class _FakeModel:
    """Returns a plausible (N, 2, H, W) flow tensor; optionally OOMs.

    ``oom_after=N`` means the backend succeeds for the first N calls, then
    OOMs on every subsequent call.  ``oom_after=0`` means every call OOMs.
    """

    def __init__(self, oom_after: int = 0):
        self._oom_after = oom_after
        self._calls = 0

    def __call__(self, fa, fb):
        self._calls += 1
        if self._calls > self._oom_after:
            raise RuntimeError("CUDA out of memory. Tried to allocate ...")
        n, _, h, w = fa._arr.shape
        # Half-resolution output like RAFT's final iteration.
        out = np.zeros((n, 2, h // 2, w // 2), dtype=np.float32)
        # Upsample back to (N, 2, H, W) so slicing [:, :, :h, :w] is a no-op.
        out = np.repeat(np.repeat(out, 2, axis=2), 2, axis=3)
        return [_FakeTensor(out), _FakeTensor(out)]


def _make_raft_backend(*, free_mb: float = 12000.0, oom_after: int = 0,
                       policy: FlowExecutionPolicy | None = None):
    """Build a RaftBackend backed by the fake torch stub."""
    import rr_vfiqa.motion.flow_estimator as fe

    torch_stub = _FakeTorch(free_mb=free_mb, oom_after=oom_after)
    fe.torch = torch_stub  # type: ignore[assignment]
    backend = RaftBackend.__new__(RaftBackend)
    backend._torch = torch_stub
    backend.device = torch_stub.device("cuda")
    backend._weights_id = "test-weights"
    backend._weights_url = "https://example.com/raft-small-01064c6d.pth"
    backend._torchvision_version = "0.18"
    backend._transforms = None
    backend._model = _FakeModel(oom_after=oom_after)
    backend.policy = policy or FlowExecutionPolicy()
    backend.telemetry = FlowTelemetry()
    backend._fallback = None
    return backend


# ---------------------------------------------------------------------------
# FlowExecutionPolicy / FlowTelemetry
# ---------------------------------------------------------------------------

class TestFlowExecutionPolicy:
    def test_defaults_target_12gb_budget(self):
        pol = FlowExecutionPolicy()
        assert pol.max_batch_size == 16
        assert pol.memory_budget_mb == 12000
        assert pol.precision == "fp32"
        assert pol.audit_max_width == 960
        assert pol.allow_oom_backoff is True

    def test_canonical_policy_matches_defaults(self):
        # USERPLAN §6 P0.3: the canonical profile is the safe default.
        assert CANONICAL_RAFT_POLICY.max_batch_size == 16
        assert CANONICAL_RAFT_POLICY.memory_budget_mb == 12000
        assert CANONICAL_RAFT_POLICY.audit_max_width == 960
        assert CANONICAL_RAFT_POLICY.allow_oom_backoff is True


class TestFlowTelemetry:
    def test_to_dict_rounds_peak_vram(self):
        tel = FlowTelemetry(peak_vram_mb=1234.567,
                            flow_batch_sizes=[16, 8, 4],
                            oom_retries=2,
                            backend_fallback="cuda_oom")
        d = tel.to_dict()
        assert d["peak_vram_mb"] == 1234.6
        assert d["flow_batch_sizes"] == [16, 8, 4]
        assert d["oom_retries"] == 2
        assert d["backend_fallback"] == "cuda_oom"
        assert d["phase_seconds"] == {}


# ---------------------------------------------------------------------------
# OOM detection helper
# ---------------------------------------------------------------------------

class TestIsOomError:
    def test_cuda_oom_message(self):
        assert _is_oom_error(RuntimeError("CUDA out of memory. Tried to allocate"))

    def test_torch_out_of_memory_class(self):
        class OutOfMemoryError(Exception):
            pass
        assert _is_oom_error(OutOfMemoryError("OOM"))

    def test_non_oom_error(self):
        assert not _is_oom_error(ValueError("shape mismatch"))

    def test_rocm_oom(self):
        assert _is_oom_error(RuntimeError("HIP out of memory"))


# ---------------------------------------------------------------------------
# Tier-3 resolution cap (config presets)
# ---------------------------------------------------------------------------

class TestTier3ResolutionCap:
    def test_fast_caps_at_480(self):
        assert FAST.tier3_max_width == 480

    def test_standard_caps_at_960(self):
        assert STANDARD.tier3_max_width == 960

    def test_audit_caps_at_960(self):
        # USERPLAN §6 P0.2: audit Tier-3 is capped even for 4K sources.
        assert AUDIT.tier3_max_width == 960

    def test_balanced_caps_at_960(self):
        # USERPLAN §6 P0.2: the default inspect path must not OOM on 16 GB GPUs.
        assert BALANCED.tier3_max_width == 960

    def test_custom_preset_respects_zero_as_unlimited(self):
        # A tier3_max_width of 0 means "no cap" (use native resolution).
        p = Preset(name="custom", tier3_max_width=0)
        assert p.tier3_max_width == 0

    def test_cap_never_exceeds_source_width(self):
        # The pipeline computes min(source_width, cap); verify the math.
        source_width = 1920
        cap = BALANCED.tier3_max_width
        assert min(source_width, cap) == 960

        source_width = 480
        assert min(source_width, cap) == 480


# ---------------------------------------------------------------------------
# collect_backend_telemetry
# ---------------------------------------------------------------------------

class TestCollectBackendTelemetry:
    def test_farneback_contributes_only_tier_widths(self):
        backend = FarnebackBackend()
        tel = collect_backend_telemetry(
            backend, tier2_flow_width=960, tier3_flow_width=960)
        assert tel["tier2_flow_width"] == 960
        assert tel["tier3_flow_width"] == 960
        # Farneback has no VRAM tracking.
        assert "peak_vram_mb" not in tel
        assert "backend_fallback" not in tel

    def test_raft_without_fallback_is_canonical(self):
        # oom_after=100 means the first 100 model calls succeed — no fallback.
        backend = _make_raft_backend(oom_after=100)
        # Run one batch so telemetry is populated.
        backend.flow_many([_pair()])
        tel = collect_backend_telemetry(
            backend, tier2_flow_width=960, tier3_flow_width=960)
        assert tel["tier2_flow_width"] == 960
        assert tel["tier3_flow_width"] == 960
        assert tel["backend_fallback"] is False
        assert tel["comparable_to_canonical"] is True
        assert "fallback_reason" not in tel

    def test_raft_with_fallback_is_not_canonical(self):
        # Force an OOM on every model call so the backend falls back.
        backend = _make_raft_backend(oom_after=0)
        backend.flow_many([_pair()])
        tel = collect_backend_telemetry(
            backend, tier2_flow_width=960, tier3_flow_width=960)
        assert tel["backend_fallback"] is True
        assert tel["comparable_to_canonical"] is False
        assert tel["fallback_reason"] == "cuda_oom"


# ---------------------------------------------------------------------------
# Memory-aware micro-batch (mocked RAFT)
# ---------------------------------------------------------------------------

class TestMemoryAwareMicroBatch:
    def test_empty_input_returns_empty(self):
        backend = _make_raft_backend(oom_after=100)
        assert backend.flow_many([]) == []

    def test_single_pair_returns_single_flow(self):
        backend = _make_raft_backend(oom_after=100)
        flows = backend.flow_many([_pair()])
        assert len(flows) == 1
        h, w = 48, 48
        assert flows[0].shape == (h, w, 2)
        assert flows[0].dtype == np.float32

    def test_batch_sizes_recorded_in_telemetry(self):
        backend = _make_raft_backend(oom_after=100)
        pairs = [_pair() for _ in range(5)]
        backend.flow_many(pairs)
        # The backend should have recorded the batch size(s) it used.
        assert len(backend.telemetry.flow_batch_sizes) >= 1
        assert sum(backend.telemetry.flow_batch_sizes) >= 5

    def test_oom_backoff_halves_batch(self):
        # Succeed on the first call, OOM on the second, then succeed after the
        # batch is halved.  Use enough pairs that a second batch is attempted.
        backend = _make_raft_backend(oom_after=1)
        pairs = [_pair() for _ in range(20)]
        flows = backend.flow_many(pairs)
        assert len(flows) == 20
        # At least one OOM retry should have been recorded.
        assert backend.telemetry.oom_retries >= 1

    def test_oom_fallback_to_farneback(self):
        # OOM on every model call → degraded Farneback fallback.
        backend = _make_raft_backend(oom_after=0)
        pairs = [_pair() for _ in range(3)]
        flows = backend.flow_many(pairs)
        assert len(flows) == 3
        for f in flows:
            assert f.shape == (48, 48, 2)
            assert f.dtype == np.float32
        # The fallback must be recorded.
        assert backend.telemetry.backend_fallback == "cuda_oom"


# ---------------------------------------------------------------------------
# inference_mode (USERPLAN §6 P0.4)
# ---------------------------------------------------------------------------

class TestInferenceMode:
    def test_forward_batch_uses_inference_mode(self):
        """Verify the backend calls torch.inference_mode(), not no_grad()."""
        entered = {"inference_mode": False}

        class _Ctx:
            def __enter__(self_):
                entered["inference_mode"] = True
                return self_

            def __exit__(self_, *a):
                return False

        backend = _make_raft_backend()
        # Patch the stub's inference_mode to record entry.
        backend._torch.inference_mode = lambda: _Ctx()  # type: ignore[assignment]
        backend.flow_many([_pair()])
        assert entered["inference_mode"] is True


# ---------------------------------------------------------------------------
# get_flow_backend canonical policy
# ---------------------------------------------------------------------------

class TestGetFlowBackend:
    def test_auto_uses_canonical_policy(self, monkeypatch):
        """USERPLAN §6 P0.3: auto + CUDA resolves to the canonical profile."""
        captured = {}

        def fake_raft_init(self, device="cuda", policy=None):
            captured["policy"] = policy
            self._torch = SimpleNamespace(
                cuda=SimpleNamespace(is_available=lambda: True))
            self.device = None
            self._weights_id = "x"
            self._weights_url = "https://x/raft-small-01064c6d.pth"
            self._torchvision_version = "0.18"
            self._transforms = None
            self._model = None
            self.policy = policy
            self.telemetry = FlowTelemetry()
            self._fallback = None

        monkeypatch.setattr(RaftBackend, "__init__", fake_raft_init)
        # Clear the backend cache so we get a fresh instance.
        from rr_vfiqa.motion import flow_estimator as fe
        fe._backends.clear()
        get_flow_backend("auto", device="cuda")
        assert captured["policy"] is CANONICAL_RAFT_POLICY


# ---------------------------------------------------------------------------
# Report.performance field (schema)
# ---------------------------------------------------------------------------

class TestReportPerformanceField:
    def test_report_accepts_performance(self):
        from rr_vfiqa.schema import Report
        rep = Report(
            overall_score=88.0, confidence=0.8,
            scores={"a": 1.0}, event_ambiguity=0.1,
            worst_windows=[], features={}, meta={},
            performance={"peak_vram_mb": 907.0, "oom_retries": 0,
                         "backend_fallback": False, "comparable_to_canonical": True},
        )
        d = rep.to_dict()
        assert d["performance"]["peak_vram_mb"] == 907.0
        assert d["performance"]["backend_fallback"] is False
        assert d["performance"]["comparable_to_canonical"] is True

    def test_report_performance_defaults_to_empty(self):
        from rr_vfiqa.schema import Report
        rep = Report(overall_score=88.0, confidence=0.8, scores={},
                     event_ambiguity=0.1, worst_windows=[], features={}, meta={})
        assert rep.performance == {}
        assert rep.to_dict()["performance"] == {}
