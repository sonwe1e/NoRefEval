from types import SimpleNamespace

import numpy as np

from rr_vfiqa.cache.source_cache import SourceCache
from rr_vfiqa.config import EvalConfig
from rr_vfiqa.io.video_reader import VideoReader
from rr_vfiqa.motion.flow_estimator import FarnebackBackend
from rr_vfiqa.schema import FrameBundle


class _BackendVariant(FarnebackBackend):
    def __init__(self, name, weight_hash):
        super().__init__()
        self.name = name
        self._weight_hash = weight_hash

    def cache_identity(self):
        return {
            "backend": self.name,
            "algorithm_version": "test-v1",
            "weights_id": "test",
            "weights_hash": self._weight_hash,
        }


def test_source_cache_isolated_by_backend_weights_and_threshold(videos, tmp_path):
    reader = VideoReader(str(videos["source"]))
    cfg = EvalConfig.build(
        str(videos["source"]), str(videos["good"]), "fast",
        cache_dir=str(tmp_path), device="cpu", flow_backend="farneback")
    a = SourceCache(reader, cfg, backend=_BackendVariant("farneback", "aaa"))
    b = SourceCache(reader, cfg, backend=_BackendVariant("raft", "bbb"))
    assert a.root != b.root
    assert "flow_farneback_aaa" in a.root.name
    assert "flow_raft_bbb" in b.root.name
    assert a.contract["occlusion_cycle_threshold"] == 3.0
    assert a.contract["camera_algorithm_version"]

    cfg2 = EvalConfig.build(
        str(videos["source"]), str(videos["good"]), "fast",
        cache_dir=str(tmp_path), device="cpu", flow_backend="farneback",
        occlusion_cycle_threshold=5.0)
    c = SourceCache(reader, cfg2, backend=_BackendVariant("farneback", "aaa"))
    assert c.root != a.root


def test_compare_passes_backend_and_quarantines_failed_candidates(
        monkeypatch, tmp_path):
    import rr_vfiqa.pipeline as pipeline

    calls = []

    def fake_eval(_source, candidate, **kwargs):
        calls.append(kwargs["flow_backend"])
        if candidate == "failed.mp4":
            return SimpleNamespace(
                overall_score=float("nan"), confidence=0.0, scores={},
                meta={"status": "failed"})
        score = 80.0 if candidate == "a.mp4" else 60.0
        return SimpleNamespace(
            overall_score=score, confidence=0.8, scores={},
            meta={"status": "ok"})

    monkeypatch.setattr(pipeline, "evaluate_vfi", fake_eval)
    rows = pipeline.compare_models(
        "source.mp4", ["a.mp4", "failed.mp4", "b.mp4"],
        labels=["a", "failed", "b"], flow_backend="raft",
        out_dir=str(tmp_path))
    assert calls == ["raft", "raft", "raft"]
    assert [r["model"] for r in rows] == ["a", "b", "failed"]
    assert rows[0]["relative_vs_mean"] == 10.0
    assert rows[1]["relative_vs_mean"] == -10.0
    assert rows[2]["overall"] is None
    assert rows[2]["relative_vs_mean"] is None


def test_dynamic_ui_metrics_ignore_world_pixels():
    from rr_vfiqa.regions.ui_detector import compute_window

    h = w = 40
    mask = np.zeros((h, w), np.uint8)
    mask[4:14, 4:14] = 1

    class FakeUI:
        def mask_at(self, _h, _w):
            return mask

    frames = np.zeros((5, h, w, 3), np.uint8)
    # Dynamic UI: clean 50/50 transition inside the mask.
    frames[1][mask > 0] = 0
    frames[2][mask > 0] = 50
    frames[3][mask > 0] = 100
    # The world is strongly out of endpoint range and must not contaminate UI.
    frames[1][mask == 0] = 0
    frames[2][mask == 0] = 220
    frames[3][mask == 0] = 100
    bundle = FrameBundle(
        indices=np.arange(5, dtype=np.int32),
        times=np.arange(5) / 120.0,
        rgb=frames, width=w, height=h)
    cfg = EvalConfig.build("source.mp4", "candidate.mp4", "standard")
    out = compute_window(bundle, FakeUI(), cfg)
    assert out["ui_mode"] == 1.0
    assert out["ui_dyn_blend_frac"] == 1.0
    assert out["ui_dyn_out_of_range_frac"] == 0.0


def test_source_cache_thread_race_computes_each_entry_once(videos, tmp_path):
    """Cold-cache concurrent access must not corrupt or duplicate entries.

    The window loops evaluate in a thread pool, so several threads race on
    the same missing pair/edge entry.  Regression: without the double-checked
    lock + atomic store writes, one thread read a half-written npz and the
    edges stage failed with ``EOFError('No data left in file')``.
    """
    import threading

    reader = VideoReader(str(videos["source"]))
    cfg = EvalConfig.build(
        str(videos["source"]), str(videos["good"]), "fast",
        cache_dir=str(tmp_path), device="cpu", flow_backend="farneback")
    cache = SourceCache(reader, cfg, backend=FarnebackBackend())

    saves: list[str] = []
    orig_save = cache.store.save_npz

    def counting_save(key, **arrays):
        saves.append(key)
        return orig_save(key, **arrays)

    cache.store.save_npz = counting_save

    errors: list[str] = []

    def worker():
        try:
            for j in range(3):
                if cache.get_edges(j).edges is None:
                    errors.append("edges None")
                if cache.get_pair(j).f_01 is None:
                    errors.append("pair None")
        except Exception as exc:  # noqa: BLE001
            errors.append(repr(exc))

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"concurrent cache access raised: {errors[:3]}"
    # Every entry is written exactly once despite 8 racing threads.
    assert len(saves) == len(set(saves)), (
        f"duplicated cache writes: {[k for k in saves if saves.count(k) > 1]}")

    # Second wave on the warm cache: reads only, no computation.
    saves.clear()
    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert saves == [], f"warm-cache wave still wrote: {saves}"
