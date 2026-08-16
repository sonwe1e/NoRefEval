"""End-to-end pipeline tests on the synthetic corpus."""

import pytest

from rr_vfiqa import evaluate_vfi
from rr_vfiqa.schema import SUBSCORE_KEYS


@pytest.fixture(scope="module")
def reports(videos, cache_dir, flow_backend):
    out = {}
    for tag in ("good", "bad"):
        out[tag] = evaluate_vfi(
            str(videos["source"]), str(videos[tag]), preset="fast",
            cache_dir=cache_dir, device="cuda", flow_backend=flow_backend,
            out_dir=None, export_clips=False)
    return out


def test_report_schema(reports):
    for tag in ("good", "bad"):
        d = reports[tag].to_dict()
        assert set(SUBSCORE_KEYS) <= set(d["scores"])
        assert 0.0 <= d["overall_score"] <= 100.0
        assert 0.0 < d["confidence"] <= 1.0
        assert 0.0 <= d["event_ambiguity"] <= 1.0
        for w in d["worst_windows"]:
            assert w["end"] >= w["start"]
            assert w["type"]
            assert 0.0 <= w["severity"] <= 1.0
        # meta carries alignment provenance
        assert d["meta"]["alignment"]["anchor_error"] < 12.0


def test_bad_scores_below_good(reports):
    g, b = reports["good"].to_dict(), reports["bad"].to_dict()
    assert g["meta"]["score_schema"] == "endpoint-reduced-reference-v3"
    assert b["overall_score"] < g["overall_score"]
    assert b["scores"]["temporal_stability"] < g["scores"]["temporal_stability"]


def test_worst_windows_localized(reports):
    b = reports["bad"].to_dict()
    assert len(b["worst_windows"]) >= 1
    # Degraded segments live in mid-frame ranges [10..16], [30..36], [60..66]
    # → candidate times roughly [0.17..1.1], so some worst window must fall in.
    assert any(w["start"] < 1.2 for w in b["worst_windows"])
    assert all(w["type"] for w in b["worst_windows"])


def test_cache_speeds_second_run(videos, cache_dir, flow_backend):
    """Second evaluation reuses the source cache (no recomputation crash)."""
    r1 = evaluate_vfi(str(videos["source"]), str(videos["good"]), preset="fast",
                      cache_dir=cache_dir, device="cuda", flow_backend=flow_backend,
                      out_dir=None, export_clips=False)
    r2 = evaluate_vfi(str(videos["source"]), str(videos["good"]), preset="fast",
                      cache_dir=cache_dir, device="cuda", flow_backend=flow_backend,
                      out_dir=None, export_clips=False)
    assert abs(r1.overall_score - r2.overall_score) < 2.0


def test_out_dir_reports(videos, cache_dir, flow_backend, tmp_path):
    evaluate_vfi(str(videos["source"]), str(videos["bad"]), preset="fast",
                  cache_dir=cache_dir, device="cuda", flow_backend=flow_backend,
                  out_dir=str(tmp_path), export_clips=True)
    assert (tmp_path / "report.json").exists()
    assert (tmp_path / "timeline.md").exists()
    assert (tmp_path / "badcases").exists()
    clips = list((tmp_path / "badcases").glob("*.mp4"))
    assert len(clips) >= 1


def test_fail_closed_on_core_failures(videos, cache_dir, flow_backend, monkeypatch):
    """A broken core stage must not silently yield a good score (USERPLAN §7)."""
    from rr_vfiqa.metrics import (cycle_reconstruction, edge_structure,
                                  flow_composition_metric,
                                  global_technical_quality, parity_frequency,
                                  temporal_compensation)

    def boom(*a, **k):
        raise RuntimeError("simulated stage failure")

    for mod in (flow_composition_metric, cycle_reconstruction,
                temporal_compensation, edge_structure):
        monkeypatch.setattr(mod, "compute", boom)
    monkeypatch.setattr(parity_frequency, "compute_window", boom)
    monkeypatch.setattr(global_technical_quality, "compute_window", boom)

    rep = evaluate_vfi(str(videos["source"]), str(videos["good"]), preset="fast",
                       cache_dir=cache_dir, device="cuda", flow_backend=flow_backend,
                       out_dir=None, export_clips=False)
    d = rep.to_dict()
    assert d["meta"]["status"] == "failed"
    assert d["overall_score"] is None            # NaN must not become a number
    assert d["confidence"] < 0.05
    assert d["meta"]["stage_errors"]
    assert d["meta"]["stage_error_samples"]
