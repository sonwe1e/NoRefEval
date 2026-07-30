"""Unit tests for the diagnostic evidence engine (USERPLAN §5, §7).

Synthetic scalar dicts only — no video decoding, no GPU.  Verifies that
diagnosis is multi-evidence, that cadence uses a conservative multiplicative
penalty, and that issues merge / rank correctly.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from rr_vfiqa.diagnosis import (
    DiagnosticIssue, Evidence, TRACKS, cadence_integrity, cadence_penalty,
    diagnose_windows, merge_issues,
)
from rr_vfiqa.diagnosis.rules import (
    RULES, Condition, Rule, affected_duration_fraction, evaluate_rules,
)
from rr_vfiqa.diagnosis.schema import quality_level, severity_band


# --------------------------------------------------------------------- rules
def _clean_nr():
    """Scalars that should NOT fire any NR issue."""
    return {
        "nr_native_duplicate_fraction": 0.01,
        "nr_native_freeze_fraction": 0.01,
        "nr_native_self_comp": 0.12,
        "nr_mct_native_mean": 16.0,
        "gtq_sharp_odd_even_ratio": 0.97,
        "parity_window_sharp_gap": 0.02,
        "edge_ghost_frac": 0.01,
        "flow_fold_frac": 0.005,
        "nr_ui_edge_instability": 0.02,
    }


def test_clean_window_fires_no_issue():
    assert evaluate_rules(_clean_nr()) == []


def test_freeze_requires_multiple_signals():
    # A lone duplicate fraction below the corroboration bar must NOT fire.
    sc = _clean_nr()
    sc["nr_native_duplicate_fraction"] = 0.20
    fired = evaluate_rules(sc)
    types = [r.issue_type for r, _, _ in fired]
    assert "duplicate_freeze" not in types


def test_freeze_fires_with_corroboration():
    sc = _clean_nr()
    sc["nr_native_duplicate_fraction"] = 0.45
    sc["nr_native_self_comp"] = 0.01     # frames carry no new content
    sc["nr_mct_native_mean"] = 2.0
    fired = evaluate_rules(sc)
    types = [r.issue_type for r, _, _ in fired]
    assert "duplicate_freeze" in types
    rule, evs, mass = [t for t in fired if t[0].issue_type == "duplicate_freeze"][0]
    assert len(evs) >= 2
    assert mass >= rule.min_mass
    for e in evs:
        assert 0.0 < e.strength <= 1.0


def test_ui_instability_rule():
    sc = _clean_nr()
    sc["nr_ui_edge_instability"] = 0.30
    sc["nr_text_stroke_instability"] = 0.40
    sc["nr_ui_component_instability"] = 0.50
    types = [r.issue_type for r, _, _ in evaluate_rules(sc)]
    assert "ui_text_instability" in types


def test_fr_color_mismatch_needs_structure_ok():
    # High chroma error ALONE is not enough (min_evidence=2); structure must
    # also speak (here we deliberately keep luma/ssim healthy so the rule's
    # lower_is_worse conditions contribute corroboration).
    sc = {"fr_scan_chroma_l1": 12.0, "fr_l1_y": 3.0, "fr_ssim": 0.96}
    types = [r.issue_type for r, _, _ in evaluate_rules(sc)]
    assert "fr_color_pipeline_mismatch" in types


def test_condition_direction_lower_is_worse():
    c = Condition("fr_ssim", threshold=0.9, weight=1.0,
                  direction="lower_is_worse", scale=0.2)
    assert c.evaluate({"fr_ssim": 0.95}) is None      # healthy -> no evidence
    ev = c.evaluate({"fr_ssim": 0.5})
    assert ev is not None and ev.strength > 0


def test_every_rule_has_track_and_causes():
    for r in RULES:
        assert r.track in TRACKS, r.issue_type
        assert r.probable_causes, r.issue_type
        assert r.min_evidence >= 2, r.issue_type     # multi-signal floor


# ------------------------------------------------------------------ cadence
def test_cadence_penalty_clean_is_identity():
    assert cadence_penalty(80.0, 0.0) == pytest.approx(80.0)


def test_cadence_penalty_collapsed_lowers_score():
    clean = cadence_penalty(80.0, 0.0)
    bad = cadence_penalty(80.0, 0.9)
    assert bad < clean
    # conservative but decisive: a near-collapsed cadence must bite hard
    assert bad < 0.5 * clean


def test_cadence_penalty_nan_propagates():
    assert math.isnan(cadence_penalty(float("nan"), 0.5))


def test_cadence_integrity_aggregation():
    clean_windows = [_clean_nr() for _ in range(8)]
    rep = cadence_integrity(clean_windows, common_overall=85.0)
    assert rep.cadence_risk < 0.15
    assert rep.cadence_integrity > 70.0
    assert rep.overall == pytest.approx(
        85.0 * math.exp(-2.2 * rep.cadence_risk), rel=1e-6)

    # collapse every window: native dup + freeze + near-zero native mct/comp
    bad = dict(_clean_nr())
    bad.update(nr_native_duplicate_fraction=0.5, nr_native_freeze_fraction=0.4,
               nr_mct_native_mean=1.5, nr_native_self_comp=0.01,
               nr_native_self_cycle=1.0)
    rep2 = cadence_integrity([bad] * 8, common_overall=85.0)
    assert rep2.cadence_risk > 0.5
    assert rep2.cadence_integrity < 40.0
    assert rep2.overall < rep.overall


def test_cadence_alternation_signal_helps():
    base = dict(_clean_nr(), nr_mct_native_mean=4.0, nr_native_self_comp=0.03)
    rep_no = cadence_integrity([base], common_overall=80.0)
    rep_alt = cadence_integrity([base], common_overall=80.0,
                                alternation_per_window=[0.9])
    assert rep_alt.cadence_risk >= rep_no.cadence_risk


# --------------------------------------------------------- issues / merging
def test_diagnose_windows_builds_and_merges():
    freeze = dict(_clean_nr(), nr_native_duplicate_fraction=0.5,
                  nr_native_freeze_fraction=0.4, nr_native_self_comp=0.01,
                  nr_mct_native_mean=2.0)
    ws = [
        (1.0, 1.1, 120, freeze, 0.9),
        (1.2, 1.3, 144, freeze, 0.9),   # close -> should merge with above
        (5.0, 5.1, 600, _clean_nr(), 0.9),  # clean -> no issue
    ]
    issues = diagnose_windows(ws, nms_seconds=0.5)
    freeze_issues = [i for i in issues if i.issue_type == "duplicate_freeze"]
    assert len(freeze_issues) == 1              # merged
    assert freeze_issues[0].end_time >= 1.3
    assert freeze_issues[0].probable_causes     # inferred causes present
    assert freeze_issues[0].evidence


def test_diagnose_severity_orders_descending():
    a = DiagnosticIssue("x", "t", 0.3, 0.8, 0.0, 0.1, "Temporal",
                        evidence=[Evidence("m", 1.0, 0.5, 0.5)])
    b = DiagnosticIssue("x", "t", 0.9, 0.8, 1.0, 1.1, "Temporal",
                        evidence=[Evidence("m", 1.0, 0.5, 0.9)])
    merged = merge_issues([a, b], 0.0)
    assert merged[0].severity >= merged[1].severity


def test_affected_duration_union():
    issues = [
        DiagnosticIssue("a", "t", 0.5, 0.8, 1.0, 2.0, "Temporal"),
        DiagnosticIssue("b", "t", 0.5, 0.8, 1.5, 2.5, "Motion"),   # overlaps
        DiagnosticIssue("c", "t", 0.5, 0.8, 5.0, 6.0, "Structure"),
    ]
    frac = affected_duration_fraction(issues, total_seconds=10.0)
    # union = [1.0,2.5] + [5.0,6.0] = 1.5 + 1.0 = 2.5 -> 0.25
    assert frac == pytest.approx(0.25)


# ------------------------------------------------------------------ schema
def test_severity_and_quality_bands():
    assert severity_band(0.9)[0] == "severe"
    assert severity_band(0.5)[0] == "high"
    assert severity_band(0.25)[0] == "medium"
    assert severity_band(0.05)[0] == "low"
    assert quality_level(95)[0] == "excellent"
    assert quality_level(70)[0] == "noticeable"
    assert quality_level(30)[0] == "severe"
    assert quality_level(float("nan"))[0] == "unknown"


def test_issue_dict_marks_causes_inferred():
    issue = DiagnosticIssue("duplicate_freeze", "t", 0.8, 0.9, 1.0, 1.2,
                            "Temporal", probable_causes=["x"],
                            evidence=[Evidence("m", 1.0, 0.5, 0.8)])
    d = issue.to_dict()
    assert d["causes_are_inferred"] is True
    assert d["severity_band"] == "severe"
    assert d["evidence"][0]["strength"] == 0.8


# ----------------------------------------------------------- HTML report
def _fake_report():
    from rr_vfiqa.schema import Report, WorstWindow
    from rr_vfiqa.diagnosis import build_diagnostics_block

    class _Meta:
        n_frames = 240
        fps = 60.0

    issues_raw = [
        (1.0, 1.2, 60, dict(_clean_nr(), nr_native_duplicate_fraction=0.5,
                            nr_native_freeze_fraction=0.4,
                            nr_native_self_comp=0.01, nr_mct_native_mean=2.0), 0.9),
        (3.0, 3.2, 180, dict(_clean_nr(), nr_ui_edge_instability=0.3,
                             nr_text_stroke_instability=0.4,
                             nr_ui_component_instability=0.5), 0.8),
    ]
    issues = diagnose_windows(issues_raw, nms_seconds=0.5)
    diag = build_diagnostics_block(issues, _Meta(), 0.8)
    return Report(
        overall_score=58.0, confidence=0.8, event_ambiguity=0.1,
        scores={"temporal_stability": 60.0, "ui_text_stability": 40.0},
        worst_windows=[WorstWindow(1.0, 1.2, 60, ["duplicate_freeze"], 0.8, 0.8)],
        features={},
        meta={"mode": "no-reference", "status": "ok",
              "limitations": ["test limit"],
              "diagnostics": diag,
              "cadence": {"cadence_risk": 0.6, "cadence_integrity": 30.0,
                          "common_time_quality": 80.0, "evidence": []}},
    ), issues


def test_html_report_renders_key_sections(tmp_path):
    from rr_vfiqa.report.html_report import render_html_report, write_html_report
    report, issues = _fake_report()
    htmlstr = render_html_report(report)
    assert "<!doctype html>" in htmlstr.lower()
    assert "Overall Quality" in htmlstr
    assert "Cadence Integrity" in htmlstr
    assert "问题卡片" in htmlstr
    assert "可能原因" in htmlstr
    assert "推断" in htmlstr                  # causes tagged as inferred
    assert "MOS" in htmlstr                    # honest semantics note
    # one card per diagnosed issue, each with a timeline target
    assert htmlstr.count('class="card band-') == len(issues)
    for track in ("Temporal", "UI/Text"):
        assert f">{track}<" in htmlstr or f'>{track}</span>' in htmlstr
    # self-contained: no external links / scripts / stylesheets
    assert "http://" not in htmlstr and "https://" not in htmlstr
    assert "<script src" not in htmlstr and "<link" not in htmlstr
    # writes to disk
    p = write_html_report(report, tmp_path / "report.html")
    assert p.exists() and p.read_text(encoding="utf-8").startswith("<!doctype")


def test_html_report_handles_no_issues():
    from rr_vfiqa.report.html_report import render_html_report
    from rr_vfiqa.schema import Report
    report = Report(overall_score=92.0, confidence=0.9, event_ambiguity=0.0,
                    scores={}, worst_windows=[], features={},
                    meta={"mode": "full-reference", "status": "ok",
                          "diagnostics": {"issues": [], "issue_count": 0,
                                          "total_seconds": 4.0}})
    htmlstr = render_html_report(report)
    assert "未" in htmlstr or "0" in htmlstr   # zero-issue path renders

