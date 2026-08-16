"""Unit tests for the diagnostic evidence engine (USERPLAN §5, §7).

Synthetic scalar dicts only — no video decoding, no GPU.  Verifies that
diagnosis is multi-evidence, that cadence uses a conservative multiplicative
penalty, and that issues merge / rank correctly.
"""

from __future__ import annotations

import math

import pytest

from rr_vfiqa.diagnosis import (
    DiagnosticIssue, Evidence, TRACKS, cadence_integrity, cadence_penalty,
    confirmed_affected_seconds, diagnose_windows, issue_level, merge_issues,
)
from rr_vfiqa.diagnosis.rules import (
    RULES, Condition, affected_duration_fraction, evaluate_rules,
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


def _collapsed_nr():
    """Full v2 cadence-collapse evidence: parent motion + strong, coherent
    parity asymmetry + duplicate/freeze corroboration."""
    return {
        "nr_parent_motion_p90": 25.0,
        "nr_parent_moving_pixel_fraction": 0.5,
        "nr_phase_asymmetry": 0.9,
        "nr_phase_gap_signed": 0.85,
        "nr_native_duplicate_fraction": 0.5,
        "nr_native_freeze_fraction": 0.4,
        "nr_mct_native_mean": 1.5,
        "nr_native_self_comp": 0.01,
        "nr_native_self_cycle": 1.0,
    }


def test_cadence_integrity_aggregation():
    # Clean windows have no parent-motion / asymmetry signals -> zero risk.
    clean_windows = [_clean_nr() for _ in range(8)]
    rep = cadence_integrity(clean_windows, common_overall=85.0)
    assert rep.cadence_risk < 0.15
    assert rep.cadence_integrity > 70.0
    # separate reporting: overall stays the artifact-quality score.
    assert rep.overall == pytest.approx(85.0, rel=1e-6)

    # collapse every window with FULL v2 evidence.
    rep2 = cadence_integrity([_collapsed_nr()] * 8, common_overall=85.0)
    assert rep2.cadence_risk > 0.5
    assert rep2.cadence_integrity < 40.0
    # separate mode: overall is NOT exponentially crushed (USERPLAN §5).
    assert rep2.overall == pytest.approx(85.0, rel=1e-6)


def test_cadence_unstable_direction_is_suppressed():
    # Same collapse evidence but the signed phase gap flips sign across
    # windows -> the coherence gate suppresses the risk (USERPLAN §4.2 gate 3).
    rep = cadence_integrity(
        [_collapsed_nr()] * 4, common_overall=85.0,
        phase_gap_signed_per_window=[0.8, -0.8, 0.8, -0.8])
    assert rep.cadence_risk == 0.0, rep.evidence
    assert rep.gates["phase_coherence"] is False
    assert rep.overall == pytest.approx(85.0, rel=1e-6)


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


def test_affected_duration_uses_support_spans():
    # A wide display span backed by narrow sampled windows must NOT count the
    # unsampled gaps (USERPLAN §9): affected-duration is the support union.
    issues = [
        DiagnosticIssue("a", "t", 0.8, 0.9, 0.0, 2.0, "Temporal",
                        support_spans=[[0.1, 0.3], [1.7, 1.9]]),
    ]
    # support union = 0.2 + 0.2 = 0.4; the display span alone would have been 2.0
    assert confirmed_affected_seconds(issues) == pytest.approx(0.4)
    assert affected_duration_fraction(issues, total_seconds=10.0) \
        == pytest.approx(0.04)


def test_merge_issues_unions_support_spans():
    a = DiagnosticIssue("x", "t", 0.3, 0.8, 0.0, 0.5, "Temporal",
                        support_spans=[[0.02, 0.09]])
    b = DiagnosticIssue("x", "t", 0.9, 0.8, 0.6, 1.4, "Temporal",
                        support_spans=[[0.61, 0.68], [1.31, 1.38]])
    merged = merge_issues([a, b], nms_seconds=0.5)
    assert len(merged) == 1
    # display span stays the whole cluster union
    assert merged[0].start_time == pytest.approx(0.0)
    assert merged[0].end_time == pytest.approx(1.4)
    # support_spans keep every sampled interval, overlap-deduped + sorted
    assert merged[0].support_spans == [
        [0.02, 0.09], [0.61, 0.68], [1.31, 1.38]]


def test_merge_issues_dedupes_overlapping_support_spans():
    a = DiagnosticIssue("x", "t", 0.3, 0.8, 0.0, 0.9, "Temporal",
                        support_spans=[[0.1, 0.5]])
    b = DiagnosticIssue("x", "t", 0.4, 0.8, 0.3, 1.4, "Temporal",
                        support_spans=[[0.4, 0.7], [1.2, 1.3]])
    merged = merge_issues([a, b], nms_seconds=0.5)
    assert len(merged) == 1
    assert merged[0].support_spans == [[0.1, 0.7], [1.2, 1.3]]


def test_diagnose_windows_records_support_spans():
    freeze = dict(_clean_nr(), nr_native_duplicate_fraction=0.5,
                  nr_native_freeze_fraction=0.4, nr_native_self_comp=0.01,
                  nr_mct_native_mean=2.0)
    issues = diagnose_windows([(1.0, 1.2, 120, freeze, 0.9)], nms_seconds=0.5)
    assert len(issues) == 1
    assert issues[0].support_spans == [[1.0, 1.2]]


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


def test_issue_level_mapping():
    # Worst-local-issue level reuses the severity-band thresholds (USERPLAN §10).
    assert issue_level(0.9) == severity_band(0.9)
    assert issue_level(0.5)[0] == "high"
    assert issue_level(0.25)[0] == "medium"
    assert issue_level(0.05)[0] == "low"


def test_issue_dict_marks_causes_inferred():
    issue = DiagnosticIssue("duplicate_freeze", "t", 0.8, 0.9, 1.0, 1.2,
                            "Temporal", probable_causes=["x"],
                            evidence=[Evidence("m", 1.0, 0.5, 0.8)])
    d = issue.to_dict()
    assert d["causes_are_inferred"] is True
    assert d["severity_band"] == "severe"
    assert d["evidence"][0]["strength"] == 0.8


def test_issue_dict_serializes_support_spans():
    issue = DiagnosticIssue("duplicate_freeze", "t", 0.8, 0.9, 1.0, 1.2,
                            "Temporal",
                            support_spans=[[1.02, 1.09], [1.51, 1.58]])
    d = issue.to_dict()
    assert d["support_spans"] == [[1.02, 1.09], [1.51, 1.58]]
    # issues without support_spans still serialize (backward compatible)
    assert DiagnosticIssue("x", "t", 0.1, 0.5, 0.0, 1.0,
                           "Temporal").to_dict()["support_spans"] == []


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


def test_build_diagnostics_block_splits_quality_levels():
    from rr_vfiqa.diagnosis import build_diagnostics_block

    class _Meta:
        n_frames = 240
        fps = 60.0

    freeze = dict(_clean_nr(), nr_native_duplicate_fraction=0.5,
                  nr_native_freeze_fraction=0.4, nr_native_self_comp=0.01,
                  nr_mct_native_mean=2.0)
    issues = diagnose_windows(
        [(1.0, 1.2, 60, freeze, 0.9), (3.0, 3.2, 180, freeze, 0.9)],
        nms_seconds=0.5)
    diag = build_diagnostics_block(issues, _Meta(), 0.8)
    # USERPLAN §10: global level (based on overall, unknown here) is separate
    # from the worst local issue level (based on the worst issue severity).
    assert diag["global_quality_level"]["key"] == "unknown"
    assert diag["global_quality_level_basis"] == "overall"
    assert diag["worst_issue_level"]["key"] in {"severe", "high", "medium", "low"}
    assert diag["worst_issue"] is not None
    # USERPLAN §9: confirmed seconds + estimated fraction from support spans.
    assert diag["confirmed_affected_seconds"] == pytest.approx(0.4)
    assert diag["estimated_affected_fraction"] == pytest.approx(0.1)
    assert diag["affected_duration_fraction"] == pytest.approx(0.1)


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
    # USERPLAN §10: global vs worst-local level shown as independent KPIs.
    assert "Global Quality" in htmlstr
    assert "Worst Local Issue" in htmlstr
    # one card per diagnosed issue, each with a timeline target
    assert htmlstr.count('class="card band-') == len(issues)
    for track in ("Temporal", "UI/Text"):
        assert f">{track}<" in htmlstr or f'>{track}</span>' in htmlstr
    # USERPLAN §9: sampled support spans shown on each card
    assert "采样" in htmlstr
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

