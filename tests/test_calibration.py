"""Calibration harness self-test: on the synthetic severity ladder the
endpoint-reference score must track full-reference quality (§12.5 harness;
absolute acceptance numbers require the real corpora from §P3)."""

import numpy as np
import pytest

from rr_vfiqa.calibration import (build_severity_corpus, pairwise_accuracy,
                                  plcc, run_synthetic_validation, srcc)


def test_severity_ladder_monotone_fr(tmp_path):
    src, entries = build_severity_corpus(tmp_path, levels=5, n_frames=120,
                                          w=320, h=192)
    psnrs = [e.fr_psnr for e in entries]
    assert all(b < a for a, b in zip(psnrs, psnrs[1:]))   # strictly worsening
    assert entries[0].defects == []


def test_correlation_helpers():
    x = np.arange(10.0)
    assert srcc(x, x * 3 + 1) == pytest.approx(1.0)
    assert plcc(x, x * 3 + 1) == pytest.approx(1.0, abs=1e-9)
    assert pairwise_accuracy(x, x) == 1.0
    assert pairwise_accuracy(x, -x) == 0.0


def test_synthetic_validation_tracks_fr(tmp_path, flow_backend):
    out = run_synthetic_validation(tmp_path, preset="fast",
                                   flow_backend=flow_backend, device="cuda",
                                   levels=5)
    # Bootstrap-level acceptance on synthetic content (§P3 thresholds like
    # SRCC 0.8 apply to calibration on REAL data with human rankings). Here
    # the harness must demonstrate a strong, significant FR relationship.
    assert out["srcc_overall_vs_psnr"] >= 0.7
    assert out["plcc_overall_vs_psnr"] >= 0.8
    assert out["pairwise_accuracy"] >= 0.8


def test_model_ranking_on_interpolator_ladder(tmp_path):
    """§12.1 loop with organic model outputs: reference interpolators at
    different flow resolutions + naive blender, ranked against FR truth.
    The primary product claim — same-source model ranking — must hold."""
    from rr_vfiqa.calibration import run_model_validation

    out = run_model_validation(tmp_path, seeds=(7,), n_frames=160, w=320,
                               h=192, interp_backend="farneback",
                               device="cpu", widths=(224, None),
                               eval_preset="fast", eval_backend="farneback")
    by_model = {r["model"]: r for r in out["rows"]}
    # The perfect interleave must beat the naive crossfade blender.
    assert by_model["perfect"]["overall"] > by_model["linear"]["overall"]
    # Same-source ranking accuracy (§11.5): evaluator order vs FR order.
    assert out["within_seed_pairwise"] >= 0.75
    assert out["srcc"] >= 0.6
    # Motion-compensated features must track FR quality strongly (the
    # route's claim that temporal consistency is the backbone). At 4
    # candidates many features tie at |SRCC|=1.0, so check strength, not rank.
    corr = out["feature_correlations"]
    backbone = {k: v for k, v in corr.items()
                if k.startswith(("mct_", "cycle_", "comp_"))}
    assert backbone and max(backbone.values()) >= 0.8


def test_defect_localization_recall(tmp_path, flow_backend):
    """Each single-defect candidate must be localized with a correct label
    in its ground-truth segment (§P3: 各坏例类别 F1, worst-case recall)."""
    from rr_vfiqa.calibration import run_detection_eval

    out = run_detection_eval(tmp_path, defects=["blur", "freeze", "rotation_tear",
                                                "head_erase", "card_freeze",
                                                "ui_drift"],
                             preset="standard", flow_backend=flow_backend,
                             device="cuda")
    assert out["recall"] >= 0.8        # at most one missed category
    assert out["f1"] >= 0.5
    assert out["per_defect"]["freeze"]["detected"]       # direct copy check
    assert out["per_defect"]["card_freeze"]["detected"]  # flip progression
