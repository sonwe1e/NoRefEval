import numpy as np
import pytest

from rr_vfiqa.fusion.feature_normalizer import normalize_error
from rr_vfiqa.fusion.score_schema import aggregate_errors, compute_scores
from rr_vfiqa.config import EvalConfig


def test_normalize_forward_monotone():
    vals = [normalize_error("comp_mean", x) for x in (0.0, 0.05, 0.1, 0.3, 1.0)]
    assert all(b > a for a, b in zip(vals, vals[1:]))
    assert vals[0] == 0.0


def test_normalize_inverted_monotone():
    # Non-increasing in quality; strictly increasing below the good reference
    # (0.75 for edge_recall), saturated at 0 above it.
    vals = [normalize_error("edge_recall", x) for x in (0.9, 0.75, 0.5, 0.2, 0.0)]
    assert all(b >= a for a, b in zip(vals, vals[1:]))
    assert all(b > a for a, b in zip(vals[1:], vals[2:]))
    assert normalize_error("edge_recall", 0.9) == 0.0


def test_normalize_ratio_features():
    assert normalize_error("edge_count_odd_even_ratio", 1.0) == 0.0
    assert normalize_error("edge_count_odd_even_ratio", 0.5) > 0.8


def test_normalize_nan():
    assert normalize_error("comp_mean", float("nan")) is None
    assert normalize_error("unknown_key", 1.0) is None


def test_aggregate_weights_tail():
    errors = [0.1] * 90 + [1.0] * 10
    agg = aggregate_errors(errors)
    plain = np.mean(errors)
    assert agg > plain          # P90/P99 weighting lifts the tail


def test_score_ordering():
    cfg = EvalConfig.build("a", "b", "fast")
    good = {c: [0.1] for c in ("motion", "temporal", "structure", "character",
                               "thin_weapon", "ui", "transition", "global")}
    bad = {c: [0.6] for c in good}
    overall_g, scores_g, _ = compute_scores(cfg, good)
    overall_b, scores_b, _ = compute_scores(cfg, bad)
    assert overall_g > overall_b
    for k in scores_g:
        assert scores_g[k] > scores_b[k]


def test_calibrator_monotone(tmp_path):
    pytest.importorskip("lightgbm")
    from rr_vfiqa.fusion.monotonic_calibrator import FEATURE_ORDER, MonotonicCalibrator

    rng = np.random.default_rng(0)
    X = rng.uniform(0, 0.8, (240, len(FEATURE_ORDER)))
    X[:, 8] = rng.uniform(0.3, 1.0, 240)          # confidence column
    y = 100 * np.exp(-2.0 * (X[:, :8].mean(1))) + 5 * X[:, 8] + rng.normal(0, 1, 240)

    cal = MonotonicCalibrator().fit(X, np.clip(y, 0, 100), num_boost_round=150)
    feats = {k: 0.1 for k in FEATURE_ORDER}
    feats["confidence"] = 0.8
    base = cal.predict_quality(feats)
    feats["A_motion"] = 0.7
    worse = cal.predict_quality(feats)
    # Monotone AND unsaturated: the old ×100 bug pinned both values at 100.
    assert worse < base
    assert 0.0 < worse < 100.0 and 0.0 < base < 100.0

    p = tmp_path / "cal.pkl"
    cal.save(p)
    cal2 = MonotonicCalibrator.load(p)
    assert abs(cal2.predict_quality(feats) - worse) < 1e-6


def test_calibrator_pairwise_ranks_winner_first():
    pytest.importorskip("lightgbm")
    from rr_vfiqa.fusion.monotonic_calibrator import FEATURE_ORDER, MonotonicCalibrator

    rng = np.random.default_rng(1)
    n = 120
    d = len(FEATURE_ORDER)
    X_win = rng.uniform(0.0, 0.3, (n, d))     # winners: low errors
    X_lose = rng.uniform(0.4, 0.8, (n, d))    # losers: high errors
    cal = MonotonicCalibrator().fit_pairwise(X_win, X_lose, num_boost_round=200)

    win_feats = {k: 0.1 for k in FEATURE_ORDER}
    lose_feats = {k: 0.6 for k in FEATURE_ORDER}
    q_win = cal.predict_quality(win_feats)
    q_lose = cal.predict_quality(lose_feats)
    assert q_win > q_lose
    assert 0.0 < q_win <= 100.0 and 0.0 <= q_lose < 100.0
