"""Monotonic LightGBM calibrator (USERPLAN §11.4).

Trained on (per-category aggregated errors, auxiliary features) → subjective
score with monotone constraints so more error can never raise the score.
Used when a trained model file exists; the bootstrap formula is the fallback.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

FEATURE_ORDER = [
    "A_motion", "A_temporal", "A_structure", "A_character", "A_thin_weapon",
    "A_ui", "A_transition", "A_global", "confidence", "event_ambiguity",
]
# all error features are monotonically non-increasing w.r.t. quality;
# confidence is non-decreasing.
MONOTONE_CONSTRAINTS = [-1, -1, -1, -1, -1, -1, -1, -1, +1, -1]


class MonotonicCalibrator:
    def __init__(self, model=None, mode: str = "mos"):
        self.model = model
        self.mode = mode          # "mos": 0–100 regression; "pair": rank logit
        self.meta: dict = {}

    # -- training -------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray, num_boost_round: int = 400):
        """MOS regression on the 0–100 scale. predict returns that scale
        directly (no rescaling — the previous ×100 saturated every output)."""
        import lightgbm as lgb

        assert X.shape[1] == len(FEATURE_ORDER)
        params = {
            "objective": "regression",
            "metric": "rmse",
            "monotone_constraints": MONOTONE_CONSTRAINTS,
            "monotone_constraints_method": "advanced",
            "learning_rate": 0.03,
            "num_leaves": 15,
            "min_data_in_leaf": 8,
            "feature_fraction": 0.9,
            "verbose": -1,
        }
        ds = lgb.Dataset(X, label=y, feature_name=FEATURE_ORDER)
        self.model = lgb.train(params, ds, num_boost_round=num_boost_round)
        self.mode = "mos"
        return self

    def fit_pairwise(self, X_win: np.ndarray, X_lose: np.ndarray,
                     num_boost_round: int = 400):
        """A/B preference training with the actual pair-rank loss (§11.4):

            L = log(1 + exp(-(s_win − s_lose)))

        via a custom LightGBM objective on stacked [winners; losers], so
        monotone constraints still apply. The learned score is a ranking
        logit; predict_quality maps it through a sigmoid onto 0–100.
        """
        import lightgbm as lgb

        n = len(X_win)
        assert len(X_lose) == n and X_win.shape[1] == len(FEATURE_ORDER)
        X = np.concatenate([X_win, X_lose], 0)

        def pair_rank_obj(preds, _ds):
            a, b = preds[:n], preds[n:]
            s = 1.0 / (1.0 + np.exp(np.clip(-(a - b), -60, 60)))   # σ(a−b)
            g = np.concatenate([s - 1.0, 1.0 - s])
            h = np.concatenate([s * (1.0 - s)] * 2) + 1e-6
            return g, h

        params = {
            "objective": pair_rank_obj,       # LightGBM ≥4.0 callable objective
            "monotone_constraints": MONOTONE_CONSTRAINTS,
            "monotone_constraints_method": "advanced",
            "learning_rate": 0.05,
            "num_leaves": 15,
            "min_data_in_leaf": 4,
            "feature_fraction": 0.9,
            "verbose": -1,
        }
        ds = lgb.Dataset(X, label=np.zeros(2 * n), feature_name=FEATURE_ORDER)
        self.model = lgb.train(params, ds, num_boost_round=num_boost_round)
        self.mode = "pair"
        return self

    # -- inference --------------------------------------------------------------

    def predict_quality(self, features: dict[str, float]) -> float:
        x = np.asarray([[features.get(k, 0.0) for k in FEATURE_ORDER]], np.float64)
        raw = float(self.model.predict(x)[0])
        if self.mode == "pair":
            return float(100.0 / (1.0 + np.exp(-np.clip(raw, -60, 60))))
        return float(np.clip(raw, 0.0, 100.0))

    # -- persistence ---------------------------------------------------------------

    def save(self, path: str | Path, meta: dict | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "mode": self.mode,
                         "meta": meta or {}}, f)
        (path.with_suffix(".meta.json")).write_text(
            json.dumps({"feature_order": FEATURE_ORDER,
                        "monotone_constraints": MONOTONE_CONSTRAINTS,
                        "mode": self.mode,
                        "meta": meta or {}}, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "MonotonicCalibrator":
        with open(path, "rb") as f:
            blob = pickle.load(f)
        cal = cls(blob["model"], mode=blob.get("mode", "mos"))
        cal.meta = blob.get("meta", {})
        return cal


def maybe_load(path: str | None) -> MonotonicCalibrator | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    return MonotonicCalibrator.load(p)
