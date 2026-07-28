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
    def __init__(self, model=None):
        self.model = model
        self.meta: dict = {}

    # -- training -------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray, num_boost_round: int = 400):
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
        return self

    def fit_pairwise(self, X_win: np.ndarray, X_lose: np.ndarray,
                     num_boost_round: int = 400):
        """A/B preference training (§11.4): winner score should beat loser.

        Implemented as regression toward {1, 0} on stacked pairs — a simple,
        monotone-constraint-preserving surrogate for the pair-rank loss.
        """
        X = np.concatenate([X_win, X_lose], 0)
        y = np.concatenate([np.ones(len(X_win)), np.zeros(len(X_lose))])
        return self.fit(X, y, num_boost_round)

    # -- inference --------------------------------------------------------------

    def predict_quality(self, features: dict[str, float]) -> float:
        x = np.asarray([[features.get(k, 0.0) for k in FEATURE_ORDER]], np.float64)
        raw = float(self.model.predict(x)[0])
        return float(np.clip(raw * 100.0, 0.0, 100.0))

    # -- persistence ---------------------------------------------------------------

    def save(self, path: str | Path, meta: dict | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "meta": meta or {}}, f)
        (path.with_suffix(".meta.json")).write_text(
            json.dumps({"feature_order": FEATURE_ORDER,
                        "monotone_constraints": MONOTONE_CONSTRAINTS,
                        "meta": meta or {}}, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "MonotonicCalibrator":
        with open(path, "rb") as f:
            blob = pickle.load(f)
        cal = cls(blob["model"])
        cal.meta = blob.get("meta", {})
        return cal


def maybe_load(path: str | None) -> MonotonicCalibrator | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    return MonotonicCalibrator.load(p)
