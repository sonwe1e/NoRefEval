"""Judgment confidence and event ambiguity (USERPLAN §2.1/§8.6/§11).

Low confidence means: much of the video was occluded or off-support, windows
were dominated by discrete events whose timing is unknowable, anchors could
not be verified, or too few windows were evaluated.
"""

from __future__ import annotations

import math

import numpy as np

from ..schema import Alignment, WindowFeatures


def compute_confidence(windows: list[WindowFeatures], alignment: Alignment,
                       n_frames: int, anchor_error: float) -> tuple[float, float]:
    """Returns (confidence 0..1, event_ambiguity 0..1)."""
    penalties = []

    # Visibility coverage across windows.
    cov = [wf.scalars.get("comp_visible_fraction") for wf in windows]
    cov = [c for c in cov if c is not None and c == c]
    if cov:
        mean_cov = float(np.mean(cov))
        penalties.append(np.clip((0.75 - mean_cov) / 0.5, 0, 1))

    # Feature availability.
    if windows:
        avail = np.mean([len(wf.scalars) for wf in windows])
        penalties.append(np.clip((12 - avail) / 12, 0, 1))

    # Event ambiguity: how much of the timeline looks like discrete switches.
    amb = [wf.scalars.get("event_ambiguity", 0.0) for wf in windows]
    event_ambiguity = float(np.clip(np.mean(amb) * 1.5, 0, 1)) if amb else 0.0
    penalties.append(0.5 * event_ambiguity)

    # Anchor trust.
    penalties.append(np.clip((anchor_error - 8.0) / 30.0, 0, 1))
    if alignment.warnings:
        penalties.append(min(0.4, 0.15 * len(alignment.warnings)))

    # Window density.
    frac = (len(windows) * 5) / max(n_frames, 1)
    penalties.append(np.clip((0.01 - frac) / 0.01, 0, 0.5))

    conf = float(np.clip(1.0 - 0.9 * float(np.mean(penalties)), 0.05, 1.0))
    return conf, event_ambiguity
