"""Deterministic motion/easing functions (PICPLAN §6.4).

Every object position, rotation, scale and alpha in a scene must be a pure
function of (scene seed, frame index, t, fixed parameters).  No real-time
physics, no wall-clock, no shared RNG state.
"""
from __future__ import annotations

import numpy as np


def clamp01(t: float) -> float:
    return 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)


def linear(t: float) -> float:
    return float(t)


def smoothstep(t: float) -> float:
    t = clamp01(t)
    return float(t * t * (3.0 - 2.0 * t))


def ease_in_out_cubic(t: float) -> float:
    t = clamp01(t)
    if t < 0.5:
        return float(4.0 * t * t * t)
    u = 2.0 * t - 2.0
    return float(1.0 + 0.5 * u * u * u)


def bezier_2d(t: float, p0, p1, p2, p3):
    """Cubic bezier through four 2D control points, returns (x, y)."""
    t = clamp01(t)
    u = 1.0 - t
    p0, p1, p2, p3 = (np.asarray(p, dtype=np.float64) for p in (p0, p1, p2, p3))
    pt = (u ** 3) * p0 + 3 * (u ** 2) * t * p1 + 3 * u * (t ** 2) * p2 + (t ** 3) * p3
    return float(pt[0]), float(pt[1])


def sinusoidal(t: float, frequency: float, amplitude: float, phase: float = 0.0) -> float:
    return float(amplitude * np.sin(2.0 * np.pi * frequency * t + phase))


def lerp(a: float, b: float, t: float) -> float:
    return float(a + (b - a) * t)


def window(t: float, t0: float, t1: float) -> float:
    """Normalized progress of t inside [t0, t1], clamped to [0, 1]."""
    if t1 <= t0:
        return 0.0
    return clamp01((t - t0) / (t1 - t0))


def frame_range(fps: int, start_time: float, end_time: float, n_frames: int) -> tuple[int, int]:
    """Inclusive/exclusive frame index range for a time interval."""
    a = max(0, int(round(start_time * fps)))
    b = min(n_frames, int(round(end_time * fps)))
    return a, max(a, b)
