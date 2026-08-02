"""PTS-based lag, triplet and optical-flow planning."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class TemporalTriplet:
    left: int
    middle: int
    right: int
    span_seconds: float


@dataclass(frozen=True)
class TemporalLagPlan:
    native: tuple[tuple[int, int], ...]
    parent: tuple[tuple[int, int], ...]      # 2× native_dt (FPS-adaptive, USERPLAN §4.1)
    lag_1_60: tuple[tuple[int, int], ...]
    lag_1_30: tuple[tuple[int, int], ...]
    native_triplets: tuple[TemporalTriplet, ...]
    common_triplets: tuple[TemporalTriplet, ...]
    native_dt: float                         # median diff(pts), seconds
    parent_dt: float                         # 2.0 * native_dt, seconds

    @classmethod
    def build(cls, times: np.ndarray) -> "TemporalLagPlan":
        times = np.asarray(times, np.float64)
        native = tuple((i, i + 1) for i in range(max(0, len(times) - 1)))
        native_dt = (
            float(np.median(np.diff(times))) if len(times) > 1 else float("nan"))
        parent_dt = 2.0 * native_dt if np.isfinite(native_dt) else float("nan")
        # FPS-adaptive parent lag (USERPLAN §4.1): 60 FPS -> 1/30 s, 120 FPS -> 1/60 s.
        parent = tuple(
            _target_pairs(times, parent_dt)) if np.isfinite(parent_dt) else ()
        return cls(
            native=native,
            parent=parent,
            lag_1_60=tuple(_target_pairs(times, 1.0 / 60.0)),
            lag_1_30=tuple(_target_pairs(times, 1.0 / 30.0)),
            native_triplets=tuple(
                TemporalTriplet(i, i + 1, i + 2,
                                float(times[i + 2] - times[i]))
                for i in range(max(0, len(times) - 2))),
            common_triplets=tuple(_target_triplets(times, 1.0 / 60.0)),
            native_dt=native_dt,
            parent_dt=parent_dt,
        )


@dataclass
class FlowPairPlan:
    reasons: dict[tuple[int, int], set[str]] = field(default_factory=dict)

    def add(self, a: int, b: int, reason: str) -> None:
        if a == b:
            return
        pair = (min(int(a), int(b)), max(int(a), int(b)))
        self.reasons.setdefault(pair, set()).add(reason)

    def extend(self, pairs, reason: str) -> None:
        for a, b in pairs:
            self.add(a, b, reason)

    def unique_pairs(self) -> list[tuple[int, int]]:
        return sorted(self.reasons)

    @classmethod
    def for_no_reference(cls, lag_plan: TemporalLagPlan) -> "FlowPairPlan":
        plan = cls()
        for tag, triplets in (
            ("self-native", lag_plan.native_triplets),
            ("self-common", lag_plan.common_triplets),
        ):
            for triplet in triplets:
                plan.add(triplet.left, triplet.middle, tag)
                plan.add(triplet.middle, triplet.right, tag)
                plan.add(triplet.left, triplet.right, tag)
        plan.extend(lag_plan.native, "mct-native")
        plan.extend(lag_plan.parent, "mct-parent")
        plan.extend(lag_plan.lag_1_60, "mct-1-60")
        plan.extend(lag_plan.lag_1_30, "mct-1-30")
        return plan


def _target_pairs(times: np.ndarray, target: float) -> list[tuple[int, int]]:
    if len(times) < 2:
        return []
    native = float(np.median(np.diff(times)))
    tolerance = max(0.35 * target, 0.35 * native)
    output = []
    for i in range(len(times) - 1):
        delta = times[i + 1:] - times[i]
        relative = int(np.argmin(np.abs(delta - target)))
        if abs(float(delta[relative]) - target) <= tolerance:
            output.append((i, i + 1 + relative))
    return output


def _target_triplets(times: np.ndarray, half_span: float) -> list[TemporalTriplet]:
    output: list[TemporalTriplet] = []
    if len(times) < 3:
        return output
    native = float(np.median(np.diff(times)))
    tolerance = max(0.35 * half_span, 0.35 * native)
    for middle in range(1, len(times) - 1):
        left = int(np.argmin(np.abs(times - (times[middle] - half_span))))
        right = int(np.argmin(np.abs(times - (times[middle] + half_span))))
        if not (left < middle < right):
            continue
        if (abs(float(times[left] - (times[middle] - half_span))) <= tolerance
                and abs(float(times[right] - (times[middle] + half_span))) <= tolerance):
            output.append(TemporalTriplet(
                left, middle, right, float(times[right] - times[left])))
    return output
