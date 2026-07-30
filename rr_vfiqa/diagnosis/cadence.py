"""Cadence integrity for no-reference mode (USERPLAN §5).

A 120 FPS stream whose odd frames merely copy the previous even frame
(``Y[2i+1] = Y[2i]``) has an *effective* cadence of 60 FPS, yet the shared
1/60 s time scale can still look healthy.  The common-time score therefore
must not be trusted on its own: we compute a native-cadence risk and apply a
*conservative multiplicative penalty* to the overall score,

    S_overall = S_common * exp(-lambda * R_cadence),

so a clean cadence leaves the score untouched while a collapsed cadence
pulls it down hard.  ``common_time_quality`` and ``cadence_integrity`` are
both surfaced so the user sees why the total moved.

The risk is built only from native-cadence scalars the metrics already emit
(``nr_native_*``), using ``max(R_abs, R_rel)`` per USERPLAN §6.4.  An optional
odd/even motion-increment alternation signal can be supplied when a caller
has frame-level access; otherwise that component is omitted (documented).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_LAMBDA = 2.2


@dataclass
class CadenceReport:
    cadence_risk: float          # 0..1 global native-cadence risk
    cadence_integrity: float     # 0..100 = 100 * exp(-lambda * risk)
    common_time_quality: float   # the un-penalised common-time overall
    overall: float               # common_time_quality * exp(-lambda * risk)
    per_window_risk: list[float]
    evidence: list[dict]         # the global sub-signals that drove the risk


def _window_cadence_risk(
    scalars: dict[str, float],
    alternation: float | None = None,
    *,
    common_time_motion: float | None = None,
) -> tuple[float, list[dict]]:
    """Native-cadence risk for one window in 0..1, plus the sub-signals.

    ``common_time_motion`` is the shared 1/60 s (or 1/30 s) motion magnitude.
    A collapsed native cadence is only penalised when there is *real motion* at
    a longer scale (or a stable odd/even alternation) — a genuinely static
    scene (pause menu, loading screen, idle character) has native duplicate
    fractions near 1 too, but is NOT a cadence failure.  This motion gate
    prevents the exponential penalty from crushing legitimate still content
    (USERPLAN P0-R1).
    """
    ev: list[dict] = []

    dup = _get(scalars, "nr_native_duplicate_fraction")
    frz = _get(scalars, "nr_native_freeze_fraction")
    mct = _get(scalars, "nr_mct_native_mean")
    comp = _get(scalars, "nr_native_self_comp")
    cycle = _get(scalars, "nr_native_self_cycle")

    # Each component is a 0..1 "this looks like a collapsed cadence" signal.
    r_dup = _clip01(dup / 0.30) if dup is not None else None
    r_frz = _clip01(frz / 0.40) if frz is not None else None
    # low native motion-compensation residual => frames carry no new content
    r_mct = _clip01(1.0 - mct / 18.0) if mct is not None else None
    # very low native self-composition error => odd frame == warped even frame
    r_comp = _clip01(1.0 - comp / 0.10) if comp is not None else None
    r_cycle = _clip01(1.0 - cycle / 14.0) if cycle is not None else None
    r_alt = _clip01(alternation) if alternation is not None else None

    contribs = []
    for name, r, thr in (
        ("native_duplicate_fraction", r_dup, 0.40),
        ("native_freeze_fraction", r_frz, 0.40),
        ("native_mct_low", r_mct, 0.45),
        ("native_self_comp_low", r_comp, 0.45),
        ("native_self_cycle_low", r_cycle, 0.45),
        ("odd_even_motion_alternation", r_alt, 0.50),
    ):
        if r is None:
            continue
        ev.append({"signal": name, "value": round(r, 4)})
        if r >= thr:
            contribs.append(r)

    if not contribs:
        return 0.0, ev
    # Require corroboration: a single weak signal is not a cadence verdict.
    # Two or more agreeing signals drive the risk; one strong duplicate/freez
    # signal alone still counts (it is near-conclusive on its own).
    strong_single = max(contribs) if any(
        (n in ("native_duplicate_fraction", "native_freeze_fraction")
         and (v or 0) >= 0.7)
        for n, v in (("native_duplicate_fraction", r_dup),
                     ("native_freeze_fraction", r_frz))) else 0.0
    if len(contribs) >= 2:
        risk = float(np.clip(0.55 * float(np.mean(contribs))
                             + 0.45 * float(np.max(contribs)), 0.0, 1.0))
    else:
        risk = float(np.clip(0.7 * contribs[0], 0.0, 1.0))
    risk = max(risk, strong_single)

    # --- motion gate (USERPLAN P0-R1) -------------------------------------
    # If there is no motion at the common 1/60 s scale AND no stable odd/even
    # alternation, the native "stillness" is more likely a genuinely static
    # scene than a collapsed cadence — suppress the risk.
    has_alternation = (r_alt is not None and r_alt >= 0.30)
    if common_time_motion is not None:
        ev.append({"signal": "common_time_motion",
                   "value": round(common_time_motion, 4)})
        if common_time_motion < 2.0 and not has_alternation:
            # No longer-scale motion and no alternation: treat as legit static.
            risk = float(np.clip(risk * 0.10, 0.0, 1.0))
            ev.append({"signal": "motion_gate", "value": "static_scene_suppress"})

    return float(np.clip(risk, 0.0, 1.0)), ev


def cadence_integrity(
    windows_scalars: list[dict[str, float]],
    common_overall: float,
    *,
    lam: float = DEFAULT_LAMBDA,
    alternation_per_window: list[float] | None = None,
    common_time_motion_per_window: list[float] | None = None,
) -> CadenceReport:
    """Aggregate native-cadence risk across windows and penalise the total.

    ``common_time_motion_per_window`` is the shared-scale (1/60 s) motion per
    window; near-zero values with no odd/even alternation mark a genuinely
    static scene, whose native duplicate/freeze signals are suppressed by the
    motion gate (USERPLAN P0-R1).
    """
    per_window: list[float] = []
    global_ev: list[dict] = []
    for i, sc in enumerate(windows_scalars):
        alt = (alternation_per_window[i]
               if alternation_per_window and i < len(alternation_per_window)
               else None)
        ctm = (common_time_motion_per_window[i]
               if common_time_motion_per_window
               and i < len(common_time_motion_per_window) else None)
        r, ev = _window_cadence_risk(sc, alt, common_time_motion=ctm)
        per_window.append(r)
    if per_window:
        # A cadence collapse is usually sustained, so use a high percentile
        # (robust to a few clean windows) but never below the median.
        risk = float(np.clip(
            max(float(np.percentile(per_window, 80)),
                float(np.median(per_window))), 0.0, 1.0))
        # attach the evidence of the worst window for the report
        worst = int(np.argmax(per_window))
        ctm_w = (common_time_motion_per_window[worst]
                 if common_time_motion_per_window
                 and worst < len(common_time_motion_per_window) else None)
        _, global_ev = _window_cadence_risk(
            windows_scalars[worst],
            alternation_per_window[worst]
            if alternation_per_window and worst < len(alternation_per_window)
            else None,
            common_time_motion=ctm_w)
    else:
        risk = 0.0
    integrity = float(100.0 * np.exp(-lam * risk))
    common = common_overall if np.isfinite(common_overall) else float("nan")
    overall = float(common * np.exp(-lam * risk)) if np.isfinite(common) else float("nan")
    return CadenceReport(
        cadence_risk=risk,
        cadence_integrity=integrity,
        common_time_quality=common,
        overall=overall,
        per_window_risk=per_window,
        evidence=global_ev,
    )


def cadence_penalty(common_overall: float, cadence_risk: float,
                    lam: float = DEFAULT_LAMBDA) -> float:
    """The multiplicative penalty in isolation (for tests / reuse)."""
    if not np.isfinite(common_overall):
        return float("nan")
    return float(common_overall * np.exp(-lam * max(0.0, cadence_risk)))


def _get(scalars: dict[str, float], key: str) -> float | None:
    v = scalars.get(key)
    if v is None:
        return None
    f = float(v)
    return f if np.isfinite(f) else None


def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))
