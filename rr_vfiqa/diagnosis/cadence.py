"""Cadence integrity v2 for no-reference mode (USERPLAN §4).

A 120 FPS stream whose odd frames merely copy the previous even frame
(``Y[2i+1] = Y[2i]``) has an *effective* cadence of 60 FPS, yet the shared
time scale can still look healthy.  However, the old v1 detector confused
three things with a cadence collapse:

1. **Smooth, accurate motion** — low MCT / self-composition / self-cycle
   residuals are exactly what high-quality motion produces (USERPLAN §4.2);
2. **Combat effects** — flicker, particles and camera shake produce transient
   odd/even energy with an unstable direction;
3. **Risk-selected windows** — aggregating over intentionally-picked high-risk
   windows overstated the *global* cadence risk (USERPLAN §4.3).

So v2 declares a cadence collapse only when ALL of the following hard gates
hold, estimated from the **full-film uniform scan** (USERPLAN §4.4), never
from risk-selected windows:

1. **Parent-scale motion is present** — ``nr_parent_motion_p90`` (t -> t+2,
   FPS-adaptive: 1/30 s at 60 FPS, 1/60 s at 120 FPS) above threshold.  A
   genuinely static scene (pause menu, idle character) has parent motion ~0
   and is NOT a cadence failure.
2. **Parity asymmetry is strong** — one phase of the native adjacent-frame
   differences carries (near-)zero new information while the other carries
   real motion (``nr_phase_asymmetry`` above threshold).
3. **The asymmetry direction is stable** — ``phase_coherence`` over the
   film's blocks above threshold.  Transient combat effects flip phase and
   are suppressed.

Low MCT / self-composition / self-cycle residuals and duplicate/freeze
fractions are only *auxiliary* corroboration, never independent evidence.

Per USERPLAN §5, before real-video calibration the exponential multiplier is
suspended: Artifact Quality and Cadence Integrity are reported as two
independent axes (``penalty_mode="separate"``).  A conservative
``penalty_mode="cap15"`` caps the penalty at 15 points via
``S * (0.85 + 0.15 * cadence/100)``.  The old ``"exponential"`` mode is kept
for A/B comparison and tests only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_LAMBDA = 2.2

# --- v2 hard-gate thresholds ------------------------------------------------
# Parent-scale motion (mean luma delta over the t -> t+2 pair, 0..255 scale).
# 3.0 sits comfortably above codec noise on low-res scan luma and far below
# real content motion.
PARENT_MOTION_P90_MIN = 3.0
# Only a sanity floor: reject content where NO pixel meaningfully moves (pure
# static / pause screen).  Deliberately NOT a "10% of the frame must move"
# bar — a small character or sparse combat motion is still real motion and a
# duplicate collapse of it must be caught (USERPLAN §4.1).
PARENT_MOVING_FRAC_MIN = 0.01
PHASE_ASYMMETRY_MIN = 0.25           # |median(d_even) - median(d_odd)| / sum
PHASE_COHERENCE_MIN = 0.50           # direction stable across the film
# Auxiliary-signal thresholds (corroboration only).
_AUX_DUP = 0.30
_AUX_FRZ = 0.40
_AUX_MCT = 18.0
_AUX_COMP = 0.10
_AUX_CYCLE = 14.0
# Scan coherence is computed over blocks of this many frames.
_COHERENCE_BLOCK_FRAMES = 16

# Penalty modes: what ``overall`` should be once a cadence risk is known.
PENALTY_SEPARATE = "separate"        # overall = artifact quality (default)
PENALTY_CAP15 = "cap15"              # overall = common * (0.85 + 0.15*cad/100)
PENALTY_EXPONENTIAL = "exponential"  # legacy: overall = common * exp(-lam*R)


@dataclass
class CadenceReport:
    cadence_risk: float          # 0..1 global native-cadence risk (v2 gates)
    cadence_integrity: float     # 0..100 diagnostic axis = 100 * exp(-lam * risk)
    common_time_quality: float   # the un-penalised artifact-quality overall
    overall: float               # common_time_quality, or capped by penalty_mode
    per_window_risk: list[float]
    evidence: list[dict]         # the gates / signals that drove the risk
    # USERPLAN §4/§5 interpretable fields
    phase_asymmetry: float       # global parity asymmetry
    phase_coherence: float       # global direction stability
    parent_motion_p90: float     # global parent-scale motion
    parent_moving_fraction: float
    gates: dict[str, bool]       # which hard gates passed
    penalty_mode: str


def _window_cadence_risk(
    scalars: dict[str, float],
    *,
    parent_motion_p90: float | None = None,
    parent_moving_fraction: float | None = None,
    phase_asymmetry: float | None = None,
) -> tuple[float, list[dict]]:
    """Native-cadence risk for one window in 0..1 (v2 gates 1+2).

    Returns (risk, evidence).  Gate 3 (phase coherence) is applied at the
    aggregate level by :func:`cadence_integrity`, which owns the cross-window
    signed-gap statistics.

    Hard gates first: no parent-scale motion -> genuinely static scene ->
    zero risk; no parity asymmetry -> no evidence of a phase-collapsed cadence
    -> zero risk.  Only then do duplicate/freeze and low-residual signals
    corroborate the magnitude (USERPLAN §4.2: they can never fire alone).
    """
    ev: list[dict] = []

    # --- Hard gate 1: parent-scale motion present (FPS-adaptive) ------------
    if parent_motion_p90 is None or not np.isfinite(parent_motion_p90) \
            or parent_motion_p90 < PARENT_MOTION_P90_MIN:
        ev.append({
            "signal": "parent_motion_p90",
            "value": _r(parent_motion_p90),
            "gate": "parent_motion",
            "verdict": "static_scene",
        })
        return 0.0, ev
    if parent_moving_fraction is not None and np.isfinite(parent_moving_fraction) \
            and parent_moving_fraction < PARENT_MOVING_FRAC_MIN:
        ev.append({
            "signal": "parent_moving_fraction",
            "value": _r(parent_moving_fraction),
            "gate": "parent_motion",
            "verdict": "static_scene",
        })
        return 0.0, ev
    ev.append({
        "signal": "parent_motion_p90",
        "value": _r(parent_motion_p90),
        "gate": "parent_motion",
        "verdict": "pass",
    })

    # --- Hard gate 2: parity asymmetry of native adjacent diffs -------------
    if phase_asymmetry is None or not np.isfinite(phase_asymmetry) \
            or phase_asymmetry < PHASE_ASYMMETRY_MIN:
        ev.append({
            "signal": "phase_asymmetry",
            "value": _r(phase_asymmetry),
            "gate": "phase_asymmetry",
            "verdict": "no_parity_structure",
        })
        return 0.0, ev
    ev.append({
        "signal": "phase_asymmetry",
        "value": _r(phase_asymmetry),
        "gate": "phase_asymmetry",
        "verdict": "pass",
    })

    # --- Auxiliary corroboration only (USERPLAN §4.2) -----------------------
    dup = _get(scalars, "nr_native_duplicate_fraction")
    frz = _get(scalars, "nr_native_freeze_fraction")
    mct = _get(scalars, "nr_mct_native_mean")
    comp = _get(scalars, "nr_native_self_comp")
    cycle = _get(scalars, "nr_native_self_cycle")
    alt = _get(scalars, "nr_native_motion_alternation")

    r_dup = _clip01(dup / _AUX_DUP) if dup is not None else None
    r_frz = _clip01(frz / _AUX_FRZ) if frz is not None else None
    r_mct = _clip01(1.0 - mct / _AUX_MCT) if mct is not None else None
    r_comp = _clip01(1.0 - comp / _AUX_COMP) if comp is not None else None
    r_cycle = _clip01(1.0 - cycle / _AUX_CYCLE) if cycle is not None else None
    r_alt = _clip01(alt) if alt is not None else None

    aux = [v for v in (r_dup, r_frz, r_mct, r_comp, r_cycle, r_alt)
           if v is not None]
    for name, r in (
        ("native_duplicate_fraction", r_dup),
        ("native_freeze_fraction", r_frz),
        ("native_mct_low", r_mct),
        ("native_self_comp_low", r_comp),
        ("native_self_cycle_low", r_cycle),
        ("odd_even_motion_alternation", r_alt),
    ):
        if r is not None:
            ev.append({"signal": name, "value": round(r, 4),
                       "gate": "auxiliary"})

    # Magnitude: the asymmetry is the primary evidence; auxiliary signals
    # corroborate the duplicate interpretation.
    risk = float(np.clip(
        0.7 * phase_asymmetry + 0.3 * (float(np.mean(aux)) if aux else 0.0),
        0.0, 1.0))
    return risk, ev


def scan_phase_stats(scan, scene_cuts=None) -> dict[str, float]:
    """Global parity statistics from the full-film Tier-1 scan.

    Estimates come from uniform per-scene sequences (USERPLAN §4.4), never
    from risk-selected windows.  Returns the global hard-gate inputs: parent
    motion (robust p90), moving-pixel fraction, parity asymmetry (median over
    scenes) and direction coherence over fixed blocks.  ``scene_cuts`` (an
    index array) is accepted for forward compatibility; block boundaries are
    not aligned to scene cuts yet.
    """
    k = np.asarray(scan.indices, np.int64)
    d = np.asarray(scan.frame_diff, np.float64)
    d_p = np.asarray(scan.frame_diff_parent, np.float64)
    m_p = np.asarray(scan.moving_frac_parent, np.float64)
    n = len(k)
    if n < 8:
        return {
            "parent_motion_p90": float("nan"),
            "parent_moving_fraction": float("nan"),
            "phase_asymmetry": float("nan"),
            "phase_coherence": float("nan"),
            "duplicate_fraction": float("nan"),
            "scenes_evaluated": 0,
        }

    # Block-based coherence: split the timeline into fixed blocks and record
    # each block's signed phase gap, so a globally phase-locked duplicate
    # stream reaches coherence ~1 while transient combat effects (direction
    # flips) drive it toward 0.
    block = _COHERENCE_BLOCK_FRAMES
    signed_gaps: list[float] = []
    asymmetry_blocks: list[float] = []
    for s in range(0, n, block):
        sl = slice(s, min(s + block, n))
        ks, ds = k[sl], d[sl]
        if len(ds) < 8:
            continue
        even = ds[ks % 2 == 0]
        odd = ds[ks % 2 == 1]
        if len(even) < 2 or len(odd) < 2:
            continue
        med_e, med_o = float(np.median(even)), float(np.median(odd))
        denom = med_e + med_o + 1e-6
        asymmetry_blocks.append(float(abs(med_e - med_o) / denom))
        signed_gaps.append(float((med_o - med_e) / denom))

    parent_p90 = float(np.percentile(d_p[2:], 90)) if n > 2 else float("nan")
    moving_parent = float(np.median(m_p[2:])) if n > 2 else float("nan")
    asymmetry = (float(np.median(asymmetry_blocks))
                 if asymmetry_blocks else float("nan"))
    if signed_gaps:
        w = np.abs(signed_gaps)
        coherence = float(abs(np.sum(signed_gaps)) / (np.sum(w) + 1e-6))
    else:
        coherence = float("nan")
    # Full-film duplicate evidence (auxiliary corroboration for the risk
    # magnitude): fraction of native adjacent diffs below the copy threshold.
    dup_frac = float(np.mean(d[1:] < 0.75)) if n > 1 else float("nan")
    return {
        "parent_motion_p90": parent_p90,
        "parent_moving_fraction": moving_parent,
        "phase_asymmetry": asymmetry,
        "phase_coherence": coherence,
        "duplicate_fraction": dup_frac,
        "scenes_evaluated": len(asymmetry_blocks),
    }


def _cadence_from_stats(
    stats: dict[str, float],
    windows_scalars: list[dict[str, float]],
    common_overall: float,
    *,
    lam: float,
    penalty_mode: str,
) -> CadenceReport:
    """Build a CadenceReport from global scan stats + per-window scalars."""
    ev: list[dict] = []
    gates = {
        "parent_motion": False,
        "phase_asymmetry": False,
        "phase_coherence": False,
    }

    parent_p90 = _get(stats, "parent_motion_p90")
    moving_frac = _get(stats, "parent_moving_fraction")
    asym = _get(stats, "phase_asymmetry")
    coh = _get(stats, "phase_coherence")

    # Hard gate 1: parent-scale motion.
    if parent_p90 is None or parent_p90 < PARENT_MOTION_P90_MIN:
        ev.append({"signal": "parent_motion_p90", "value": _r(parent_p90),
                   "gate": "parent_motion", "verdict": "static_scene"})
        gates["parent_motion"] = False
        per_window = [0.0 for _ in windows_scalars]
        return _report(0.0, per_window, ev, common_overall, lam,
                       penalty_mode, asym, coh, parent_p90, moving_frac, gates)
    gates["parent_motion"] = True
    ev.append({"signal": "parent_motion_p90", "value": _r(parent_p90),
               "gate": "parent_motion", "verdict": "pass"})
    if moving_frac is not None and moving_frac < PARENT_MOVING_FRAC_MIN:
        ev.append({"signal": "parent_moving_fraction",
                   "value": _r(moving_frac), "gate": "parent_motion",
                   "verdict": "static_scene"})
        gates["parent_motion"] = False
        per_window = [0.0 for _ in windows_scalars]
        return _report(0.0, per_window, ev, common_overall, lam,
                       penalty_mode, asym, coh, parent_p90, moving_frac, gates)
    ev.append({"signal": "parent_moving_fraction",
               "value": _r(moving_frac), "gate": "parent_motion",
               "verdict": "pass"})

    # Hard gate 2: parity asymmetry.
    if asym is None or asym < PHASE_ASYMMETRY_MIN:
        ev.append({"signal": "phase_asymmetry", "value": _r(asym),
                   "gate": "phase_asymmetry", "verdict": "no_parity_structure"})
        per_window = [0.0 for _ in windows_scalars]
        return _report(0.0, per_window, ev, common_overall, lam,
                       penalty_mode, asym, coh, parent_p90, moving_frac, gates)
    gates["phase_asymmetry"] = True
    ev.append({"signal": "phase_asymmetry", "value": _r(asym),
               "gate": "phase_asymmetry", "verdict": "pass"})

    # Hard gate 3: direction coherence.
    if coh is None or coh < PHASE_COHERENCE_MIN:
        ev.append({"signal": "phase_coherence", "value": _r(coh),
                   "gate": "phase_coherence",
                   "verdict": "unstable_direction"})
        per_window = [0.0 for _ in windows_scalars]
        return _report(0.0, per_window, ev, common_overall, lam,
                       penalty_mode, asym, coh, parent_p90, moving_frac, gates)
    gates["phase_coherence"] = True
    ev.append({"signal": "phase_coherence", "value": _r(coh),
               "gate": "phase_coherence", "verdict": "pass"})

    # All three hard gates hold: build per-window risks (auxiliary evidence)
    # and aggregate robustly.
    per_window: list[float] = []
    for sc in windows_scalars:
        r, _ = _window_cadence_risk(
            sc,
            parent_motion_p90=parent_p90,
            parent_moving_fraction=moving_frac,
            phase_asymmetry=asym,
        )
        per_window.append(r)
    risk = float(np.clip(
        max(float(np.percentile(per_window, 80)) if per_window else 0.0,
            float(np.median(per_window)) if per_window else 0.0),
        0.0, 1.0)) if per_window else 0.0
    # The scan-level duplicate fraction corroborates the risk magnitude.
    dup = _get(stats, "duplicate_fraction")
    if dup is not None:
        ev.append({"signal": "scan_duplicate_fraction", "value": _r(dup),
                   "gate": "auxiliary"})
        risk = float(np.clip(
            max(risk, 0.35 * dup), 0.0, 1.0))

    return _report(risk, per_window, ev, common_overall, lam, penalty_mode,
                   asym, coh, parent_p90, moving_frac, gates)


def _report(risk, per_window, ev, common_overall, lam, penalty_mode,
            asym, coh, parent_p90, moving_frac, gates) -> CadenceReport:
    risk = float(np.clip(risk, 0.0, 1.0))
    integrity = float(100.0 * np.exp(-lam * risk))
    common = common_overall if np.isfinite(common_overall) else float("nan")
    if penalty_mode == PENALTY_EXPONENTIAL:
        overall = (float(common * np.exp(-lam * risk))
                   if np.isfinite(common) else float("nan"))
    elif penalty_mode == PENALTY_CAP15:
        overall = (float(common * (0.85 + 0.15 * integrity / 100.0))
                   if np.isfinite(common) else float("nan"))
    else:  # separate
        overall = common
    return CadenceReport(
        cadence_risk=risk,
        cadence_integrity=integrity,
        common_time_quality=common,
        overall=overall,
        per_window_risk=[float(v) for v in per_window],
        evidence=ev,
        phase_asymmetry=_r(asym) if asym is not None else None,
        phase_coherence=_r(coh) if coh is not None else None,
        parent_motion_p90=_r(parent_p90) if parent_p90 is not None else None,
        parent_moving_fraction=(_r(moving_frac)
                                if moving_frac is not None else None),
        gates=dict(gates),
        penalty_mode=penalty_mode,
    )


def cadence_integrity(
    windows_scalars: list[dict[str, float]],
    common_overall: float,
    *,
    lam: float = DEFAULT_LAMBDA,
    penalty_mode: str = PENALTY_SEPARATE,
    scan_stats: dict[str, float] | None = None,
    phase_gap_signed_per_window: list[float] | None = None,
) -> CadenceReport:
    """Aggregate native-cadence risk with v2 hard gates (USERPLAN §4).

    ``scan_stats`` is the full-film estimate from :func:`scan_phase_stats`
    (preferred — USERPLAN §4.4).  When it is ``None`` the global gates are
    estimated from the per-window scalars instead (tests / degenerate inputs):
    parent motion and asymmetry come from the window-level ``nr_*`` features,
    and coherence from ``phase_gap_signed_per_window`` (falling back to no
    coherence gate when the sign is unavailable).
    """
    if scan_stats is not None:
        return _cadence_from_stats(scan_stats, windows_scalars,
                                   common_overall, lam=lam,
                                   penalty_mode=penalty_mode)

    # --- window-only fallback ----------------------------------------------
    per_window: list[float] = []
    ev: list[dict] = []
    signed: list[float] = []
    for i, sc in enumerate(windows_scalars):
        parent_p90 = _get(sc, "nr_parent_motion_p90")
        moving = _get(sc, "nr_parent_moving_pixel_fraction")
        asym = _get(sc, "nr_phase_asymmetry")
        r, _ = _window_cadence_risk(
            sc, parent_motion_p90=parent_p90,
            parent_moving_fraction=moving, phase_asymmetry=asym)
        per_window.append(r)
        g = (phase_gap_signed_per_window[i]
             if phase_gap_signed_per_window
             and i < len(phase_gap_signed_per_window) else None)
        if g is not None and np.isfinite(g):
            signed.append(float(g))
    # Gate 3 (coherence) applied at the aggregate level.
    if signed:
        w = np.abs(signed)
        coh = float(abs(np.sum(signed)) / (np.sum(w) + 1e-6))
        if coh < PHASE_COHERENCE_MIN:
            ev.append({"signal": "phase_coherence", "value": _r(coh),
                       "gate": "phase_coherence",
                       "verdict": "unstable_direction"})
            return _report(0.0, per_window, ev, common_overall, lam,
                           penalty_mode, float("nan"), coh,
                           float("nan"), float("nan"),
                           {"parent_motion": False, "phase_asymmetry": False,
                            "phase_coherence": False})
        ev.append({"signal": "phase_coherence", "value": _r(coh),
                   "gate": "phase_coherence", "verdict": "pass"})
    risk = (float(np.clip(
        max(float(np.percentile(per_window, 80)),
            float(np.median(per_window))), 0.0, 1.0))
        if per_window else 0.0)
    # Attach the evidence of the worst window for the report.
    worst = int(np.argmax(per_window)) if per_window else -1
    if worst >= 0:
        sc = windows_scalars[worst]
        _, wev = _window_cadence_risk(
            sc,
            parent_motion_p90=_get(sc, "nr_parent_motion_p90"),
            parent_moving_fraction=_get(sc, "nr_parent_moving_pixel_fraction"),
            phase_asymmetry=_get(sc, "nr_phase_asymmetry"))
        ev.extend(wev)
    return _report(risk, per_window, ev, common_overall, lam, penalty_mode,
                   float("nan"), float("nan"), float("nan"), float("nan"),
                   {"parent_motion": True, "phase_asymmetry": True,
                    "phase_coherence": True})


def cadence_penalty(common_overall: float, cadence_risk: float,
                    lam: float = DEFAULT_LAMBDA) -> float:
    """Legacy exponential penalty in isolation (A/B comparison / tests)."""
    if not np.isfinite(common_overall):
        return float("nan")
    return float(common_overall * np.exp(-lam * max(0.0, cadence_risk)))


def _get(stats: dict, key: str) -> float | None:
    v = stats.get(key)
    if v is None:
        return None
    f = float(v)
    return f if np.isfinite(f) else None


def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def _r(v, n: int = 4):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(f):
        return None
    return round(f, n)
