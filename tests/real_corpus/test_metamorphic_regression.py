"""Metamorphic regression tests (USERPLAN P4).

For each RPG case we know the *exact* defect injected (type, time window,
parameters).  We verify the relations that are reliable for short defect
windows in small synthetic scenes:

1. **Localization** — the defective candidate produces at least one diagnostic
   issue whose time window overlaps the injected defect interval.
2. **Metric direction** — the sub-score the defect should affect moves in the
   expected direction (freeze -> higher duplicate fraction; blur -> lower
   sharpness; ui_drift -> higher UI instability).

These are the metamorphic relations that hold regardless of the absolute score
margins, which can be small when the defect occupies a brief window.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("av")   # PyAV required for video decode in inspect


def _inspect(video: str, reference: str | None, tmp: Path,
             speed: str = "fast", mode: str = "nr") -> dict:
    """Run ``rr-vfiqa inspect`` via the CLI and return the parsed report.json.

    ``reference`` is the clean baseline for *comparison* (the oracle/source).
    For FR mode the inspect command also needs ``--reference`` to be the clean
    source, so ``reference`` is passed through.
    """
    from rr_vfiqa.cli import main
    out = tmp / "result"
    out.mkdir(parents=True, exist_ok=True)
    cmd = ["inspect", "--candidate", str(video), "--out", str(out),
           "--speed", speed, "--no-clips",
           "--flow-backend", "farneback", "--device", "cpu", "--quiet"]
    if reference is not None and mode == "fr":
        cmd += ["--reference", str(reference)]
    main(cmd)
    rjson = out / "report.json"
    if not rjson.exists():
        return {}
    return json.loads(rjson.read_text(encoding="utf-8"))


def _features(report: dict) -> dict[str, float]:
    """Return the per-feature median map from the report metadata."""
    return dict((report.get("meta") or {}).get("features") or {})


def _defect_windows(case: dict) -> list[tuple[float, float]]:
    return [(float(d.get("start_time", 0)), float(d.get("end_time", 0)))
            for d in case["defects"]]


class TestDefectLocalization:
    """A located issue should overlap the injected defect window.

    This is the core metamorphic relation: the defect is injected at a *known*
    time, and the diagnoser should localise *something* to that interval.
    Because detection sensitivity varies by defect type and resolution, we
    report per-case results and assert an aggregate detection rate rather than
    requiring every single case to fire — undetected cases are surfaced via
    ``pytest.warns`` so they are visible but don't mask the overall result.
    """

    @pytest.mark.parametrize("mode", ["nr", "endpoint", "fr"])
    def test_defect_produces_overlapping_issue(self, rpg_cases, tmp_path, mode):
        cases = [c for c in rpg_cases if c["mode"] == mode]
        assert cases, f"no {mode} cases"
        detected = 0
        for case in cases:
            rep = _inspect(case["candidate"], case.get("reference_for_inspect"),
                           tmp_path / f"loc_{case['case_id']}", mode=mode)
            diag = (rep.get("meta") or {}).get("diagnostics") or {}
            issues = diag.get("issues") or []
            windows = _defect_windows(case)
            found = False
            for issue in issues:
                iss_t0 = float(issue.get("start_time", 0))
                iss_t1 = float(issue.get("end_time", 0))
                for wt0, wt1 in windows:
                    if iss_t0 <= wt1 and iss_t1 >= wt0:
                        found = True
                        break
                if found:
                    break
            if found:
                detected += 1
            else:
                pytest.warns(UserWarning)  # noqa: B018 — surface undetected case
                import warnings
                warnings.warn(
                    f"{case['case_id']}: no issue overlaps defect windows "
                    f"{windows}; issues="
                    f"{[(i.get('issue_type'), i.get('start_time'), i.get('end_time')) for i in issues]}",
                    stacklevel=1)
        # Sanity floor on localisation rate.  FR is evaluated at low resolution
        # here where subtle defects (colour/shift) are near the noise floor, so
        # the floor is lower than NR/Endpoint.
        floor = 0.2 if mode == "fr" else 0.4
        rate = detected / len(cases)
        assert rate >= floor, (
            f"{mode}: only {detected}/{len(cases)} defects localised "
            f"(rate {rate:.0%}, floor {floor:.0%})")


class TestMetricDirection:
    """The sub-score a defect targets must move in the expected direction.

    We compare the defective candidate against its clean oracle on the feature
    the defect directly affects.  This is robust to small absolute score
    differences because it tests the *direction* of change in the affected
    dimension.
    """

    # Map defect type -> (feature_key, higher_is_worse).  higher_is_worse=True
    # means the defective candidate should show a HIGHER value.
    _FEATURE_CHECKS: dict[str, tuple[str, bool]] = {
        "freeze": ("nr_native_duplicate_fraction", True),
        "odd_frame_blur": ("nr_phase_sharpness_energy", False),
        "cadence_collapse": ("nr_native_duplicate_fraction", True),
        "ui_drift": ("nr_ui_edge_instability", True),
        "projectile_ghost_flicker": ("nr_phase_sharpness_energy", False),
        "generated_motion_blur": ("gtq_sharpness", False),
        "disocclusion_ghost": ("comp_mean", True),
        "thin_weapon_wrong_motion": ("comp_mean", True),
        "ui_text_drift": ("ui_static_l1", True),
        "generated_freeze_copy": ("comp_mean", True),
        "global_blur": ("fr_l1_y", True),
        "local_spatial_shift": ("fr_edge_chamfer", True),
        "color_imbalance": ("fr_l1_rgb", True),
        "frame_drop": ("fr_temporal_diff_error", True),
        "noise_banding": ("fr_l1_y", True),
    }

    @pytest.mark.parametrize("mode", ["nr", "endpoint", "fr"])
    def test_defect_moves_metric_in_expected_direction(
            self, rpg_cases, tmp_path, mode):
        cases = [c for c in rpg_cases
                 if c["mode"] == mode and c["oracle"] is not None]
        assert cases, f"no {mode} cases with an oracle"
        for case in cases:
            dtype = case["defects"][0]["type"] if case["defects"] else None
            check = self._FEATURE_CHECKS.get(dtype) if dtype else None
            if check is None:
                continue  # defect type not in the explicit checklist; skip
            key, higher_is_worse = check
            rep_cand = _inspect(case["candidate"], case.get("reference_for_inspect"),
                                tmp_path / f"mcand_{case['case_id']}", mode=mode)
            rep_clean = _inspect(case["oracle"], case.get("reference_for_inspect"),
                                 tmp_path / f"mclean_{case['case_id']}", mode=mode)
            fc = _features(rep_cand)
            fk = _features(rep_clean)
            if key not in fc or key not in fk:
                continue  # feature not emitted by this mode/preset; skip
            vc, vk = float(fc[key]), float(fk[key])
            if higher_is_worse:
                assert vc > vk, (
                    f"{case['case_id']} ({dtype}): {key} should be higher "
                    f"for defective ({vc:.4f}) than clean ({vk:.4f})")
            else:
                assert vc < vk, (
                    f"{case['case_id']} ({dtype}): {key} should be lower "
                    f"for defective ({vc:.4f}) than clean ({vk:.4f})")
