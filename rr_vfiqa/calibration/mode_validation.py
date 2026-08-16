"""Independent directional validation harness for NR and FR schemas."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .provenance import report_provenance
from ..config import EvaluationMode, parse_mode
from ..fusion.feature_registry import (
    definitions,
    feature_contract_hash,
)
from ..multimode import evaluate


def _localization_hit(report, case: dict[str, Any]) -> bool | None:
    interval = case.get("expected_interval_seconds")
    expected = set(case.get("expected_types") or [])
    if interval is None and not expected:
        return None
    start, end = interval if interval is not None else (-float("inf"), float("inf"))
    for window in getattr(report, "worst_windows", []):
        overlaps = float(window.end) >= start and float(window.start) <= end
        types = set(window.types)
        if overlaps and (not expected or types & expected):
            return True
    return False


def _feature_direction_rows(mode: EvaluationMode, better, worse) -> list[dict]:
    registry = {row.name: row for row in definitions(mode)}
    better_features = getattr(better, "features", {}) or {}
    worse_features = getattr(worse, "features", {}) or {}
    rows = []
    for name in sorted(set(better_features) & set(worse_features) & set(registry)):
        a, b = float(better_features[name]), float(worse_features[name])
        if not np.isfinite(a) or not np.isfinite(b):
            continue
        definition = registry[name]
        delta = a - b
        passed = (
            delta < 0.0
            if definition.direction == "lower_is_better"
            else delta > 0.0)
        rows.append({
            "feature": name,
            "direction": definition.direction,
            "better_value": a,
            "worse_value": b,
            "delta": delta,
            "passed": bool(passed),
            "diagnostic": "diagnostic" in definition.category,
        })
    return rows


def validate_manifest(
    manifest: dict[str, Any],
    *,
    preset: str = "fast",
    flow_backend: str = "farneback",
    device: str = "cpu",
) -> dict[str, Any]:
    mode = parse_mode(manifest["mode"])
    if mode is EvaluationMode.ENDPOINT_2X:
        raise ValueError("use the endpoint calibration harness for endpoint-2x")
    cases = []
    correct = 0
    first_report = None
    localization_values: list[bool] = []
    feature_rows: list[dict] = []
    for case in manifest.get("cases", []):
        reference = case.get("reference")
        common = dict(
            mode=mode,
            reference_video=reference,
            preset=preset,
            flow_backend=flow_backend,
            device=device,
            export_clips=False,
        )
        better = evaluate(candidate_video=case["better"], **common)
        worse = evaluate(candidate_video=case["worse"], **common)
        first_report = first_report or better
        margin = float(better.overall_score - worse.overall_score)
        min_margin = float(case.get("min_score_margin", 0.0))
        passed = (
            better.overall_score == better.overall_score
            and worse.overall_score == worse.overall_score
            and better.meta.get("status") != "failed"
            and worse.meta.get("status") != "failed"
            and margin > min_margin
        )
        localization_hit = _localization_hit(worse, case)
        if localization_hit is not None:
            localization_values.append(localization_hit)
        current_feature_rows = _feature_direction_rows(
            mode, better, worse)
        for row in current_feature_rows:
            feature_rows.append({"case": case.get("name"), **row})
        correct += int(passed)
        cases.append({
            "name": case.get("name", Path(case["worse"]).stem),
            "better_score": better.to_dict()["overall_score"],
            "worse_score": worse.to_dict()["overall_score"],
            "score_margin": margin,
            "min_score_margin": min_margin,
            "passed": passed,
            "better_status": better.meta.get("status"),
            "worse_status": worse.meta.get("status"),
            "better_confidence": getattr(better, "confidence", None),
            "worse_confidence": getattr(worse, "confidence", None),
            "localization_hit": localization_hit,
            "feature_directional_accuracy": (
                sum(row["passed"] for row in current_feature_rows)
                / len(current_feature_rows)
                if current_feature_rows else None),
            "better_schema": better.meta["score_schema"],
            "worse_schema": worse.meta["score_schema"],
        })
    total = len(cases)
    result = {
        "validation_schema": "mode-directional-validation-v1",
        "mode": mode.value,
        "score_schema": cases[0]["better_schema"] if cases else None,
        "directional_accuracy": correct / total if total else None,
        "passed": correct,
        "total": total,
        "mean_score_margin": (
            float(np.mean([case["score_margin"] for case in cases]))
            if cases else None),
        "localization_recall": (
            sum(localization_values) / len(localization_values)
            if localization_values else None),
        "feature_directional_accuracy": (
            sum(row["passed"] for row in feature_rows) / len(feature_rows)
            if feature_rows else None),
        "feature_directions": feature_rows,
        "production_gate": False,
        "cases": cases,
    }
    if first_report is not None:
        result.update(report_provenance(
            mode=mode.value,
            score_schema=first_report.meta["score_schema"],
            metric_contract=first_report.meta["metric_contract"],
            preset_contract=first_report.meta["preset_contract"],
            feature_contract_hash=feature_contract_hash(mode),
            backend_contract=first_report.meta["backend_contract"],
        ))
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--preset", default="fast")
    parser.add_argument("--flow-backend", default="farneback")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    result = validate_manifest(
        manifest, preset=args.preset, flow_backend=args.flow_backend,
        device=args.device)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0 if result["total"] and result["passed"] == result["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
