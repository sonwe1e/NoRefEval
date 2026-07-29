"""Independent directional validation harness for NR and FR schemas."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .provenance import report_provenance
from ..config import EvaluationMode, parse_mode
from ..fusion.feature_registry import feature_contract_hash
from ..multimode import evaluate


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
        passed = (
            better.overall_score == better.overall_score
            and worse.overall_score == worse.overall_score
            and better.overall_score > worse.overall_score
        )
        correct += int(passed)
        cases.append({
            "name": case.get("name", Path(case["worse"]).stem),
            "better_score": better.to_dict()["overall_score"],
            "worse_score": worse.to_dict()["overall_score"],
            "passed": passed,
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
