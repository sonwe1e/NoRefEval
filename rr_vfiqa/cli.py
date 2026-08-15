"""Command-line interface: explicit no-reference / endpoint / full-reference."""

from __future__ import annotations

import argparse
import json
import sys

from .config import EvaluationMode
from .mode_router import SPEED_ALIASES, fail_closed_report, preset_from_speed, route_mode
from .multimode import compare, evaluate
from ._version import VERSION


def _mode_kwargs(mode: str, *, vqa_backend: str, calibrator: str | None,
                 geometry_policy: str) -> dict:
    """Per-mode extra kwargs for ``evaluate`` — single source of truth."""
    if mode == EvaluationMode.NO_REFERENCE.value:
        return {"vqa_backend": vqa_backend}
    if mode == EvaluationMode.ENDPOINT_2X.value:
        return {"calibrator_path": calibrator}
    return {"geometry_policy": geometry_policy}   # full-reference


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--preset", default=None,
                   choices=["fast", "standard", "audit"],
                   help="internal resource preset (see also --speed)")
    p.add_argument("--speed", default=None, choices=sorted(SPEED_ALIASES),
                   help="speed alias: fast | balanced | thorough (see --preset)")
    p.add_argument("--cache-dir", default="./cache")
    p.add_argument("--device", default="cuda")
    p.add_argument("--flow-backend", default="auto",
                   choices=["auto", "raft", "farneback"])
    p.add_argument("--no-clips", action="store_true",
                   help="skip badcase/overlay clip export")
    p.add_argument("--quiet", action="store_true")


def _resolve_preset(args: argparse.Namespace) -> str:
    try:
        return preset_from_speed(getattr(args, "speed", None),
                                 getattr(args, "preset", None))
    except ValueError as exc:
        raise SystemExit(str(exc))


def _run_inspect(args: argparse.Namespace, preset: str, progress) -> int:
    summary, rc = _inspect_core(
        candidate=args.candidate, reference=args.reference, mode=args.mode,
        out_dir=args.out, speed=getattr(args, "speed", None), preset=preset,
        no_clips=args.no_clips, cache_dir=args.cache_dir, device=args.device,
        flow_backend=args.flow_backend, calibrator=args.calibrator,
        vqa_backend=args.vqa_backend, geometry_policy=args.geometry_policy,
        quiet=args.quiet, progress=progress)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.out is None and not args.quiet:
        print("[rr-vfiqa] 未指定 --out：报告文件（report.html / report.json / "
              "timeline / heatmaps / badcases）未写入。"
              "加 --out <目录> 保存完整报告。", file=sys.stderr)
    return rc


def _inspect_core(*, candidate: str, reference: str | None, mode: str = "auto",
                  out_dir: str | None = None, speed: str | None = None,
                  preset: str | None = None, no_clips: bool = False,
                  cache_dir: str = "./cache", device: str = "cpu",
                  flow_backend: str = "auto", calibrator: str | None = None,
                  vqa_backend: str = "none", geometry_policy: str = "strict",
                  quiet: bool = True, progress=None
                  ) -> tuple[dict, int]:
    """Run one auto-safe inspection; return (summary_dict, exit_code).

    Pure (no stdout) so the batch runner can call it per job.  ``preset`` here
    is already resolved from --speed by the caller; when running auto with the
    default speed we still upgrade to the auto_audit tier (USERPLAN §3).
    """
    progress = progress or (lambda _: None)
    resolved = preset_from_speed(speed, preset)
    route = None
    if mode == "auto":
        route = route_mode(candidate, reference, geometry_policy=geometry_policy)
        progress(f"auto-route -> {route.mode.value if route.mode else 'NONE'}: "
                 f"{route.reason}")
        if not route.ok:
            refused = fail_closed_report(route)
            if out_dir:
                from pathlib import Path
                from .report import write_html_report, write_json_report
                out = Path(out_dir)
                out.mkdir(parents=True, exist_ok=True)
                obj = _refused_report_obj(refused)
                write_json_report(obj, out / "report.json")
                write_html_report(obj, out / "report.html")
            return ({**refused, "report_dir": out_dir,
                     "auto_route": route.to_dict()}, 1)
        mode = route.mode.value

    if speed is None and resolved == "standard":
        resolved = "balanced"

    extra = _mode_kwargs(
        mode, vqa_backend=vqa_backend, calibrator=calibrator,
        geometry_policy=geometry_policy)
    # USERPLAN P0-R2: any failure past routing must still leave a complete
    # report (report.json + report.html) with status=failed and the reason —
    # never crash the command with an unhandled traceback.
    try:
        report = evaluate(
            candidate_video=candidate, reference_video=reference, mode=mode,
            preset=resolved, cache_dir=cache_dir, device=device, out_dir=out_dir,
            flow_backend=flow_backend, export_clips=not no_clips,
            progress=progress, **extra)
    except Exception as exc:  # noqa: BLE001
        report = _failed_report(mode, route, out_dir, repr(exc))
    d = report.to_dict()
    diag = (d.get("meta") or {}).get("diagnostics") or {}
    summary = {
        "overall": d.get("overall_score"),
        "confidence": d.get("confidence"),
        "mode": mode,
        # USERPLAN §10: global (whole video) vs worst local issue level are
        # reported independently; ``quality_level`` is kept as the worst local
        # issue label for backward compatibility with older consumers.
        "quality_level": ((diag.get("worst_issue") or {}).get("severity_label")
                          if diag.get("worst_issue") else None),
        "global_quality_level": (diag.get("global_quality_level") or {}).get("label"),
        "worst_issue_level": (diag.get("worst_issue_level") or {}).get("label"),
        "issues": diag.get("issue_count", 0),
        "affected_duration_fraction": diag.get("affected_duration_fraction", 0.0),
        "confirmed_affected_seconds": diag.get("confirmed_affected_seconds", 0.0),
        "report_dir": out_dir,
        "auto_route": route.to_dict() if route else None,
    }
    return summary, (1 if d.get("meta", {}).get("status") == "failed" else 0)


def _failed_report(mode: str, route, out_dir: str | None, reason: str):
    """Build a fail-closed Report when evaluation raises past routing.

    Writes report.json + report.html so the user always gets an artefact with
    status=failed and the human-readable failure reason (USERPLAN P0-R2).
    """
    from pathlib import Path
    from .schema import Report
    from .report import write_html_report, write_json_report
    meta = {
        "mode": mode,
        "status": "failed",
        "failure_reason": reason,
        "auto_route": route.to_dict() if route else None,
    }
    report = Report(
        overall_score=None, confidence=0.0, scores={},
        event_ambiguity=0.0, worst_windows=[], features={}, meta=meta)
    if out_dir:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        write_json_report(report, out / "report.json")
        write_html_report(report, out / "report.html")
    return report


def _refused_report_obj(refused: dict):
    """Wrap a fail-closed dict into a Report so the HTML/JSON writers accept it."""
    from .schema import Report
    meta = dict(refused.get("meta") or {})
    return Report(
        overall_score=refused.get("overall_score"),  # None -> null in JSON
        confidence=refused.get("confidence", 0.0),
        scores=refused.get("scores", {}),
        event_ambiguity=refused.get("event_ambiguity", 0.0),
        worst_windows=[],
        features=refused.get("features", {}),
        meta=meta,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="rr-vfiqa",
        description="Mode-aware VFI quality assessment")
    parser.add_argument("--version", action="version",
                        version=f"%(prog)s {VERSION}")
    sub = parser.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("evaluate", help="evaluate one candidate with an explicit contract")
    pe.add_argument("--mode", required=True,
                    choices=[mode.value for mode in EvaluationMode])
    pe.add_argument("--reference", "--source", dest="reference", default=None,
                    help="reference video (required except in no-reference mode)")
    pe.add_argument("--candidate", required=True)
    pe.add_argument("--out", default=None, help="report output directory")
    pe.add_argument("--calibrator", default=None, help="trained LightGBM pickle")
    pe.add_argument("--vqa-backend", default="none",
                    choices=["none", "pyiqa-niqe"],
                    help="weak learned prior for no-reference mode")
    pe.add_argument("--geometry-policy", default="strict",
                    choices=[
                        "strict", "resize-candidate", "common-resolution"],
                    help="full-reference frame geometry policy")
    _add_common(pe)

    pc = sub.add_parser("compare", help="rank candidates within one evaluation mode")
    pc.add_argument("--mode", required=True,
                    choices=[mode.value for mode in EvaluationMode])
    pc.add_argument("--reference", "--source", dest="reference", default=None)
    pc.add_argument("--candidates", nargs="+", required=True)
    pc.add_argument("--labels", nargs="*", default=None)
    pc.add_argument("--out", default=None)
    pc.add_argument("--vqa-backend", default="none",
                    choices=["none", "pyiqa-niqe"])
    pc.add_argument("--geometry-policy", default="strict",
                    choices=[
                        "strict", "resize-candidate", "common-resolution"])
    pc.add_argument("--allow-cross-content", action="store_true",
                    help="NR only: emit independent reports without ranking")
    pc.add_argument(
        "--comparison-group-id",
        default=None,
        help=(
            "NR only: trusted same-content group; bypass visual fingerprint "
            "rejection while retaining FPS/PTS/geometry checks"),
    )
    _add_common(pc)

    pins = sub.add_parser(
        "inspect",
        help="one-command auto-safe evaluation: route the mode, "
             "score, diagnose and write an HTML report")
    pins.add_argument("--candidate", required=True)
    pins.add_argument("--reference", "--source", dest="reference", default=None)
    pins.add_argument("--mode", default="auto",
                      choices=["auto"] + [m.value for m in EvaluationMode],
                      help="auto (default) safely picks the contract")
    pins.add_argument("--out", default=None,
                      help="report output directory (required for report.html "
                           "/ heatmaps / badcase clips; without it only the "
                           "JSON summary is printed)")
    pins.add_argument("--calibrator", default=None)
    pins.add_argument("--vqa-backend", default="none",
                      choices=["none", "pyiqa-niqe"])
    pins.add_argument("--geometry-policy", default="strict",
                      choices=["strict", "resize-candidate", "common-resolution"])
    _add_common(pins)

    pb = sub.add_parser(
        "inspect-batch",
        help="run auto-safe inspect over a manifest of jobs and write an index")
    pb.add_argument("--manifest", required=True, help="jobs.json with {items:[...]}")
    pb.add_argument("--out", required=True, help="batch output root")
    _add_common(pb)

    args = parser.parse_args(argv)
    progress = None if args.quiet else (lambda m: print(f"[rr-vfiqa] {m}"))
    preset = _resolve_preset(args)

    if args.cmd == "inspect-batch":
        from .execution.batch_runner import inspect_batch
        res = inspect_batch(
            args.manifest, args.out,
            inspect_fn=lambda **kw: _inspect_core(**kw),
            speed=getattr(args, "speed", None), preset=preset,
            no_clips=args.no_clips, flow_backend=args.flow_backend,
            device=args.device, progress=progress)
        print(json.dumps({"jobs": len(res["rows"]), "out_root": res["out_root"]},
                         indent=2, ensure_ascii=False))
        return 0

    if args.cmd == "inspect":
        return _run_inspect(args, preset, progress)

    if args.cmd == "evaluate":
        if args.mode == EvaluationMode.NO_REFERENCE.value and args.reference:
            parser.error("--reference is not allowed with --mode no-reference")
        if args.mode != EvaluationMode.NO_REFERENCE.value and not args.reference:
            parser.error(f"--reference is required with --mode {args.mode}")
        extra = _mode_kwargs(
            args.mode, vqa_backend=args.vqa_backend,
            calibrator=args.calibrator, geometry_policy=args.geometry_policy)
        report = evaluate(
            candidate_video=args.candidate,
            reference_video=args.reference,
            mode=args.mode,
            preset=preset,
            cache_dir=args.cache_dir,
            device=args.device,
            out_dir=args.out,
            flow_backend=args.flow_backend,
            export_clips=not args.no_clips,
            progress=progress,
            **extra,
        )
        print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
        # Non-zero exit on a fail-closed evaluation so the CLI can gate CI.
        return 1 if report.meta.get("status") == "failed" else 0

    if args.cmd == "compare":
        if args.mode == EvaluationMode.NO_REFERENCE.value and args.reference:
            parser.error("--reference is not allowed with --mode no-reference")
        if args.mode != EvaluationMode.NO_REFERENCE.value and not args.reference:
            parser.error(f"--reference is required with --mode {args.mode}")
        compare_extra = _mode_kwargs(
            args.mode, vqa_backend=args.vqa_backend,
            calibrator=None, geometry_policy=args.geometry_policy)
        results = compare(
            args.candidates,
            reference_video=args.reference,
            mode=args.mode,
            preset=preset,
            cache_dir=args.cache_dir,
            device=args.device,
            out_dir=args.out,
            labels=args.labels,
            flow_backend=args.flow_backend,
            progress=progress,
            allow_cross_content=args.allow_cross_content,
            comparison_group_id=args.comparison_group_id,
            **compare_extra,
        )
        print(f"{'rank':<5}{'model':<28}{'overall':>9}{'relative':>10}{'conf':>7}")
        for i, r in enumerate(results, 1):
            overall = f"{r['overall']:.2f}" if r["overall"] is not None else "FAILED"
            relative = (f"{r['relative_vs_mean']:.2f}"
                        if r["relative_vs_mean"] is not None else "—")
            print(f"{i:<5}{r['model']:<28}{overall:>9}"
                  f"{relative:>10}{r['confidence']:>7.2f}")
        return 1 if any(r["status"] == "failed" for r in results) else 0

    return 2


if __name__ == "__main__":
    sys.exit(main())
