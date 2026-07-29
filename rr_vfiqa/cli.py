"""Command-line interface: explicit no-reference / endpoint / full-reference."""

from __future__ import annotations

import argparse
import json
import sys

from .config import EvaluationMode
from .multimode import compare, evaluate


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--preset", default="standard",
                   choices=["fast", "standard", "audit"])
    p.add_argument("--cache-dir", default="./cache")
    p.add_argument("--device", default="cuda")
    p.add_argument("--flow-backend", default="auto",
                   choices=["auto", "raft", "farneback"])
    p.add_argument("--quiet", action="store_true")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="rr-vfiqa",
        description="Mode-aware VFI quality assessment")
    sub = parser.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("evaluate", help="evaluate one candidate with an explicit contract")
    pe.add_argument("--mode", required=True,
                    choices=[mode.value for mode in EvaluationMode])
    pe.add_argument("--reference", "--source", dest="reference", default=None,
                    help="reference video (required except in no-reference mode)")
    pe.add_argument("--candidate", required=True)
    pe.add_argument("--out", default=None, help="report output directory")
    pe.add_argument("--no-clips", action="store_true")
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

    args = parser.parse_args(argv)
    progress = None if args.quiet else (lambda m: print(f"[rr-vfiqa] {m}"))

    if args.cmd == "evaluate":
        if args.mode == EvaluationMode.NO_REFERENCE.value and args.reference:
            parser.error("--reference is not allowed with --mode no-reference")
        if args.mode != EvaluationMode.NO_REFERENCE.value and not args.reference:
            parser.error(f"--reference is required with --mode {args.mode}")
        extra = {}
        if args.mode == EvaluationMode.NO_REFERENCE.value:
            extra["vqa_backend"] = args.vqa_backend
        elif args.mode == EvaluationMode.ENDPOINT_2X.value:
            extra["calibrator_path"] = args.calibrator
        elif args.mode == EvaluationMode.FULL_REFERENCE.value:
            extra["geometry_policy"] = args.geometry_policy
        report = evaluate(
            candidate_video=args.candidate,
            reference_video=args.reference,
            mode=args.mode,
            preset=args.preset,
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
        compare_extra = {}
        if args.mode == EvaluationMode.NO_REFERENCE.value:
            compare_extra["vqa_backend"] = args.vqa_backend
        elif args.mode == EvaluationMode.FULL_REFERENCE.value:
            compare_extra["geometry_policy"] = args.geometry_policy
        results = compare(
            args.candidates,
            reference_video=args.reference,
            mode=args.mode,
            preset=args.preset,
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
