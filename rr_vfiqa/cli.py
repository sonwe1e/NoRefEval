"""Command-line interface: rr-vfiqa evaluate / compare."""

from __future__ import annotations

import argparse
import json
import sys

from .pipeline import compare_models, evaluate_vfi


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
        description="Endpoint-referenced VFI quality assessment (60→120 FPS)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("evaluate", help="evaluate one candidate against its source")
    pe.add_argument("--source", required=True)
    pe.add_argument("--candidate", required=True)
    pe.add_argument("--out", default=None, help="report output directory")
    pe.add_argument("--no-clips", action="store_true")
    pe.add_argument("--calibrator", default=None, help="trained LightGBM pickle")
    _add_common(pe)

    pc = sub.add_parser("compare", help="rank several candidates on one source")
    pc.add_argument("--source", required=True)
    pc.add_argument("--candidates", nargs="+", required=True)
    pc.add_argument("--labels", nargs="*", default=None)
    pc.add_argument("--out", default=None)
    _add_common(pc)

    args = parser.parse_args(argv)
    progress = None if args.quiet else (lambda m: print(f"[rr-vfiqa] {m}"))

    if args.cmd == "evaluate":
        report = evaluate_vfi(
            args.source, args.candidate, preset=args.preset,
            cache_dir=args.cache_dir, device=args.device, out_dir=args.out,
            flow_backend=args.flow_backend, export_clips=not args.no_clips,
            calibrator_path=args.calibrator, progress=progress)
        print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
        # Non-zero exit on a fail-closed evaluation so the CLI can gate CI.
        return 1 if report.meta.get("status") == "failed" else 0

    if args.cmd == "compare":
        results = compare_models(
            args.source, args.candidates, preset=args.preset,
            cache_dir=args.cache_dir, device=args.device, out_dir=args.out,
            labels=args.labels, flow_backend=args.flow_backend, progress=progress)
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
