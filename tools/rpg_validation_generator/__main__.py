"""Command-line entry point (PICPLAN §18).

    python -m tools.rpg_validation_generator generate --output validation --mode all
    python -m tools.rpg_validation_generator generate --draft --mode all
    python -m tools.rpg_validation_generator validate --root validation
"""
from __future__ import annotations

import argparse
import json
import sys

from .config import GeneratorConfig
from .orchestration import generate
from .validate import validate_root


def _parse_resolution(s: str) -> tuple[int, int]:
    w, h = s.lower().split("x")
    return int(w), int(h)


def _build_config(args: argparse.Namespace) -> GeneratorConfig:
    if args.draft:
        cfg = GeneratorConfig.draft(seed=args.seed)
    else:
        cfg = GeneratorConfig.production(seed=args.seed)
    if args.resolution:
        w, h = _parse_resolution(args.resolution)
        cfg = GeneratorConfig(width=w, height=h,
                              duration_seconds=cfg.duration_seconds,
                              master_fps=cfg.master_fps, seed=cfg.seed,
                              world_width=w + 320, world_height=h + 280)
    if args.duration is not None:
        cfg = GeneratorConfig(width=cfg.width, height=cfg.height,
                              duration_seconds=args.duration,
                              master_fps=cfg.master_fps, seed=cfg.seed,
                              world_width=cfg.world_width,
                              world_height=cfg.world_height)
    if args.master_fps is not None:
        cfg = GeneratorConfig(width=cfg.width, height=cfg.height,
                              duration_seconds=cfg.duration_seconds,
                              master_fps=args.master_fps, seed=cfg.seed,
                              world_width=cfg.world_width,
                              world_height=cfg.world_height)
    return cfg


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="rpg_validation_generator")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="generate validation data")
    g.add_argument("--output", default="validation")
    g.add_argument("--resolution", default=None,
                   help="WxH, e.g. 1280x720 (overrides preset)")
    g.add_argument("--duration", type=float, default=None)
    g.add_argument("--master-fps", type=int, default=None)
    g.add_argument("--seed", type=int, default=20260729)
    g.add_argument("--mode", choices=["all", "nr", "endpoint", "fr"],
                   default="all")
    g.add_argument("--draft", action="store_true",
                   help="640x360 / 2s development preset")
    g.add_argument("--overwrite", action="store_true")
    g.add_argument("--workdir", default=None,
                   help="persistent scratch dir for master memmaps")

    v = sub.add_parser("validate", help="validate a generated tree")
    v.add_argument("--root", default="validation")
    v.add_argument("--full", action="store_true",
                   help="decode every frame for agreement checks (slow; default samples)")

    args = p.parse_args(argv)

    if args.cmd == "generate":
        cfg = _build_config(args)
        result = generate(args.output, cfg, mode=args.mode,
                          overwrite=args.overwrite, workdir=args.workdir)
        print(json.dumps(result, indent=2))
        return 0

    if args.cmd == "validate":
        errs = validate_root(args.root, full_decode=getattr(args, "full", False))
        if errs:
            print(f"VALIDATION FAILED ({len(errs)} issue(s)):")
            for e in errs:
                print("  -", e)
            return 1
        print("VALIDATION OK")
        return 0

    return 2


if __name__ == "__main__":
    sys.exit(main())
