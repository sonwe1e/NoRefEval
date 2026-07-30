"""Validation of a generated tree (PICPLAN §19).

Operates purely on the on-disk output (videos + manifests + masks); it never
re-renders.  Returns a list of failure strings (empty == pass).  Post-encode
pixel comparisons use a tolerance that absorbs H.264/yuv420p chroma error;
the *exact* contracts are checked against the pre-encode raw hashes stored
in each manifest (and re-derived here where possible).
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .case_specs import cases_for_mode, spec_by_id
from .encode import decode_frames, probe
from .hashing import file_sha256
from .schema import MODE_DIR

_TOL_MEAN = 4.0      # mean abs per-pixel tolerance after H.264 round-trip
_DECODE_STRIDE = 6   # sample every Nth frame for post-encode agreement checks


def _load_manifest(case_dir: Path) -> dict:
    return json.loads((case_dir / "manifest.json").read_text(encoding="utf-8"))


def _mask_for(case_dir: Path, frame_index: int, h: int, w: int) -> np.ndarray | None:
    # prefer full-resolution contract masks (already include the effect halo)
    p = case_dir / "_contract_masks" / f"{frame_index:06d}.png"
    if not p.exists():
        p = case_dir / "defect_masks" / f"{frame_index:06d}.png"
    if not p.exists():
        return None
    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if img.shape != (h, w):
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)
    return img > 127


def _mean_abs(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean())


def validate_case(case_dir: Path, manifest: dict, full_decode: bool = False) -> list[str]:
    errs: list[str] = []
    cid = manifest["case_id"]
    fps_by_role = manifest["fps_by_role"]
    fc_by_role = manifest["frame_count_by_role"]
    res = manifest["resolution"]

    # --- per-file openability / metadata / audio / hashes -----------------
    decoded_meta: dict[str, dict] = {}
    for role, fname in manifest["files"].items():
        fpath = case_dir / fname
        if not fpath.exists():
            errs.append(f"{cid}: missing {fname}")
            continue
        if manifest["hashes"]["video_sha256"].get(role):
            if file_sha256(fpath) != manifest["hashes"]["video_sha256"][role]:
                errs.append(f"{cid}: video hash mismatch {role}")
        info = probe(fpath)
        decoded_meta[role] = info
        if info["audio_streams"] != 0:
            errs.append(f"{cid}: {role} has audio stream")
        if abs(info["fps"] - fps_by_role[role]) > 0.6:
            errs.append(f"{cid}: {role} fps {info['fps']} != {fps_by_role[role]}")
        if info["frames"] != fc_by_role[role]:
            errs.append(f"{cid}: {role} frames {info['frames']} != {fc_by_role[role]}")
        if (info["width"], info["height"]) != tuple(res):
            errs.append(f"{cid}: {role} resolution {info['width']}x{info['height']} != {res}")

    # --- defect interval within duration ----------------------------------
    for d in manifest["defects"]:
        if d["start_time"] < 0 or d["end_time"] > manifest["duration_seconds"] + 1e-6:
            errs.append(f"{cid}: defect interval out of range")

    mode = manifest["mode"]
    w, h = res  # manifest resolution is [width, height]

    stride = 1 if full_decode else _DECODE_STRIDE

    # --- endpoint even-frame contract (post-encode, tolerant) + raw hash --
    if mode == "endpoint" and {"candidate", "source"} <= set(manifest["files"]):
        n_src = fc_by_role["source"]
        even_all = np.arange(0, 2 * n_src, 2)
        even_idx = even_all[::stride]
        src_idx = np.arange(n_src)[::stride]
        c_ev = decode_frames(case_dir / manifest["files"]["candidate"], even_idx)
        src = decode_frames(case_dir / manifest["files"]["source"], src_idx)
        if c_ev.shape[0] == src.shape[0] and c_ev.shape[0] > 0:
            m = _mean_abs(c_ev, src)
            if m > _TOL_MEAN:
                errs.append(f"{cid}: endpoint even-frame decode diff {m:.2f} > {_TOL_MEAN}")
        rh = manifest["hashes"].get("raw_rgb_sha256", {})
        if rh.get("candidate_even") and rh.get("source") and \
                rh["candidate_even"] != rh["source"]:
            errs.append(f"{cid}: endpoint even-raw hash mismatch (pre-encode contract)")

    # --- NR / FR outside-mask agreement (post-encode, tolerant) -----------
    if mode in ("nr", "fr") and "candidate" in manifest["files"]:
        ref_role = "oracle" if mode == "nr" else "reference"
        if ref_role in manifest["files"]:
            n = fc_by_role["candidate"]
            idx = np.arange(n)[::stride]
            cand = decode_frames(case_dir / manifest["files"]["candidate"], idx)
            ref = decode_frames(case_dir / manifest["files"][ref_role], idx)
            worst = 0.0
            for j, i in enumerate(idx):
                mask = _mask_for(case_dir, i, h, w)
                if mask is not None and mask.mean() > 0.999:
                    continue
                if mask is not None:
                    d = _mean_abs(cand[j][~mask], ref[j][~mask])
                else:
                    d = _mean_abs(cand[j], ref[j])
                worst = max(worst, d)
            if worst > _TOL_MEAN:
                errs.append(f"{cid}: outside-mask decode diff {worst:.2f} > {_TOL_MEAN}")

    # --- FR geometry parity ----------------------------------------------
    if mode == "fr":
        if fc_by_role.get("reference") != fc_by_role.get("candidate"):
            errs.append(f"{cid}: FR ref/cand frame count mismatch")
        if fps_by_role.get("reference") != fps_by_role.get("candidate"):
            errs.append(f"{cid}: FR ref/cand fps mismatch")

    return errs


def validate_root(root: str | Path, full_decode: bool = False) -> list[str]:
    root = Path(root)
    errs: list[str] = []
    if not root.exists():
        return [f"root {root} does not exist"]
    for mode in ("nr", "endpoint", "fr"):
        mode_dir = root / MODE_DIR[mode]
        if not mode_dir.exists():
            errs.append(f"missing mode dir {mode_dir}")
            continue
        jsonl = mode_dir / "manifest.jsonl"
        if not jsonl.exists():
            errs.append(f"missing {jsonl}")
            continue
        manifests = [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
        # mode-level counts (PICPLAN §19)
        if len(manifests) != 5:
            errs.append(f"{mode}: expected 5 cases, got {len(manifests)}")
        if mode == "nr":
            cand_fps = [m["fps_by_role"]["candidate"] for m in manifests]
            if sorted(cand_fps) != sorted([60, 60, 120, 120, 120]):
                errs.append(f"nr: candidate fps distribution wrong: {cand_fps}")
        for m in manifests:
            case_dir = mode_dir / f"case_{int(m['case_id'].split('_')[-1]):02d}"
            errs += validate_case(case_dir, m, full_decode=full_decode)
    return errs
