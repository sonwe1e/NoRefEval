"""Reproducibility metadata shared by every calibration/validation artifact."""

from __future__ import annotations

import importlib.metadata
from hashlib import sha256
import platform
from pathlib import Path
import subprocess
import sys
from typing import Any

from ..motion.flow_estimator import get_flow_backend

CALIBRATION_SCHEMA_VERSION = 1


def _git_state() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True,
            capture_output=True, text=True, timeout=5)
        diff = subprocess.run(
            ["git", "diff", "--binary", "--", "."], cwd=root, check=True,
            capture_output=True, timeout=10)
        staged = subprocess.run(
            ["git", "diff", "--cached", "--binary", "--", "."], cwd=root,
            check=True, capture_output=True, timeout=10)
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, check=True,
            capture_output=True, text=True, timeout=5)
        return {
            "commit_sha": head.stdout.strip(),
            "working_tree_dirty": bool(status.stdout.strip()),
            "tracked_diff_sha256": (
                sha256(diff.stdout + staged.stdout).hexdigest()
                if diff.stdout or staged.stdout else None),
        }
    except (OSError, subprocess.SubprocessError):
        return {
            "commit_sha": "unknown",
            "working_tree_dirty": None,
            "tracked_diff_sha256": None,
        }


def _versions() -> dict[str, str]:
    out: dict[str, str] = {}
    for name in ("rr-vfiqa", "numpy", "opencv-python-headless", "av", "torch",
                 "torchvision", "scipy", "lightgbm"):
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = "not-installed"
    return out


def calibration_provenance(
        evidence_kind: str, *, preset: str, flow_backend: str, device: str,
        dataset: dict[str, Any],
) -> dict[str, Any]:
    backend = get_flow_backend(flow_backend, device)
    provenance = {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "evidence_kind": evidence_kind,
        "scope": "synthetic_bootstrap"
        if evidence_kind.startswith("synthetic") else "unspecified",
        "environment": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "device": device,
            "packages": _versions(),
        },
        "evaluation": {
            "preset": preset,
            "flow_backend": backend.cache_identity(),
        },
        "dataset": dataset,
    }
    provenance.update(_git_state())
    return provenance


def report_provenance(
    *,
    mode: str,
    score_schema: str,
    metric_contract: str,
    preset_contract: str,
    feature_contract_hash: str,
    backend_contract: dict[str, Any],
    calibration_id: str | None = None,
) -> dict[str, Any]:
    """Minimal reproducibility contract embedded in every evaluation report."""
    git = _git_state()
    return {
        "metric_contract": metric_contract,
        "calibration_id": calibration_id,
        "preset_contract": preset_contract,
        "feature_contract_hash": feature_contract_hash,
        "backend_contract": backend_contract,
        "code_commit": git["commit_sha"],
        "working_tree_dirty": git["working_tree_dirty"],
        "tracked_diff_sha256": git["tracked_diff_sha256"],
        "runtime_contract": {
            "python": sys.version.split()[0],
            "packages": _versions(),
        },
        "production_gate": False,
        "mode": mode,
        "score_schema": score_schema,
    }
