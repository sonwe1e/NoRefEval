"""Reproducibility metadata shared by every calibration/validation artifact."""

from __future__ import annotations

import importlib.metadata
from functools import lru_cache
from hashlib import sha256
import os
import platform
from pathlib import Path
import subprocess
import sys
from typing import Any

from .._build_info import BUILD_COMMIT, BUILD_DIRTY
from .._version import VERSION
from ..motion.flow_estimator import get_flow_backend

CALIBRATION_SCHEMA_VERSION = 1


@lru_cache(maxsize=1)
def _build_contract() -> dict[str, Any]:
    package_root = Path(__file__).resolve().parents[1]
    digest = sha256()
    for path in sorted(package_root.rglob("*.py")):
        relative = path.relative_to(package_root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    try:
        version = importlib.metadata.version("rr-vfiqa")
    except importlib.metadata.PackageNotFoundError:
        version = "source-tree"
    return {
        "source_version": VERSION,
        "distribution_version": version,
        "package_source_sha256": digest.hexdigest(),
        "embedded_commit": os.environ.get(
            "RR_VFIQA_BUILD_COMMIT", BUILD_COMMIT),
        "embedded_dirty": BUILD_DIRTY,
    }


def file_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def calibrator_provenance(
    path: str | Path | None,
    *,
    meta: dict[str, Any] | None,
    expected_feature_contract_hash: str,
) -> dict[str, Any] | None:
    if path is None:
        return None
    resolved = Path(path).resolve()
    if not resolved.exists():
        return None
    metadata = meta or {}
    return {
        "path": str(resolved),
        "sha256": file_sha256(resolved),
        "feature_contract_hash": metadata.get(
            "feature_contract_hash", "unverified"),
        "expected_feature_contract_hash": expected_feature_contract_hash,
        "training_manifest_hash": metadata.get("training_manifest_hash"),
    }


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
        embedded = _build_contract()["embedded_commit"]
        return {
            "commit_sha": embedded,
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
        "build_contract": _build_contract(),
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
    artifact_contract: str = "artifact-contract-v1",
    diagnosis_contract: str = "diagnosis-contract-v2",
) -> dict[str, Any]:
    """Minimal reproducibility contract embedded in every evaluation report."""
    git = _git_state()
    return {
        "metric_contract": metric_contract,
        "calibration_id": calibration_id,
        "preset_contract": preset_contract,
        "artifact_contract": artifact_contract,
        "diagnosis_contract": diagnosis_contract,
        "feature_contract_hash": feature_contract_hash,
        "backend_contract": backend_contract,
        "code_commit": git["commit_sha"],
        "working_tree_dirty": git["working_tree_dirty"],
        "tracked_diff_sha256": git["tracked_diff_sha256"],
        "runtime_contract": {
            "python": sys.version.split()[0],
            "packages": _versions(),
        },
        "build_contract": _build_contract(),
        "production_gate": False,
        "mode": mode,
        "score_schema": score_schema,
    }
