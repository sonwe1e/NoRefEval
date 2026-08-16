"""Cache schema versioning. Bump SCHEMA_VERSION to invalidate old caches."""

from __future__ import annotations

import json
from pathlib import Path

SCHEMA_VERSION = 2


def read_meta(cache_dir: Path) -> dict:
    meta_path = cache_dir / "meta.json"
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def write_meta(cache_dir: Path, meta: dict) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta = dict(meta)
    meta["schema_version"] = SCHEMA_VERSION
    (cache_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def is_valid(cache_dir: Path) -> bool:
    return read_meta(cache_dir).get("schema_version") == SCHEMA_VERSION
