"""SHA256 helpers (PICPLAN §14).

``raw_rgb_sha256`` streams frame-by-frame over a (n, h, w, 3) uint8 array
(or memmap), so the full raw tensor never needs to be materialized as one
bytes object.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

_CHUNK_FRAMES = 16


def raw_rgb_sha256(frames: np.ndarray) -> str:
    h = hashlib.sha256()
    n = frames.shape[0]
    for i in range(0, n, _CHUNK_FRAMES):
        block = np.ascontiguousarray(frames[i:i + _CHUNK_FRAMES])
        h.update(block.tobytes())
    return h.hexdigest()


def raw_rgb_sha256_subset(frames: np.ndarray, indices) -> str:
    """Hash selected frames in order (used for 60/30 FPS derivatives)."""
    h = hashlib.sha256()
    for i in indices:
        h.update(np.ascontiguousarray(frames[i]).tobytes())
    return h.hexdigest()


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def json_sha256(obj: dict) -> str:
    import json
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()
