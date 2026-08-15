"""Tiny on-disk array store: .npy / .npz files with optional read-only mmap.

Writes are atomic (temp file + ``os.replace``): a concurrent reader either
sees the old file or the complete new file, never a half-written one.  The
evaluation window loops run in a thread pool, so several threads can race to
compute the same missing cache entry.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np


class FeatureStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str, ext: str) -> Path:
        safe = key.replace("/", "__")
        return self.root / f"{safe}.{ext}"

    def has(self, key: str, ext: str = "npy") -> bool:
        return self._path(key, ext).exists()

    def _atomic_write(self, target: Path, write: object) -> None:
        """Write via a sibling temp file, then atomically replace the target."""
        fd, tmp = tempfile.mkstemp(dir=str(self.root), suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                write(f)
            os.replace(tmp, target)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def save_npy(self, key: str, arr: np.ndarray) -> None:
        target = self._path(key, "npy")
        self._atomic_write(target, lambda f: np.save(f, arr))

    def load_npy(self, key: str, mmap: bool = True) -> np.ndarray:
        mode = "r" if mmap else None
        return np.load(self._path(key, "npy"), mmap_mode=mode)

    def save_npz(self, key: str, **arrays) -> None:
        # Plain (uncompressed) zip: float16 flows barely compress (~28% at
        # best) but deflate costs ~20x on write; old compressed entries remain
        # readable by np.load, so no cache migration is needed.
        target = self._path(key, "npz")
        self._atomic_write(target, lambda f: np.savez(f, **arrays))

    def load_npz(self, key: str) -> dict[str, np.ndarray]:
        with np.load(self._path(key, "npz")) as z:
            return {k: z[k] for k in z.files}

    def save_json(self, key: str, obj) -> None:
        import json
        self._path(key, "json").write_text(json.dumps(obj), encoding="utf-8")

    def load_json(self, key: str):
        import json
        return json.loads(self._path(key, "json").read_text(encoding="utf-8"))
