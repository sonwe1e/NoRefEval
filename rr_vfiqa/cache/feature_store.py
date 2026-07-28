"""Tiny on-disk array store: .npy / .npz files with optional read-only mmap."""

from __future__ import annotations

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

    def save_npy(self, key: str, arr: np.ndarray) -> None:
        np.save(self._path(key, "npy"), arr)

    def load_npy(self, key: str, mmap: bool = True) -> np.ndarray:
        mode = "r" if mmap else None
        return np.load(self._path(key, "npy"), mmap_mode=mode)

    def save_npz(self, key: str, **arrays) -> None:
        np.savez_compressed(self._path(key, "npz"), **arrays)

    def load_npz(self, key: str) -> dict[str, np.ndarray]:
        with np.load(self._path(key, "npz")) as z:
            return {k: z[k] for k in z.files}

    def save_json(self, key: str, obj) -> None:
        import json
        self._path(key, "json").write_text(json.dumps(obj), encoding="utf-8")

    def load_json(self, key: str):
        import json
        return json.loads(self._path(key, "json").read_text(encoding="utf-8"))
