"""Master rendering and explicit temporal sampling (PICPLAN §8).

The master at 120 FPS is the single clean temporal truth.  60 FPS and 30 FPS
versions are derived by *frame-index* sampling in Python — never via an
FFmpeg framerate filter:

    clean_120 = master[0::1]
    clean_60  = master[0::2]
    clean_30  = master[0::4]
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import GeneratorConfig
from .renderer.renderer import Scene


@dataclass
class Master:
    """RGB + label + world-only memmaps for one scene's clean 120 FPS master."""

    scene_id: str
    rgb: np.ndarray          # (n, h, w, 3) uint8 memmap
    labels: np.ndarray       # (n, h, w) uint8 memmap
    world_only: np.ndarray   # (n, h, w, 3) uint8 memmap (no screen UI)
    fps: int
    path_rgb: Path
    path_labels: Path
    path_world: Path

    @property
    def n_frames(self) -> int:
        return self.rgb.shape[0]

    def close(self) -> None:
        del self.rgb
        del self.labels
        del self.world_only


def render_master(scene: Scene, config: GeneratorConfig, workdir: Path,
                  keep_labels: bool = True) -> Master:
    """Render the full clean master into memmaps (bounded RAM)."""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    n, h, w = config.n_master_frames, config.height, config.width
    rgb_path = workdir / f"{scene.scene_id}_master_rgb.raw"
    lab_path = workdir / f"{scene.scene_id}_master_labels.raw"
    world_path = workdir / f"{scene.scene_id}_master_world.raw"
    rgb = np.memmap(rgb_path, dtype=np.uint8, mode="w+", shape=(n, h, w, 3))
    labels = np.memmap(lab_path, dtype=np.uint8, mode="w+", shape=(n, h, w))
    world = np.memmap(world_path, dtype=np.uint8, mode="w+", shape=(n, h, w, 3))
    for i in range(n):
        frame_rgb, frame_lab, frame_world = scene.render_frame(i)
        rgb[i] = frame_rgb
        labels[i] = frame_lab
        world[i] = frame_world
    rgb.flush()
    labels.flush()
    world.flush()
    if not keep_labels:
        labels = None
    return Master(scene.scene_id, rgb, labels, world, config.master_fps,
                  rgb_path, lab_path, world_path)


def sample_indices(master_fps: int, target_fps: int, n_master: int) -> np.ndarray:
    """Frame indices for exact integer-ratio downsampling."""
    if master_fps % target_fps != 0:
        raise ValueError(
            f"target fps {target_fps} must divide master fps {master_fps}")
    step = master_fps // target_fps
    return np.arange(0, n_master, step)


def gather(frames: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Materialize sampled frames as a contiguous uint8 array."""
    return np.ascontiguousarray(frames[indices])
