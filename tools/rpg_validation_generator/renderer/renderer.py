"""Scene base class and per-frame compositing pipeline (PICPLAN §6.2).

Layer order is fixed:

    background -> ground decoration -> rear building -> character/enemy
    -> weapon/projectile -> foreground occluder -> world text
    -> screen UI -> post effects

World layers are drawn on the large world canvas and warped by the camera;
screen-space UI is drawn afterwards in viewport pixels.
"""
from __future__ import annotations

import numpy as np

from ..config import GeneratorConfig
from .camera import Camera
from .primitives import Canvas


class Scene:
    scene_id: str = "scene_base"
    scene_title: str = "base"

    def __init__(self, config: GeneratorConfig, seed: int):
        self.config = config
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        # Feet line as a fraction of world height (PICPLAN §7 scenes reuse
        # one vertical anchor that scales with the world canvas).
        self.ground_y = round(config.world_height * 0.64)
        # Reusable draw buffers (reset each frame in render_frame).
        self._world = Canvas(config.world_width, config.world_height)
        self._vp = Canvas(config.width, config.height)
        self.setup()

    # -- to be implemented by concrete scenes -------------------------------
    def setup(self) -> None:
        """Build static objects / particle pools from ``self.rng``."""

    def camera_at(self, t: float) -> Camera:
        return Camera(self.config.world_width / 2, self.config.world_height / 2)

    def draw_background(self, cv: Canvas, t: float) -> None:
        pass

    def draw_actors(self, cv: Canvas, t: float) -> None:
        """character/enemy/weapon/projectile/particle layers."""

    def draw_occluders(self, cv: Canvas, t: float) -> None:
        pass

    def draw_world_text(self, cv: Canvas, t: float) -> None:
        pass

    def draw_ui(self, cv: Canvas, t: float) -> None:
        """Screen-space HUD, drawn after the camera warp."""

    def post_effects(self, cv: Canvas, t: float) -> None:
        pass

    # -- pipeline -----------------------------------------------------------
    def clamp_camera(self, cam: Camera) -> Camera:
        half_w = (self.config.width / 2) / cam.zoom
        half_h = (self.config.height / 2) / cam.zoom
        cam.cx = min(max(cam.cx, half_w), self.config.world_width - half_w)
        cam.cy = min(max(cam.cy, half_h), self.config.world_height - half_h)
        return cam

    def render_frame(self, frame_index: int):
        """Return (rgb, labels, world_only) viewport arrays for one frame.

        ``world_only`` is the camera-warped world *after* post effects but
        *before* screen-space UI is composited.  UI defects use it to refill
        the scene content that a moving HUD element uncovers, keeping the
        world region bit-identical to the clean master.
        """
        t = frame_index / float(self.config.master_fps)
        world = self._world
        world.reset()
        self.draw_background(world, t)
        self.draw_actors(world, t)
        self.draw_occluders(world, t)
        self.draw_world_text(world, t)
        cam = self.clamp_camera(self.camera_at(t))
        rgb, labels = cam.warp(world.rgb, world.labels,
                               self.config.width, self.config.height)
        vp = self._vp
        vp.rgb[:] = rgb
        vp.labels[:] = labels
        self.post_effects(vp, t)             # world-space flashes, etc.
        world_only = vp.rgb.copy()
        self.draw_ui(vp, t)                  # screen-space HUD on top
        return vp.rgb, vp.labels, world_only

    # -- spec / hashing ------------------------------------------------------
    def spec(self) -> dict:
        """Canonical, JSON-serializable description of this scene run."""
        c = self.config
        return {
            "scene_id": self.scene_id,
            "scene_title": self.scene_title,
            "seed": self.seed,
            "resolution": [c.width, c.height],
            "world_size": [c.world_width, c.world_height],
            "duration_seconds": c.duration_seconds,
            "master_fps": c.master_fps,
            "n_master_frames": c.n_master_frames,
        }
