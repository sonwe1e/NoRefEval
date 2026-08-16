"""Projectiles, particles and impact effects (PICPLAN §6.3, Scene 05).

Particle motion is analytic: each particle's parameters are drawn once from
a seeded RNG at setup, and its position at time t is a closed-form function
of (birth, velocity, t).  No per-frame RNG, no integration.
"""
from __future__ import annotations

import numpy as np

from ..motion import bezier_2d, window
from .primitives import Canvas


class Projectile:
    """Magic bolt flying along a cubic bezier; labelled ``projectile``."""

    def __init__(self, p0, p1, p2, p3, radius: float = 9.0,
                 color=(255, 200, 90), core=(255, 255, 220)):
        self.p0, self.p1, self.p2, self.p3 = p0, p1, p2, p3
        self.r = radius
        self.color = tuple(int(c) for c in color)
        self.core = tuple(int(c) for c in core)
        self.last_pos = None

    def position(self, u: float) -> tuple[float, float]:
        return bezier_2d(u, self.p0, self.p1, self.p2, self.p3)

    def draw(self, cv: Canvas, u: float, alpha: float = 1.0) -> tuple[float, float]:
        x, y = self.position(u)
        self.last_pos = (x, y)
        r = self.r
        if alpha < 1.0:
            halo = (np.array(self.color) * alpha).astype(int).tolist()
            core = (np.array(self.core) * alpha).astype(int).tolist()
        else:
            halo, core = self.color, self.core
        cv.circle(x, y, r * 1.9, halo, label="projectile")
        cv.circle(x, y, r, self.color, label="projectile")
        cv.circle(x, y, r * 0.45, core, label="projectile")
        # motion tail
        for k in range(1, 4):
            tu = max(0.0, u - 0.035 * k)
            tx, ty = self.position(tu)
            cv.circle(tx, ty, max(1.5, r * (1 - 0.22 * k)), halo,
                      label="projectile")
        return x, y


class ParticleEmitter:
    """Burst of analytic particles spawned at ``burst_time``."""

    def __init__(self, rng: np.random.Generator, origin, n: int = 46,
                 burst_time: float = 2.0, speed=(60.0, 240.0),
                 life=(0.25, 0.6), gravity: float = 260.0,
                 colors=((120, 200, 255), (180, 230, 255), (90, 160, 250))):
        self.ox, self.oy = float(origin[0]), float(origin[1])
        self.burst = burst_time
        self.g = gravity
        ang = rng.uniform(0, 2 * np.pi, n)
        spd = rng.uniform(speed[0], speed[1], n)
        self.vx = np.cos(ang) * spd
        self.vy = np.sin(ang) * spd
        self.life = rng.uniform(life[0], life[1], n)
        self.size = rng.uniform(1.5, 4.0, n)
        self.colors = [colors[i % len(colors)] for i in range(n)]

    def draw(self, cv: Canvas, t: float) -> None:
        dt = t - self.burst
        if dt < 0:
            return
        alive = dt < self.life
        if not np.any(alive):
            return
        xs = self.ox + self.vx * dt
        ys = self.oy + self.vy * dt + 0.5 * self.g * dt * dt
        fade = np.clip(1.0 - dt / self.life, 0, 1)
        for i in np.nonzero(alive)[0]:
            col = tuple(int(c * fade[i]) for c in self.colors[i])
            cv.circle(float(xs[i]), float(ys[i]), float(self.size[i]), col,
                      label="particle")


class ShockRing:
    """Expanding impact ring (labelled particle)."""

    def __init__(self, origin, start_time: float, duration: float = 0.35,
                 max_r: float = 70.0, color=(200, 230, 255)):
        self.ox, self.oy = origin
        self.t0 = start_time
        self.dur = duration
        self.max_r = max_r
        self.color = color

    def draw(self, cv: Canvas, t: float) -> None:
        u = window(t, self.t0, self.t0 + self.dur)
        if u <= 0 or u >= 1:
            return
        r = self.max_r * u
        alpha = 1.0 - u
        col = tuple(int(c * alpha) for c in self.color)
        cv.circle(self.ox, self.oy, max(2.0, r), col, thickness=3,
                  label="particle")


class Flash:
    """Short full-region brightness flash around a point (Scene 03 hit)."""

    def __init__(self, origin, start_time: float, duration: float = 0.12,
                 radius: float = 90.0):
        self.ox, self.oy = origin
        self.t0 = start_time
        self.dur = duration
        self.radius = radius

    def intensity(self, t: float) -> float:
        u = window(t, self.t0, self.t0 + self.dur)
        if u <= 0 or u >= 1:
            return 0.0
        return float(np.sin(np.pi * u))  # 0→1→0

    def draw(self, cv: Canvas, t: float) -> None:
        k = self.intensity(t)
        if k <= 0.01:
            return
        h, w = cv.rgb.shape[:2]
        ys, xs = np.mgrid[0:h, 0:w]
        d2 = (xs - self.ox) ** 2 + (ys - self.oy) ** 2
        wgt = np.exp(-d2 / (2 * (self.radius ** 2))) * k
        add = (wgt * 130)[:, :, None]
        cv.rgb = np.clip(cv.rgb.astype(np.float32) + add, 0, 255).astype(np.uint8)
