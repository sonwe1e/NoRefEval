"""Characters, enemies, NPCs and the sword (PICPLAN §6.3).

Humanoids are composed of circles/ellipses/polygons/line segments with an
analytic sinusoidal gait; no sprites, no external art.
"""
from __future__ import annotations

import numpy as np

from ..motion import sinusoidal
from .primitives import Canvas


class Character:
    """Player avatar.  ``label`` lets scenes re-use this for NPC-ish roles."""

    def __init__(self, body=(196, 128, 60), head=(140, 180, 235),
                 scale: float = 1.0, label: str = "character"):
        self.body = tuple(int(c) for c in body)
        self.head = tuple(int(c) for c in head)
        self.scale = scale
        self.label = label

    def draw(self, cv: Canvas, x: float, y: float, t: float,
             facing: float = 1.0, speed: float = 1.0,
             arm_angle_deg: float | None = None) -> None:
        """Draw at feet position (x, y).  ``t`` drives the walk cycle."""
        s = self.scale
        f = facing
        gait = sinusoidal(t, 3.2 * max(speed, 0.05), 1.0)
        leg = 10 * s * gait
        # legs
        cv.line(x - 4 * s, y - 18 * s, x - 5 * s + leg, y, (60, 60, 70), int(4 * s),
                label=self.label)
        cv.line(x + 4 * s, y - 18 * s, x + 5 * s - leg, y, (50, 50, 62), int(4 * s),
                label=self.label)
        # torso
        cv.ellipse(x, y - 30 * s, 10 * s, 15 * s, 0, self.body, label=self.label)
        # belt
        cv.rect(x - 10 * s, y - 22 * s, 20 * s, 4 * s, (40, 60, 120), label=self.label)
        # head + helmet
        cv.circle(x, y - 52 * s, 9 * s, self.head, label=self.label)
        cv.ellipse(x, y - 56 * s, 10 * s, 6 * s, 0, (90, 100, 130), label=self.label)
        # facing indicator (eye)
        cv.circle(x + 4 * s * f, y - 53 * s, 1.6 * s, (30, 30, 30))
        # arms
        sh = y - 38 * s
        if arm_angle_deg is None:
            swing = 14 * s * gait
            cv.line(x - 8 * s, sh, x - 12 * s - swing, sh + 16 * s, self.body,
                    int(3.5 * s), label=self.label)
            cv.line(x + 8 * s, sh, x + 12 * s + swing, sh + 16 * s, self.body,
                    int(3.5 * s), label=self.label)
        else:
            a = np.radians(arm_angle_deg)
            ex = x + f * 12 * s * np.cos(a)
            ey = sh + 12 * s * np.sin(a)
            cv.line(x + 8 * s * f, sh, ex, ey, self.body, int(3.5 * s),
                    label=self.label)
            cv.line(x - 8 * s * f, sh, x - 12 * s * f, sh + 16 * s, self.body,
                    int(3.5 * s), label=self.label)
            return ex, ey
        return None


class Enemy(Character):
    def __init__(self, scale: float = 1.0):
        super().__init__(body=(70, 70, 130), head=(80, 120, 190), scale=scale,
                         label="enemy")

    def draw(self, cv, x, y, t, facing=-1.0, speed=1.0, arm_angle_deg=None):
        res = super().draw(cv, x, y, t, facing=facing, speed=speed,
                           arm_angle_deg=arm_angle_deg)
        # horns to read as "enemy"
        s = self.scale
        cv.line(x - 6 * s, y - 60 * s, x - 10 * s, y - 70 * s, (60, 60, 110),
                int(2.5 * s), label="enemy")
        cv.line(x + 6 * s, y - 60 * s, x + 10 * s, y - 70 * s, (60, 60, 110),
                int(2.5 * s), label="enemy")
        return res


class NPC(Character):
    def __init__(self, scale: float = 1.0):
        super().__init__(body=(90, 160, 160), head=(150, 190, 220), scale=scale,
                         label="character")


class Sword:
    """Thin weapon (PICPLAN Scene 03): long, narrow, labelled ``weapon``."""

    def __init__(self, length: float = 86.0, width: float = 3.0):
        self.length = length
        self.width = width

    def draw(self, cv: Canvas, hx: float, hy: float, angle_deg: float) -> tuple:
        """Draw from hand point (hx, hy) at ``angle_deg``; returns tip."""
        a = np.radians(angle_deg)
        tx = hx + self.length * np.cos(a)
        ty = hy + self.length * np.sin(a)
        cv.line(hx, hy, tx, ty, (190, 205, 225), self.width, label="weapon")
        # glint along the blade
        mx, my = hx + 0.6 * self.length * np.cos(a), hy + 0.6 * self.length * np.sin(a)
        cv.line(hx + 0.25 * (mx - hx), hy + 0.25 * (my - hy), mx, my,
                (235, 245, 255), max(1.0, self.width - 1.5), label="weapon")
        # guard + grip
        gx = hx - 8 * np.cos(a)
        gy = hy - 8 * np.sin(a)
        pa = a + np.pi / 2
        cv.line(hx + 8 * np.cos(pa), hy + 8 * np.sin(pa),
                hx - 8 * np.cos(pa), hy - 8 * np.sin(pa), (40, 120, 180), 3,
                label="weapon")
        cv.line(hx, hy, gx, gy, (50, 80, 140), 4, label="weapon")
        return tx, ty
