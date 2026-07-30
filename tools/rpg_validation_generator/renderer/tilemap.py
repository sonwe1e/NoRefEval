"""Ground and static world props (PICPLAN §6.3).

All geometry is derived from a seeded RNG *once* at scene setup; per-frame
drawing is a pure function of (t, camera), never of RNG state.
"""
from __future__ import annotations

import numpy as np

from .primitives import Canvas, checker_fill


class TileMap:
    """Grass ground with a deterministic dirt road and scattered stones."""

    def __init__(self, width: int, height: int, rng: np.random.Generator,
                 road_y: float | None = None, palette="grass"):
        self.w = width
        self.h = height
        palettes = {
            "grass": ((62, 110, 58), (56, 100, 52)),     # BGR greens
            "ruins": ((78, 84, 92), (70, 76, 84)),       # stone greys
            "town": ((96, 116, 138), (88, 108, 128)),    # paving
            "battle": ((52, 58, 74), (46, 52, 66)),      # dark arena
        }
        self.c0, self.c1 = palettes[palette]
        self.road_y = road_y if road_y is not None else height * 0.62
        n = max(12, width // 90)
        self.stones = [
            (float(rng.integers(20, width - 20)), float(rng.integers(20, height - 20)),
             float(rng.integers(4, 11)), int(rng.integers(40, 70)))
            for _ in range(n)
        ]
        n_tufts = max(30, width // 25)
        self.tufts = [
            (float(rng.integers(0, width)), float(rng.integers(0, height)),
             float(rng.uniform(0.4, 1.0)))
            for _ in range(n_tufts)
        ]

    def draw(self, cv: Canvas) -> None:
        bg = checker_fill(self.w, self.h, 32, self.c0, self.c1)
        cv.rgb[:] = bg
        # dirt road band
        ry = int(self.road_y)
        rh = max(36, self.h // 12)
        road = np.array((58, 92, 128), dtype=np.uint8)  # brownish
        cv.rgb[ry:ry + rh] = road
        cv.rgb[ry + 2:ry + rh - 2][::6] = (road * 0.85).astype(np.uint8)
        # grass tufts
        for x, y, s in self.tufts:
            if abs(y - self.road_y) < rh:
                continue
            cv.line(x, y, x - 2 * s, y - 6 * s, (48, 122, 44), 1)
            cv.line(x, y, x + 1, y - 7 * s, (52, 132, 50), 1)
        # stones
        for x, y, r, g in self.stones:
            hi = min(g + 30, 255)
            cv.ellipse(x, y, r, r * 0.7, 0, (g, g + 6, g + 10), label="background")
            cv.ellipse(x - r * 0.25, y - r * 0.2, r * 0.4, r * 0.25, 0, (hi, hi, hi))


class Rock:
    def __init__(self, x: float, y: float, r: float):
        self.x, self.y, self.r = x, y, r

    def draw(self, cv: Canvas) -> None:
        cv.ellipse(self.x, self.y, self.r, self.r * 0.75, 0, (70, 78, 88),
                   label="background")
        cv.ellipse(self.x - self.r * 0.25, self.y - self.r * 0.25,
                   self.r * 0.45, self.r * 0.3, 0, (95, 105, 118))


class Tree:
    def __init__(self, x: float, y: float, scale: float = 1.0):
        self.x, self.y, self.s = x, y, scale

    def draw(self, cv: Canvas, sway: float = 0.0) -> None:
        s = self.s
        cv.rect(self.x - 5 * s, self.y - 30 * s, 10 * s, 30 * s, (40, 70, 92))
        for i, (dx, dy, r, col) in enumerate([
            (0, -52, 26, (40, 96, 44)), (-16, -40, 18, (36, 88, 40)),
            (16, -42, 18, (44, 104, 48))]):
            cv.circle(self.x + dx * s + sway * (i + 1) * 0.4, self.y + dy * s,
                      r * s, col, label="background")


class Pillar:
    """Foreground occluder (Scene 02).  Labelled foreground_occluder."""

    def __init__(self, x: float, y_top: float, y_bottom: float, w: float):
        self.x, self.y0, self.y1, self.w = x, y_top, y_bottom, w

    def draw(self, cv: Canvas) -> None:
        x, w = self.x, self.w
        cv.rect(x - w / 2 - 6, self.y0 - 14, w + 12, 16, (88, 96, 108),
                label="foreground_occluder")
        cv.rect(x - w / 2, self.y0, w, self.y1 - self.y0, (96, 104, 116),
                label="foreground_occluder")
        # fluting lines
        for k in range(1, 4):
            lx = x - w / 2 + k * w / 4
            cv.line(lx, self.y0 + 4, lx, self.y1 - 4, (78, 86, 98), 2)
        # cracks
        cv.polyline([(x - w * 0.2, self.y0 + 40), (x, self.y0 + 70),
                     (x - w * 0.1, self.y0 + 95)], (70, 78, 90), 2)
        cv.rect(x - w / 2 - 8, self.y1 - 2, w + 16, 16, (88, 96, 108),
                label="foreground_occluder")
