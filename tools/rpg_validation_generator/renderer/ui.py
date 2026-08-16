"""Screen-space HUD / UI widgets (PICPLAN §6.3).

Drawn directly in viewport pixels *after* the camera warp, so the camera
never moves them.  Everything here is labelled ``ui`` or ``text`` so defect
operators can isolate UI from world content.
"""
from __future__ import annotations

import cv2
import numpy as np

from .primitives import Canvas

PANEL_BG = (46, 38, 30)        # dark blue-grey (BGR)
PANEL_EDGE = (120, 100, 70)
GOLD = (60, 200, 235)


class HealthBar:
    def __init__(self, x, y, w=240, h=18, name="PLAYER", color=(70, 80, 220)):
        self.x, self.y, self.w, self.h = x, y, w, h
        self.name = name
        self.color = tuple(int(c) for c in color)

    def draw(self, cv: Canvas, frac: float) -> None:
        x, y, w, h = self.x, self.y, self.w, self.h
        cv.rect(x - 2, y - 2, w + 4, h + 4, PANEL_EDGE, label="ui")
        cv.rect(x, y, w, h, (30, 30, 34), label="ui")
        fw = int(w * max(0.0, min(1.0, frac)))
        if fw > 0:
            cv.rect(x, y, fw, h, self.color, label="ui")
            cv.rect(x, y, fw, h // 3, tuple(min(c + 40, 255) for c in self.color),
                    label="ui")
        cv.text(x + 6, y + h - 5, self.name, (255, 255, 255), scale=0.45,
                thickness=1, label="text")
        cv.text(x + w - 8, y + h - 5, f"{int(frac * 100)}%", (255, 255, 255),
                scale=0.45, thickness=1, label="text", anchor="tr")


class SkillButton:
    def __init__(self, cx, cy, r=26, glyph="A", color=(160, 90, 60)):
        self.cx, self.cy, self.r = cx, cy, r
        self.glyph = glyph
        self.color = tuple(int(c) for c in color)

    def draw(self, cv: Canvas, cooldown_frac: float = 0.0) -> None:
        cv.circle(self.cx, self.cy, self.r + 3, PANEL_EDGE, label="ui")
        cv.circle(self.cx, self.cy, self.r, self.color, label="ui")
        cv.circle(self.cx, self.cy, self.r * 0.62,
                  tuple(min(c + 50, 255) for c in self.color), label="ui")
        cv.text(self.cx, self.cy, self.glyph, (255, 255, 255), scale=0.7,
                thickness=2, label="text", anchor="center")
        if cooldown_frac > 0:
            # darkened pie slice sweeping as cooldown recovers
            ang = 360.0 * cooldown_frac
            pts = [(self.cx, self.cy)]
            a0 = -90.0
            for a in np.linspace(a0, a0 - ang, 24):
                rad = np.radians(a)
                pts.append((self.cx + self.r * np.cos(rad),
                            self.cy + self.r * np.sin(rad)))
            m = np.zeros((cv.height, cv.width), dtype=np.uint8)
            cv2.fillPoly(m, [np.asarray(pts, dtype=np.int32)], 255)
            sel = (m > 0) & (cv.labels == 6)
            cv.rgb[sel] = (cv.rgb[sel] * 0.45).astype(np.uint8)


class Joystick:
    def __init__(self, cx, cy, r=52):
        self.cx, self.cy, self.r = cx, cy, r

    def draw(self, cv: Canvas, knob_dx: float = 0.0, knob_dy: float = 0.0) -> None:
        cv.circle(self.cx, self.cy, self.r, (90, 80, 70), thickness=3, label="ui")
        base = np.zeros_like(cv.rgb)
        cv2.circle(base, (int(self.cx), int(self.cy)), int(self.r), (90, 80, 70), -1)
        cv.blend_masked((cv.rgb * 0.55 + base * 0.45).astype(np.uint8),
                        base[:, :, 0], 1.0)
        cv.labels[base[:, :, 0] > 0] = 6
        kx = self.cx + knob_dx * self.r * 0.55
        ky = self.cy + knob_dy * self.r * 0.55
        cv.circle(kx, ky, self.r * 0.42, (150, 140, 120), label="ui")
        cv.circle(kx, ky, self.r * 0.2, (200, 190, 170), label="ui")


class CooldownNumber:
    """Monotonic countdown text on a skill button."""

    def __init__(self, cx, cy):
        self.cx, self.cy = cx, cy

    def draw(self, cv: Canvas, value: float) -> None:
        s = f"{max(0.0, value):.1f}"
        cv.text(self.cx, self.cy - 34, s, (80, 220, 255), scale=0.6,
                thickness=2, label="text", anchor="center")


class FloatingName:
    """Name tag + small bar above a character (world text layer)."""

    def __init__(self, name="MERCHANT", color=(120, 230, 180)):
        self.name = name
        self.color = tuple(int(c) for c in color)

    def draw(self, cv: Canvas, x: float, y: float) -> None:
        cv.text(x, y, self.name, self.color, scale=0.5, thickness=1,
                label="text", anchor="center")
        w = 46
        cv.rect(x - w / 2, y + 4, w, 4, (30, 30, 34), label="ui")
        cv.rect(x - w / 2, y + 4, w * 2 // 3, 4, self.color, label="ui")


class DialogPanel:
    def __init__(self, x, y, w, h):
        self.x, self.y, self.w, self.h = x, y, w, h

    def draw(self, cv: Canvas, speaker: str, lines: list[str],
             progress: float = 1.0) -> None:
        x, y, w, h = self.x, self.y, self.w, self.h
        cv.rect(x, y, w, h, PANEL_BG, label="ui")
        cv.rect(x, y, w, h, PANEL_EDGE, thickness=2, label="ui")
        cv.rect(x + 10, y + 10, 120, 22, (80, 120, 190), label="ui")
        cv.text(x + 16, y + 27, speaker, (255, 255, 255), scale=0.55,
                thickness=1, label="text")
        nchars = int(progress * sum(len(s) for s in lines))
        acc, shown = 0, []
        for s in lines:
            take = min(len(s), max(0, nchars - acc))
            shown.append(s[:take])
            acc += len(s)
        for i, s in enumerate(shown):
            cv.text(x + 16, y + 56 + i * 22, s, (235, 235, 235), scale=0.5,
                    thickness=1, label="text")


class ShopPanel:
    def __init__(self, x, y, w=250, h=300):
        self.x, self.y, self.w, self.h = x, y, w, h
        self.items = [
            ("POTION", 120, (80, 120, 220)),
            ("ETHER", 240, (180, 120, 80)),
            ("ANTIDOTE", 90, (90, 200, 120)),
            ("BOMB", 310, (60, 90, 200)),
        ]

    def draw(self, cv: Canvas, open_frac: float = 1.0) -> None:
        x, y, w, h = self.x, self.y, self.w, int(self.h * max(0.0, min(1.0, open_frac)))
        if h <= 6:
            return
        cv.rect(x, y, w, h, PANEL_BG, label="ui")
        cv.rect(x, y, w, h, PANEL_EDGE, thickness=2, label="ui")
        cv.rect(x, y, w, 26, (70, 90, 150), label="ui")
        cv.text(x + 10, y + 19, "SHOP", (255, 255, 255), scale=0.6, thickness=1,
                label="text")
        row_h = 52
        for i, (name, price, col) in enumerate(self.items):
            ry = y + 34 + i * row_h
            if ry + row_h > y + h - 4:
                break
            # procedural icon: gem shape
            ix, iy = x + 24, ry + 20
            cv.polygon([(ix, iy - 12), (ix + 11, iy - 4), (ix + 7, iy + 10),
                        (ix - 7, iy + 10), (ix - 11, iy - 4)], col, label="ui")
            cv.polyline([(ix - 11, iy - 4), (ix + 11, iy - 4),
                         (ix, iy - 12)], tuple(min(c + 60, 255) for c in col), 1)
            cv.text(x + 46, ry + 14, name, (235, 235, 235), scale=0.5,
                    thickness=1, label="text")
            cv.text(x + 46, ry + 34, f"{price} G", GOLD, scale=0.5, thickness=1,
                    label="text")
