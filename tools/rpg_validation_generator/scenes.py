"""The five base RPG scenes, shared by all three modes (PICPLAN §7).

All motion is analytic in t (PICPLAN §3, §6.4).  Scenes draw only clean
content — defects are never baked into scene code (PICPLAN §5).
"""
from __future__ import annotations

import numpy as np

from .motion import (ease_in_out_cubic, lerp, sinusoidal, smoothstep, window)
from .renderer.character import Character, Enemy, NPC, Sword
from .renderer.effects import Flash, ParticleEmitter, Projectile, ShockRing
from .renderer.renderer import Scene
from .renderer.tilemap import Pillar, Rock, TileMap, Tree
from .renderer.ui import (CooldownNumber, DialogPanel, FloatingName, HealthBar,
                          Joystick, ShopPanel, SkillButton)

# The feet line is a fraction of the world height so it stays on-canvas for
# both the production world (1000 px) and the smaller draft world.  Scenes
# read it via ``self.ground_y`` (set in Scene.__init__).


def _hud(cfg, n_buttons=3):
    """Standard HUD set used by most scenes."""
    hp = HealthBar(20, 18, w=max(170, cfg.width // 5))
    buttons = []
    for i in range(n_buttons):
        bx = cfg.width - 60 - i * 66
        buttons.append(SkillButton(bx, cfg.height - 70, r=26,
                                   glyph="ASD"[i],
                                   color=[(160, 90, 60), (90, 140, 70),
                                          (70, 100, 170)][i]))
    joy = Joystick(100, cfg.height - 110, r=max(38, cfg.height // 14))
    return hp, buttons, joy


class Scene01GrassChase(Scene):
    """草原追击: constant-speed run with slight acceleration, camera pan."""

    scene_id = "scene_01_grass_chase"
    scene_title = "grassland_chase"

    def setup(self):
        c = self.config
        self.map = TileMap(c.world_width, c.world_height, self.rng,
                           road_y=self.ground_y - 40, palette="grass")
        props_rng = np.random.default_rng(self.seed + 11)
        self.trees = [Tree(float(x), self.ground_y + float(dy), float(s))
                      for x, dy, s in props_rng.uniform((40, 60, 0.8),
                                                        (1560, 260, 1.4),
                                                        size=(9, 3))]
        self.rocks = [Rock(float(x), self.ground_y + float(dy), float(r))
                      for x, dy, r in props_rng.uniform((60, -20, 8),
                                                        (1540, 220, 20),
                                                        size=(7, 3))]
        self.player = Character()
        self.enemy = Enemy()
        self.hp, self.buttons, self.joy = _hud(c)
        self.x0, self.x1 = c.world_width * 0.12, c.world_width * 0.9

    def player_x(self, t):
        u = t / self.config.duration_seconds
        # mostly linear, slight acceleration via smoothstep blend
        return lerp(self.x0, self.x1, 0.82 * u + 0.18 * smoothstep(u))

    def camera_at(self, t):
        from .renderer.camera import Camera
        px = self.player_x(t)
        return Camera(px + 60, self.config.world_height * 0.52, zoom=1.0)

    def draw_background(self, cv, t):
        self.map.draw(cv)
        for tr in self.trees:
            tr.draw(cv, sway=sinusoidal(t, 0.3, 2.0, phase=tr.x))
        for rk in self.rocks:
            rk.draw(cv)

    def draw_actors(self, cv, t):
        px = self.player_x(t)
        ex = px - 180 + 14 * sinusoidal(t, 0.9, 1.0)
        self.enemy.draw(cv, ex, self.ground_y, t + 0.35, facing=1.0, speed=1.05)
        self.player.draw(cv, px, self.ground_y, t, facing=1.0, speed=1.0)

    def draw_ui(self, cv, t):
        self.hp.draw(cv, 0.86 + 0.02 * sinusoidal(t, 0.2, 1.0))
        for b in self.buttons:
            b.draw(cv, 0.0)
        self.joy.draw(cv, knob_dx=0.85, knob_dy=0.05 * sinusoidal(t, 1.3, 1.0))


class Scene02RuinsOcclusion(Scene):
    """遗迹遮挡: camera rotation+zoom, pillars occlude and reveal the hero."""

    scene_id = "scene_02_ruins_occlusion"
    scene_title = "ruins_occlusion"

    def setup(self):
        c = self.config
        self.map = TileMap(c.world_width, c.world_height, self.rng,
                           road_y=self.ground_y - 30, palette="ruins")
        self.pillars = [
            Pillar(c.world_width * 0.42, c.world_height * 0.16,
                   self.ground_y + 120, 64),
            Pillar(c.world_width * 0.66, c.world_height * 0.12,
                   self.ground_y + 140, 72),
        ]
        rear_rng = np.random.default_rng(self.seed + 23)
        self.rear_pillars = [(float(x), float(w)) for x, w in
                             rear_rng.uniform((120, 26), (1480, 44),
                                              size=(6, 2))]
        self.bands = [float(y) for y in
                      np.linspace(c.world_height * 0.3, self.ground_y - 60, 4)]
        self.player = Character(body=(180, 140, 90))
        self.hp, _, _ = _hud(c, n_buttons=1)
        self.x0, self.x1 = c.world_width * 0.15, c.world_width * 0.85

    def player_x(self, t):
        u = t / self.config.duration_seconds
        return lerp(self.x0, self.x1, smoothstep(u))

    def camera_at(self, t):
        from .renderer.camera import Camera
        u = t / self.config.duration_seconds
        px = self.player_x(t)
        rot = 3.0 * sinusoidal(u, 1.0, 1.0)                 # ±3 deg over scene
        zoom = 1.07 + 0.05 * sinusoidal(u, 1.0, 1.0, phase=1.5708)
        return Camera(px, self.config.world_height * 0.5, zoom=zoom,
                      rotation_deg=rot)

    def draw_background(self, cv, t):
        self.map.draw(cv)
        # depth bands (parallax decoration layers)
        for i, y in enumerate(self.bands):
            shade = 58 + 8 * i
            cv.rect(0, int(y), cv.width, 10, (shade, shade + 4, shade + 10))
        # rear (parallax) pillars — lighter, behind actors
        for x, w in self.rear_pillars:
            cv.rect(x - w / 2, cv.height * 0.2, w, self.ground_y - cv.height * 0.2 + 40,
                    (110, 116, 126), label="background")

    def draw_actors(self, cv, t):
        self.player.draw(cv, self.player_x(t), self.ground_y, t, facing=1.0,
                         speed=0.7)

    def draw_occluders(self, cv, t):
        for p in self.pillars:
            p.draw(cv)

    def draw_ui(self, cv, t):
        self.hp.draw(cv, 0.92)


class Scene03BossSword(Scene):
    """Boss 挥剑: fast thin-weapon rotation, boss retreat, hit flash."""

    scene_id = "scene_03_boss_sword"
    scene_title = "boss_sword_strike"

    APPROACH_END = 1.2
    SWING_START = 1.2
    SWING_END = 1.9
    HIT_TIME = 1.72

    def setup(self):
        c = self.config
        self.map = TileMap(c.world_width, c.world_height, self.rng,
                           road_y=self.ground_y + 10, palette="battle")
        self.player = Character(scale=1.05)
        self.boss = Enemy(scale=1.5)
        self.sword = Sword(length=96.0, width=3.0)
        self.flash = Flash((c.world_width * 0.58, self.ground_y - 60),
                           self.HIT_TIME, duration=0.14, radius=110)
        self.hp, self.buttons, _ = _hud(c)
        self.px0, self.px1 = c.world_width * 0.2, c.world_width * 0.44
        self.boss_x0 = c.world_width * 0.6

    def player_x(self, t):
        u = window(t, 0.0, self.APPROACH_END)
        return lerp(self.px0, self.px1, ease_in_out_cubic(u))

    def boss_x(self, t):
        u = window(t, self.HIT_TIME, self.HIT_TIME + 0.9)
        return self.boss_x0 + 150 * smoothstep(u)

    def boss_facing(self, t):
        # turns away after the hit
        u = window(t, self.HIT_TIME, self.HIT_TIME + 0.6)
        return -1.0 if u < 0.5 else 1.0

    def camera_at(self, t):
        from .renderer.camera import Camera
        zoom = 1.0 + 0.06 * np.exp(-6.0 * max(0.0, t - self.HIT_TIME)) * (
            1.0 if t > self.HIT_TIME else 0.0)
        return Camera(self.config.world_width * 0.42,
                      self.config.world_height * 0.5, zoom=max(1.0, zoom))

    def sword_angle(self, t):
        if t < self.SWING_START:
            return -35.0 + 6.0 * sinusoidal(t, 1.1, 1.0)   # idle guard
        u = window(t, self.SWING_START, self.SWING_END)
        if u < 1.0:
            # fast slash: wind-up high-left -> horizontal at the boss
            # torso near HIT_TIME -> follow-through down-right
            return lerp(-160.0, 30.0, ease_in_out_cubic(u))
        return lerp(30.0, -35.0, smoothstep(window(t, self.SWING_END,
                                                   self.SWING_END + 0.7)))

    def draw_background(self, cv, t):
        self.map.draw(cv)
        # arena pillars at the back
        cv.rect(cv.width * 0.06, cv.height * 0.1, 34, self.ground_y - cv.height * 0.1,
                (72, 78, 92), label="background")
        cv.rect(cv.width * 0.9, cv.height * 0.1, 34, self.ground_y - cv.height * 0.1,
                (72, 78, 92), label="background")

    def draw_actors(self, cv, t):
        bx = self.boss_x(t)
        self.boss.draw(cv, bx, self.ground_y, t * 0.8, facing=self.boss_facing(t),
                       speed=0.4)
        px = self.player_x(t)
        ang = self.sword_angle(t)
        arm = (ang * 0.7) if self.SWING_START <= t <= self.SWING_END + 0.7 else None
        hand = self.player.draw(cv, px, self.ground_y, t, facing=1.0, speed=0.6,
                                arm_angle_deg=arm)
        if hand is None:
            hx, hy = px + 16, self.ground_y - 34
        else:
            hx, hy = hand
        self.sword.draw(cv, hx, hy, ang)

    def draw_ui(self, cv, t):
        self.hp.draw(cv, 0.78)
        cd = max(0.0, 1.0 - window(t, self.SWING_END, 3.4))
        for i, b in enumerate(self.buttons):
            b.draw(cv, cd if i == 0 else 0.0)

    def post_effects(self, cv, t):
        self.flash.draw(cv, t)


class Scene04TownShop(Scene):
    """城镇 NPC 与商店: text, floating name, dialog + shop popups, cooldowns."""

    scene_id = "scene_04_town_shop"
    scene_title = "town_npc_shop"

    WALK_END = 2.2
    DIALOG_START = 1.2
    SHOP_START = 1.6

    def setup(self):
        c = self.config
        self.map = TileMap(c.world_width, c.world_height, self.rng,
                           road_y=self.ground_y - 20, palette="town")
        props_rng = np.random.default_rng(self.seed + 41)
        self.houses = [(float(x), float(w), float(h),
                        tuple(int(v) for v in col))
                       for x, w, h, col in zip(
                           props_rng.uniform(80, 1450, 5),
                           props_rng.uniform(120, 220, 5),
                           props_rng.uniform(140, 240, 5),
                           props_rng.integers((90, 90, 100), (150, 140, 150),
                                              (5, 3)))]
        self.player = Character(body=(200, 150, 90))
        self.npc = NPC(scale=1.0)
        self.npc_x = c.world_width * 0.66
        self.name_tag = FloatingName("MERCHANT")
        self.hp, self.buttons, self.joy = _hud(c)
        self.cooldown = CooldownNumber(self.buttons[0].cx, self.buttons[0].cy)
        pw = min(420, int(c.width * 0.66))
        self.dialog = DialogPanel((c.width - pw) // 2, c.height - 150, pw, 120)
        self.shop = ShopPanel(c.width - 270, 90, w=250, h=min(330, c.height - 160))
        self.px0, self.px1 = c.world_width * 0.18, self.npc_x - 90

    def player_x(self, t):
        u = window(t, 0.2, self.WALK_END)
        return lerp(self.px0, self.px1, smoothstep(u))

    def walk_speed(self, t):
        return 0.8 if t < self.WALK_END else 0.02

    def camera_at(self, t):
        from .renderer.camera import Camera
        px = self.player_x(t)
        cx = lerp(self.config.world_width * 0.35, self.config.world_width * 0.52,
                  smoothstep(window(t, 0.0, self.WALK_END)))
        return Camera(cx + 0.2 * (px - cx), self.config.world_height * 0.5)

    def draw_background(self, cv, t):
        self.map.draw(cv)
        for x, w, h, col in self.houses:
            y0 = self.ground_y - h
            cv.rect(x, y0, w, h, col, label="background")
            roof = [(x - 12, y0), (x + w / 2, y0 - 46), (x + w + 12, y0)]
            cv.polygon(roof, (60, 70, 120), label="background")
            cv.rect(x + w * 0.2, y0 + h * 0.45, w * 0.22, h * 0.4, (50, 60, 70))
            cv.rect(x + w * 0.58, y0 + h * 0.3, w * 0.24, w * 0.24, (160, 190, 210))

    def draw_actors(self, cv, t):
        bob = sinusoidal(t, 0.5, 2.0) if t >= self.WALK_END else 0.0
        self.npc.draw(cv, self.npc_x, self.ground_y + bob * 0.0, t * 0.15,
                      facing=-1.0, speed=0.05)
        self.player.draw(cv, self.player_x(t), self.ground_y, t, facing=1.0,
                         speed=self.walk_speed(t))

    def draw_world_text(self, cv, t):
        bob = sinusoidal(t, 1.2, 3.0)
        self.name_tag.draw(cv, self.npc_x, self.ground_y - 105 + bob)

    def draw_ui(self, cv, t):
        self.hp.draw(cv, 1.0)
        cd_value = max(0.0, 5.0 - 2.3 * t)  # monotonic countdown
        for i, b in enumerate(self.buttons):
            b.draw(cv, min(1.0, cd_value / 5.0) if i == 0 else 0.0)
        self.cooldown.draw(cv, cd_value)
        self.joy.draw(cv, knob_dx=0.6 if t < self.WALK_END else 0.0, knob_dy=0.0)
        if t >= self.DIALOG_START:
            self.dialog.draw(cv, "MERCHANT",
                             ["Welcome, traveler.",
                              "Potions are cheap today!",
                              "Stay safe out there."],
                             progress=window(t, self.DIALOG_START,
                                             self.DIALOG_START + 1.6))
        if t >= self.SHOP_START:
            self.shop.draw(cv, open_frac=smoothstep(
                window(t, self.SHOP_START, self.SHOP_START + 0.5)))


class Scene05MagicCombat(Scene):
    """魔法战斗: curved projectile, particles, camera shake, cooldowns."""

    scene_id = "scene_05_magic_combat"
    scene_title = "magic_combat"

    CAST_START = 0.8
    CAST_END = 1.9
    HIT_TIME = 1.9

    def setup(self):
        c = self.config
        self.map = TileMap(c.world_width, c.world_height, self.rng,
                           road_y=self.ground_y + 20, palette="battle")
        self.player = Character(body=(220, 160, 80))
        self.enemy = Enemy(scale=1.25)
        ex = c.world_width * 0.72
        ey = self.ground_y - 70
        self.enemy_pos = (ex, ey)
        self.projectile = Projectile(
            (c.world_width * 0.28, self.ground_y - 60),
            (c.world_width * 0.45, self.ground_y - 260),
            (c.world_width * 0.62, self.ground_y - 220),
            (ex, ey), radius=10)
        part_rng = np.random.default_rng(self.seed + 57)
        self.particles = ParticleEmitter(part_rng, (ex, ey), n=52,
                                         burst_time=self.HIT_TIME)
        self.ring = ShockRing((ex, ey), self.HIT_TIME, duration=0.4, max_r=90)
        self.hp, self.buttons, _ = _hud(c)
        self.enemy_hp = HealthBar(c.width // 2 - 130, 18, w=260, h=16,
                                  name="ENEMY", color=(60, 60, 200))
        self.cooldown = CooldownNumber(self.buttons[0].cx, self.buttons[0].cy)

    def camera_at(self, t):
        from .renderer.camera import Camera
        cx, cy = (self.config.world_width * 0.5, self.config.world_height * 0.5)
        dt = t - self.HIT_TIME
        if dt > 0:
            amp = 12.0 * np.exp(-5.5 * dt)
            cx += amp * np.sin(2 * np.pi * 11 * dt)
            cy += amp * 0.7 * np.sin(2 * np.pi * 13 * dt + 1.0)
        return Camera(cx, cy)

    def enemy_hp_frac(self, t):
        u = window(t, self.HIT_TIME, self.HIT_TIME + 0.45)
        return lerp(1.0, 0.32, smoothstep(u))

    def cast_u(self, t):
        return window(t, self.CAST_START, self.CAST_END)

    def draw_background(self, cv, t):
        self.map.draw(cv)
        # arcane circle under the enemy
        ex, ey = self.enemy_pos
        cv.ellipse(ex, self.ground_y + 6, 70, 18, 0, (140, 90, 60), thickness=2)

    def draw_actors(self, cv, t):
        ex, ey = self.enemy_pos
        hurt = window(t, self.HIT_TIME, self.HIT_TIME + 0.3)
        self.enemy.draw(cv, ex + 8 * hurt, self.ground_y, t * 0.6, facing=-1.0,
                        speed=0.2)
        casting = self.CAST_START <= t <= self.CAST_END
        arm = -60.0 if casting else None
        self.player.draw(cv, self.config.world_width * 0.26, self.ground_y, t,
                         facing=1.0, speed=0.05, arm_angle_deg=arm)
        u = self.cast_u(t)
        if 0.0 < u < 1.0:
            self.projectile.draw(cv, u)
        self.particles.draw(cv, t)
        self.ring.draw(cv, t)

    def draw_ui(self, cv, t):
        self.hp.draw(cv, 0.95)
        self.enemy_hp.draw(cv, self.enemy_hp_frac(t))
        cd = max(0.0, 1.0 - window(t, self.CAST_END, 3.4))
        for i, b in enumerate(self.buttons):
            b.draw(cv, cd if i == 0 else 0.0)
        if cd > 0:
            self.cooldown.draw(cv, cd * 4.0)


SCENES: list[type[Scene]] = [
    Scene01GrassChase,
    Scene02RuinsOcclusion,
    Scene03BossSword,
    Scene04TownShop,
    Scene05MagicCombat,
]


def make_scene(index: int, config) -> Scene:
    cls = SCENES[index]
    return cls(config, config.scene_seed(index))
