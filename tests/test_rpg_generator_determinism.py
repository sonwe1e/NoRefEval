"""Determinism: same seed -> identical raw RGB (PICPLAN §3, §20.5)."""
from __future__ import annotations

import numpy as np

from tools.rpg_validation_generator.hashing import json_sha256, raw_rgb_sha256
from tools.rpg_validation_generator.scenes import make_scene


def _render_all_frames(scene):
    n = scene.config.n_master_frames
    return np.stack([scene.render_frame(i)[0] for i in range(n)])


def _with_seed(cfg, seed):
    return cfg.__class__(width=cfg.width, height=cfg.height,
                         duration_seconds=cfg.duration_seconds,
                         master_fps=cfg.master_fps, seed=seed,
                         world_width=cfg.world_width, world_height=cfg.world_height)


def test_same_seed_identical_rgb(tiny_cfg):
    # tiny config keeps this cheap; determinism holds at any resolution.
    for i in range(5):
        a = _render_all_frames(make_scene(i, tiny_cfg))
        b = _render_all_frames(make_scene(i, tiny_cfg))
        assert np.array_equal(a, b), i
        assert raw_rgb_sha256(a) == raw_rgb_sha256(b)


def test_different_seed_differs(tiny_cfg):
    a = _render_all_frames(make_scene(0, tiny_cfg))
    b = _render_all_frames(make_scene(0, _with_seed(tiny_cfg, tiny_cfg.seed + 1)))
    assert raw_rgb_sha256(a) != raw_rgb_sha256(b)


def test_scene_spec_hash_stable(rpg_cfg):
    s1 = make_scene(2, rpg_cfg).spec()
    s2 = make_scene(2, rpg_cfg).spec()
    assert json_sha256(s1) == json_sha256(s2)
    assert s1["scene_id"] == "scene_03_boss_sword"


def test_no_external_assets_in_scenes():
    """Scene code must not reference any image/font loader (PICPLAN §4, §22)."""
    import inspect
    from tools.rpg_validation_generator import scenes
    src = inspect.getsource(scenes)
    for banned in ("imread", "fromfile", "urllib", "requests", ".ttf", "freetype"):
        assert banned not in src, banned
