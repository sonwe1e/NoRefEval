import os

import pytest

# rr_vfiqa's synthetic-video fixtures are only available when the package's
# testing helpers import cleanly (they pull in scipy/torch-heavy paths).  If
# the import fails we silently skip them so the rpg_validation_generator
# tests below still run.
try:
    from rr_vfiqa.testing.synth import (build_scene_cut_set, build_test_set,
                                        interleave, make_defective_mids,
                                        make_source_and_truth, render_scene,
                                        write_video)
    _HAVE_RR_SYNTH = True
except Exception:  # pragma: no cover
    _HAVE_RR_SYNTH = False

# Tests default to the dependency-light Farneback backend; set
# RR_VFIQA_TEST_FLOW=raft to run the accurate (GPU) path.
TEST_FLOW = os.environ.get("RR_VFIQA_TEST_FLOW", "farneback")

if _HAVE_RR_SYNTH:
    @pytest.fixture(scope="session")
    def videos(tmp_path_factory):
        d = tmp_path_factory.mktemp("videos")
        return build_test_set(d, n_frames=160, w=320, h=192, seed=7)

    @pytest.fixture(scope="session")
    def defect_videos(tmp_path_factory):
        """Two bad candidates grouping structural vs flicker defects (§6)."""
        d = tmp_path_factory.mktemp("defects")
        frames, meta = render_scene(n_frames=160, w=320, h=192, seed=7,
                                    return_meta=True)
        source, truth = make_source_and_truth(frames)
        struct_mids, _ = make_defective_mids(
            source, truth, meta,
            ["rotation_tear", "head_erase", "pole_wrong_motion", "disocc_fill"])
        flick_mids, _ = make_defective_mids(
            source, truth, meta,
            ["blur", "ghost", "freeze", "ui_drift", "text_merge", "shop_jump",
             "sword_flicker", "card_freeze"])
        return {
            "source": write_video(d / "source_60.mp4", source, 60),
            "good": write_video(d / "candidate_good_120.mp4", frames, 120),
            "structural": write_video(d / "candidate_struct_120.mp4",
                                      interleave(source, struct_mids), 120),
            "flicker": write_video(d / "candidate_flicker_120.mp4",
                                   interleave(source, flick_mids), 120),
        }

    @pytest.fixture(scope="session")
    def cut_videos(tmp_path_factory):
        d = tmp_path_factory.mktemp("cuts")
        return build_scene_cut_set(d, n_frames=240, w=320, h=192)


@pytest.fixture(scope="session")
def flow_backend():
    return TEST_FLOW


@pytest.fixture(scope="session")
def cache_dir(tmp_path_factory):
    return str(tmp_path_factory.mktemp("cache"))


# ---------------------------------------------------------------- RPG gen ---

from tools.rpg_validation_generator.config import GeneratorConfig  # noqa: E402
from tools.rpg_validation_generator.scenes import make_scene  # noqa: E402
from tools.rpg_validation_generator.timeline import render_master  # noqa: E402

# Fast config: master at 120 FPS so the 60/120 candidate ratios in the plan
# are exact integer downsamples, but small geometry and a 2 s window keep it
# quick.  Determinism tests use an even smaller one.
_TEST_CFG = GeneratorConfig(width=320, height=180, duration_seconds=2.0,
                            master_fps=120, seed=20260729,
                            world_width=400, world_height=250)
_TINY_CFG = GeneratorConfig(width=160, height=96, duration_seconds=1.0,
                            master_fps=60, seed=20260729,
                            world_width=220, world_height=140)


@pytest.fixture(scope="session")
def rpg_cfg():
    return _TEST_CFG


@pytest.fixture(scope="session")
def tiny_cfg():
    return _TINY_CFG


@pytest.fixture(scope="session")
def rpg_masters(rpg_cfg, tmp_path_factory):
    d = tmp_path_factory.mktemp("rpg_masters")
    return {i: render_master(make_scene(i, rpg_cfg), rpg_cfg, d / f"s{i}")
            for i in range(5)}
