import os

import pytest

from rr_vfiqa.testing.synth import (build_scene_cut_set, build_test_set,
                                    interleave, make_defective_mids,
                                    make_source_and_truth, render_scene,
                                    write_video)

# Tests default to the dependency-light Farneback backend; set
# RR_VFIQA_TEST_FLOW=raft to run the accurate (GPU) path.
TEST_FLOW = os.environ.get("RR_VFIQA_TEST_FLOW", "farneback")


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
         "sword_flicker"])
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
