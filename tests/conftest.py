import os

import pytest

from rr_vfiqa.testing.synth import build_test_set

# Tests default to the dependency-light Farneback backend; set
# RR_VFIQA_TEST_FLOW=raft to run the accurate (GPU) path.
TEST_FLOW = os.environ.get("RR_VFIQA_TEST_FLOW", "farneback")


@pytest.fixture(scope="session")
def videos(tmp_path_factory):
    d = tmp_path_factory.mktemp("videos")
    return build_test_set(d, n_frames=160, w=320, h=192, seed=7)


@pytest.fixture(scope="session")
def flow_backend():
    return TEST_FLOW


@pytest.fixture(scope="session")
def cache_dir(tmp_path_factory):
    return str(tmp_path_factory.mktemp("cache"))
