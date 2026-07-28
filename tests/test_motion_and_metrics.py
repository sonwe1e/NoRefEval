import numpy as np
import pytest

from rr_vfiqa.motion.flow_estimator import FarnebackBackend
from rr_vfiqa.motion.flow_composition import composition_error
from rr_vfiqa.motion.occlusion import cycle_occlusion
from rr_vfiqa.motion.global_camera_motion import estimate_camera_motion
from rr_vfiqa.sampling.temporal_nms import temporal_nms
from rr_vfiqa.metrics.parity_frequency import _alternation_energy


@pytest.fixture(scope="module")
def texture():
    rng = np.random.default_rng(3)
    img = rng.integers(0, 255, (128, 192, 3), np.uint8).astype(np.float32)
    import cv2
    return cv2.GaussianBlur(img.astype(np.uint8), (3, 3), 0.8)


def _shift(img, dx, dy):
    import cv2
    m = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, m, (img.shape[1], img.shape[0]),
                          borderMode=cv2.BORDER_REFLECT_101)


def test_flow_translation(texture):
    be = FarnebackBackend()
    f = be.flow(texture, _shift(texture, -6, 0))
    inner = f[20:-20, 20:-20]
    assert abs(np.median(inner[..., 0]) + 6) < 1.5
    assert abs(np.median(inner[..., 1])) < 1.0


def test_occlusion_clean_for_translation(texture):
    be = FarnebackBackend()
    a, b = texture, _shift(texture, -5, 0)
    f_ab, f_ba = be.flow(a, b), be.flow(b, a)
    occ = cycle_occlusion(f_ab, f_ba)
    assert occ.occ_ab[20:-20, 20:-20].mean() < 0.1
    assert occ.conf_ab.mean() > 0.5


def test_camera_motion_recovers_translation(texture):
    b = _shift(texture, -8, 3)
    cam = estimate_camera_motion(texture, b)
    assert cam.model in ("partial_affine", "affine", "translation", "identity")
    flow = cam.warp_flow(texture.shape[0], texture.shape[1])
    inner = flow[20:-20, 20:-20]
    assert abs(np.median(inner[..., 0]) + 8) < 2.0
    assert abs(np.median(inner[..., 1]) - 3) < 2.0


def _split_shift(img, dx_left, dx_right):
    """Left half shifted by dx_left, right half by dx_right — a torn field."""
    w = img.shape[1]
    out = img.copy()
    out[:, : w // 2] = _shift(img, dx_left, 0)[:, : w // 2]
    out[:, w // 2 :] = _shift(img, dx_right, 0)[:, w // 2 :]
    return out


def test_composition_clean_on_realistic_motion(texture):
    # Realistic small motion (~2 source px per interval): a correct midpoint
    # must compose tightly.
    be = FarnebackBackend()
    a = texture
    m = _shift(texture, -1.5, 0)
    b = _shift(texture, -3, 0)
    res = composition_error(be.flow(a, b), be.flow(a, m), be.flow(m, b))
    assert np.median(res.error_map[20:-20, 20:-20]) < 0.35


def test_composition_detects_torn_content(texture):
    # Pure translations compose for ANY M, so composition consistency catches
    # content-level errors: regions filled from the wrong place (USERPLAN §3:
    # tearing, "region copied from prev/next frame"). Here the right half of M
    # is filled with left-half content — flows can no longer chain.
    be = FarnebackBackend()
    w = texture.shape[1]
    a = texture
    b = _split_shift(texture, -8, 8)       # halves move apart
    m = _split_shift(texture, -4, 8)
    m[:, w // 2 :] = a[:, : w // 2 - 0][:, ::-1][:, : w - w // 2]  # foreign content
    res = composition_error(be.flow(a, b), be.flow(a, m), be.flow(m, b))
    right = res.error_map[20:-20, 3 * w // 4 :]
    left = res.error_map[20:-20, 20: w // 4]
    assert np.median(right) > 0.8
    assert np.median(right) > 2 * np.median(left)


def test_temporal_nms_spacing():
    vals = np.zeros(100)
    vals[10] = vals[12] = 5.0       # adjacent tied peaks: only one survives
    vals[60] = 4.0
    times = np.arange(100) / 10.0   # 0.1 s per step
    picked = temporal_nms(vals, times, top_k=5, min_gap_seconds=0.5)
    assert len(picked) == 2
    assert 60 in picked
    assert {10, 12}.intersection(picked)      # either of the tied peaks is fine
    assert not ({10, 12} <= set(picked))      # but not both


def test_alternation_energy_detects_parity():
    k = np.arange(64)
    smooth = np.ones(64) * 10.0 + 0.01 * k
    alternating = 10.0 + 3.0 * ((-1.0) ** k)
    e_smooth = _alternation_energy(smooth, k)
    e_alt = _alternation_energy(alternating, k)
    assert e_alt["e_pi"] > 0.8
    assert e_smooth["e_pi"] < 0.05
