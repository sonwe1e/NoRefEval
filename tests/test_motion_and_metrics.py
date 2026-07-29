import cv2
import numpy as np
import pytest

from rr_vfiqa.motion.flow_estimator import FarnebackBackend
from rr_vfiqa.motion.flow_composition import composition_error
from rr_vfiqa.motion.occlusion import cycle_occlusion
from rr_vfiqa.motion.global_camera_motion import estimate_camera_motion
from rr_vfiqa.sampling.temporal_nms import temporal_nms
from rr_vfiqa.metrics.parity_frequency import _alternation_energy
from rr_vfiqa.schema import backward_warp, forward_splat


def _bar_image(h=32, w=48, col=10):
    img = np.zeros((h, w), np.float32)
    img[:, col: col + 2] = 200.0
    return img


def test_backward_warp_moves_content_opposite_flow():
    # target_to_source flow (+5, 0): out(x) = img(x + 5) — the bar at col 10
    # is SAMPLED from the right, so it APPEARS at col 5.
    img = _bar_image(col=10)
    flow = np.zeros((32, 48, 2), np.float32)
    flow[..., 0] = 5.0
    out = backward_warp(img, flow)
    assert out[16, 5] > 150
    assert out[16, 10] < 10


def test_forward_splat_moves_content_along_flow():
    # source_to_target flow (+5, 0): source pixel at col 10 lands at col 15.
    img = _bar_image(col=10)
    flow = np.zeros((32, 48, 2), np.float32)
    flow[..., 0] = 5.0
    out, cov = forward_splat(img, flow)
    assert cov[16, 15] > 0.9 and cov[16, 16] > 0.9
    assert out[16, 15] > 150
    assert out[16, 10] < 10           # bar content moved on
    assert out[16, 0] == 0.0          # nothing splatted out of thin air


def test_forward_backward_roundtrip_on_translation():
    # b = a shifted right by 6 px, so F_ab = (+6, 0) everywhere. Resampling b
    # onto a's grid (target a, source b) takes the target→source flow F_ab.
    a = _bar_image(col=20)
    b = np.zeros_like(a)
    b[:, 6:] = a[:, :-6]
    f_ab = np.zeros(a.shape + (2,), np.float32)
    f_ab[..., 0] = 6.0
    rec = backward_warp(b, f_ab)
    inner = (slice(4, 28), slice(8, 40))
    assert np.abs(rec[inner] - a[inner]).mean() < 1.0


def test_forward_splat_half_flow_reaches_midpoint():
    # Content halfway to target: splat with half the flow.
    img = _bar_image(col=10)
    flow = np.zeros((32, 48, 2), np.float32)
    flow[..., 0] = 10.0
    out, cov = forward_splat(img, 0.5 * flow)
    assert cov[16, 15] > 0.9
    assert out[16, 20] < 10           # full-flow target got no bar content


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


def test_edge_fscore_cross_distance():
    """F-score must compare each edge set against the OTHER's distance map.

    On structured content: identical ≈ 1, displaced edges score low, and a
    candidate with no edges at all yields NaN (guarded, not 1 — the old
    self-distance version returned 1 regardless).
    """
    import cv2
    from rr_vfiqa.regions.ui_detector import _edge_fscore

    ref = np.full((96, 128), 40, np.uint8)
    cv2.rectangle(ref, (20, 20), (60, 70), 220, 2)
    cv2.rectangle(ref, (80, 30), (110, 60), 200, 2)
    mask = np.ones_like(ref)
    assert _edge_fscore(ref, ref, mask) > 0.95
    shifted = np.roll(ref, 6, axis=1)
    assert _edge_fscore(shifted, ref, mask) < 0.7   # buggy version was exactly 1.0
    flat = np.full_like(ref, 128)
    assert np.isnan(_edge_fscore(flat, ref, mask))


def test_audit_tracker_degrades_to_klt():
    """Without the optional cotracker package, the audit tier must fall back
    to KLT with an explanatory note — never crash (§P2 optional dependency)."""
    from rr_vfiqa.models.tracker_backend import TrackerBackend, get_audit_tracker
    backend, note = get_audit_tracker(device="cpu")
    assert isinstance(backend, TrackerBackend)
    assert note in ("cotracker",) or "fell back" in note

    # The fallback (or real) tracker must actually track a translating dot.
    frames = []
    for t in range(5):
        f = np.zeros((64, 96), np.uint8)
        cv2.circle(f, (30 + 4 * t, 32), 4, 255, -1)
        frames.append(f)
    tracks, vis = backend.track(frames, np.float32([[30, 32]]))
    assert tracks.shape == (5, 1, 2)
    assert vis[-1, 0]
    assert abs(tracks[-1, 0, 0] - 46) < 4


def test_cotracker_video_layout_is_btchw():
    from rr_vfiqa.models.tracker_backend import _cotracker_video_array

    frames = [np.full((12, 20), t, np.uint8) for t in range(5)]
    video = _cotracker_video_array(frames)
    assert video.shape == (1, 5, 3, 12, 20)
    assert np.all(video[0, 3] == 3)


def test_audit_tracker_falls_back_on_constructor_error(monkeypatch):
    import rr_vfiqa.models.tracker_backend as tracker_backend

    class BrokenCoTracker:
        def __init__(self, **_):
            raise RuntimeError("checkpoint unavailable")

    monkeypatch.setattr(tracker_backend, "CoTrackerBackend", BrokenCoTracker)
    backend, note = tracker_backend.get_audit_tracker(device="cpu")
    assert isinstance(backend, tracker_backend.KLTTracker)
    assert "RuntimeError" in note and "fell back" in note


def test_alpha_blend_fit():
    """Optimal-α fit: a real 50/50 mix is detected; hard switches and novel
    content are not (§3.5)."""
    from rr_vfiqa.imutils import alpha_blend_fit

    rng = np.random.default_rng(9)
    xi = rng.integers(0, 255, (24, 32, 3), np.uint8)
    xj = rng.integers(0, 255, (24, 32, 3), np.uint8)

    mix = np.clip(0.5 * xi.astype(np.float32) + 0.5 * xj.astype(np.float32),
                  0, 255).astype(np.uint8)
    alpha, resid = alpha_blend_fit(xi, mix, xj)
    assert np.median(resid) < 2.0
    assert np.abs(np.median(alpha) - 0.5) < 0.05

    alpha, resid = alpha_blend_fit(xi, xj, xj)     # hard switch: M == X_j
    assert np.median(alpha) < 0.05                  # α collapses to one side

    novel = rng.integers(0, 255, (24, 32, 3), np.uint8)
    _, resid = alpha_blend_fit(xi, novel, xj)       # novel mid content
    assert np.median(resid) > 20.0                  # no affine mixture explains it
