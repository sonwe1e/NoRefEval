import numpy as np
import pytest

from rr_vfiqa.config import EvalConfig
from rr_vfiqa.io.video_reader import VideoReader
from rr_vfiqa.io.timestamp_alignment import build_alignment
from rr_vfiqa.io.color_normalization import estimate_color_transform


def test_video_meta(videos):
    src = VideoReader(str(videos["source"]))
    cand = VideoReader(str(videos["good"]))
    assert src.meta.fps == pytest.approx(60, abs=1.0)
    assert cand.meta.fps == pytest.approx(120, abs=1.0)
    assert src.meta.n_frames == 80
    assert cand.meta.n_frames == 160
    assert np.all(np.diff(src.meta.pts_seconds) > 0)


def test_alignment(videos):
    cfg = EvalConfig.build(str(videos["source"]), str(videos["good"]), "fast")
    src = VideoReader(str(videos["source"]))
    cand = VideoReader(str(videos["good"]))
    al = build_alignment(cfg, src, cand)
    assert al.fps_ratio == pytest.approx(2.0, abs=0.05)
    assert al.first_anchor_offset == 0
    # even candidate frames map to source, odd frames to pairs
    assert al.anchor_of_candidate[0] == 0
    assert al.anchor_of_candidate[1] == -1
    assert al.pair_of_candidate[1] == 0
    assert len(al.scene_cuts) == 0
    # anchors are re-encoded source frames: mismatch stays under codec noise
    assert al.anchor_error < 12.0


def test_color_transform_identity(videos):
    src = VideoReader(str(videos["source"]))
    a = src.read_one(10, width=320)
    tf = estimate_color_transform(a, a)
    assert tf.is_trivial()


def test_color_transform_inverts_to_source_space():
    """apply() must map candidate→source: fitted cand = 1.1·src + 6."""
    rng = np.random.default_rng(1)
    src = rng.integers(20, 235, (64, 96, 3), np.uint8)
    cand = np.clip(1.1 * src.astype(np.float32) + 6.0, 0, 255).astype(np.uint8)
    tf = estimate_color_transform(src, cand)
    assert not tf.is_trivial()
    rec = tf.apply(cand)                       # candidate → source space
    assert np.abs(rec.astype(int) - src.astype(int)).mean() < 2.0
    fwd = tf.apply_forward(src)                # source → candidate direction
    assert np.abs(fwd.astype(int) - cand.astype(int)).mean() < 2.0


def test_random_access_consistency(videos):
    cand = VideoReader(str(videos["good"]))
    seq = cand.decode_all(width=160)
    rnd = cand.read_frames([0, 5, 17, 63, 159], width=160)
    for pos, idx in enumerate(rnd.indices):
        assert np.mean(np.abs(seq[idx].astype(int) - rnd.rgb[pos].astype(int))) < 2.0
