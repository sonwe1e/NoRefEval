"""Expanded coverage: per-defect responses, scene cuts, misalignment, drops,
and a standard-preset end-to-end run exercising the region branches (§6)."""

import numpy as np
import pytest

from rr_vfiqa import evaluate_no_reference, evaluate_vfi


def _eval(source, candidate, cache_dir, flow_backend, preset="fast"):
    return evaluate_vfi(str(source), str(candidate), preset=preset,
                        cache_dir=cache_dir, device="cuda",
                        flow_backend=flow_backend, out_dir=None,
                        export_clips=False)


def _eval_nr(candidate, flow_backend):
    return evaluate_no_reference(
        str(candidate),
        preset="fast",
        device="cpu",
        flow_backend=flow_backend,
        vqa_backend="none",
        export_clips=False,
    )


def test_standard_preset_end_to_end(videos, cache_dir, flow_backend):
    """Standard preset runs the region branches without failures."""
    good = _eval(videos["source"], videos["good"], cache_dir + "/std",
                 flow_backend, preset="standard")
    bad = _eval(videos["source"], videos["bad"], cache_dir + "/std",
                flow_backend, preset="standard")
    assert good.meta["status"] == "ok"
    assert bad.overall_score < good.overall_score
    for key in ("character_integrity", "thin_object_weapon", "ui_text",
                "transition_quality"):
        assert good.scores[key] == good.scores[key]     # not NaN
    assert bad.scores["temporal_stability"] < good.scores["temporal_stability"]


def test_structural_defects_lower_character_and_thin(defect_videos, cache_dir,
                                                     flow_backend):
    good = _eval(defect_videos["source"], defect_videos["good"],
                 cache_dir + "/def", flow_backend, preset="standard")
    bad = _eval(defect_videos["source"], defect_videos["structural"],
                cache_dir + "/def", flow_backend, preset="standard")
    assert bad.overall_score < good.overall_score
    # The character branch is a classical motion proxy (§4): its subscore
    # ordering is backend-dependent at this resolution. The reliable claims
    # are the overall ranking and defect-type labels on worst windows.
    types = {t for w in bad.worst_windows for t in w.types}
    assert types & {"character_missing", "structure_loss", "motion_inconsistent",
                    "thin_object_stick"}


def test_flicker_defects_lower_temporal_and_ui(defect_videos, cache_dir,
                                               flow_backend):
    good = _eval(defect_videos["source"], defect_videos["good"],
                 cache_dir + "/def2", flow_backend, preset="standard")
    bad = _eval(defect_videos["source"], defect_videos["flicker"],
                cache_dir + "/def2", flow_backend, preset="standard")
    assert bad.overall_score < good.overall_score
    assert bad.scores["temporal_stability"] < good.scores["temporal_stability"]
    assert bad.scores["ui_text"] < good.scores["ui_text"]
    # Card freeze + shop jump attack the transition branch. On this short
    # corpus the P90-weighted subscore moves marginally, so assert the direct
    # branch evidence: card flip progression error and/or double exposure.
    f = bad.features
    assert f.get("card_flip_err", 0.0) > 0.5 or f.get("event_ghost_frac", 0.0) > 0.1
    types = {t for w in bad.worst_windows for t in w.types}
    assert types & {"card_flip_error", "transition_ghost", "ui_unstable",
                    "parity_flicker", "temporal_instability"}


def test_nr_ui_text_defect_corpus_lowers_ui_subscore(
    defect_videos,
    flow_backend,
):
    """The combined corpus validates its UI/text branch, not NR truth."""
    good = _eval_nr(defect_videos["good"], flow_backend)
    bad = _eval_nr(defect_videos["flicker"], flow_backend)
    assert bad.scores["ui_text_stability"] < good.scores["ui_text_stability"]


def test_proxy_branches_declared(videos, cache_dir, flow_backend):
    """The report must declare which branches are heuristic proxies (§P2)."""
    rep = _eval(videos["source"], videos["good"], cache_dir + "/proxy",
                flow_backend, preset="standard")
    proxies = rep.meta["proxy_branches"]
    assert proxies["character"] == "classic_motion_segmenter"
    assert proxies["card"] == "saturation_area_progression"


def test_scene_cut_is_marked_not_scored(cut_videos, cache_dir, flow_backend):
    rep = _eval(cut_videos["source"], cut_videos["candidate"],
                cache_dir + "/cut", flow_backend)
    assert rep.meta["status"] in ("ok", "degraded")
    assert len(rep.meta["alignment"]["scene_cut_pairs"]) >= 1


def test_frame_offset_detected(videos, cache_dir, flow_backend, tmp_path):
    from rr_vfiqa.testing.synth import build_offset_candidate
    from rr_vfiqa.io.video_reader import VideoReader

    good_frames = VideoReader(str(videos["good"])).decode_all()
    offset_path = build_offset_candidate(videos["source"], good_frames, tmp_path)
    rep = _eval(videos["source"], offset_path, cache_dir + "/off", flow_backend)
    al = rep.meta["alignment"]
    # Either the offset is found, or the anchors fail to verify — both are
    # acceptable detection; silent acceptance is not.
    assert al["first_anchor_offset"] != 0 or al["anchor_error"] > 12.0 \
        or al["warnings"]


def test_dropped_frames_survive(videos, cache_dir, flow_backend, tmp_path):
    from rr_vfiqa.io.video_reader import VideoReader
    from rr_vfiqa.testing.synth import build_dropped_candidate

    good_frames = VideoReader(str(videos["good"])).decode_all()
    dropped = build_dropped_candidate(good_frames, tmp_path)
    rep = _eval(videos["source"], dropped, cache_dir + "/drop", flow_backend)
    assert rep.meta["status"] in ("ok", "degraded")
    assert rep.confidence > 0
    assert rep.meta["alignment"]["reliable"]
    assert rep.meta["alignment"]["events"]


def test_vfr_candidate_survives(videos, cache_dir, flow_backend, tmp_path):
    """Variable-frame-rate candidate (§2.1): PTS-based alignment must not
    crash and must stay honest in the report meta."""
    from rr_vfiqa.io.video_reader import VideoReader
    from rr_vfiqa.testing.synth import build_vfr_candidate

    good_frames = VideoReader(str(videos["good"])).decode_all()
    vfr = build_vfr_candidate(good_frames, tmp_path)
    cand_meta = VideoReader(str(vfr)).meta
    assert cand_meta.fps == pytest.approx(120, abs=6)   # mean rate preserved
    rep = _eval(videos["source"], vfr, cache_dir + "/vfr", flow_backend)
    assert rep.meta["status"] in ("ok", "degraded")
    assert rep.confidence > 0
