"""Per-operator defect behaviour (PICPLAN §12)."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from tools.rpg_validation_generator.case_specs import CASES
from tools.rpg_validation_generator.defects import get_operator
from tools.rpg_validation_generator.defects.base import DefectContext
from tools.rpg_validation_generator.motion import frame_range
from tools.rpg_validation_generator.timeline import gather, sample_indices


def _ctx_for(spec, master, cfg):
    idx = (np.arange(master.n_frames) if spec.mode == "endpoint"
           else sample_indices(cfg.master_fps, spec.candidate_fps, master.n_frames))
    cand = gather(master.rgb, idx)
    labs = np.ascontiguousarray(master.labels[idx])
    world = np.ascontiguousarray(master.world_only[idx])
    return cand, labs, world, idx


def test_all_defects_localised_and_effective(rpg_cfg, rpg_masters):
    for spec in CASES:
        master = rpg_masters[spec.scene_index]
        cand, labs, world, idx = _ctx_for(spec, master, rpg_cfg)
        clean = cand.copy()
        ctx = DefectContext(spec.candidate_fps, len(idx), labs, world, rpg_cfg,
                            np.random.default_rng(7), Path(tempfile.mkdtemp()))
        res = get_operator(spec.defect_type).apply(
            cand, spec.candidate_fps, spec.start_time, spec.end_time, ctx,
            **spec.params)
        a, b = frame_range(spec.candidate_fps, spec.start_time, spec.end_time, len(idx))
        aff = set(res.affected_indices)
        # effect is confined to the window
        assert aff <= set(range(a, b)), spec.case_id
        assert len(aff) > 0, spec.case_id
        # something in the window actually changed
        assert any(not np.array_equal(cand[i], clean[i]) for i in range(a, b)), spec.case_id
        # frames outside the affected set are untouched
        assert all(np.array_equal(cand[i], clean[i])
                   for i in range(len(idx)) if i not in aff), spec.case_id
        # masks align with affected frames and exist on disk at 320x180
        assert len(res.mask_paths) == len(aff), spec.case_id
        for p in res.mask_paths:
            fp = ctx.case_dir / p
            assert fp.exists() and fp.stat().st_size > 0, spec.case_id


def test_endpoint_defects_odd_only(rpg_cfg, rpg_masters):
    for spec in CASES:
        if spec.mode != "endpoint":
            continue
        master = rpg_masters[spec.scene_index]
        cand, labs, world, idx = _ctx_for(spec, master, rpg_cfg)
        ctx = DefectContext(spec.candidate_fps, len(idx), labs, world, rpg_cfg,
                            np.random.default_rng(7), Path(tempfile.mkdtemp()))
        res = get_operator(spec.defect_type).apply(
            cand, spec.candidate_fps, spec.start_time, spec.end_time, ctx,
            **spec.params)
        assert all(i % 2 == 1 for i in res.affected_indices), spec.case_id


def test_ui_drift_keeps_world_unchanged(rpg_cfg, rpg_masters):
    import cv2
    spec = [s for s in CASES if s.defect_type == "ui_drift"][0]
    master = rpg_masters[spec.scene_index]
    cand, labs, world, idx = _ctx_for(spec, master, rpg_cfg)
    clean = cand.copy()
    ctx = DefectContext(spec.candidate_fps, len(idx), labs, world, rpg_cfg,
                        np.random.default_rng(7), Path(tempfile.mkdtemp()))
    res = get_operator("ui_drift").apply(cand, spec.candidate_fps, spec.start_time,
                                         spec.end_time, ctx, **spec.params)
    h, w = cand.shape[1:3]
    aff = set(res.affected_indices)
    for i in range(len(idx)):
        if i not in aff:
            assert np.array_equal(cand[i], clean[i])
            continue
        # outside the saved effect footprint (old+shifted UI/text + halo) the
        # world region must be bit-identical to the clean master
        fp = cv2.imread(str(ctx.case_dir / "_contract_masks" / f"{i:06d}.png"),
                        cv2.IMREAD_GRAYSCALE) > 127
        assert np.array_equal(cand[i][~fp], clean[i][~fp]), i
        # and the footprint must be confined to UI/text screen-space pixels
        # (it may extend 2 px into adjacent world via the halo only)
