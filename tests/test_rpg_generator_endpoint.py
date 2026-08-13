"""Endpoint-2x contracts (PICPLAN §10, §19.3)."""
from __future__ import annotations

import json


from tools.rpg_validation_generator.builder import build_case
from tools.rpg_validation_generator.case_specs import cases_for_mode
from tools.rpg_validation_generator.defects.base import MASK_SIZE


def test_endpoint_even_frame_contract_pre_encode(rpg_cfg, rpg_masters, tmp_path):
    for spec in cases_for_mode("endpoint"):
        master = rpg_masters[spec.scene_index]
        mode_dir = tmp_path / "endpoint_real"
        record, prov, contract = build_case(
            spec, master, rpg_cfg, mode_dir, rpg_cfg.seed + spec.case_index,
            "pytest")
        assert contract.endpoint_even_exact is True, spec.case_id
        # manifest / provenance must record it too
        assert prov["contracts"]["endpoint_even_exact_pre_encode"] is True
        rh = record.raw_rgb_sha256
        assert rh["candidate_even"] == rh["source"], spec.case_id
        # every affected (defect) frame index is odd
        case_dir = mode_dir / spec.case_dir_name
        m = json.loads((case_dir / "manifest.json").read_text(encoding="utf-8"))
        aff = m["defects"][0]["roi_boxes"]  # present
        # defect mask files only on odd frames
        for p in (case_dir / "defect_masks").glob("*.png"):
            assert int(p.stem) % 2 == 1, (spec.case_id, p.name)
        assert aff  # non-empty rois


def test_endpoint_frame_counts_and_fps(rpg_cfg, rpg_masters, tmp_path):
    spec = cases_for_mode("endpoint")[0]
    master = rpg_masters[spec.scene_index]
    record, _, _ = build_case(spec, master, rpg_cfg, tmp_path / "ep",
                              rpg_cfg.seed, "pytest")
    n_src = master.n_frames // 2
    assert record.frame_count_by_role == {"source": n_src, "candidate": master.n_frames,
                                          "oracle": master.n_frames}
    assert record.fps_by_role == {"source": 60, "candidate": 120, "oracle": 120}


def test_endpoint_masks_are_320x180(rpg_cfg, rpg_masters, tmp_path):
    import cv2
    spec = cases_for_mode("endpoint")[2]
    record, _, _ = build_case(spec, rpg_masters[spec.scene_index], rpg_cfg,
                              tmp_path / "ep", rpg_cfg.seed, "pytest")
    case_dir = tmp_path / "ep" / spec.case_dir_name
    png = next((case_dir / "defect_masks").glob("*.png"))
    img = cv2.imread(str(png), cv2.IMREAD_GRAYSCALE)
    assert img.shape == (MASK_SIZE[1], MASK_SIZE[0])
