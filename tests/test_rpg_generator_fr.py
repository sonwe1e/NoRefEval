"""Full-Reference contracts (PICPLAN §11, §19.4)."""
from __future__ import annotations

from tools.rpg_validation_generator.builder import build_case
from tools.rpg_validation_generator.case_specs import cases_for_mode


def test_fr_geometry_and_fps_parity(rpg_cfg, rpg_masters, tmp_path):
    for spec in cases_for_mode("fr"):
        master = rpg_masters[spec.scene_index]
        record, prov, contract = build_case(
            spec, master, rpg_cfg, tmp_path / "fr", rpg_cfg.seed + spec.case_index,
            "pytest")
        assert record.fps_by_role["reference"] == record.fps_by_role["candidate"] == 60
        assert record.fps_by_role["source"] == 30
        assert (record.frame_count_by_role["reference"]
                == record.frame_count_by_role["candidate"])
        assert record.frame_count_by_role["source"] == master.n_frames // 4
        assert contract.fr_outside_mask_exact is True, spec.case_id
        assert prov["contracts"]["fr_outside_mask_exact_pre_encode"] is True


def test_fr_outside_defect_decoded_close(rpg_cfg, rpg_masters, tmp_path):
    import numpy as np
    from tools.rpg_validation_generator.encode import decode_all
    spec = cases_for_mode("fr")[0]  # global blur (full-frame inside window)
    record, _, _ = build_case(spec, rpg_masters[spec.scene_index], rpg_cfg,
                              tmp_path / "fr", rpg_cfg.seed, "pytest")
    case_dir = tmp_path / "fr" / spec.case_dir_name
    cand = decode_all(case_dir / record.files["candidate"])
    ref = decode_all(case_dir / record.files["reference"])
    assert cand.shape == ref.shape
    # frames fully outside the defect window must decode nearly identical
    a, b = int(spec.start_time * 60) - 2, int(spec.start_time * 60) - 1
    a = max(0, a)
    assert np.abs(cand[a].astype(int) - ref[a].astype(int)).mean() < 3.0
