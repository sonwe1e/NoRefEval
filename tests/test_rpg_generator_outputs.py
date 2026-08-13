"""End-to-end output tree + CLI (PICPLAN §19, §20)."""
from __future__ import annotations

import json


from tools.rpg_validation_generator import __main__ as cli
from tools.rpg_validation_generator.orchestration import generate
from tools.rpg_validation_generator.validate import validate_root


def _manifests(root, modes=("nr", "endpoint", "fr")):
    dirmap = {"nr": "nr_real", "endpoint": "endpoint_real", "fr": "fr_real"}
    out = {}
    for mode in modes:
        p = root / dirmap[mode] / "manifest.jsonl"
        out[mode] = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    return out


def test_full_draft_tree_valid(rpg_cfg, tmp_path):
    out = tmp_path / "validation"
    res = generate(out, rpg_cfg, mode="all", overwrite=True)
    assert res["cases_per_mode"] == {"nr": 5, "endpoint": 5, "fr": 5}
    # small draft tree -> full per-frame decode is cheap and strict
    assert validate_root(out, full_decode=True) == []


def test_manifest_origin_flags(rpg_cfg, tmp_path):
    out = tmp_path / "validation"
    generate(out, rpg_cfg, mode="all", overwrite=True)
    for mode, ms in _manifests(out).items():
        assert len(ms) == 5
        for m in ms:
            assert m["data_origin"] == "procedural_synthetic_rpg"
            assert m["real_capture"] is False
            assert m["production_gate"] is False
            assert m["schema_version"] == "rpg-procedural-validation-v1"


def test_nr_fps_distribution(rpg_cfg, tmp_path):
    out = tmp_path / "validation"
    generate(out, rpg_cfg, mode="nr", overwrite=True)
    ms = _manifests(out, modes=("nr",))["nr"]
    fps = sorted(m["fps_by_role"]["candidate"] for m in ms)
    assert fps == [60, 60, 120, 120, 120]


def test_no_audio_anywhere(rpg_cfg, tmp_path):
    from tools.rpg_validation_generator.encode import probe
    dirmap = {"nr": "nr_real", "endpoint": "endpoint_real", "fr": "fr_real"}
    out = tmp_path / "validation"
    generate(out, rpg_cfg, mode="all", overwrite=True)
    for mode, ms in _manifests(out).items():
        for m in ms:
            case_dir = out / dirmap[mode] / f"case_{int(m['case_id'].split('_')[-1]):02d}"
            for role, fname in m["files"].items():
                assert probe(case_dir / fname)["audio_streams"] == 0


def test_cli_generate_and_validate(rpg_cfg, tmp_path, monkeypatch):
    out = tmp_path / "cli_out"
    rc = cli.main(["generate", "--output", str(out), "--draft", "--mode", "all",
                   "--overwrite"])
    assert rc == 0
    rc = cli.main(["validate", "--root", str(out)])
    assert rc == 0


def test_cli_validate_missing_root_returns_nonzero(tmp_path):
    rc = cli.main(["validate", "--root", str(tmp_path / "nope")])
    assert rc == 1
