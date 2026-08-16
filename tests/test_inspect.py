"""End-to-end P0 integration: the ``inspect`` one-command path (USERPLAN §1).

Renders a tiny synthetic video with the package's own dependency-light synth
helpers, runs ``inspect`` (auto-safe -> no-reference), and checks that the
diagnostic engine, cadence block and HTML report are all wired through.
Skipped if the synth helpers are unavailable.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("rr_vfiqa.testing.synth")
from rr_vfiqa.testing.synth import render_scene, write_video  # noqa: E402

from rr_vfiqa.cli import main as cli_main  # noqa: E402
from rr_vfiqa.mode_router import route_mode  # noqa: E402
from rr_vfiqa.config import EvaluationMode  # noqa: E402


@pytest.fixture(scope="module")
def nr_video(tmp_path_factory):
    d = tmp_path_factory.mktemp("inspect_nr")
    frames, _ = render_scene(n_frames=96, w=320, h=192, seed=11, return_meta=True)
    path = write_video(d / "cand_60.mp4", frames, 60)
    return str(path)


def test_route_no_reference_without_reference_decode(nr_video):
    # candidate-only -> NR; the reference is never opened.
    r = route_mode(nr_video, None)
    assert r.ok and r.mode is EvaluationMode.NO_REFERENCE


def test_cli_version_flag(capsys):
    # ``--version`` prints the source-of-truth version and exits 0.
    from rr_vfiqa._version import VERSION

    with pytest.raises(SystemExit) as exc:
        cli_main(["--version"])
    assert exc.value.code == 0
    assert VERSION in capsys.readouterr().out


def test_route_fail_closed_on_missing_candidate():
    # A missing candidate yields a clear fail-closed refusal, never a guess.
    r = route_mode("___no_such_file___.mp4", None)
    assert not r.ok and r.mode is None


def test_inspect_nr_end_to_end(nr_video, tmp_path):
    out = tmp_path / "result"
    rc = cli_main([
        "inspect", "--candidate", nr_video, "--out", str(out),
        "--speed", "fast", "--no-clips", "--flow-backend", "farneback",
        "--device", "cpu", "--quiet",
    ])
    assert rc in (0, 1)                      # 1 only if fail-closed/degraded
    html = out / "report.html"
    rjson = out / "report.json"
    assert html.exists() and rjson.exists()
    doc = json.loads(rjson.read_text(encoding="utf-8"))
    meta = doc["meta"]
    assert meta["mode"] == "no-reference"
    # USERPLAN §5 + §7 wired through the public report
    assert "cadence" in meta and "diagnostics" in meta
    assert "cadence_integrity" in doc["scores"]
    assert "common_time_quality" in doc["scores"]
    # HTML is the interactive diagnostic report
    txt = html.read_text(encoding="utf-8")
    assert "Overall Quality" in txt and "问题卡片" in txt
    # honest semantics, never MOS
    assert "MOS" in txt


def test_inspect_fail_closed_writes_refused_report(tmp_path):
    out = tmp_path / "refused"
    rc = cli_main([
        "inspect", "--candidate", "___missing___.mp4", "--out", str(out),
        "--quiet",
    ])
    assert rc == 1
    doc = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert doc["overall_score"] is None
    assert doc["meta"]["status"] == "failed"
    assert doc["meta"]["auto_route"]["mode"] is None
    assert (out / "report.html").exists()


def test_inspect_batch_writes_index(nr_video, tmp_path):
    manifest = tmp_path / "jobs.json"
    manifest.write_text(json.dumps({"items": [
        {"id": "job_a", "candidate": nr_video},
        {"id": "job_b", "candidate": nr_video},
    ]}), encoding="utf-8")
    out = tmp_path / "batch"
    rc = cli_main([
        "inspect-batch", "--manifest", str(manifest), "--out", str(out),
        "--speed", "fast", "--no-clips", "--flow-backend", "farneback",
        "--device", "cpu", "--quiet",
    ])
    assert rc == 0
    idx = out / "index.html"
    idxj = out / "index.json"
    assert idx.exists() and idxj.exists()
    rows = json.loads(idxj.read_text(encoding="utf-8"))["rows"]
    assert len(rows) == 2
    assert all(r["mode"] == "no-reference" for r in rows)
    assert all((out / r["report"]).exists() for r in rows)
    htmltxt = idx.read_text(encoding="utf-8")
    assert "批量评测" in htmltxt
    assert "不自动排名" in htmltxt          # honest: no cross-model ranking
    assert "job_a" in htmltxt and "job_b" in htmltxt


def test_overlay_clip_export(nr_video, tmp_path):
    # export_clips path: original + overlay companions land in badcases/.
    from rr_vfiqa.report.overlay_exporter import export_overlay_clips
    out = tmp_path / "ov"
    paths = export_overlay_clips(
        nr_video,
        [{"issue_type": "duplicate_freeze", "title": "freeze", "start_time": 0.2,
          "end_time": 0.5, "severity_band": "high", "severity_label": "明显问题"}],
        out, out_fps=60.0)
    assert len(paths) == 1 and paths[0].exists() and paths[0].stat().st_size > 0
