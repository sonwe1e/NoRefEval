"""Documentation contract tests (USERPLAN §6).

Verifies that README and docs/index.html stay in sync with the production
code.  These tests catch schema drift, stale CLI references, and broken
resources before they reach users.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
README = _PROJECT_ROOT / "README.md"
HTML = _PROJECT_ROOT / "docs" / "index.html"


# ----------------------------------------------------------------- fixtures
@pytest.fixture(scope="session")
def readme_text() -> str:
    return README.read_text(encoding="utf-8")


@pytest.fixture(scope="session")
def html_text() -> str:
    if not HTML.exists():
        pytest.skip("docs/index.html not built yet")
    return HTML.read_text(encoding="utf-8")


@pytest.fixture(scope="session")
def schema_ids() -> dict[str, str]:
    """Read the canonical schema IDs from production code."""
    from rr_vfiqa.fusion.mode_score_schemas import ENDPOINT_SCHEMA_ID, SCHEMA_IDS
    from rr_vfiqa.config import EvaluationMode
    return {
        "nr": SCHEMA_IDS[EvaluationMode.NO_REFERENCE],
        "fr": SCHEMA_IDS[EvaluationMode.FULL_REFERENCE],
        "endpoint": ENDPOINT_SCHEMA_ID,
    }


# ------------------------------------------------------- schema sync tests
def test_readme_has_current_nr_schema(readme_text, schema_ids):
    """README must reference the current NR schema version."""
    assert schema_ids["nr"] in readme_text, (
        f"README missing current NR schema {schema_ids['nr']}")


def test_readme_has_current_endpoint_schema(readme_text, schema_ids):
    """README must reference the current Endpoint schema version."""
    assert schema_ids["endpoint"] in readme_text, (
        f"README missing current Endpoint schema {schema_ids['endpoint']}")


def test_readme_has_current_fr_schema(readme_text, schema_ids):
    """README must reference the current FR schema version."""
    assert schema_ids["fr"] in readme_text, (
        f"README missing current FR schema {schema_ids['fr']}")


def test_readme_no_stale_schemas(readme_text):
    """README must not reference old schema versions."""
    # These old versions should no longer appear in README.
    stale = [
        "nr-stability-risk-v3",
        "nr-v3",
        "endpoint-reduced-reference-v2",
        "fr-same-rate-fidelity-v2",
    ]
    for old in stale:
        assert old not in readme_text, (
            f"README contains stale schema {old!r}; update to current version")


def test_html_has_current_schemas(html_text, schema_ids):
    """HTML tutorial must reference all current schema versions."""
    for mode, schema in schema_ids.items():
        assert schema in html_text, (
            f"docs/index.html missing {mode} schema {schema}")


# ---------------------------------------------------------- HTML structure
def test_html_has_no_external_cdn(html_text):
    """Page must not depend on external CDNs or runtime fetch."""
    forbidden = [
        "cdn.jsdelivr.net",
        "unpkg.com",
        "cdnjs.cloudflare.com",
        "fonts.googleapis.com",
        "ajax.googleapis.com",
    ]
    for bad in forbidden:
        assert bad not in html_text, (
            f"docs/index.html references external CDN: {bad}")


def test_html_has_no_external_stylesheet(html_text):
    """Page must not reference external stylesheets."""
    # Allow http/https links to other sites, but not <link rel=stylesheet>
    assert not re.search(
        r'<link[^>]+rel=["\']stylesheet["\'][^>]+href=["\']https?://',
        html_text, re.IGNORECASE), (
        "docs/index.html references external stylesheet")


def test_html_images_have_alt(html_text):
    """All <img> tags must have alt attributes."""
    imgs = re.findall(r'<img[^>]*>', html_text)
    for img in imgs:
        assert 'alt=' in img, f"<img> missing alt: {img}"


def test_html_videos_have_controls(html_text):
    """All <video> tags must have controls attribute."""
    videos = re.findall(r'<video[^>]*>', html_text)
    for vid in videos:
        assert 'controls' in vid, f"<video> missing controls: {vid}"


def test_html_internal_links_valid(html_text):
    """All internal #href anchors must exist as id attributes."""
    anchors = re.findall(r'href=["\']#([^"\']+)["\']', html_text)
    for anchor in anchors:
        assert f'id="{anchor}"' in html_text or f"id='{anchor}'" in html_text, (
            f"docs/index.html links to #{anchor} but no element has that id")


# ---------------------------------------------------------- README structure
def test_readme_has_inspect_first(readme_text):
    """README's first `inspect` command should appear before any `evaluate`
    command (inspect is the primary entry)."""
    # Find the first bash code block that contains a rr-vfiqa command
    blocks = re.findall(r'```bash\n(.*?)```', readme_text, re.DOTALL)
    assert blocks, "README has no bash code block"
    # Find first block containing inspect and first containing evaluate
    inspect_pos = None
    evaluate_pos = None
    for i, block in enumerate(blocks):
        if 'inspect' in block and inspect_pos is None:
            inspect_pos = i
        if 'evaluate' in block and evaluate_pos is None:
            evaluate_pos = i
    assert inspect_pos is not None, "README has no `inspect` code block"
    if evaluate_pos is not None:
        assert inspect_pos < evaluate_pos, (
            "`inspect` should appear before `evaluate` in README")


def test_readme_links_to_html_docs(readme_text):
    """README must link to docs/index.html."""
    assert "docs/index.html" in readme_text, (
        "README missing link to docs/index.html")


def test_readme_has_output_directory(readme_text):
    """README should show the output directory structure."""
    assert "report.json" in readme_text
    assert "report.html" in readme_text
    assert "badcases/" in readme_text


def test_readme_preset_claims_are_valid_cli_choices(readme_text):
    """Every `--preset X` in README must be a real argparse choice.

    Catches doc drift like the old "tier-3 自动审计请用 --preset balanced"
    claim: argparse only accepts fast/standard/audit.
    """
    import argparse

    import rr_vfiqa.cli as cli

    probe = argparse.ArgumentParser()
    cli._add_common(probe)
    preset_action = next(a for a in probe._actions if a.dest == "preset")
    choices = set(preset_action.choices)
    for m in re.finditer(r'--preset\s+([a-z0-9-]+)', readme_text):
        assert m.group(1) in choices, (
            f"README documents --preset {m.group(1)!r} but the CLI accepts "
            f"only {sorted(choices)}")


def test_readme_badcase_naming_matches_exporter(readme_text):
    """The badcases tree in README must use the real clip naming scheme."""
    from rr_vfiqa.report.badcase_exporter import export_badcase_clips  # noqa: F401
    # The real name embeds (index, start_time, tag); README must show it.
    assert "badcase_" in readme_text, (
        "README badcases tree is stale: original clips are named "
        "badcase_NN_<start>s_<tag>.mp4, not issue_NNN_original.mp4")
    assert "issue_000_original" not in readme_text


# ---------------------------------------------------------- code block test
def test_readme_quick_start_runs(tmp_path):
    """README's Quick Start command should execute successfully."""
    from rr_vfiqa.cli import main
    from rr_vfiqa.testing.synth import render_scene, write_video

    # NR requires 60/120 FPS; use 60 FPS so the pipeline accepts the input.
    frames = render_scene(n_frames=60, w=160, h=96, seed=42)
    video = tmp_path / "test.mp4"
    write_video(video, frames, 60)

    out = tmp_path / "out"
    rc = main([
        "inspect", "--candidate", str(video), "--out", str(out),
        "--speed", "fast", "--no-clips",
        "--flow-backend", "farneback", "--device", "cpu", "--quiet",
    ])
    assert rc == 0, f"README quick start command failed with rc={rc}"
    assert (out / "report.json").exists(), "report.json not generated"
    assert (out / "report.html").exists(), "report.html not generated"

    rep = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert rep["meta"]["mode"] == "no-reference"
    assert rep["meta"]["status"] == "ok"
