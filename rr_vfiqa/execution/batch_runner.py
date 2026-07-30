"""Batch evaluation entry point (USERPLAN §11).

Reads a small manifest of jobs, runs the same auto-safe ``inspect`` pipeline
on each, and writes a single browsable index page.  Modes are decided per-job
by ``route_mode`` (auto-safe), so a manifest may freely mix no-reference and
reference jobs.

Manifest format::

    {"items": [{"id": "...", "candidate": "a.mp4", "reference": "ref.mp4"}, ...]}

Cross-model ranking is deliberately NOT performed here (USERPLAN §11): scores
from different modes / references / contents are not comparable, so the index
only *lists* results and links to each per-job report.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Callable

ProgressFn = Callable[[str], None]


def inspect_batch(
    manifest_path: str | Path,
    out_root: str | Path,
    *,
    inspect_fn: Callable[..., tuple[dict, int]],
    speed: str | None = None,
    preset: str | None = None,
    no_clips: bool = False,
    flow_backend: str = "auto",
    device: str = "cpu",
    progress: ProgressFn | None = None,
) -> dict:
    """Run ``inspect_fn`` per manifest item; return {rows, out_root}."""
    say = progress or (lambda _: None)
    manifest_path = Path(manifest_path)
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    items = manifest.get("items") or []

    rows: list[dict[str, Any]] = []
    for item in items:
        job_id = str(item.get("id") or item.get("candidate") or f"job_{len(rows)}")
        job_out = out_root / _safe(job_id)
        say(f"batch [{job_id}] -> {item.get('candidate')}")
        try:
            summary, rc = inspect_fn(
                candidate=item["candidate"],
                reference=item.get("reference"),
                out_dir=str(job_out),
                speed=speed, preset=preset, no_clips=no_clips,
                flow_backend=flow_backend, device=device, quiet=True,
                geometry_policy=item.get("geometry_policy", "strict"),
            )
        except Exception as exc:  # noqa: BLE001
            summary, rc = ({
                "overall": None, "confidence": 0.0, "mode": None,
                "issues": 0, "affected_duration_fraction": 0.0,
                "error": repr(exc),
            }, 1)
        row = {
            "id": job_id,
            "mode": summary.get("mode"),
            "overall": summary.get("overall"),
            "confidence": summary.get("confidence"),
            "issues": summary.get("issues", 0),
            "affected_duration_fraction": summary.get(
                "affected_duration_fraction", 0.0),
            "worst": ((summary.get("auto_route") or {}).get("reason")
                      if summary.get("mode") is None else None),
            "report": f"{_safe(job_id)}/report.html",
            "status": "failed" if rc != 0 else "ok",
        }
        rows.append(row)

    write_batch_index(rows, out_root)
    (out_root / "index.json").write_text(
        json.dumps({"rows": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"rows": rows, "out_root": str(out_root)}


def write_batch_index(rows: list[dict], out_root: str | Path) -> Path:
    """Write a self-contained ``index.html`` listing every job (no ranking)."""
    out_root = Path(out_root)
    body = []
    for r in rows:
        overall = r.get("overall")
        ov = f"{overall:.1f}" if isinstance(overall, (int, float)) else "FAILED"
        conf = r.get("confidence")
        cf = f"{conf:.2f}" if isinstance(conf, (int, float)) else "—"
        aff = float(r.get("affected_duration_fraction") or 0) * 100
        worst = r.get("worst") or "—"
        body.append(
            f"<tr class='st-{_esc(r.get('status'))}'>"
            f"<td>{_esc(r.get('id'))}</td><td>{_esc(r.get('mode'))}</td>"
            f"<td class='num'>{ov}</td><td class='num'>{cf}</td>"
            f"<td class='num'>{r.get('issues', 0)}</td>"
            f"<td class='num'>{aff:.1f}%</td>"
            f"<td>{_esc(worst)}</td>"
            f"<td><a href='{_esc(r.get('report'))}'>report</a></td></tr>")
    htmlstr = (
        "<!doctype html><html lang='zh'><head><meta charset='utf-8'>"
        "<title>rr-vfiqa 批量评测</title><style>"
        "body{font:14px system-ui,sans-serif;margin:24px;background:#0f1320;color:#e8ecf5}"
        "h1{font-size:18px}table{border-collapse:collapse;width:100%}"
        "th,td{padding:6px 10px;border-bottom:1px solid #2a3147;text-align:left}"
        "th{color:#9aa4bd;font-size:12px;text-transform:uppercase}.num{text-align:right;"
        "font-variant-numeric:tabular-nums}a{color:#7fd1ff}"
        ".st-failed td{background:#3a1d1d}.note{color:#9aa4bd;font-size:12px}"
        "@media (prefers-color-scheme: light){body{background:#f4f6fb;color:#1a1f2e}"
        "th,td{border-color:#dde3ee}.st-failed td{background:#fbeaea}}</style></head>"
        f"<body><h1>批量评测（{len(rows)} 项）</h1>"
        "<p class='note'>不同模式 / 参考 / 内容的分数不可跨行比较，本表仅列示与跳转，"
        "不自动排名。</p>"
        "<table><tr><th>ID</th><th>Mode</th><th>Overall</th><th>Conf</th>"
        "<th>Issues</th><th>Affected</th><th>Worst / 原因</th><th></th></tr>"
        f"{''.join(body)}</table></body></html>")
    p = out_root / "index.html"
    p.write_text(htmlstr, encoding="utf-8")
    return p


def _esc(v: Any) -> str:
    return html.escape("" if v is None else str(v))


def _safe(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in str(name))[:80]
