"""Self-contained interactive HTML diagnostic report (USERPLAN §9).

No external assets, no network, no JS framework: one ``report.html`` with
inline CSS + a few lines of vanilla JS for the clickable timeline.  It reads
the structured diagnosis from ``report.meta["diagnostics"]`` (produced by
``diagnosis.build_diagnostics_block``) and the cadence block from
``report.meta["cadence"]`` when present.

The report is intentionally honest about what the numbers mean: the header
labels the overall score as an *engineering risk* level, never a human MOS,
and every probable cause is tagged as inferred.
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

from ..diagnosis.schema import issue_level, quality_level
from ..diagnosis.rules import TRACKS

_TRACK_COLOR = {
    "Temporal": "#d9534f",
    "Motion": "#f0ad4e",
    "Structure": "#9b59b6",
    "UI/Text": "#3498db",
    "Technical": "#7f8c8d",
    "Cadence": "#16a085",
}
_BAND_COLOR = {"severe": "#c0392b", "high": "#e67e22",
               "medium": "#f1c40f", "low": "#95a5a6"}


def render_html_report(report, *, mode_label: str | None = None) -> str:
    """Return a complete HTML document string for ``report``."""
    d = report.to_dict()
    meta = d.get("meta", {}) or {}
    diag = meta.get("diagnostics", {}) or {}
    cad = meta.get("cadence", {}) or {}
    overall = d.get("overall_score")
    conf = d.get("confidence")
    # USERPLAN §10: the global (whole-video) level and the worst local issue
    # level are two independent KPIs — one severe local issue is not a claim
    # that the whole video is severely bad.
    global_key, global_label = quality_level(
        overall if overall is not None else float("nan"))
    issues = diag.get("issues", []) or []
    total_seconds = float(diag.get("total_seconds") or 0.0)
    affected = diag.get("affected_duration_fraction", 0.0)
    confirmed = float(diag.get("confirmed_affected_seconds") or 0.0)
    worst = diag.get("worst_issue")
    wl = diag.get("worst_issue_level") or {}
    worst_key = wl.get("key") if isinstance(wl, dict) else None
    worst_label = wl.get("label") if isinstance(wl, dict) else None
    if not worst_label and worst is not None:
        worst_key, worst_label = issue_level(float(worst.get("severity") or 0.0))
    worst_label = worst_label or "—"

    mode = mode_label or meta.get("mode", "?")
    status = meta.get("status", "?")

    parts: list[str] = [_HEAD]
    parts.append("<body>")
    parts.append(f'<header class="topbar status-{_esc(status)}">')
    parts.append(f"<h1>{_esc(mode)} 诊断报告</h1>")
    parts.append('<div class="kpis">')
    parts.append(_kpi("Overall Quality", _fmt(overall, 1) + " / 100",
                      _BAND_COLOR.get(_score_band(overall), "#888")))
    parts.append(_kpi("Confidence", _fmt(conf, 2)))
    parts.append(_kpi("Global Quality", global_label,
                      _BAND_COLOR.get(global_key, "#888")))
    parts.append(_kpi("Worst Local Issue", worst_label,
                      _BAND_COLOR.get(worst_key, "#888")))
    parts.append(_kpi("Mode", mode))
    parts.append(_kpi("Issues", str(len(issues))))
    parts.append(_kpi("Affected", f"{affected * 100:.1f}%"
                      + (f"（确认 {confirmed:.1f}s）" if confirmed > 0 else "")))
    parts.append(_kpi("Worst", _fmt_interval(worst) if worst else "—"))
    parts.append("</div>")
    if cad:
        parts.append('<div class="cadence">')
        parts.append(f"<span>Cadence Integrity <b>{_fmt(cad.get('cadence_integrity'), 1)}</b></span>")
        parts.append(f"<span>Common-time <b>{_fmt(cad.get('common_time_quality'), 1)}</b></span>")
        parts.append(f"<span>cadence risk <b>{_fmt(cad.get('cadence_risk'), 2)}</b></span>")
        parts.append("</div>")
    parts.append("</header>")

    # ---- subscores ------------------------------------------------------
    subs = d.get("scores", {}) or {}
    if subs:
        parts.append('<section><h2>子分数</h2><div class="subscores">')
        for k, v in subs.items():
            parts.append(_bar(_esc(k), v))
        parts.append("</div></section>")

    # ---- timeline -------------------------------------------------------
    parts.append('<section><h2>时间线 <span class="hint">（点击色块跳到对应问题卡片）</span></h2>')
    parts.append(_render_timeline(issues, total_seconds))
    parts.append("</section>")

    # ---- issue cards ----------------------------------------------------
    parts.append(f'<section><h2>问题卡片（{len(issues)}）</h2>')
    if not issues:
        parts.append('<p class="ok">未在所选窗口中检测到达到阈值的问题组合。</p>')
    for i, issue in enumerate(issues):
        parts.append(_render_card(i, issue))
    parts.append("</section>")

    # ---- provenance / limits -------------------------------------------
    parts.append('<section class="meta"><h2>说明与限制</h2><ul>')
    for lim in meta.get("limitations", []) or []:
        parts.append(f"<li>{_esc(lim)}</li>")
    parts.append("<li>分数为确定性工程风险等级，<b>不</b>等价于人类主观 MOS。</li>")
    parts.append("<li>“可能原因”均为推断，不代表已知模型内部根因。</li>")
    parts.append(f"<li>score_schema: {_esc(meta.get('score_schema'))}</li>")
    parts.append("</ul></section>")

    parts.append(_SCRIPT)
    parts.append("</body></html>")
    return "\n".join(parts)


def write_html_report(report, path: str | Path, **kw) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_html_report(report, **kw), encoding="utf-8")
    return path


# ---------------------------------------------------------------- helpers
def _esc(v: Any) -> str:
    return html.escape("" if v is None else str(v))


def _fmt(v: Any, n: int = 2) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v):.{n}f}"
    except (TypeError, ValueError):
        return "—"


def _score_band(overall) -> str:
    if overall is None or overall != overall:
        return "low"
    if overall >= 80:
        return "low"
    if overall >= 65:
        return "medium"
    if overall >= 45:
        return "high"
    return "severe"


def _fmt_interval(issue: dict | None) -> str:
    if not issue:
        return "—"
    return f"{_ts(issue.get('start_time'))}–{_ts(issue.get('end_time'))}"


def _ts(t) -> str:
    try:
        t = float(t)
    except (TypeError, ValueError):
        return "—:—"
    m, s = divmod(t, 60)
    return f"{int(m):02d}:{s:05.2f}"


def _kpi(label: str, value: str, color: str | None = None) -> str:
    style = f' style="border-top:4px solid {color}"' if color else ""
    return (f'<div class="kpi"{style}><div class="kpi-v">{_esc(value)}</div>'
            f'<div class="kpi-l">{_esc(label)}</div></div>')


def _bar(label: str, value) -> str:
    v = float(value) if value is not None else float("nan")
    w = max(0.0, min(100.0, v)) if v == v else 0.0
    txt = _fmt(v, 1) if v == v else "—"
    return (f'<div class="sbar"><span class="sbar-l">{label}</span>'
            f'<span class="sbar-t"><i style="width:{w:.1f}%"></i></span>'
            f'<span class="sbar-v">{txt}</span></div>')


def _render_timeline(issues: list[dict], total_seconds: float) -> str:
    if total_seconds <= 0:
        total_seconds = max((float(i.get("end_time") or 0) for i in issues),
                            default=1.0) or 1.0
    rows = []
    for track in TRACKS:
        bars = []
        for i, issue in enumerate(issues):
            if issue.get("track") != track:
                continue
            s = float(issue.get("start_time") or 0)
            e = float(issue.get("end_time") or s)
            left = 100.0 * s / total_seconds
            width = max(0.5, 100.0 * (e - s) / total_seconds)
            band = issue.get("severity_band", "low")
            color = _BAND_COLOR.get(band, "#888")
            bars.append(
                f'<a class="tbar band-{band}" href="#issue-{i}" '
                f'title="{_esc(issue.get("title"))} {_fmt_interval(issue)}" '
                f'style="left:{left:.2f}%;width:{width:.2f}%;background:{color}"></a>')
        rows.append(
            f'<div class="trow"><span class="tlabel" '
            f'style="color:{_TRACK_COLOR.get(track, "#888")}">{_esc(track)}</span>'
            f'<div class="ttrack">{"".join(bars)}</div></div>')
    axis = "".join(
        f'<span style="left:{100 * k / 4:.0f}%">{_ts(total_seconds * k / 4)}</span>'
        for k in range(5))
    return (f'<div class="timeline">{"".join(rows)}'
            f'<div class="taxis">{axis}</div></div>')


def _render_card(i: int, issue: dict) -> str:
    band = issue.get("severity_band", "low")
    color = _BAND_COLOR.get(band, "#888")
    ev = issue.get("evidence", []) or []
    ev_rows = "".join(
        f'<li><code>{_esc(e.get("metric"))}</code> = {_fmt(e.get("observed"), 3)} '
        f'(阈值 {_fmt(e.get("threshold"), 3)}, 强度 {_fmt(e.get("strength"), 2)})'
        + (f' — {_esc(e.get("note"))}' if e.get("note") else "") + "</li>"
        for e in ev)
    causes = "".join(f"<li>{_esc(c)} <em>(推断)</em></li>"
                     for c in issue.get("probable_causes", []) or [])
    maps = issue.get("maps", []) or []
    map_imgs = "".join(
        f'<a href="{_esc(m)}" target="_blank">'
        f'<img class="heat" src="{_esc(m)}" alt="{_esc(m)}" '
        f'loading="lazy"></a>'
        for m in maps)
    map_block = (f'<h4>热力图</h4><div class="heatmaps">{map_imgs}</div>'
                 if map_imgs else "")
    # USERPLAN §10: embed the companion videos + keyframe thumbnail.
    clips = issue.get("clip_paths") or {}
    clip_rows = "".join(
        f'<li><a href="{_esc(p)}" target="_blank">{html.escape(label)}</a></li>'
        for label, p in clips.items() if p)
    clip_block = (f'<h4>坏例视频</h4><ul class="clips">{clip_rows or "<li>—</li>"}</ul>'
                  if clip_rows else "")
    thumb = issue.get("thumbnail", "")
    thumb_block = (f'<h4>关键帧</h4><a href="{_esc(thumb)}" target="_blank">'
                   f'<img class="thumb" src="{_esc(thumb)}" alt="keyframe" '
                   f'loading="lazy"></a>' if thumb else "")
    # USERPLAN §9: the display span (merged card) is shown separately from the
    # actually-sampled support spans, so the reader can see what really backs
    # the affected-duration figure (unsampled gaps are never counted).
    spans = issue.get("support_spans") or []
    span_parts = [f"{_ts(p[0])}–{_ts(p[1])}" for p in spans
                  if isinstance(p, (list, tuple)) and len(p) == 2]
    span_block = f"<span>采样 {'、'.join(span_parts)}</span>" if span_parts else ""
    return (
        f'<article class="card band-{band}" id="issue-{i}">'
        f'<div class="card-h" style="border-left:6px solid {color}">'
        f'<h3>{_esc(issue.get("title"))}</h3>'
        f'<span class="badge" style="background:{color}">{_esc(issue.get("severity_label"))}</span>'
        f'</div>'
        f'<div class="card-m">'
        f'<span>⏱ {_fmt_interval(issue)}</span>'
        f'{span_block}'
        f'<span>严重度 {_fmt(issue.get("severity"), 2)}</span>'
        f'<span>置信度 {_fmt(issue.get("confidence"), 2)}</span>'
        f'<span>轨道 {_esc(issue.get("track"))}</span>'
        f'</div>'
        f'<div class="card-b"><h4>证据</h4><ul>{ev_rows or "<li>—</li>"}</ul>'
        f'<h4>可能原因</h4><ul>{causes or "<li>—</li>"}</ul>'
        f'{map_block}{thumb_block}{clip_block}</div>'
        f'</article>')


_HEAD = """<!doctype html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>rr-vfiqa 诊断报告</title>
<style>
:root{--bg:#0f1320;--panel:#171c2e;--ink:#e8ecf5;--mut:#9aa4bd;--line:#2a3147}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:14px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,"Microsoft YaHei",sans-serif}
h1{font-size:18px;margin:0}h2{font-size:15px;margin:18px 0 8px;color:var(--mut);
text-transform:uppercase;letter-spacing:.04em}section{max-width:1080px;margin:0 auto;padding:0 16px}
.topbar{max-width:1080px;margin:0 auto;padding:18px 16px 8px;border-bottom:1px solid var(--line)}
.topbar.status-failed{border-left:6px solid #c0392b}
.topbar.status-degraded{border-left:6px solid #e67e22}
.kpis{display:flex;flex-wrap:wrap;gap:10px;margin-top:12px}
.kpi{background:var(--panel);border-radius:8px;padding:10px 14px;min-width:120px;flex:1}
.kpi-v{font-size:20px;font-weight:700}.kpi-l{color:var(--mut);font-size:12px}
.cadence{display:flex;gap:18px;margin-top:10px;color:var(--mut);font-size:13px}
.cadence b{color:var(--ink)}
.subscores{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:6px 18px}
.sbar{display:grid;grid-template-columns:140px 1fr 44px;align-items:center;gap:8px}
.sbar-l{color:var(--mut);font-size:12px}.sbar-v{text-align:right;font-variant-numeric:tabular-nums}
.sbar-t{background:#222a40;border-radius:4px;height:8px;overflow:hidden}
.sbar-t i{display:block;height:100%;background:linear-gradient(90deg,#e67e22,#27ae60)}
.timeline{background:var(--panel);border-radius:8px;padding:10px}
.trow{display:grid;grid-template-columns:84px 1fr;align-items:center;gap:8px;margin:3px 0}
.tlabel{font-size:12px;font-weight:600;text-align:right}
.ttrack{position:relative;height:16px;background:#10152400;border-bottom:1px solid var(--line)}
.tbar{position:absolute;top:1px;height:14px;border-radius:3px;cursor:pointer;opacity:.92}
.tbar:hover{opacity:1;outline:2px solid #fff}
.taxis{position:relative;height:16px;margin-left:92px;color:var(--mut);font-size:11px}
.taxis span{position:absolute;transform:translateX(-50%)}
.hint{color:var(--mut);font-size:11px;text-transform:none;letter-spacing:0}
.card{background:var(--panel);border-radius:10px;margin:10px 0;overflow:hidden;scroll-margin-top:12px}
.card:target{outline:2px solid #3498db}
.card-h{display:flex;justify-content:space-between;align-items:center;padding:10px 14px}
.card-h h3{margin:0;font-size:15px}
.badge{color:#fff;border-radius:12px;padding:2px 10px;font-size:12px;font-weight:600}
.card-m{display:flex;flex-wrap:wrap;gap:14px;padding:0 14px 6px;color:var(--mut);font-size:12px}
.card-b{padding:0 14px 12px}.card-b h4{margin:8px 0 2px;color:var(--mut);font-size:12px}
.card-b ul{margin:0;padding-left:18px}.card-b code{color:#7fd1ff}
.card-b em{color:var(--mut);font-style:normal;font-size:11px}
.heatmaps{display:flex;flex-wrap:wrap;gap:8px;margin-top:4px}
.heatmaps img.heat{width:180px;height:auto;border-radius:6px;
border:1px solid var(--line);cursor:zoom-in;transition:transform .15s}
.heatmaps img.heat:hover{transform:scale(1.5);z-index:2;position:relative}
.clips{list-style:square;padding-left:18px}.clips a{color:#7fd1ff}
.thumb{width:240px;height:auto;border-radius:6px;border:1px solid var(--line);
margin-top:4px}
.meta{color:var(--mut);padding-bottom:40px}.meta li{margin:3px 0}
.ok{color:#27ae60}
@media (prefers-color-scheme: light){
:root{--bg:#f4f6fb;--panel:#fff;--ink:#1a1f2e;--mut:#5a6478;--line:#dde3ee}
.sbar-t{background:#e7ecf5}.ttrack{background:#eef1f8}}
</style></head>"""

_SCRIPT = """<script>
document.querySelectorAll('.tbar').forEach(function(b){
  b.addEventListener('click',function(e){
    var t=document.querySelector(b.getAttribute('href'));
    if(t){e.preventDefault();t.scrollIntoView({behavior:'smooth',block:'center'});}
  });
});
</script>"""
