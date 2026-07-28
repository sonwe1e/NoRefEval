"""Timeline reports: markdown summary + optional matplotlib PNG."""

from __future__ import annotations

from pathlib import Path

from ..sampling.window_selector import window_times
from ..schema import Report, VideoMeta, WindowFeatures


def render_timeline_md(windows: list[WindowFeatures], cand_meta: VideoMeta,
                       report: Report) -> str:
    lines = [
        "# rr_vfiqa evaluation timeline",
        "",
        f"overall: **{report.overall_score:.1f}** / confidence "
        f"{report.confidence:.2f} / event_ambiguity {report.event_ambiguity:.2f}",
        "",
        "| t_start | t_end | src | risk | severity | types |",
        "|--------:|------:|-----|-------:|---------:|-------|",
    ]
    sev_by_center = {w.center_index: w for w in report.worst_windows}
    for wf in windows:
        t0, t1 = window_times(wf.window, cand_meta)
        worst = sev_by_center.get(wf.window.center)
        sev = f"{worst.severity:.2f}" if worst else "-"
        types = ",".join(worst.types) if worst else ""
        lines.append(f"| {t0:6.2f} | {t1:6.2f} | {wf.window.source} | "
                     f"{wf.window.risk:5.2f} | {sev:>8} | {types} |")
    lines += ["", "## subscores", ""]
    for k, v in report.scores.items():
        lines.append(f"- {k}: {v:.1f}")
    lines += ["", "## worst windows", ""]
    for w in report.worst_windows:
        lines.append(f"- [{w.start:.2f}s–{w.end:.2f}s] severity {w.severity:.2f} "
                     f"conf {w.confidence:.2f}: {', '.join(w.types)}")
    return "\n".join(lines) + "\n"


def render_timeline_png(windows: list[WindowFeatures], cand_meta: VideoMeta,
                        report: Report, path: str | Path) -> Path | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    n = len(windows)
    if n == 0:
        return None
    sev = {w.center_index: w.severity for w in report.worst_windows}
    times = [window_times(wf.window, cand_meta)[0] for wf in windows]
    risks = [wf.window.risk for wf in windows]
    sevs = [sev.get(wf.window.center, 0.0) for wf in windows]

    fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
    axes[0].bar(range(n), risks, width=1.0, color="#6b7280")
    axes[0].set_ylabel("tier-1 risk")
    axes[1].bar(range(n), sevs, width=1.0, color="#dc2626")
    axes[1].set_ylabel("defect severity")
    axes[1].set_xticks(range(0, n, max(1, n // 12)))
    axes[1].set_xticklabels([f"{times[i]:.1f}s" for i in range(0, n, max(1, n // 12))],
                            rotation=30)
    fig.suptitle(f"rr_vfiqa — overall {report.overall_score:.1f}")
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path
