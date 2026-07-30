"""Safe automatic mode routing (USERPLAN §2).

``route_mode`` decides the evaluation contract from the inputs *without*
being asked, but it never guesses silently: when the evidence for a
reference-bearing mode is insufficient it returns ``mode=None`` with a
human-readable reason and the caller must NOT emit a misleading score.

Rules:

* candidate only -> ``no-reference``;
* reference:candence FPS ≈ 2 and the endpoint alignment is reliable ->
  ``endpoint-2x``;
* reference:candence FPS ≈ 1 and the same-rate alignment passes every
  fail-closed gate (FPS, geometry, descriptor error, coverage, gaps) ->
  ``full-reference``;
* anything else -> fail closed (``mode=None``).

The same-rate / endpoint decisions reuse the existing alignment builders, so
the routing verdict is exactly as strict as the pipelines' own fail-closed
paths.  Speed presets are aliased per USERPLAN §3 (``--speed``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import EvalConfig, EvaluationMode

# USERPLAN §3: public speed names -> internal presets.  ``balanced`` is the
# default and maps to the existing ``standard`` preset.
SPEED_ALIASES: dict[str, str] = {
    "fast": "fast",
    "balanced": "standard",
    "thorough": "audit",
}


def preset_from_speed(speed: str | None, preset: str | None) -> str:
    """Resolve the (speed, preset) pair to a single preset name."""
    if speed is not None:
        key = speed.strip().lower()
        if key not in SPEED_ALIASES:
            raise ValueError(
                f"unknown --speed {speed!r}; choose from {sorted(SPEED_ALIASES)}")
        return SPEED_ALIASES[key]
    return preset or "standard"


@dataclass
class RouteResult:
    mode: EvaluationMode | None
    reason: str
    candidate_fps: float | None = None
    reference_fps: float | None = None
    reliable: bool = False
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.mode is not None

    def to_dict(self) -> dict:
        return {
            "mode": self.mode.value if self.mode else None,
            "reason": self.reason,
            "reliable": self.reliable,
            "candidate_fps": self.candidate_fps,
            "reference_fps": self.reference_fps,
            "warnings": list(self.warnings),
        }


def route_mode(
    candidate_video: str,
    reference_video: str | None,
    *,
    geometry_policy: str = "strict",
) -> RouteResult:
    """Decide the contract for ``inspect`` (auto-safe)."""
    from .io.video_reader import VideoReader

    try:
        cand = VideoReader(candidate_video)
    except Exception as exc:  # noqa: BLE001
        return RouteResult(None, f"无法打开候选视频：{exc!r}")

    c_fps = float(cand.meta.fps)
    if reference_video is None:
        return RouteResult(
            EvaluationMode.NO_REFERENCE,
            "仅提供候选视频，自动选择 no-reference。",
            candidate_fps=c_fps, reliable=True)

    try:
        ref = VideoReader(reference_video)
    except Exception as exc:  # noqa: BLE001
        return RouteResult(None, f"无法打开参考视频：{exc!r}",
                           candidate_fps=c_fps)
    r_fps = float(ref.meta.fps)
    ratio = c_fps / max(r_fps, 1e-6)

    if 1.9 <= ratio <= 2.1:
        ok, why, warns = _endpoint_ok(candidate_video, reference_video)
        if ok:
            return RouteResult(
                EvaluationMode.ENDPOINT_2X,
                f"参考/候选 FPS 比≈2（{r_fps:.2f}/{c_fps:.2f}）且端点对齐可靠。",
                candidate_fps=c_fps, reference_fps=r_fps,
                reliable=True, warnings=warns)
        return RouteResult(
            None,
            f"FPS 比≈2，但端点对齐不可靠，未自动判定为 endpoint-2x：{why}",
            candidate_fps=c_fps, reference_fps=r_fps, warnings=warns)

    if 0.95 <= ratio <= 1.05:
        ok, why, warns = _fr_ok(ref, cand, geometry_policy)
        if ok:
            return RouteResult(
                EvaluationMode.FULL_REFERENCE,
                f"参考/候选 FPS 相同（{r_fps:.2f}≈{c_fps:.2f}）且逐帧对齐通过"
                "全部 fail-closed 校验。",
                candidate_fps=c_fps, reference_fps=r_fps,
                reliable=True, warnings=warns)
        return RouteResult(
            None,
            "无法安全确定为 Full Reference。两条视频帧率相近，但逐帧内容/几何"
            f"不对应，未生成误导性分数。原因：{why}",
            candidate_fps=c_fps, reference_fps=r_fps, warnings=warns)

    return RouteResult(
        None,
        f"参考/候选 FPS 比={ratio:.3f}，既不≈1 也不≈2，无法安全选择模式。",
        candidate_fps=c_fps, reference_fps=r_fps)


# ----------------------------------------------------- fail-closed probes
def _endpoint_ok(candidate_video: str, reference_video: str
                 ) -> tuple[bool, str, list[str]]:
    from .io.timestamp_alignment import build_alignment
    from .io.video_reader import VideoReader
    try:
        cfg = EvalConfig.build_mode(
            candidate_video=candidate_video,
            reference_video=reference_video,
            mode=EvaluationMode.ENDPOINT_2X, preset="fast")
        alignment = build_alignment(
            cfg, VideoReader(reference_video), VideoReader(candidate_video))
    except Exception as exc:  # noqa: BLE001
        return False, f"端点对齐异常：{exc!r}", []
    return (bool(alignment.reliable),
            "" if alignment.reliable else "; ".join(alignment.warnings) or
            "对齐覆盖率/误差未达标",
            list(alignment.warnings))


def _fr_ok(ref, cand, geometry_policy: str) -> tuple[bool, str, list[str]]:
    from .io.full_reference_alignment import build_full_reference_alignment
    try:
        alignment = build_full_reference_alignment(
            ref, cand, geometry_policy=geometry_policy)
    except Exception as exc:  # noqa: BLE001
        return False, f"逐帧对齐异常：{exc!r}", []
    return (bool(alignment.reliable),
            "" if alignment.reliable else "; ".join(alignment.warnings) or
            "描述符误差/匹配覆盖率/gap 未达标",
            list(alignment.warnings))


def fail_closed_report(route: RouteResult) -> dict:
    """A report-shaped dict for a refused auto-route (no misleading score)."""
    return {
        "overall_score": None,
        "confidence": 0.0,
        "mode": None,
        "status": "failed",
        "scores": {},
        "event_ambiguity": 0.0,
        "worst_windows": [],
        "features": {},
        "meta": {
            "mode": None,
            "status": "failed",
            "auto_route": route.to_dict(),
            "score_semantics": "no score produced: mode could not be safely "
                               "determined (USERPLAN §2 fail-closed)",
        },
    }
