"""Multi-evidence diagnosis rules (USERPLAN §7).

Each rule maps a *combination* of feature signals to a named issue family.
No single feature is allowed to fire an issue on its own: a rule requires a
minimum weighted evidence mass (``min_mass``), which enforces the
"diagnose from multiple signals" requirement.  Probable causes are stored as
*inferred* explanations.

The rules consume plain ``{feature: value}`` dicts, so they apply uniformly to
NR / endpoint / FR windows (each family's metric keys simply coexist).  Unknown
keys are skipped, which means a rule degrades gracefully when a mode does not
produce a given signal.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .schema import DiagnosticIssue, Evidence

# Timeline tracks (USERPLAN §9).
TRACKS = ("Temporal", "Motion", "Structure", "UI/Text", "Technical", "Cadence")


@dataclass(frozen=True)
class Condition:
    metric: str
    threshold: float
    weight: float
    direction: str = "higher_is_worse"   # or "lower_is_worse"
    scale: float = 1.0                   # normalisation span for the strength
    note: str = ""

    def evaluate(self, scalars: dict[str, float]) -> Evidence | None:
        if self.metric not in scalars:
            return None
        v = float(scalars[self.metric])
        if not np.isfinite(v):
            return None
        if self.direction == "higher_is_worse":
            if v <= self.threshold:
                return None
            raw = (v - self.threshold) / max(self.scale, 1e-6)
        else:
            if v >= self.threshold:
                return None
            raw = (self.threshold - v) / max(self.scale, 1e-6)
        strength = float(np.clip(raw, 0.0, 1.0))
        if strength <= 0.0:
            return None
        return Evidence(
            metric=self.metric, observed=v, threshold=self.threshold,
            strength=strength, direction=self.direction,
            note=self.note or _default_note(self.metric))


@dataclass(frozen=True)
class Rule:
    issue_type: str
    title: str
    track: str
    conditions: tuple[Condition, ...]
    min_mass: float = 0.6                # weighted evidence required to fire
    min_evidence: int = 2                # hard multi-signal floor
    probable_causes: tuple[str, ...] = ()

    def evaluate(self, scalars: dict[str, float]) -> tuple[list[Evidence], float]:
        """Return (fired evidence, weighted mass)."""
        evs: list[Evidence] = []
        mass = 0.0
        for cond in self.conditions:
            ev = cond.evaluate(scalars)
            if ev is not None:
                evs.append(ev)
                mass += cond.weight * ev.strength
        return evs, mass


def _default_note(metric: str) -> str:
    return metric.replace("_", " ")


# ---------------------------------------------------------------- rule table
RULES: tuple[Rule, ...] = (
    Rule(
        issue_type="duplicate_freeze",
        title="生成帧冻结 / 复制前一帧",
        track="Temporal",
        min_mass=0.8,
        conditions=(
            Condition("nr_native_duplicate_fraction", 0.10, 1.2,
                      scale=0.30, note="native 相邻帧重复比例偏高"),
            Condition("nr_native_freeze_fraction", 0.10, 1.0,
                      scale=0.30, note="native 冻结比例偏高"),
            Condition("nr_native_self_comp", 0.06, 0.9,
                      direction="lower_is_worse", scale=0.06,
                      note="native 自补偿残差接近 0（帧几乎不动）"),
            Condition("nr_mct_native_mean", 6.0, 0.8,
                      direction="lower_is_worse", scale=6.0,
                      note="native 运动补偿残差极低"),
            Condition("nr_duplicate_fraction", 0.10, 0.6,
                      scale=0.20, note="公共尺度也见重复"),
        ),
        probable_causes=(
            "模型复制了端点帧",
            "编码 / 导出过程丢帧或重复写帧",
            "推理循环重复写入同一帧",
            "时间戳错误导致帧被复用",
        ),
    ),
    Rule(
        issue_type="generated_blur",
        title="生成帧模糊 / 过度平滑",
        track="Structure",
        min_mass=0.7,
        conditions=(
            Condition("gtq_sharp_odd_even_ratio", 0.80, 1.0,
                      direction="lower_is_worse", scale=0.35,
                      note="奇偶帧清晰度差异"),
            Condition("parity_window_sharp_gap", 0.10, 1.0,
                      scale=0.20, note="生成帧清晰度低于端点"),
            Condition("nr_phase_sharp_gap", 0.18, 0.8,
                      scale=0.20, note="相位清晰度在生成帧塌陷"),
            Condition("edge_recall", 0.70, 0.7, direction="lower_is_worse",
                      scale=0.30, note="边缘召回下降"),
        ),
        probable_causes=(
            "插帧结果过度平滑",
            "blend mask 过软",
            "motion compensation 不准确",
            "编码质量不足",
        ),
    ),
    Rule(
        issue_type="ghost_double_exposure",
        title="重影 / 双重曝光",
        track="Structure",
        min_mass=0.7,
        conditions=(
            Condition("edge_ghost_frac", 0.05, 1.2, scale=0.10,
                      note="额外双边缘比例"),
            # USERPLAN §6.2: edge_chamfer_sup_to_em is now normalized to the
            # frame diagonal (typical 0.01-0.05); threshold in the same unit.
            Condition("edge_chamfer_sup_to_em", 3.0 / 559.77, 0.9,
                      scale=4.0 / 559.77, note="边缘 chamfer 升高"),
            Condition("cycle_resid_p90", 8.0, 0.8, scale=8.0,
                      note="循环重建残差偏高"),
            Condition("comp_p90_max", 12.0, 0.7, scale=10.0,
                      note="合成误差峰值偏高"),
        ),
        probable_causes=(
            "双向 warp 错位",
            "blend mask 错误",
            "遮挡处理失败",
            "前后运动层被错误混合",
        ),
    ),
    Rule(
        issue_type="tearing_flow_folding",
        title="撕裂 / 光流折叠",
        track="Motion",
        min_mass=0.7,
        conditions=(
            Condition("flow_fold_frac", 0.02, 1.2, scale=0.05,
                      note="光流折叠比例"),
            Condition("flow_jdet_low_frac", 0.05, 1.0, scale=0.10,
                      note="Jacobian 行列式过低比例"),
            Condition("nr_flow_fold_fraction", 0.03, 0.9, scale=0.06,
                      note="native 光流折叠"),
            Condition("nr_flow_jdet_low_fraction", 0.08, 0.8, scale=0.12,
                      note="native Jacobian 异常"),
        ),
        probable_causes=(
            "光流局部错误",
            "运动边界归属错误",
            "形变过强",
            "遮挡区域错误传播",
        ),
    ),
    Rule(
        issue_type="ui_text_instability",
        title="UI / 文字不稳定",
        track="UI/Text",
        min_mass=0.7,
        conditions=(
            Condition("nr_ui_edge_instability", 0.08, 1.2, scale=0.12,
                      note="静态 UI 边缘 XOR 抖动"),
            Condition("nr_text_stroke_instability", 0.12, 1.2, scale=0.20,
                      note="文字笔画断裂 / 漂移"),
            Condition("nr_ui_component_instability", 0.20, 0.9, scale=0.30,
                      note="UI 组件数量交替"),
            Condition("edge_count_odd_even_ratio", 0.70, 0.6,
                      direction="lower_is_worse", scale=0.40,
                      note="奇偶帧边缘计数不一致"),
        ),
        probable_causes=(
            "UI 被当成场景运动处理",
            "UI 未被单独处理",
            "文字区域融合 / 缩放错误",
            "奇偶帧处理不一致",
        ),
    ),
    Rule(
        issue_type="fr_color_pipeline_mismatch",
        title="色彩管线不一致",
        track="Technical",
        min_mass=0.8,
        conditions=(
            Condition("fr_scan_chroma_l1", 6.0, 1.3, scale=6.0,
                      note="色度误差偏高"),
            Condition("fr_l1_y", 8.0, 0.8, direction="lower_is_worse",
                      scale=8.0, note="亮度误差相对较小"),
            Condition("fr_ssim", 0.90, 0.5, direction="lower_is_worse",
                      scale=0.30, note="结构仍基本正常"),
        ),
        probable_causes=(
            "色彩空间转换错误",
            "RGB / BGR 顺序错误",
            "limited / full range 不一致",
            "编码器色彩矩阵 / gamma 差异",
        ),
    ),
    Rule(
        issue_type="fr_spatial_reference_error",
        title="空间保真误差",
        track="Structure",
        min_mass=0.8,
        conditions=(
            Condition("fr_l1_rgb", 10.0, 1.0, scale=8.0, note="RGB 误差偏高"),
            Condition("fr_ssim", 0.90, 0.8, direction="lower_is_worse",
                      scale=0.25, note="SSIM 下降"),
            Condition("fr_edge_f1", 0.70, 0.7, direction="lower_is_worse",
                      scale=0.30, note="边缘 F1 下降"),
        ),
        probable_causes=(
            "生成帧空间细节偏离参考",
            "纹理 / 边缘重建不准",
        ),
    ),
    Rule(
        issue_type="fr_temporal_fidelity_error",
        title="时序保真误差 / 闪烁",
        track="Temporal",
        min_mass=0.8,
        conditions=(
            Condition("fr_temporal_diff_error", 8.0, 1.0, note="时序差分误差"),
            Condition("fr_flicker_excess", 2.0, 1.0, note="闪烁过量"),
            Condition("fr_mcr_difference", 8.0, 0.8, note="运动补偿残差差"),
        ),
        probable_causes=(
            "相邻生成帧不一致",
            "闪烁 / 亮度抖动",
        ),
    ),
)


def _rule_for(issue_type: str) -> Rule | None:
    for r in RULES:
        if r.issue_type == issue_type:
            return r
    return None


def evaluate_rules(scalars: dict[str, float]) -> list[tuple[Rule, list[Evidence], float]]:
    """Return [(rule, evidence, mass)] for every rule that crosses its bar."""
    fired = []
    for rule in RULES:
        evs, mass = rule.evaluate(scalars)
        if len(evs) >= rule.min_evidence and mass >= rule.min_mass:
            fired.append((rule, evs, mass))
    return fired


def diagnose_windows(
    windows_scalars: list[tuple[float, float, int, dict[str, float], float]],
    nms_seconds: float = 0.5,
    severity_floor: float = 0.10,
    *,
    error_maps: dict[int, np.ndarray] | None = None,
) -> list[DiagnosticIssue]:
    """Build issues from per-window data.

    Each element is ``(start_time, end_time, center_index, scalars,
    window_confidence)``.  Windows that fire one or more rules become issues;
    same-type issues close in time are merged so a sustained defect reads as
    one card rather than a stack.

    ``error_maps`` maps ``center_index`` -> a dense (H, W) field; when supplied
    the hottest issue per window gets spatial ``boxes`` extracted from that
    window's map so the report can annotate *where* the defect sits
    (USERPLAN P1).
    """
    raw: list[DiagnosticIssue] = []
    for start, end, center, scalars, win_conf in windows_scalars:
        emap = error_maps.get(center) if error_maps else None
        fired = evaluate_rules(scalars)
        for rule, evs, mass in fired:
            severity = float(np.clip(mass, 0.0, 1.0))
            if severity < severity_floor:
                continue
            strength_mean = float(np.mean([e.strength for e in evs])) if evs else 0.0
            confidence = float(np.clip(
                0.55 * win_conf + 0.45 * min(1.0, len(evs) / 3.0)
                + 0.10 * strength_mean, 0.0, 1.0))
            # Attach boxes from the error map to the strongest issue only, so
            # the annotation reflects the dominant defect in this window.
            boxes: list[list[int]] = []
            if emap is not None and rule is fired[0][0]:
                boxes = boxes_from_error_map(emap)
            raw.append(DiagnosticIssue(
                issue_type=rule.issue_type,
                title=rule.title,
                severity=severity,
                confidence=confidence,
                start_time=start,
                end_time=end,
                track=rule.track,
                evidence=evs,
                probable_causes=list(rule.probable_causes),
                center_index=center,
                boxes=boxes,
            ))
    return merge_issues(raw, nms_seconds)


def boxes_from_error_map(error_map: np.ndarray, *,
                         threshold: float = 0.55,
                         min_area: int = 64,
                         max_boxes: int = 3) -> list[list[int]]:
    """Extract up to ``max_boxes`` bounding boxes of the high-value region.

    Returns a list of ``[x, y, w, h]`` boxes in pixel coordinates of
    ``error_map``.  Used to populate ``DiagnosticIssue.boxes`` so the report
    can annotate *where* the defect sits (USERPLAN P1).
    """
    if error_map is None or error_map.size == 0:
        return []
    m = np.nan_to_num(error_map.astype(np.float32), nan=0.0)
    vmax = float(np.percentile(m, 99)) if m.max() > 0 else 1.0
    if vmax <= 0:
        return []
    binary = (m >= threshold * vmax).astype(np.uint8)
    # Light clean-up so broad hot regions become tight contours.
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    boxes: list[list[int]] = []
    for cnt in sorted(contours, key=cv2.contourArea, reverse=True):
        if len(boxes) >= max_boxes:
            break
        area = int(cv2.contourArea(cnt))
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        boxes.append([int(x), int(y), int(w), int(h)])
    return boxes


def _cluster_issues(issues: list[DiagnosticIssue], nms_seconds: float,
                    ) -> list[list[DiagnosticIssue]]:
    """Group issues into clusters by type and temporal proximity.

    Uses the same greedy rule as the original sequential merge: issues of the
    same type whose ``start_time`` falls within ``nms_seconds`` of the running
    cluster end are folded into the current cluster. Returning explicit clusters
    lets the merge step pick a single representative window, so the resulting
    issue's span, severity, center_index, boxes, maps and evidence all describe
    one consistent source window.
    """
    ordered = sorted(issues, key=lambda i: (i.issue_type, i.start_time))
    clusters: list[list[DiagnosticIssue]] = []
    cluster_end: float = -1.0
    for issue in ordered:
        if clusters and clusters[-1][-1].issue_type == issue.issue_type and \
                issue.start_time - cluster_end <= nms_seconds:
            clusters[-1].append(issue)
            cluster_end = max(cluster_end, issue.end_time)
        else:
            clusters.append([issue])
            cluster_end = issue.end_time
    return clusters


def merge_issues(issues: list[DiagnosticIssue], nms_seconds: float) -> list[DiagnosticIssue]:
    """Merge same-type issues within ``nms_seconds`` (cluster → representative).

    Issues are first grouped into clusters by type and temporal proximity, then
    each cluster is collapsed to a single issue:

    * ``start_time``/``end_time`` — union of the whole cluster's time span.
    * ``severity``/``confidence`` — max over the cluster.
    * ``center_index``, ``boxes``, ``maps``, ``evidence`` — taken from one
      *representative* issue (the worst by ``(severity, confidence, evidence
      count)``), so these fields stay mutually consistent instead of being
      drawn from different source windows.
    * ``clip_paths`` — merged from every issue in the cluster.
    * ``thumbnail`` — taken from whichever issue has one.

    The merged issue also carries ``representative_center_index`` and
    ``supporting_center_indices`` so downstream consumers can see which window
    the spatial fields came from.
    """
    merged: list[DiagnosticIssue] = []
    for cluster in _cluster_issues(issues, nms_seconds):
        representative = max(
            cluster, key=lambda x: (x.severity, x.confidence, len(x.evidence)))
        # Merge clip_paths from all issues in the cluster.
        merged_clip_paths: dict[str, str] = {}
        for iss in cluster:
            merged_clip_paths.update(iss.clip_paths)
        # Take the thumbnail from whichever issue has one.
        thumbnail = ""
        for iss in cluster:
            if iss.thumbnail:
                thumbnail = iss.thumbnail
                break
        merged.append(DiagnosticIssue(
            issue_type=representative.issue_type, title=representative.title,
            severity=max(iss.severity for iss in cluster),
            confidence=max(iss.confidence for iss in cluster),
            start_time=min(iss.start_time for iss in cluster),
            end_time=max(iss.end_time for iss in cluster),
            track=representative.track, evidence=list(representative.evidence),
            probable_causes=list(representative.probable_causes),
            center_index=representative.center_index,
            boxes=list(representative.boxes), maps=list(representative.maps),
            clip_paths=merged_clip_paths, thumbnail=thumbnail,
            representative_center_index=representative.center_index,
            supporting_center_indices=[int(iss.center_index) for iss in cluster]))
    merged.sort(key=lambda i: -i.severity)
    return merged


def affected_duration_fraction(issues: list[DiagnosticIssue],
                               total_seconds: float) -> float:
    """Fraction of the timeline covered by any issue (union of spans)."""
    if total_seconds <= 0 or not issues:
        return 0.0
    spans = sorted((i.start_time, i.end_time) for i in issues)
    covered = 0.0
    cs, ce = -1.0, -1.0
    for s, e in spans:
        if s > ce:
            covered += max(0.0, ce - cs)
            cs, ce = s, e
        else:
            ce = max(ce, e)
    covered += max(0.0, ce - cs)
    return float(np.clip(covered / total_seconds, 0.0, 1.0))


def build_diagnostics_block(issues: list[DiagnosticIssue], meta,
                            confidence: float) -> dict:
    """Serialize a diagnosis for ``report.meta`` / the HTML report.

    ``meta`` only needs ``.n_frames`` and ``.fps`` (a VideoMeta), so this is
    reusable from every pipeline without importing them back.
    """
    from .schema import quality_level
    total_seconds = (float(meta.n_frames) / float(meta.fps)
                     if getattr(meta, "fps", 0) > 0 else 0.0)
    by_track: dict[str, int] = {}
    by_type: dict[str, int] = {}
    for issue in issues:
        by_track[issue.track] = by_track.get(issue.track, 0) + 1
        by_type[issue.issue_type] = by_type.get(issue.issue_type, 0) + 1
    worst = issues[0] if issues else None
    return {
        "issues": [i.to_dict() for i in issues],
        "issue_count": len(issues),
        "affected_duration_fraction": round(
            affected_duration_fraction(issues, total_seconds), 4),
        "total_seconds": round(total_seconds, 3),
        "by_track": by_track,
        "by_type": by_type,
        "worst_issue": worst.to_dict() if worst else None,
        "overall_confidence": round(float(confidence), 4),
        "quality_level": (quality_level(float("nan"))
                          if worst is None else None),
    }
