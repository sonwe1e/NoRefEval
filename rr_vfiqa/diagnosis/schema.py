"""Structured diagnostic objects (USERPLAN §7)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# Severity bands used both for the timeline colour and the human-readable
# quality level.  These are *engineering risk* bands, never a claim about
# subjective MOS (USERPLAN §4).
SEVERITY_BANDS = (
    (0.75, "severe", "严重问题"),
    (0.45, "high", "明显问题"),
    (0.20, "medium", "有可见问题"),
    (0.00, "low", "轻微问题"),
)


def severity_band(severity: float) -> tuple[str, str]:
    """Return (level_key, label_zh) for a 0..1 severity value."""
    for threshold, key, label in SEVERITY_BANDS:
        if severity >= threshold:
            return key, label
    return "low", "轻微问题"


def quality_level(overall: float) -> tuple[str, str]:
    """Map an overall 0..100 score to an engineering risk level (USERPLAN §4)."""
    if overall != overall:  # NaN
        return "unknown", "无法判定"
    if overall >= 90:
        return "excellent", "很好"
    if overall >= 80:
        return "good", "基本良好"
    if overall >= 65:
        return "noticeable", "有可见问题"
    if overall >= 45:
        return "obvious", "明显问题"
    return "severe", "严重问题"


@dataclass
class Evidence:
    """One measurable signal supporting (or weakening) a diagnosis.

    ``strength`` is 0..1: how far the observed value sits past the rule's
    threshold.  ``direction`` records whether *higher* or *lower* values are
    the bad direction, so the report can phrase the evidence correctly.
    """

    metric: str
    observed: float
    threshold: float
    strength: float
    direction: str = "higher_is_worse"   # or "lower_is_worse"
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "observed": _r(self.observed),
            "threshold": _r(self.threshold),
            "strength": _r(self.strength),
            "direction": self.direction,
            "note": self.note,
        }


@dataclass
class DiagnosticIssue:
    """A located problem with the evidence that justifies it (USERPLAN §7).

    ``probable_causes`` are *inferred* root causes; the report must label them
    as such and never claim to know the model's true internals.
    """

    issue_type: str
    title: str
    severity: float
    confidence: float
    start_time: float
    end_time: float
    track: str
    evidence: list[Evidence] = field(default_factory=list)
    probable_causes: list[str] = field(default_factory=list)
    center_index: int = -1
    boxes: list[list[int]] = field(default_factory=list)
    maps: list[str] = field(default_factory=list)   # names of heatmaps in out/heatmaps/
    # USERPLAN §10: paths (relative to result root) to the companion videos and
    # keyframe thumbnail for this issue, populated by the report writer.
    clip_paths: dict[str, str] = field(default_factory=dict)
    # original / overlay / compare -> relative path
    thumbnail: str = ""   # keyframe with boxes drawn, relative path
    # P0-4: for merged issues, the center_index of the representative (worst)
    # window and the list of every center_index that was merged in. For a
    # standalone issue these default to the issue's own center_index / [] so
    # the dict output stays stable.
    representative_center_index: int = -1
    supporting_center_indices: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        band_key, band_label = severity_band(self.severity)
        return {
            "issue_type": self.issue_type,
            "title": self.title,
            "severity": _r(self.severity),
            "severity_band": band_key,
            "severity_label": band_label,
            "confidence": _r(self.confidence),
            "start_time": _r(self.start_time),
            "end_time": _r(self.end_time),
            "track": self.track,
            "center_index": int(self.center_index),
            "evidence": [e.to_dict() for e in self.evidence],
            "probable_causes": list(self.probable_causes),
            "boxes": [list(b) for b in self.boxes],
            "maps": list(self.maps),
            "clip_paths": dict(self.clip_paths),
            "thumbnail": self.thumbnail,
            "causes_are_inferred": True,
            "representative_center_index": int(self.representative_center_index),
            "supporting_center_indices": list(self.supporting_center_indices),
        }


def _r(v: float, n: int = 4):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f != f or f in (float("inf"), float("-inf")):
        return None
    return round(f, n)
