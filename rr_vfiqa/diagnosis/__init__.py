"""Diagnostic evidence engine (USERPLAN §7).

Turns the per-window scalar features that the metrics already compute into
structured, multi-evidence ``DiagnosticIssue`` objects with named probable
causes (always marked as *inferred*).  This is what moves the project from a
scorer to a diagnoser: the report shows *what is wrong and why*, not just a
number.

The engine is deliberately decoupled from the metrics: it consumes plain
``{feature: value}`` dicts, so it can be unit-tested with synthetic inputs and
re-used by every mode without touching the heavy pipelines.
"""

from .schema import DiagnosticIssue, Evidence, issue_level
from .rules import (
    TRACKS,
    affected_duration_fraction,
    build_diagnostics_block,
    confirmed_affected_seconds,
    diagnose_windows,
    merge_issues,
    preferred_maps_for,
    select_map,
    select_map_name,
)
from .cadence import (
    cadence_integrity,
    cadence_penalty,
    scan_phase_stats,
)

__all__ = [
    "DiagnosticIssue",
    "Evidence",
    "issue_level",
    "TRACKS",
    "diagnose_windows",
    "merge_issues",
    "preferred_maps_for",
    "select_map",
    "select_map_name",
    "affected_duration_fraction",
    "confirmed_affected_seconds",
    "build_diagnostics_block",
    "cadence_integrity",
    "cadence_penalty",
    "scan_phase_stats",
]
