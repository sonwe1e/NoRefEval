"""Regression contracts for the current USERPLAN P0 reliability batch."""

from __future__ import annotations

from types import SimpleNamespace

import cv2
import numpy as np

from rr_vfiqa.calibration.provenance import (
    calibrator_provenance,
    file_sha256,
    report_provenance,
)
from rr_vfiqa.fusion.feature_registry import feature_contract_hash
from rr_vfiqa.fusion.monotonic_calibrator import MonotonicCalibrator
from rr_vfiqa.multimode import _metric_diagnostics
from rr_vfiqa.metrics.no_reference import (
    _camera_stabilize_tracks,
    _screen_static_ui_mask,
)
from rr_vfiqa.metrics.full_reference import _edge_metrics
from rr_vfiqa.sampling.full_reference_scan import scan_full_reference
from rr_vfiqa.schema import FullReferenceAlignment, Window, WindowFeatures


class _StreamingReader:
    def __init__(self, frames):
        self.frames = frames
        self.meta = SimpleNamespace(n_frames=len(frames))

    def iter_frames(self, width=None):
        for index, frame in enumerate(self.frames):
            yield index, frame

    def decode_all(self, width=None):
        raise AssertionError("full-reference scan must remain streaming")


def test_full_reference_scan_streams_and_uses_sobel_gradient():
    base = np.zeros((24, 32, 3), np.uint8)
    cv2.rectangle(base, (5, 5), (20, 18), (255, 255, 255), -1)
    shifted = np.roll(base, 1, axis=1)
    reference = _StreamingReader([base, base])
    candidate = _StreamingReader([base, shifted])
    alignment = FullReferenceAlignment(
        reference_of_candidate=np.asarray([0, 1], np.int32),
        candidate_of_reference=np.asarray([0, 1], np.int32),
        fps_ratio=1.0,
        matched_fraction=1.0,
    )
    scan = scan_full_reference(
        reference,
        candidate,
        alignment,
        width=32,
        target_size=(30, 20),
    )
    assert scan.y_l1[0] == 0.0
    assert scan.gradient_l1[1] > 0.0
    assert scan.per_frame_risk.shape == (2,)


def test_full_reference_one_sided_edges_have_finite_chamfer():
    reference = np.zeros((24, 32), np.uint8)
    cv2.rectangle(reference, (5, 5), (20, 18), 255, 1)
    candidate = np.zeros_like(reference)
    recall, precision, f1, chamfer = _edge_metrics(reference, candidate)
    assert recall == 0.0
    assert np.isfinite(chamfer)
    assert chamfer > 0.0


def test_metric_failures_are_aggregated_for_reports():
    wf = WindowFeatures(
        Window(2, np.asarray([0, 1, 2, 3, 4]), -1),
        labels={
            "metric_status": "failed",
            "metric_warnings": ["missing required metric feature(s): x"],
            "metric_coverage": 0.75,
            "metric_confidence": 0.0,
        },
    )
    diagnostics = _metric_diagnostics([wf])
    assert diagnostics["failed_windows"] == 1
    assert diagnostics["exception_windows"] == 0
    assert diagnostics["warning_count"] == 1
    assert diagnostics["mean_metric_coverage"] == 0.75
    assert diagnostics["samples"][0]["center_index"] == 2


def test_build_and_calibrator_hashes_are_recorded(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"dataset":"synthetic"}', encoding="utf-8")
    model = tmp_path / "calibrator.pkl"
    MonotonicCalibrator().save(
        model,
        meta={"training_manifest_path": str(manifest)},
    )
    loaded = MonotonicCalibrator.load(model)
    expected_contract = feature_contract_hash("endpoint-2x")
    contract = calibrator_provenance(
        model,
        meta=loaded.meta,
        expected_feature_contract_hash=expected_contract,
    )
    assert contract is not None
    assert contract["sha256"] == file_sha256(model)
    assert contract["feature_contract_hash"] == expected_contract
    assert contract["training_manifest_hash"] == file_sha256(manifest)

    report = report_provenance(
        mode="endpoint-2x",
        score_schema="endpoint-reduced-reference-v2",
        metric_contract="endpoint-metrics-v2",
        preset_contract="endpoint-fast-v1",
        feature_contract_hash=expected_contract,
        backend_contract={"flow": {"backend": "farneback"}},
    )
    build = report["build_contract"]
    assert build["source_version"] == "0.4.0"
    assert len(build["package_source_sha256"]) == 64


def test_camera_tracks_are_stabilized_for_rotation_and_scale():
    yy, xx = np.mgrid[0:4, 0:5].astype(np.float32)
    reference = np.stack(
        [20.0 + 18.0 * xx.ravel(), 15.0 + 16.0 * yy.ravel()],
        axis=1,
    )
    tracks = []
    center = np.asarray([60.0, 40.0])
    for angle, scale in ((0.0, 1.0), (4.0, 1.02), (8.0, 1.04), (12.0, 1.06)):
        radians = np.deg2rad(angle)
        rotation = np.asarray([
            [np.cos(radians), -np.sin(radians)],
            [np.sin(radians), np.cos(radians)],
        ])
        tracks.append(
            (reference - center) @ rotation.T * scale
            + center
            + np.asarray([2.0 * angle, -0.5 * angle]))
    stabilized = _camera_stabilize_tracks(np.asarray(tracks))
    error = np.linalg.norm(stabilized - stabilized[0], axis=2)
    assert np.percentile(error, 90) < 0.25


def test_ui_mask_covers_sides_and_compact_center_components():
    edge = np.zeros((80, 120), bool)
    edge[35:55, 3] = True
    edge[30:55, -4] = True
    edge[38:45, 56:65] = True
    maps = [edge.copy() for _ in range(5)]
    mask, _ = _screen_static_ui_mask(maps)
    assert mask[42, 3]
    assert mask[42, -4]
    assert mask[41, 60]
