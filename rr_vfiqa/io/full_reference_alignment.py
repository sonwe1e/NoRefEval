"""Same-rate reference ↔ candidate alignment for full-reference evaluation."""

from __future__ import annotations

import numpy as np

from ..schema import FullReferenceAlignment
from .video_reader import VideoReader

_DESCRIPTOR_WIDTH = 96
_RADIUS = 4


def _descriptors(reader: VideoReader) -> np.ndarray:
    # Process-level memo: routing and the pipeline share one full decode.
    from .video_reader import frame_descriptors

    return frame_descriptors(reader, width=_DESCRIPTOR_WIDTH)


def _monotonic_match(
    ref_desc: np.ndarray,
    cand_desc: np.ndarray,
    ref_pts: np.ndarray,
    cand_pts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Banded sequence alignment with explicit gaps on either timeline."""
    nr, nc = len(ref_desc), len(cand_desc)
    if nr == 0 or nc == 0:
        empty = np.zeros(0, np.int32)
        return empty, empty, np.zeros(0, np.float32)
    rt = ref_pts[:nr] - ref_pts[0]
    ct = cand_pts[:nc] - cand_pts[0]
    fps = 1.0 / max(float(np.median(np.diff(rt))) if nr > 1 else 1.0, 1e-6)

    states: dict[tuple[int, int], tuple[float, tuple[int, int] | None, float]] = {}
    for i in range(nr):
        nearest = int(np.clip(np.searchsorted(ct, rt[i]), 0, nc - 1))
        for k in range(max(0, nearest - _RADIUS), min(nc, nearest + _RADIUS + 1)):
            image_error = float(np.mean(np.abs(ref_desc[i] - cand_desc[k])))
            pts_frames = abs(float(ct[k] - rt[i])) * fps
            local = image_error + 3.0 * pts_frames
            best: tuple[float, tuple[int, int] | None, float] | None = None
            if i <= 1 and k <= 1:
                best = (local + 25.0 * (i + k), None, image_error)
            for di in (1, 2, 3):
                for dk in (1, 2, 3):
                    previous_key = (i - di, k - dk)
                    if previous_key not in states:
                        continue
                    previous = states[previous_key][0]
                    gap_penalty = 25.0 * ((di - 1) + (dk - 1))
                    total = previous + local + gap_penalty
                    if best is None or total < best[0]:
                        best = (total, previous_key, image_error)
            if best is not None:
                states[(i, k)] = best

    if not states:
        empty = np.zeros(0, np.int32)
        return empty, empty, np.zeros(0, np.float32)
    terminal = min(
        states,
        key=lambda key: states[key][0]
        + 25.0 * ((nr - 1 - key[0]) + (nc - 1 - key[1])),
    )
    pairs: list[tuple[int, int, float]] = []
    current: tuple[int, int] | None = terminal
    while current is not None:
        cost, previous, image_error = states[current]
        pairs.append((current[0], current[1], image_error))
        current = previous
    pairs.reverse()
    return (
        np.asarray([item[0] for item in pairs], np.int32),
        np.asarray([item[1] for item in pairs], np.int32),
        np.asarray([item[2] for item in pairs], np.float32),
    )


def build_full_reference_alignment(
    reference: VideoReader,
    candidate: VideoReader,
    *,
    scene_cuts_candidate: np.ndarray | None = None,
    geometry_policy: str = "strict",
    cand_desc: np.ndarray | None = None,
) -> FullReferenceAlignment:
    """Build and validate a one-to-one same-rate alignment.

    Small isolated candidate drops/duplicates are recorded and the affected
    intervals are excluded.  A different cadence, geometry, or insufficient
    match coverage fails closed rather than being treated as full reference.
    """
    rm, cm = reference.meta, candidate.meta
    warnings: list[str] = []
    ratio = cm.fps / max(rm.fps, 1e-6)
    if not (0.95 <= ratio <= 1.05):
        warnings.append(
            f"fps ratio {ratio:.3f} is not same-rate "
            f"(reference {rm.fps:.3f}, candidate {cm.fps:.3f})")
    ref_aspect = rm.width / max(rm.height, 1)
    cand_aspect = cm.width / max(cm.height, 1)
    same_geometry = rm.width == cm.width and rm.height == cm.height
    if geometry_policy not in (
            "strict", "resize-candidate", "common-resolution"):
        raise ValueError(f"unknown geometry policy: {geometry_policy!r}")
    geometry_ok = (
        same_geometry if geometry_policy == "strict"
        else abs(ref_aspect - cand_aspect) <= 0.005)
    if not geometry_ok:
        warnings.append(
            f"frame geometry rejected by {geometry_policy!r}: reference "
            f"{rm.width}x{rm.height}, candidate {cm.width}x{cm.height}")

    if cand_desc is None:
        cand_desc = _descriptors(candidate)
    ref_mapping, cand_mapping, errors = _monotonic_match(
        _descriptors(reference),
        cand_desc,
        rm.pts_seconds,
        cm.pts_seconds,
    )
    reference_of_candidate = np.full(cm.n_frames, -1, np.int32)
    candidate_of_reference = np.full(rm.n_frames, -1, np.int32)
    for i, k in zip(ref_mapping, cand_mapping):
        candidate_of_reference[int(i)] = int(k)
        reference_of_candidate[int(k)] = int(i)

    events: list[dict[str, int | str]] = []
    for ra, rb, ca, cb in zip(
        ref_mapping[:-1], ref_mapping[1:],
        cand_mapping[:-1], cand_mapping[1:],
    ):
        ref_gap = int(rb - ra)
        cand_gap = int(cb - ca)
        if ref_gap != 1 or cand_gap != 1:
            kind = (
                "candidate_drop" if ref_gap > cand_gap
                else "candidate_duplicate_or_gap"
            )
            events.append({
                "reference_from": int(ra),
                "reference_to": int(rb),
                "reference_gap": ref_gap,
                "candidate_from": int(ca),
                "candidate_to": int(cb),
                "candidate_gap": cand_gap,
                "type": kind,
            })
    matched_fraction = float(len(ref_mapping) / max(rm.n_frames, cm.n_frames, 1))
    image_error = float(np.median(errors)) if len(errors) else float("inf")
    max_events = max(2, int(round(0.05 * max(len(ref_mapping) - 1, 1))))
    # Same-rate FR is only valid for the same capture.  A uniformly large
    # low-resolution descriptor error means the DP merely matched two videos
    # with similar timing, not corresponding frames.
    content_ok = image_error <= 16.0
    reliable = (
        0.95 <= ratio <= 1.05
        and geometry_ok
        and content_ok
        and matched_fraction >= 0.95
        and len(events) <= max_events
    )
    if image_error > 12.0:
        warnings.append(
            f"median alignment descriptor error {image_error:.1f} is high; "
            "verify that both videos depict the same capture")
    if events:
        warnings.append(
            f"alignment recovered {len(events)} local gap event(s); "
            "affected windows are excluded")
    if not reliable:
        warnings.append("same-rate alignment is unreliable; evaluation must fail closed")

    return FullReferenceAlignment(
        reference_of_candidate=reference_of_candidate,
        candidate_of_reference=candidate_of_reference,
        fps_ratio=float(ratio),
        matched_fraction=matched_fraction,
        image_error=image_error,
        scene_cuts=np.asarray(
            scene_cuts_candidate if scene_cuts_candidate is not None else [],
            np.int32,
        ),
        warnings=warnings,
        events=events,
        reliable=reliable,
    )
