"""Source-video feature cache (USERPLAN.md §1, §13).

Everything that depends only on the 60 FPS source is computed once and reused
across all candidate models:

    cache/<content_hash>/flow_w<width>/
        pair_{i:06d}_flow.npz   f_01, f_10 (float16), occ/conf masks
        pair_{i:06d}_camera.npz global camera motion parameters
        edge_{j:06d}.npz        endpoint edge map + gradient energy

Total cost over N candidate models: T_source + N · T_candidate.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import cv2
import numpy as np

from ..config import EvalConfig
from ..io.video_reader import VideoReader
from ..motion.flow_estimator import FlowBackend, get_flow_backend, compute_pair_flow
from ..motion.occlusion import cycle_occlusion
from ..motion.global_camera_motion import estimate_camera_motion
from ..schema import CameraMotion, OcclusionMasks
from .cache_schema import is_valid, read_meta, write_meta
from .feature_store import FeatureStore


@dataclass
class SourcePairData:
    pair: int
    f_01: np.ndarray            # (Hf, Wf, 2) float32
    f_10: np.ndarray
    occ: OcclusionMasks
    camera: CameraMotion        # X_i -> X_{i+1}, native-resolution pixels
    flow_height: int
    flow_width: int
    scale: float                # flow res / native res

    @property
    def conf_visible_01(self) -> np.ndarray:
        return self.occ.conf_ab * (1.0 - self.occ.occ_ab.astype(np.float32))


@dataclass
class SourceEdges:
    frame: int
    edges: np.ndarray           # (Hf, Wf) uint8 {0,1}, multi-scale Canny union
    grad_energy: np.ndarray     # (Hf, Wf) float32, local gradient magnitude


class SourceCache:
    def __init__(self, reader: VideoReader, cfg: EvalConfig,
                 flow_width: int | None = None):
        self.reader = reader
        self.cfg = cfg
        self.flow_width = flow_width or cfg.preset.flow_width
        root = Path(cfg.cache_dir) / reader.meta.content_hash / f"flow_w{self.flow_width}"
        self.store = FeatureStore(root)
        self._backend: FlowBackend | None = None
        self._frames_at_flow_res: dict[int, np.ndarray] = {}

        meta = read_meta(root)
        if not is_valid(root) or meta.get("video_hash") != reader.meta.content_hash \
                or meta.get("flow_width") != self.flow_width:
            write_meta(root, {
                "video_hash": reader.meta.content_hash,
                "video_path": str(reader.meta.path),
                "flow_width": self.flow_width,
                "source_fps": reader.meta.fps,
                "source_frames": reader.meta.n_frames,
                "width": reader.meta.width,
                "height": reader.meta.height,
            })

    # -- lazy backend ---------------------------------------------------------

    @property
    def backend(self) -> FlowBackend:
        if self._backend is None:
            self._backend = get_flow_backend(self.cfg.flow_backend, self.cfg.device)
        return self._backend

    # -- frames ---------------------------------------------------------------

    def source_frame_flow_res(self, i: int) -> np.ndarray:
        """Source frame X_i at the flow working resolution (cached in memory)."""
        if i not in self._frames_at_flow_res:
            img = self.reader.read_one(i, width=self.flow_width)
            self._frames_at_flow_res[i] = img
            if len(self._frames_at_flow_res) > 48:  # bounded LRU-ish
                oldest = next(iter(self._frames_at_flow_res))
                del self._frames_at_flow_res[oldest]
        return self._frames_at_flow_res[i]

    # -- pairs ----------------------------------------------------------------

    def pair_key(self, i: int) -> str:
        return f"pair_{i:06d}"

    def has_pair(self, i: int) -> bool:
        return self.store.has(f"{self.pair_key(i)}_flow", "npz") \
            and self.store.has(f"{self.pair_key(i)}_camera", "npz")

    def ensure_pairs(self, pairs: Iterable[int],
                     progress: Callable[[int, int], None] | None = None,
                     skip: Iterable[int] = ()) -> None:
        """Compute flow/occlusion/camera for all missing pairs.

        `skip` is used for scene-cut pairs where flow is meaningless.
        """
        skip_set = set(skip)
        pairs = sorted({int(p) for p in pairs} - skip_set)
        todo = [p for p in pairs if not self.has_pair(p)]
        for n, i in enumerate(todo):
            self._compute_pair(i)
            if progress:
                progress(n + 1, len(todo))

    def _compute_pair(self, i: int) -> None:
        a = self.source_frame_flow_res(i)
        b = self.source_frame_flow_res(i + 1)
        f_01, f_10, scale = compute_pair_flow(self.backend, a, b, self.flow_width)
        occ = cycle_occlusion(f_01, f_10,
                              threshold_px=self.cfg.occlusion_cycle_threshold)
        camera = estimate_camera_motion(a, b)
        key = self.pair_key(i)
        self.store.save_npz(
            f"{key}_flow",
            f_01=f_01.astype(np.float16),
            f_10=f_10.astype(np.float16),
            occ_ab=occ.occ_ab,
            occ_ba=occ.occ_ba,
            conf_ab=occ.conf_ab.astype(np.float16),
            conf_ba=occ.conf_ba.astype(np.float16),
            meta=np.array([f_01.shape[0], f_01.shape[1], scale], np.float64),
        )
        self.store.save_npz(
            f"{key}_camera",
            matrix=camera.matrix,
            meta=np.array([{"identity": 0, "translation": 1, "partial_affine": 2,
                            "affine": 3, "homography": 4}[camera.model],
                           camera.inlier_ratio, camera.residual_px], np.float64),
        )

    def get_pair(self, i: int) -> SourcePairData:
        if not self.has_pair(i):
            self._compute_pair(i)
        fl = self.store.load_npz(f"{self.pair_key(i)}_flow")
        h, w, scale = (float(v) for v in fl["meta"])
        occ = OcclusionMasks(
            occ_ab=fl["occ_ab"].astype(np.uint8),
            occ_ba=fl["occ_ba"].astype(np.uint8),
            conf_ab=fl["conf_ab"].astype(np.float32),
            conf_ba=fl["conf_ba"].astype(np.float32),
        )
        cm = self.store.load_npz(f"{self.pair_key(i)}_camera")
        model_name = {0: "identity", 1: "translation", 2: "partial_affine",
                      3: "affine", 4: "homography"}[int(cm["meta"][0])]
        camera = CameraMotion(model_name, cm["matrix"].astype(np.float32),
                              float(cm["meta"][1]), float(cm["meta"][2]))
        return SourcePairData(
            pair=i,
            f_01=fl["f_01"].astype(np.float32),
            f_10=fl["f_10"].astype(np.float32),
            occ=occ,
            camera=camera,
            flow_height=int(h),
            flow_width=int(w),
            scale=float(scale),
        )

    # -- endpoint edges ---------------------------------------------------------

    def edge_key(self, j: int) -> str:
        return f"edge_{j:06d}"

    def get_edges(self, j: int) -> SourceEdges:
        key = self.edge_key(j)
        if not self.store.has(key, "npz"):
            img = self.source_frame_flow_res(j)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            e1 = cv2.Canny(gray, 60, 160)
            blur = cv2.GaussianBlur(gray, (0, 0), 1.0)
            e2 = cv2.Canny(blur, 30, 100)
            edges = ((e1 > 0) | (e2 > 0)).astype(np.uint8)
            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            grad = np.sqrt(gx * gx + gy * gy)
            self.store.save_npz(key, edges=edges, grad=grad.astype(np.float16))
        z = self.store.load_npz(key)
        return SourceEdges(frame=j, edges=z["edges"].astype(np.uint8),
                           grad_energy=z["grad"].astype(np.float32))
