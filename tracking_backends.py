"""
Pluggable camera-tracking backends — same pattern as depth_backbones.py.

A TrackingBackend turns RGB frames into a per-frame camera trajectory
(and, optionally, a 3D map). The dashboard pipeline calls .run() and
renders whatever the backend chose to produce.

Today the only registered backend is `ORBRansacBackend` (zero deps,
drifts on long clips). Future SOTA trackers (Endo-2DTAM, EndoGSLAM,
etc.) plug in by subclassing TrackingBackend and registering with
make_tracking_backend.

Use via:
    bb = make_tracking_backend("orb")
    result = bb.run(frames, hfov_deg=90.0)
"""

from dataclasses import dataclass, field  # noqa: F401
from typing import List, Optional, Sequence  # noqa: F401
import numpy as np


@dataclass
class TrackingResult:
    poses: List[np.ndarray]                # 4x4 c2w per frame, mm units
    scale_mm: float = 1.0                  # cumulative scale factor applied
    gs_map_path: Optional[str] = None      # path to .ply if backend made a map
    n_frames: int = 0
    note: str = ""                         # diagnostic note for the dashboard

    def __post_init__(self):
        if not self.n_frames and self.poses is not None:
            self.n_frames = len(self.poses)


class TrackingBackend:
    """Abstract base — every backend must implement run()."""

    name: str = "abstract"
    requires_depth_backbone: bool = False
    produces_gs_map: bool = False

    def run(self, frames: Sequence[np.ndarray], hfov_deg: float,
            depth_backbone=None,
            scale_per_frame_mm: Optional[Sequence[float]] = None,
            output_dir: Optional[str] = None) -> TrackingResult:
        raise NotImplementedError


class ORBRansacBackend(TrackingBackend):
    """ORB + RANSAC essential-matrix monocular VO. Zero new deps."""

    name = "orb"
    requires_depth_backbone = False
    produces_gs_map = False

    def __init__(self, n_features: int = 2000, min_matches: int = 30):
        self.n_features = n_features
        self.min_matches = min_matches

    def run(self, frames, hfov_deg, depth_backbone=None,
            scale_per_frame_mm=None, output_dir=None):
        from pose_estimation import MonocularVO  # lazy
        if not frames:
            return TrackingResult(poses=[], scale_mm=1.0, n_frames=0,
                                  note="no frames")
        H, W = frames[0].shape[:2]
        vo = MonocularVO(hfov_deg=hfov_deg, image_hw=(H, W),
                         n_features=self.n_features,
                         min_matches=self.min_matches)
        poses = vo.estimate(frames, scale_per_frame_mm=scale_per_frame_mm)
        return TrackingResult(
            poses=poses, scale_mm=1.0, n_frames=len(poses),
            note="ORB+RANSAC VO (drifts; placeholder until Endo-2DTAM)",
        )


# ---- factory ----

_REGISTRY = {
    ORBRansacBackend.name: ORBRansacBackend,
}


def make_tracking_backend(name: str, **kwargs) -> TrackingBackend:
    """Construct a backend by name. kwargs are forwarded to the constructor."""
    name = name.lower()
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown tracking backend {name!r}. "
            f"Available: {list(_REGISTRY)}.")
    return _REGISTRY[name](**kwargs)


def list_backends():
    return list(_REGISTRY.keys())
