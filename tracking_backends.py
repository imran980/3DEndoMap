"""
Pluggable camera-tracking backends — same pattern as depth_backbones.py.

A TrackingBackend turns RGB frames (+ optional depth) into a per-frame
camera trajectory and, optionally, a 3D map (TSDF mesh, Gaussian splats,
etc.). The dashboard pipeline calls .run() and then renders whatever
geometry the backend chose to produce.

Concrete backends:

  - ORBRansacBackend (default, zero-deps): wraps pose_estimation.MonocularVO.
    Drifts on long clips; no loop closure. Always-works smoke test.

  - Endo2DTAMBackend (recommended once installed): wraps Endo-2DTAM's
    tracking + 2D-Gaussian map for endoscopy. Clinical-grade quality,
    but requires:
        git clone https://github.com/lastbasket/Endo-2DTAM external/Endo-2DTAM
        # follow their README to set up their env + checkpoints

Use via:
    bb = make_tracking_backend("orb", hfov_deg=90.0)
    result = bb.run(frames)              # TrackingResult(poses, ...)

The TrackingResult shape is identical for every backend, so the
dashboard never needs to special-case which tracker produced the
trajectory.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Sequence
import importlib
import os
import sys
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


class Endo2DTAMBackend(TrackingBackend):
    """Endo-2DTAM (2D-Gaussian SLAM for endoscopy) — primary tracking + map.

    Lazy-imports from a local clone so the rest of the codebase stays
    runnable without it. The wrapper expects the upstream repo's
    inference API; if upstream changes shape, error messages here will
    point at the precise import that broke.

    Repo: https://github.com/lastbasket/Endo-2DTAM
    """

    name = "endo2dtam"
    requires_depth_backbone = True   # Endo-2DTAM uses depth as init prior
    produces_gs_map = True

    def __init__(self, repo_dir: str,
                 checkpoint_dir: Optional[str] = None,
                 device: Optional[str] = None,
                 config: Optional[str] = None):
        self.repo_dir = os.path.abspath(repo_dir)
        if not os.path.isdir(self.repo_dir):
            raise FileNotFoundError(
                f"Endo-2DTAM repo not found at {self.repo_dir}. Clone it: "
                f"git clone https://github.com/lastbasket/Endo-2DTAM "
                f"{self.repo_dir}")
        self.checkpoint_dir = checkpoint_dir
        self.device = device
        self.config = config
        if self.repo_dir not in sys.path:
            sys.path.insert(0, self.repo_dir)
        # Lazy probe — surface a clear error early if their package
        # layout doesn't match what this wrapper assumes.
        try:
            self._mod = importlib.import_module("endo2dtam")
        except Exception:
            try:
                self._mod = importlib.import_module("src")
            except Exception as e:
                raise ImportError(
                    f"Could not import Endo-2DTAM package from "
                    f"{self.repo_dir}. Inspect their repo layout and "
                    f"update Endo2DTAMBackend.__init__ accordingly.\n  {e}")

    def run(self, frames, hfov_deg, depth_backbone=None,
            scale_per_frame_mm=None, output_dir=None):
        # The actual run() is intentionally a stub for now: it requires
        # mapping our (frames, depth, intrinsics) onto Endo-2DTAM's
        # dataloader + inference loop, which is real integration work
        # rather than something we can guess at.
        #
        # When you're ready to fill this in, the upstream entrypoint to
        # mirror is their `evaluate_*.py` or `inference.py` (see their
        # README for the exact filename in the release you cloned).
        raise NotImplementedError(
            "Endo2DTAMBackend.run() is a stub. To wire it in:\n"
            "  1. Read external/Endo-2DTAM/README.md for the inference\n"
            "     entrypoint (likely evaluate.py or inference.py).\n"
            "  2. Build a dataset wrapper that feeds in our `frames` list\n"
            "     + intrinsics derived from --hfov.\n"
            "  3. Call their tracker, collect c2w per frame, save the\n"
            "     final 2D-GS map to `output_dir/endo2dtam_map.ply`.\n"
            "  4. Return TrackingResult(poses, scale_mm,\n"
            "     gs_map_path=<that .ply>, ...).\n"
            "Until then, run with --tracking orb."
        )


# ---- factory ----

_REGISTRY = {
    ORBRansacBackend.name: ORBRansacBackend,
    Endo2DTAMBackend.name: Endo2DTAMBackend,
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
