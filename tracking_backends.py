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
    """Endo-2DTAM (ICRA 2025) — surface-normal-aware Gaussian SLAM for endoscopy.

    Per-sequence optimizer (NOT a pretrained inference model). Reads
    `color/`, `depth/`, `pose.txt` from disk, runs tracking + 2D-Gaussian
    mapping, writes `params.npz` (refined poses + Gaussian params) and
    `means3D.npy` (Gaussian centers) into `experiments/<workdir>/<run_name>/`.

    Repo: https://github.com/lastbasket/Endo-2DTAM (ICRA 2025)

    Why this is invoked via subprocess:
    Endo-2DTAM ships pinned to python 3.10 + torch 1.13 + custom CUDA
    rasterizers (diff-surfel-rasterization, simple-knn). Forcing those
    pins into our existing dashboard env breaks EndoDAC. We instead call
    their `scripts/main.py` with their own python interpreter and read
    the artifacts back from disk. Two envs, one repo, no conflict.
    """

    name = "endo2dtam"
    requires_depth_backbone = True
    produces_gs_map = True

    def __init__(self, repo_dir: str,
                 checkpoint_dir: Optional[str] = None,
                 python_bin: Optional[str] = None,
                 device: Optional[str] = None,
                 config: Optional[str] = None):
        self.repo_dir = os.path.abspath(repo_dir)
        if not os.path.isdir(self.repo_dir):
            raise FileNotFoundError(
                f"Endo-2DTAM repo not found at {self.repo_dir}. Clone it: "
                f"git clone https://github.com/lastbasket/Endo-2DTAM "
                f"{self.repo_dir}")
        # python_bin = path to the Endo-2DTAM env's python interpreter.
        # We default to whatever's on $PATH (assumes the user activated the
        # endo2dtam conda env when launching us; subprocess inherits it).
        self.python_bin = python_bin or sys.executable
        self.device = device or "cuda:0"
        self.config_template = config

    def _stage_inputs(self, frames, hfov_deg, depth_backbone,
                      output_dir, seq_name, png_depth_scale=2.55):
        """Write `color/`, `depth/`, and `pose.txt` in Endo-2DTAM's
        expected layout under <repo>/data/<seq>/. Depth comes from the
        provided depth backbone (EndoDAC).
        """
        import cv2
        seq_dir = os.path.join(self.repo_dir, "data", seq_name)
        color_dir = os.path.join(seq_dir, "color")
        depth_dir = os.path.join(seq_dir, "depth")
        os.makedirs(color_dir, exist_ok=True)
        os.makedirs(depth_dir, exist_ok=True)

        H, W = frames[0].shape[:2]
        for i, rgb in enumerate(frames):
            cv2.imwrite(os.path.join(color_dir, f"{i:04d}_color.png"),
                        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            if depth_backbone is not None:
                # Backbone returns disparity; for Endo-2DTAM we need a
                # proper depth-PNG. The caller is responsible for
                # passing in a backbone whose .predict() returns a
                # depth-like value. If no GT-mm calibration was done,
                # we save the raw values scaled to fit uint16.
                pred = depth_backbone.predict(rgb).astype(np.float32)
                # Treat pred as already-mm if backbone is metric, else
                # use a 1/disparity heuristic anchored to ~30 mm median.
                if not getattr(depth_backbone, "is_metric", False):
                    eps = 1e-6
                    inv = 1.0 / np.clip(pred, eps, None)
                    inv_med = float(np.median(inv))
                    if inv_med > 0:
                        depth_mm = inv * (30.0 / inv_med)
                    else:
                        depth_mm = pred
                else:
                    depth_mm = pred * 1000.0
                depth_u16 = np.clip(depth_mm * png_depth_scale, 0, 65535) \
                    .astype(np.uint16)
                cv2.imwrite(os.path.join(depth_dir, f"{i:04d}_depth.png"),
                            depth_u16)

        # pose.txt: 16 floats per line, identity per frame (Endo-2DTAM
        # ignores the values when use_gt_poses=False — it just needs the
        # file to exist and have the right line count).
        I = np.eye(4)
        with open(os.path.join(seq_dir, "pose.txt"), "w") as f:
            for _ in range(len(frames)):
                f.write(" ".join(f"{v:.8f}" for v in I.flatten()) + "\n")

        # Per-sequence intrinsic YAML
        fx = fy = W / (2.0 * np.tan(np.radians(hfov_deg) / 2.0))
        yaml_path = os.path.join(seq_dir, "_intrinsics.yaml")
        with open(yaml_path, "w") as f:
            f.write(
                "dataset_name: 'c3vd'\n"
                "camera_params:\n"
                f"  image_height: {H}\n"
                f"  image_width: {W}\n"
                f"  fx: {fx:.4f}\n"
                f"  fy: {fy:.4f}\n"
                f"  cx: {W/2.0:.4f}\n"
                f"  cy: {H/2.0:.4f}\n"
                f"  png_depth_scale: {png_depth_scale}\n"
                f"  crop_edge: 0\n"
            )
        return seq_dir, yaml_path

    def _write_config_py(self, seq_name, yaml_path, output_dir):
        """Generate the experiment config module Endo-2DTAM expects."""
        cfg_path = os.path.join(output_dir, f"_endo2dtam_{seq_name}.py")
        cfg_text = f"""\
config = dict(
    workdir={output_dir!r},
    run_name={seq_name!r},
    seed=0,
    primary_device={self.device!r},
    map_every=1, ba_every=100, keyframe_every=8,
    distance_keyframe_selection=True,
    distance_current_frame_prob=0.1,
    mapping_window_size=-1,
    report_global_progress_every=2000,
    scene_radius_depth_ratio=3,
    mean_sq_dist_method="projective",
    report_iter_progress=False,
    load_checkpoint=False,
    checkpoint_time_idx=0,
    save_checkpoints=False,
    checkpoint_interval=int(1e10),
    data=dict(
        basedir={os.path.join(self.repo_dir, 'data')!r},
        gradslam_data_cfg={yaml_path!r},
        sequence={seq_name!r},
        desired_image_height=None,
        desired_image_width=None,
        start=0, end=-1, stride=1, num_frames=-1,
        train_or_test="train",
    ),
    tracking=dict(num_iters=15, use_gt_poses=False, forward_prop=True),
    mapping=dict(num_iters=15, add_new_gaussians=True),
    ba=dict(do_ba=False, num_iters=200),
    viz=dict(render_mode='color', gaussian_simplification=False),
    use_dep=True, use_normal=True, use_dam=False, dam_dep_scale=100,
)
"""
        with open(cfg_path, "w") as f:
            f.write(cfg_text)
        return cfg_path

    def _read_outputs(self, output_dir, seq_name, n_frames):
        """Pull the refined poses + Gaussian centers from Endo-2DTAM's
        outputs and convert them into our TrackingResult fields."""
        run_dir = os.path.join(output_dir, seq_name)
        params_path = os.path.join(run_dir, "params.npz")
        means_path = os.path.join(run_dir, "means3D.npy")
        if not os.path.isfile(params_path):
            raise RuntimeError(
                f"Endo-2DTAM finished but {params_path} is missing — "
                f"check the subprocess log for failures.")

        npz = np.load(params_path, allow_pickle=True)
        # The convention from their save code: per-frame camera poses in
        # 'cam_unnorm_rots' + 'cam_trans', or a flat 'gt_w2c_all_frames'.
        # Try the names we expect, fall back gracefully.
        if "cam_unnorm_rots" in npz.files and "cam_trans" in npz.files:
            from scipy.spatial.transform import Rotation as R
            quats = npz["cam_unnorm_rots"]   # Nx4 (or T x 4)
            ts    = npz["cam_trans"]         # Nx3
            quats = np.asarray(quats); ts = np.asarray(ts)
            quats = quats / np.linalg.norm(quats, axis=1, keepdims=True)
            poses = []
            for q, t in zip(quats, ts):
                w2c = np.eye(4)
                # Endo-2DTAM stores quats as (w,x,y,z) per their convention
                w2c[:3, :3] = R.from_quat(
                    [q[1], q[2], q[3], q[0]]).as_matrix()
                w2c[:3, 3] = t
                poses.append(np.linalg.inv(w2c))
        elif "gt_w2c_all_frames" in npz.files:
            w2cs = npz["gt_w2c_all_frames"]
            poses = [np.linalg.inv(w2cs[i]) for i in range(len(w2cs))]
        else:
            raise RuntimeError(
                f"params.npz keys: {list(npz.files)} — couldn't find pose "
                f"arrays. Tell me the keys to update Endo2DTAMBackend.")

        # Save the Gaussian centers as a .ply for the dashboard
        gs_ply = os.path.join(run_dir, "endo2dtam_map.ply")
        try:
            import open3d as o3d
            if os.path.isfile(means_path):
                pts = np.load(means_path)
            else:
                pts = npz["means3D"] if "means3D" in npz.files else None
            if pts is not None and len(pts) > 0:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(np.asarray(pts))
                o3d.io.write_point_cloud(gs_ply, pcd)
            else:
                gs_ply = None
        except Exception as e:
            print(f"[endo2dtam] couldn't write GS map .ply ({e}); "
                  f"dashboard will fall back to TSDF.")
            gs_ply = None
        return poses, gs_ply

    def run(self, frames, hfov_deg, depth_backbone=None,
            scale_per_frame_mm=None, output_dir=None):
        import subprocess, time
        if depth_backbone is None:
            raise ValueError(
                "Endo2DTAMBackend.run() needs a depth_backbone (e.g. "
                "EndoDAC) — Endo-2DTAM consumes depth as input.")
        if output_dir is None:
            raise ValueError(
                "Endo2DTAMBackend.run() needs an output_dir to stage data "
                "+ collect outputs.")

        os.makedirs(output_dir, exist_ok=True)
        seq_name = f"video_{int(time.time())}"
        print(f"[endo2dtam] staging {len(frames)} frames as sequence "
              f"{seq_name}...")
        _, yaml_path = self._stage_inputs(
            frames, hfov_deg, depth_backbone, output_dir, seq_name)
        cfg_path = self._write_config_py(seq_name, yaml_path, output_dir)
        print(f"[endo2dtam] config: {cfg_path}")
        print(f"[endo2dtam] running scripts/main.py via {self.python_bin} ...")

        cmd = [self.python_bin, "scripts/main.py", cfg_path]
        proc = subprocess.run(cmd, cwd=self.repo_dir)
        if proc.returncode != 0:
            raise RuntimeError(
                f"Endo-2DTAM exited with status {proc.returncode}. "
                f"Re-run manually: cd {self.repo_dir} && "
                f"{self.python_bin} scripts/main.py {cfg_path}")

        print(f"[endo2dtam] reading outputs ...")
        poses, gs_ply = self._read_outputs(output_dir, seq_name, len(frames))
        return TrackingResult(
            poses=poses, scale_mm=1.0, n_frames=len(poses),
            gs_map_path=gs_ply,
            note=f"Endo-2DTAM (subprocess via {self.python_bin})",
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
