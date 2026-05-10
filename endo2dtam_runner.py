"""
Endo-2DTAM integration — in-process call to their rgbd_slam().

Mirrors what configs/c3vd/c3vd_base.py + scripts/main.py do upstream,
but with a programmatic API instead of running their bash script.

Steps:
  1. Stage RGB + EndoDAC depth + identity pose.txt + intrinsics yaml in
     <repo>/data/<seq_name>/{color, depth, pose.txt}.
  2. Build the same nested config dict their c3vd_base.py builds, with
     workdir/run_name/intrinsics swapped for ours.
  3. sys.path-hack the cloned Endo-2DTAM repo, import scripts.main,
     call rgbd_slam(config) in-process. Same conda env.
  4. Read params.npz + means3D.npy. Convert
     (cam_unnorm_rots, cam_trans) into per-frame c2w. Save Gaussian
     centers as .ply for the dashboard.

Requires the Endo-2DTAM env (python 3.10, torch 1.13.1, plus
diff-surfel-rasterization and simple-knn). Install per their README;
this wrapper assumes those imports succeed.

Repo: https://github.com/lastbasket/Endo-2DTAM
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import cv2
import open3d as o3d


def _intrinsics_yaml(path: str, hfov_deg: float, image_hw,
                     png_depth_scale: float = 1000.0):
    """Write a per-sequence intrinsics yaml in Endo-2DTAM's format.

    png_depth_scale is the divisor that maps a stored depth value to
    meters. We save depth as uint16 mm, so meters = png / 1000.
    """
    H, W = image_hw
    fx = fy = W / (2.0 * np.tan(np.radians(hfov_deg) / 2.0))
    with open(path, "w") as f:
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


def _stage_data(frames: List[np.ndarray],
                depth_mm_per_frame: List[np.ndarray],
                repo_dir: str, seq_name: str,
                hfov_deg: float,
                png_depth_scale: float = 1000.0) -> Tuple[str, str]:
    """Write RGB + depth + identity pose.txt + intrinsics yaml into
    <repo>/data/<seq_name>/. Returns (sequence_dir, yaml_path)."""
    seq_dir = os.path.join(repo_dir, "data", seq_name)
    color_dir = os.path.join(seq_dir, "color")
    depth_dir = os.path.join(seq_dir, "depth")
    os.makedirs(color_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)

    H, W = frames[0].shape[:2]
    for i, (rgb, dep_mm) in enumerate(zip(frames, depth_mm_per_frame)):
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(color_dir, f"{i:05d}.png"), bgr)
        # Save depth as 16-bit TIFF (their dataset reader globs *.tiff).
        # Endo-2DTAM yaml interprets png_depth_scale as the divisor so
        # depth_meters = stored / scale. We store depth_mm directly and
        # set scale=1000 ⇒ meters = mm/1000. Round-trip-stable.
        d = np.clip(np.nan_to_num(dep_mm, nan=0.0), 0, 65535).astype(np.uint16)
        cv2.imwrite(os.path.join(depth_dir, f"{i:05d}.tiff"), d)

    # Identity pose.txt — Endo-2DTAM ignores values when use_gt_poses=False
    # but the file must exist with one matrix per frame.
    I = np.eye(4)
    with open(os.path.join(seq_dir, "pose.txt"), "w") as f:
        for _ in range(len(frames)):
            # Their loader does reshape(4,4).T after splitting on commas/whitespace.
            f.write(",".join(f"{v:.8f}" for v in I.T.reshape(-1).tolist()) + "\n")

    yaml_path = os.path.join(seq_dir, "_intrinsics.yaml")
    _intrinsics_yaml(yaml_path, hfov_deg, (H, W), png_depth_scale)
    return seq_dir, yaml_path


def _build_config(workdir: str, run_name: str, repo_dir: str,
                  yaml_path: str, seq_name: str,
                  image_hw, device: str = "cuda:0",
                  num_iters_tracking: int = 15,
                  num_iters_mapping: int = 15) -> Dict[str, Any]:
    """Mirror configs/c3vd/c3vd_base.py with our workdir + run_name +
    intrinsics. Tracking lr / mapping settings unchanged — they're tuned
    for endoscopy already by the upstream authors."""
    H, W = image_hw
    return dict(
        workdir=workdir,
        run_name=run_name,
        seed=0,
        primary_device=device,
        map_every=1,
        ba_every=100,
        keyframe_every=8,
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
            basedir=os.path.join(repo_dir, "data"),
            gradslam_data_cfg=yaml_path,
            sequence=seq_name,
            desired_image_height=H,
            desired_image_width=W,
            start=0, end=-1, stride=1, num_frames=-1,
            train_or_test="train",
            ignore_bad=False,
            use_train_split=True,
        ),
        tracking=dict(
            use_gt_poses=False, forward_prop=True,
            num_iters=num_iters_tracking,
            use_sil_for_loss=True, sil_thres=0.99,
            use_l1=True, ignore_outlier_depth_loss=False,
            loss_weights=dict(im=0.5, depth=1.0, point2plane=1.0),
            lrs=dict(means3D=0.0, rgb_colors=0.0, unnorm_rotations=0.0,
                     logit_opacities=0.0, log_scales=0.0,
                     cam_unnorm_rots=0.002, cam_trans=0.005,
                     exp_a=0.01, exp_b=0.01),
        ),
        mapping=dict(
            num_iters=num_iters_mapping, add_new_gaussians=True,
            sil_thres=0.5, use_l1=True, use_sil_for_loss=False,
            ignore_outlier_depth_loss=False,
            loss_weights=dict(im=1.0, depth=1.0, normal=1.0,
                              depth_dist=1000.0),
            lambda_dist=0.0, lambda_normal=0.05, opacity_cull=0.05,
            lrs=dict(means3D=0.0001, rgb_colors=0.005,
                     unnorm_rotations=0.001, logit_opacities=0.05,
                     log_scales=0.001, cam_unnorm_rots=0.0,
                     cam_trans=0.0, exp_a=0.001, exp_b=0.001),
            prune_gaussians=True,
            pruning_dict=dict(start_after=500, remove_big_after=0,
                              stop_after=20, prune_every=100,
                              removal_opacity_threshold=0.005,
                              final_removal_opacity_threshold=0.005,
                              reset_opacities=False,
                              reset_opacities_every=int(1e10)),
            use_gaussian_splatting_densification=False,
            densify_dict=dict(start_after=500, remove_big_after=3000,
                              stop_after=5000, densify_every=100,
                              grad_thresh=0.0002, num_to_split_into=2,
                              removal_opacity_threshold=0.005,
                              final_removal_opacity_threshold=0.005,
                              reset_opacities_every=3000),
        ),
        ba=dict(
            do_ba=False, num_iters=200, prune_gaussians=True,
            lrs=dict(means3D=0.0001, rgb_colors=0.005,
                     unnorm_rotations=0.001, logit_opacities=0.05,
                     log_scales=0.001, cam_unnorm_rots=0.002,
                     cam_trans=0.005, exp_a=0.001, exp_b=0.001),
            loss_weights=dict(im=1.0, depth=1.0, normal=1.0,
                              depth_dist=1000.0, point2plane=1.0),
            pruning_dict=dict(start_after=0, remove_big_after=0,
                              stop_after=50, prune_every=50,
                              removal_opacity_threshold=0.005,
                              final_removal_opacity_threshold=0.005,
                              reset_opacities=False,
                              reset_opacities_every=int(1e10)),
        ),
        viz=dict(render_mode='color', offset_first_viz_cam=True,
                 show_sil=False, visualize_cams=False,
                 viz_w=320, viz_h=320, viz_near=0.01, viz_far=100.0,
                 view_scale=2, viz_fps=30,
                 enter_interactive_post_online=True,
                 gaussian_simplification=False),
        gaussian_simplification=False,
        use_dep=True, use_normal=True, use_dam=False, dam_dep_scale=100,
    )


def _read_outputs(workdir: str, run_name: str
                  ) -> Tuple[List[np.ndarray], Optional[str]]:
    """Pull poses + Gaussian centers out of params.npz / means3D.npy and
    convert to (list-of-c2w, gs_map.ply path)."""
    run_dir = os.path.join(workdir, run_name)
    params_path = os.path.join(run_dir, "params.npz")
    means_path = os.path.join(run_dir, "means3D.npy")
    if not os.path.isfile(params_path):
        raise RuntimeError(
            f"Endo-2DTAM finished but {params_path} is missing — "
            f"check the rgbd_slam log for failures.")

    npz = np.load(params_path, allow_pickle=True)
    keys = npz.files
    if "cam_unnorm_rots" in keys and "cam_trans" in keys:
        rots = np.asarray(npz["cam_unnorm_rots"])
        trs = np.asarray(npz["cam_trans"])
        # Endo-2DTAM stores cam_unnorm_rots/cam_trans in shapes that
        # vary slightly across releases: (4, 1, T), (4, T), (T, 4), or
        # nested. Normalize to (T, 4) and (T, 3) by inspection.
        rots = np.squeeze(rots)
        trs = np.squeeze(trs)
        if rots.ndim == 2 and rots.shape[0] == 4 and rots.shape[1] != 4:
            rots = rots.T
        if trs.ndim == 2 and trs.shape[0] == 3 and trs.shape[1] != 3:
            trs = trs.T
        if rots.ndim != 2 or rots.shape[1] != 4:
            raise RuntimeError(
                f"Unexpected cam_unnorm_rots shape after squeeze: "
                f"{rots.shape}. Update _read_outputs() to match.")
        if trs.ndim != 2 or trs.shape[1] != 3:
            raise RuntimeError(
                f"Unexpected cam_trans shape after squeeze: {trs.shape}. "
                f"Update _read_outputs() to match.")
        rots = rots / np.linalg.norm(rots, axis=1, keepdims=True)

        from scipy.spatial.transform import Rotation as R
        poses: List[np.ndarray] = []
        for q, t in zip(rots, trs):
            # Endo-2DTAM stores quaternions as (w, x, y, z) per
            # build_rotation; scipy expects (x, y, z, w).
            w2c = np.eye(4, dtype=np.float64)
            w2c[:3, :3] = R.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
            w2c[:3, 3] = t
            poses.append(np.linalg.inv(w2c))
    elif "gt_w2c_all_frames" in keys:
        ws = npz["gt_w2c_all_frames"]
        poses = [np.linalg.inv(ws[i]) for i in range(len(ws))]
    else:
        raise RuntimeError(
            f"params.npz keys {keys}: couldn't find pose arrays. "
            "Update _read_outputs() to match.")

    # Endo-2DTAM ran with png_depth_scale=1000, so its world frame is in
    # METERS. The atlas (and every other organ-space consumer in this
    # repo) is in MILLIMETERS. Convert here so downstream code never has
    # to think about it. Rotations untouched; only translations scale.
    for p in poses:
        p[:3, 3] *= 1000.0

    # Save the Gaussian centers as a .ply for the dashboard's GPS panel.
    gs_ply: Optional[str] = None
    pts: Optional[np.ndarray] = None
    if os.path.isfile(means_path):
        pts = np.load(means_path)
    elif "means3D" in keys:
        pts = np.asarray(npz["means3D"])
    if pts is not None and len(pts) > 0:
        # Endo-2DTAM stores means3D as (3, N) sometimes; flip to (N, 3).
        if pts.shape[0] == 3 and pts.shape[1] != 3:
            pts = pts.T
        # Same meters->mm scaling as the poses above.
        pts = pts.astype(np.float64) * 1000.0
        gs_ply = os.path.join(run_dir, "endo2dtam_gs_map.ply")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        o3d.io.write_point_cloud(gs_ply, pcd)

    return poses, gs_ply


def run_endo2dtam(frames: List[np.ndarray],
                  depth_mm_per_frame: List[np.ndarray],
                  hfov_deg: float,
                  endo2dtam_repo: str,
                  output_dir: str,
                  device: str = "cuda:0",
                  num_iters_tracking: int = 15,
                  num_iters_mapping: int = 15,
                  cleanup: bool = True,
                  ) -> Tuple[List[np.ndarray], Optional[str], str]:
    """End-to-end: stage data, call rgbd_slam, return poses + gs_map_path.

    Returns (poses, gs_map_path, run_dir).
    """
    if not os.path.isdir(endo2dtam_repo):
        raise FileNotFoundError(
            f"Endo-2DTAM repo not found at {endo2dtam_repo}")
    if endo2dtam_repo not in sys.path:
        sys.path.insert(0, endo2dtam_repo)

    # Lazy-import their entry function. Triggers their package-level
    # imports (torch, gradslam, custom CUDA rasterizers); if the env
    # isn't set up, the error from here is the clearest signal.
    try:
        from scripts.main import rgbd_slam  # type: ignore
    except Exception as e:
        raise ImportError(
            f"Could not import scripts.main.rgbd_slam from "
            f"{endo2dtam_repo}. Install Endo-2DTAM's env per their "
            f"README (python 3.10 + torch 1.13.1 + "
            f"diff-surfel-rasterization + simple-knn).\n  {e}"
        ) from e

    seq_name = f"video_{int(time.time())}"
    print(f"[endo2dtam] staging {len(frames)} frames as sequence {seq_name}")
    seq_dir, yaml_path = _stage_data(
        frames, depth_mm_per_frame, endo2dtam_repo, seq_name,
        hfov_deg=hfov_deg)
    print(f"[endo2dtam] staged at {seq_dir}")

    workdir = os.path.abspath(output_dir)
    os.makedirs(workdir, exist_ok=True)
    H, W = frames[0].shape[:2]
    config = _build_config(
        workdir=workdir, run_name=seq_name,
        repo_dir=endo2dtam_repo, yaml_path=yaml_path, seq_name=seq_name,
        image_hw=(H, W), device=device,
        num_iters_tracking=num_iters_tracking,
        num_iters_mapping=num_iters_mapping,
    )

    # rgbd_slam expects to be invoked with the cwd at the repo root so
    # its relative paths in get_dataset / saving resolve correctly.
    saved_cwd = os.getcwd()
    try:
        os.chdir(endo2dtam_repo)
        print(f"[endo2dtam] running rgbd_slam(...) "
              f"(cwd={endo2dtam_repo})")
        rgbd_slam(config)
    finally:
        os.chdir(saved_cwd)

    poses, gs_ply = _read_outputs(workdir, seq_name)
    run_dir = os.path.join(workdir, seq_name)
    print(f"[endo2dtam] {len(poses)} poses; gs map: {gs_ply}")

    # ---- Disk cleanup ----------------------------------------------------
    # Endo-2DTAM stages a full RGB+depth copy under <repo>/data/<seq> and
    # writes large checkpoints (params.npz, means3D.npy) under <workdir>/
    # <seq>. Once we've extracted poses + the GS map .ply, the rest is
    # dead weight — multi-GB per run, will fill the disk fast on repeat
    # invocations. Move the .ply out, delete the rest.
    if cleanup:
        import shutil
        # 1. Drop the staged input dataset (color/, depth/, pose.txt, yaml)
        staged = os.path.join(endo2dtam_repo, "data", seq_name)
        if os.path.isdir(staged):
            shutil.rmtree(staged, ignore_errors=True)
            print(f"[endo2dtam] cleaned staged input: {staged}")
        # 2. Move the GS map .ply up one level so we can drop run_dir
        new_gs_ply = None
        if gs_ply and os.path.isfile(gs_ply):
            new_gs_ply = os.path.join(workdir, f"{seq_name}_gs_map.ply")
            try:
                shutil.move(gs_ply, new_gs_ply)
                gs_ply = new_gs_ply
            except Exception as e:
                print(f"[endo2dtam] couldn't relocate gs_map ({e}); "
                      f"keeping run_dir.")
                new_gs_ply = None
        # 3. Drop the entire run dir (checkpoints, eval, params.npz, etc.)
        if new_gs_ply is not None and os.path.isdir(run_dir):
            shutil.rmtree(run_dir, ignore_errors=True)
            print(f"[endo2dtam] cleaned run dir: {run_dir}")
            run_dir = workdir   # caller doesn't index into run_dir, just stores it

    return poses, gs_ply, run_dir
