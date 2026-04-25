"""
Build a colon mesh from RGB + camera poses using Depth-Anything-V2 for
per-frame depth — *no* trained Endo-4DGS required.

Two depth modes:

  --variant metric_indoor_b   uses the metric DAv2 (output is meters,
                              we convert to mm). Works on any sequence,
                              quality depends on indoor-trained prior
                              transferring to colonoscopy.

  --variant vitb              uses relative DAv2 (disparity output) and
                              fits scale per-frame against C3VD GT depth
                              if --calib_dir is given. Highest quality
                              when GT is available.

For non-C3VD data, pass --variant metric_indoor_b.

Usage:
    python build_colon_from_dav2.py \
        --c3vd_dir dataset/trans_t1_b \
        --output_dir output/colon_dav2_trans \
        --variant metric_indoor_b
"""

import os
import sys
import glob
import json
import numpy as np
import cv2
import open3d as o3d
import torch
from tqdm import tqdm
from argparse import ArgumentParser

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare_c3vd import parse_c3vd_poses
from build_colon_from_c3vd import _find_pairs, _pose_to_organ_space
from dav2_depth import DepthAnythingV2, fit_disparity_to_depth, disparity_to_depth


def _read_c3vd_depth_mm(path, target_hw=None):
    """Decode a C3VD depth file (uint16 PNG/TIFF or float32 TIFF) into mm."""
    raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if raw is None:
        return None
    if raw.ndim == 3:
        raw = raw[..., 0]
    if target_hw is not None and raw.shape[:2] != target_hw:
        raw = cv2.resize(raw, (target_hw[1], target_hw[0]),
                         interpolation=cv2.INTER_NEAREST)
    if raw.dtype in (np.float32, np.float64):
        return raw.astype(np.float32)
    return raw.astype(np.float32) * (100.0 / 65535.0)


def build(c3vd_dir, output_dir, voxel_size=0.5, depth_trunc=120.0,
          hfov_deg=140.0, variant="metric_indoor_b",
          skip_every=1, max_frames=None, align_to_organ=True,
          calibration_frames=20):
    os.makedirs(output_dir, exist_ok=True)

    pairs_with_gt = _find_pairs(c3vd_dir)
    rgbs_alone = sorted(glob.glob(os.path.join(c3vd_dir, "*_color.png"))) \
        or sorted(glob.glob(os.path.join(c3vd_dir, "rgb", "*_color.png"))) \
        or sorted(glob.glob(os.path.join(c3vd_dir, "rgb", "*.png"))) \
        or sorted(glob.glob(os.path.join(c3vd_dir, "*.png")))
    rgb_paths = [p[0] for p in pairs_with_gt] if pairs_with_gt else rgbs_alone
    gt_depth_paths = ([p[1] for p in pairs_with_gt]
                      if pairs_with_gt else [None] * len(rgb_paths))
    if not rgb_paths:
        sys.exit(f"ERROR: no RGB frames found under {c3vd_dir}")

    pose_path = os.path.join(c3vd_dir, "pose.txt")
    if not os.path.exists(pose_path):
        sys.exit(f"ERROR: missing {pose_path}")
    poses = parse_c3vd_poses(pose_path)
    if align_to_organ:
        poses = [_pose_to_organ_space(p) for p in poses]

    n = min(len(rgb_paths), len(poses))
    rgb_paths, gt_depth_paths, poses = (
        rgb_paths[:n], gt_depth_paths[:n], poses[:n])
    if skip_every > 1:
        idx = list(range(0, n, skip_every))
        rgb_paths = [rgb_paths[i] for i in idx]
        gt_depth_paths = [gt_depth_paths[i] for i in idx]
        poses = [poses[i] for i in idx]
    if max_frames and len(rgb_paths) > max_frames:
        idx = np.linspace(0, len(rgb_paths) - 1, max_frames, dtype=int)
        rgb_paths = [rgb_paths[i] for i in idx]
        gt_depth_paths = [gt_depth_paths[i] for i in idx]
        poses = [poses[i] for i in idx]
    n = len(rgb_paths)

    sample = cv2.imread(rgb_paths[0])
    H, W = sample.shape[:2]
    fx = fy = W / (2.0 * np.tan(np.radians(hfov_deg) / 2.0))
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        W, H, fx, fy, W / 2.0, H / 2.0)

    print(f"Frames: {n}  Image: {W}x{H}  hFOV: {hfov_deg}°  "
          f"variant: {variant}")
    print(f"TSDF voxel: {voxel_size} mm  depth_trunc: {depth_trunc} mm")

    print("Loading Depth-Anything-V2...")
    dav2 = DepthAnythingV2(variant=variant)

    # --- Calibration (only needed for relative-depth variants) ---
    a_avg, b_avg = None, None
    if not dav2.is_metric:
        gt_indices = [i for i, p in enumerate(gt_depth_paths) if p]
        if not gt_indices:
            sys.exit("ERROR: relative variant requires GT depth for "
                     "calibration. Either pass --variant metric_indoor_b or "
                     "use a sequence with C3VD-style depth files.")
        cal_idx = gt_indices[::max(1, len(gt_indices) // calibration_frames)] \
            [:calibration_frames]
        a_list, b_list = [], []
        for i in tqdm(cal_idx, desc="DAv2 calibration"):
            rgb = cv2.cvtColor(cv2.imread(rgb_paths[i]), cv2.COLOR_BGR2RGB)
            pred = dav2.predict(rgb)
            gt_mm = _read_c3vd_depth_mm(gt_depth_paths[i], target_hw=(H, W))
            a, b = fit_disparity_to_depth(pred, gt_mm)
            if a is not None:
                a_list.append(a)
                b_list.append(b)
        if not a_list:
            sys.exit("ERROR: calibration failed on all frames.")
        a_avg = float(np.median(a_list))
        b_avg = float(np.median(b_list))
        print(f"Calibration on {len(a_list)} frames: a={a_avg:.3f}, b={b_avg:.3f}")

    # --- TSDF fusion ---
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_size,
        sdf_trunc=4.0 * voxel_size,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )
    fused = 0
    depth_min = 0.5
    metric_scale_to_mm = 1000.0  # default: meters -> mm
    first_logged = False
    for f_idx, (rgb_p, c2w) in enumerate(
            tqdm(list(zip(rgb_paths, poses)), desc="TSDF fuse (DAv2)")):
        rgb = cv2.cvtColor(cv2.imread(rgb_p), cv2.COLOR_BGR2RGB)
        if rgb.shape[:2] != (H, W):
            rgb = cv2.resize(rgb, (W, H))
        pred = dav2.predict(rgb)
        if dav2.is_metric:
            depth_mm = pred * metric_scale_to_mm
        else:
            depth_mm = disparity_to_depth(pred, a_avg, b_avg)
        depth_mm = depth_mm.astype(np.float32)
        depth_mm[~np.isfinite(depth_mm)] = 0.0

        if not first_logged:
            valid = depth_mm[depth_mm > 0]
            pred_stats = (float(pred.min()), float(pred.max()),
                          float(np.median(pred)))
            dmm_stats = ((float(valid.min()), float(valid.max()),
                          float(np.median(valid))) if valid.size else
                         (0.0, 0.0, 0.0))
            print(f"\n  First frame DAv2 pred:    "
                  f"min={pred_stats[0]:.3f}, "
                  f"max={pred_stats[1]:.3f}, "
                  f"median={pred_stats[2]:.3f}")
            print(f"  First frame depth (mm):  "
                  f"min={dmm_stats[0]:.2f}, "
                  f"max={dmm_stats[1]:.2f}, "
                  f"median={dmm_stats[2]:.2f}")
            # If metric output is clearly out of clipping range, auto-rescale
            if dav2.is_metric and dmm_stats[1] > depth_trunc * 5:
                ratio = (depth_trunc * 0.5) / max(dmm_stats[2], 1e-6)
                metric_scale_to_mm *= ratio
                depth_mm = (pred * metric_scale_to_mm).astype(np.float32)
                depth_mm[~np.isfinite(depth_mm)] = 0.0
                print(f"  WARNING: metric DAv2 output is way larger than "
                      f"depth_trunc={depth_trunc} mm. Auto-rescaling: "
                      f"new metric_scale = {metric_scale_to_mm:.4f} "
                      f"(prefer --variant vitb with GT calibration for "
                      f"quality).")
            first_logged = True

        depth_mm[depth_mm <= depth_min] = 0.0
        depth_mm[depth_mm >= depth_trunc] = 0.0
        if not np.any(depth_mm > 0):
            continue
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.ascontiguousarray(rgb)),
            o3d.geometry.Image(np.ascontiguousarray(depth_mm)),
            depth_scale=1.0, depth_trunc=depth_trunc,
            convert_rgb_to_intensity=False)
        w2c = np.linalg.inv(c2w).astype(np.float64)
        volume.integrate(rgbd, intrinsic, w2c)
        fused += 1

    if fused == 0:
        print("\nERROR: 0 frames were fused — every frame's depth was "
              "outside [{}, {}] mm.".format(depth_min, depth_trunc))
        print("  Most likely the DAv2 variant doesn't match the scene scale.")
        print("  Re-run with: --variant vitb --calibration_frames 30  "
              "(needs GT depth files alongside RGB).")
        sys.exit(1)

    print(f"\nFused {fused}/{n} frames. Extracting mesh...")
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    mesh_path = os.path.join(output_dir, "colon_mesh_dav2.ply")
    o3d.io.write_triangle_mesh(mesh_path, mesh)
    print(f"Mesh: {len(mesh.vertices):,} verts -> {mesh_path}")

    pcd_path = os.path.join(output_dir, "colon_pointcloud_dav2.ply")
    o3d.io.write_point_cloud(pcd_path, volume.extract_point_cloud())
    print(f"Point cloud: {pcd_path}")

    meta = {
        "source": "Depth-Anything-V2 (HF) + C3VD poses",
        "variant": variant,
        "n_frames_fused": fused,
        "voxel_size_mm": voxel_size,
        "depth_trunc_mm": depth_trunc,
        "hfov_deg": hfov_deg,
        "image_size": [W, H], "fx": float(fx),
        "calibration": (None if dav2.is_metric
                        else {"a_median": a_avg, "b_median": b_avg,
                              "n_calibration_frames": calibration_frames}),
        "align_to_organ": align_to_organ,
    }
    with open(os.path.join(output_dir, "build_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    p = ArgumentParser(description="Fuse RGB + DAv2 depth + poses into a mesh")
    p.add_argument("--c3vd_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--variant", default="metric_indoor_b",
                   choices=["vits", "vitb", "vitl",
                            "metric_indoor_s", "metric_indoor_b",
                            "metric_indoor_l"],
                   help="DAv2 variant. metric_* outputs meters directly; "
                        "vits/b/l are relative and need GT for calibration.")
    p.add_argument("--voxel_size", default=0.5, type=float)
    p.add_argument("--depth_trunc", default=120.0, type=float)
    p.add_argument("--hfov", default=140.0, type=float)
    p.add_argument("--skip_every", default=1, type=int)
    p.add_argument("--max_frames", default=None, type=int)
    p.add_argument("--no_align_to_organ", action="store_true")
    p.add_argument("--calibration_frames", default=20, type=int)
    a = p.parse_args()
    build(a.c3vd_dir, a.output_dir,
          voxel_size=a.voxel_size, depth_trunc=a.depth_trunc,
          hfov_deg=a.hfov, variant=a.variant,
          skip_every=a.skip_every, max_frames=a.max_frames,
          align_to_organ=not a.no_align_to_organ,
          calibration_frames=a.calibration_frames)
