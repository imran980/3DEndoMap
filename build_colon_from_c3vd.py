"""
Build a colon mesh directly from C3VD ground-truth depth + poses.

This bypasses Endo-4DGS entirely and uses the per-frame 16-bit depth PNGs
+ 4x4 camera-to-world matrices that C3VD ships. Useful as:
  1. A reference of what the dynamic-organ pipeline *should* look like
     once the trained Gaussian model produces good depth, and
  2. A standalone way to get a real, anatomically correct colon mesh
     from any C3VD sequence in seconds.

Usage:
    python build_colon_from_c3vd.py \
        --c3vd_dir dataset/trans_t1_b \
        --output_dir output/colon_gt_trans \
        --voxel_size 0.5            # mm; 0.3 for crisper, 1.0 for faster
"""

import os
import sys
import glob
import json
import numpy as np
import cv2
import open3d as o3d
from argparse import ArgumentParser
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare_c3vd import parse_c3vd_poses


def build(c3vd_dir, output_dir, voxel_size=0.5, depth_trunc=120.0,
          hfov_deg=140.0, skip_every=1, max_frames=None):
    os.makedirs(output_dir, exist_ok=True)

    # ---- Load images, poses, GT depths ----
    rgb_dir = os.path.join(c3vd_dir, "rgb")
    if not os.path.isdir(rgb_dir):
        rgb_dir = c3vd_dir
    rgb_paths = sorted(glob.glob(os.path.join(rgb_dir, "*_color.png")))
    if not rgb_paths:
        rgb_paths = sorted(glob.glob(os.path.join(rgb_dir, "*.png")))
    depth_paths = sorted(glob.glob(os.path.join(c3vd_dir, "depth", "*.png")))
    pose_path = os.path.join(c3vd_dir, "pose.txt")

    if not depth_paths:
        sys.exit(f"ERROR: no depth PNGs under {c3vd_dir}/depth")
    if not os.path.exists(pose_path):
        sys.exit(f"ERROR: missing {pose_path}")

    poses = parse_c3vd_poses(pose_path)  # list of 4x4 c2w
    n = min(len(rgb_paths), len(depth_paths), len(poses))
    rgb_paths, depth_paths, poses = (
        rgb_paths[:n], depth_paths[:n], poses[:n])

    if skip_every > 1:
        idx = list(range(0, n, skip_every))
        rgb_paths = [rgb_paths[i] for i in idx]
        depth_paths = [depth_paths[i] for i in idx]
        poses = [poses[i] for i in idx]
    if max_frames and len(rgb_paths) > max_frames:
        idx = np.linspace(0, len(rgb_paths)-1, max_frames, dtype=int)
        rgb_paths = [rgb_paths[i] for i in idx]
        depth_paths = [depth_paths[i] for i in idx]
        poses = [poses[i] for i in idx]
    n = len(rgb_paths)

    sample = cv2.imread(rgb_paths[0])
    H, W = sample.shape[:2]
    fx = fy = W / (2.0 * np.tan(np.radians(hfov_deg) / 2.0))
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        W, H, fx, fy, W / 2.0, H / 2.0)

    print(f"Frames: {n}  Image: {W}x{H}  hFOV: {hfov_deg}° (fx={fx:.1f})")
    print(f"TSDF voxel: {voxel_size} mm  depth_trunc: {depth_trunc} mm")

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_size,
        sdf_trunc=4.0 * voxel_size,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    # OpenCV / Open3D camera convention: +x right, +y down, +z forward.
    # C3VD post-2023 poses are also +x right, +y down, +z forward, so c2w
    # can be used directly. Pass extrinsic = world->camera = inv(c2w).
    fused = 0
    for rgb_p, dep_p, c2w in tqdm(list(zip(rgb_paths, depth_paths, poses)),
                                  desc="TSDF fuse"):
        rgb = cv2.cvtColor(cv2.imread(rgb_p), cv2.COLOR_BGR2RGB)
        if rgb.shape[:2] != (H, W):
            rgb = cv2.resize(rgb, (W, H))

        dep16 = cv2.imread(dep_p, cv2.IMREAD_UNCHANGED)
        if dep16 is None:
            continue
        if dep16.shape[:2] != (H, W):
            dep16 = cv2.resize(dep16, (W, H), interpolation=cv2.INTER_NEAREST)
        # C3VD: 16-bit linear, full range = 100 mm
        depth_mm = dep16.astype(np.float32) * (100.0 / 65535.0)
        depth_mm[depth_mm <= 0.5] = 0.0
        depth_mm[depth_mm >= depth_trunc] = 0.0
        if not np.any(depth_mm > 0):
            continue

        color_o3d = o3d.geometry.Image(np.ascontiguousarray(rgb))
        depth_o3d = o3d.geometry.Image(np.ascontiguousarray(depth_mm))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d, depth_o3d,
            depth_scale=1.0, depth_trunc=depth_trunc,
            convert_rgb_to_intensity=False)

        w2c = np.linalg.inv(c2w).astype(np.float64)
        volume.integrate(rgbd, intrinsic, w2c)
        fused += 1

    print(f"\nFused {fused}/{n} frames. Extracting mesh...")
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    print(f"Mesh: {len(mesh.vertices):,} verts, {len(mesh.triangles):,} tris")

    mesh_path = os.path.join(output_dir, "colon_mesh_gt.ply")
    o3d.io.write_triangle_mesh(mesh_path, mesh)
    print(f"Saved: {mesh_path}")

    pcd = volume.extract_point_cloud()
    pcd_path = os.path.join(output_dir, "colon_pointcloud_gt.ply")
    o3d.io.write_point_cloud(pcd_path, pcd)
    print(f"Saved: {pcd_path}  ({len(pcd.points):,} pts)")

    meta = {
        "source": "C3VD GT depth + poses",
        "n_frames_fused": fused,
        "voxel_size_mm": voxel_size,
        "depth_trunc_mm": depth_trunc,
        "hfov_deg": hfov_deg,
        "image_size": [W, H],
        "fx": float(fx),
    }
    with open(os.path.join(output_dir, "build_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    p = ArgumentParser(description="Fuse C3VD GT depth + poses into a colon mesh")
    p.add_argument("--c3vd_dir", required=True,
                   help="Path to a C3VD sequence (must contain rgb/, depth/, pose.txt)")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--voxel_size", default=0.5, type=float,
                   help="TSDF voxel size in mm")
    p.add_argument("--depth_trunc", default=120.0, type=float,
                   help="Truncate depth values beyond this (mm)")
    p.add_argument("--hfov", default=140.0, type=float,
                   help="Horizontal FOV (deg) — match prepare_c3vd")
    p.add_argument("--skip_every", default=1, type=int)
    p.add_argument("--max_frames", default=None, type=int)
    a = p.parse_args()
    build(a.c3vd_dir, a.output_dir,
          voxel_size=a.voxel_size, depth_trunc=a.depth_trunc,
          hfov_deg=a.hfov, skip_every=a.skip_every, max_frames=a.max_frames)
