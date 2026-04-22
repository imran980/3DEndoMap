"""
Convert C3VD data to EndoNeRF format for Endo-4DGS training.

C3VD provides:
  - rgb/NNNN_color.png  — video frames
  - pose.txt            — 4x4 camera-to-world matrices (row-major, one per line)
  - depth/NNNN_depth.png — GT depth maps (16-bit, 0-100mm linear)
  - coverage_mesh.obj    — local 3D model

Endo-4DGS (EndoNeRF format) expects:
  - images/              — RGB frames  
  - masks/               — binary tool masks (all white for C3VD — no tools)
  - poses_bounds.npy     — LLFF format: Nx17 (3x5 pose flattened + near/far)
  - depth_dam/           — Depth-Anything pseudo-depths (generated separately)

LLFF poses_bounds.npy format:
  Each row = 17 floats:
    - 15 floats: 3x5 matrix [R|t|hwf] flattened column-major
      R = 3x3 rotation (camera-to-world), axes = [down, right, backwards]
      t = 3x1 translation
      hwf = [image_height, image_width, focal_length]
    - 2 floats: near_bound, far_bound

C3VD camera convention (after May 2023 revision):
  +x right, +y down, +z into screen (forward)

LLFF camera convention:
  Column 0 = down, Column 1 = right, Column 2 = backwards

Usage:
    python prepare_c3vd.py \
        --c3vd_dir /path/to/trans_t1_b \
        --organ_model /path/to/trans_model.obj \
        --output_dir data/c3vd/trans_t1_b
"""

import os
import sys
import glob
import json
import shutil
import numpy as np
import cv2
from argparse import ArgumentParser


def parse_c3vd_poses(pose_path):
    """
    Parse C3VD pose.txt.
    Each line: 16 floats = 4x4 camera-to-world homogeneous matrix (row-major).
    Returns list of 4x4 numpy arrays.
    """
    poses = []
    with open(pose_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # vals = [float(v) for v in line.split()]
            vals = [float(v) for v in line.replace(',', ' ').split()]
            if len(vals) == 16:
                # c2w = np.array(vals).reshape(4, 4)
                c2w = np.array(vals).reshape(4, 4).T
                poses.append(c2w)
            elif len(vals) == 12:
                c2w = np.eye(4)
                c2w[:3, :] = np.array(vals).reshape(3, 4)
                poses.append(c2w)
    return poses


def c3vd_to_llff_pose(c2w, H, W, focal):
    """
    Convert a C3VD 4x4 camera-to-world matrix to LLFF 3x5 format.
    
    C3VD axes: x=right, y=down, z=forward (into screen)
    LLFF axes: col0=down, col1=right, col2=backwards
    
    So:  LLFF_R[:,0] = C3VD_R[:,1]   (down = y)
         LLFF_R[:,1] = C3VD_R[:,0]   (right = x)  
         LLFF_R[:,2] = -C3VD_R[:,2]  (backwards = -z)
    """
    R = c2w[:3, :3]
    t = c2w[:3, 3]
    
    # Remap columns: [down, right, backwards] = [y, x, -z]
    R_llff = np.stack([R[:, 1], R[:, 0], -R[:, 2]], axis=1)
    
    # 3x5 = [R | t | hwf]
    hwf = np.array([H, W, focal]).reshape(3, 1)
    pose_3x5 = np.concatenate([R_llff, t.reshape(3, 1), hwf], axis=1)
    
    return pose_3x5


def build_poses_bounds(c2w_list, H, W, focal, near=0.01, far=100.0):
    """
    Build the Nx17 poses_bounds.npy array.
    
    Args:
        c2w_list: list of 4x4 camera-to-world matrices
        H, W: image dimensions
        focal: focal length in pixels
        near, far: depth bounds in scene units (mm for C3VD)
    """
    rows = []
    for c2w in c2w_list:
        pose_3x5 = c3vd_to_llff_pose(c2w, H, W, focal)
        # Flatten column-major (Fortran order), then append bounds
        row = np.concatenate([pose_3x5.ravel(order='F'), [near, far]])
        rows.append(row)
    return np.array(rows)  # Nx17


def prepare_c3vd(c3vd_dir, organ_model_path, output_dir,
                 focal_length=None, near=0.01, far=100.0,
                 skip_every=1, max_frames=None):
    """
    Convert C3VD sequence to EndoNeRF format for Endo-4DGS.
    """
    print("=" * 60)
    print("PREPARE C3VD → ENDONERF FORMAT")
    print("=" * 60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # ---- Find RGB frames ----
    rgb_dir = os.path.join(c3vd_dir, "rgb")
    if not os.path.isdir(rgb_dir):
        rgb_dir = c3vd_dir
    
    frames = sorted(glob.glob(os.path.join(rgb_dir, "*_color.png")))
    if not frames:
        frames = sorted(glob.glob(os.path.join(rgb_dir, "*.png")))
    if not frames:
        print(f"ERROR: No frames found in {rgb_dir}")
        sys.exit(1)
    
    print(f"Found {len(frames)} frames in {rgb_dir}")
    
    # ---- Parse poses ----
    pose_path = os.path.join(c3vd_dir, "pose.txt")
    if not os.path.exists(pose_path):
        print(f"ERROR: {pose_path} not found")
        sys.exit(1)
    
    all_poses = parse_c3vd_poses(pose_path)
    print(f"Parsed {len(all_poses)} poses")
    
    # Match frames to poses
    n = min(len(frames), len(all_poses))
    frames = frames[:n]
    all_poses = all_poses[:n]
    
    # Apply frame skipping
    if skip_every > 1:
        indices = list(range(0, n, skip_every))
        frames = [frames[i] for i in indices]
        all_poses = [all_poses[i] for i in indices]
        print(f"After skip_every={skip_every}: {len(frames)} frames")
    
    if max_frames and len(frames) > max_frames:
        indices = np.linspace(0, len(frames)-1, max_frames, dtype=int)
        frames = [frames[i] for i in indices]
        all_poses = [all_poses[i] for i in indices]
        print(f"Capped to {len(frames)} frames")
    
    n = len(frames)
    
    # ---- Get image dimensions ----
    sample = cv2.imread(frames[0])
    H, W = sample.shape[:2]
    print(f"Image size: {W}x{H}")
    
    # ---- Estimate focal length ----
    if focal_length is None:
        # C3VD colonoscope (Olympus CF-HQ190L) has wide FOV ~140-170 degrees.
        # For the rendered/registered images, approximate with pinhole.
        # f ≈ W * 0.5 gives ~90 degree hFOV — reasonable starting point.
        # Adjust if reconstruction quality is poor.
        focal_length = W * 0.5
        print(f"  Auto focal length: {focal_length:.1f} px "
              f"(~{2*np.degrees(np.arctan(W/(2*focal_length))):.0f}° hFOV)")
        print(f"  TIP: For better accuracy, extract from C3VD calibration file")
    else:
        print(f"  Focal length: {focal_length:.1f} px")
    
    # ---- Copy RGB frames ----
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    print(f"\nCopying {n} frames...")
    for i, src in enumerate(frames):
        dst = os.path.join(images_dir, f"frame_{i:05d}.png")
        shutil.copy2(src, dst)
    
    # ---- Create masks (all white — no surgical tools in C3VD phantoms) ----
    masks_dir = os.path.join(output_dir, "masks")
    os.makedirs(masks_dir, exist_ok=True)
    
    print(f"Creating {n} blank masks")
    white_mask = np.ones((H, W), dtype=np.uint8) * 255
    for i in range(n):
        cv2.imwrite(os.path.join(masks_dir, f"frame_{i:05d}.png"), white_mask)
    
    # ---- Build poses_bounds.npy ----
    print(f"\nBuilding poses_bounds.npy")
    print(f"  Focal={focal_length:.1f}, Near={near}, Far={far}")
    
    poses_bounds = build_poses_bounds(all_poses, H, W, focal_length, near, far)
    
    pb_path = os.path.join(output_dir, "poses_bounds.npy")
    np.save(pb_path, poses_bounds)
    print(f"  Saved: {pb_path}  shape={poses_bounds.shape}")
    
    # ---- Copy C3VD depth maps for reference ----
    depth_src = os.path.join(c3vd_dir, "depth")
    if os.path.isdir(depth_src):
        depth_dst = os.path.join(output_dir, "depth_gt")
        os.makedirs(depth_dst, exist_ok=True)
        depth_files = sorted(glob.glob(os.path.join(depth_src, "*.png")))
        copied = 0
        for i in range(n):
            src_idx = i * skip_every if skip_every > 1 else i
            if src_idx < len(depth_files):
                shutil.copy2(depth_files[src_idx],
                           os.path.join(depth_dst, f"frame_{i:05d}_depth.png"))
                copied += 1
        print(f"Copied {copied} GT depth maps")
    
    # ---- Copy organ model ----
    if organ_model_path and os.path.exists(organ_model_path):
        shutil.copy2(organ_model_path, os.path.join(output_dir, "organ_model.obj"))
        mtl = organ_model_path.replace('.obj', '.mtl')
        if os.path.exists(mtl):
            shutil.copy2(mtl, os.path.join(output_dir, "organ_model.mtl"))
        print(f"Copied organ model")
    
    coverage = os.path.join(c3vd_dir, "coverage_mesh.obj")
    if os.path.exists(coverage):
        shutil.copy2(coverage, os.path.join(output_dir, "coverage_mesh.obj"))
    
    # ---- Save metadata + trajectory ----
    positions = np.array([p[:3, 3] for p in all_poses])
    total_dist = np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))
    
    meta = {
        "source": "C3VD", "sequence": os.path.basename(c3vd_dir),
        "n_frames": n, "image_size": [W, H],
        "focal_length": focal_length, "near": near, "far": far,
        "camera_path_length_mm": float(total_dist),
    }
    with open(os.path.join(output_dir, "c3vd_meta.json"), 'w') as f:
        json.dump(meta, f, indent=2)
    
    traj = {"source": "C3VD", "n_frames": n,
            "image_width": W, "image_height": H, "frames": []}
    for i, pose in enumerate(all_poses):
        traj["frames"].append({
            "frame_id": i, "c2w": pose.tolist(),
            "w2c": np.linalg.inv(pose).tolist(),
            "position": pose[:3, 3].tolist(),
            "time": float(i) / max(n - 1, 1),
        })
    with open(os.path.join(output_dir, "camera_trajectory.json"), 'w') as f:
        json.dump(traj, f, indent=2)
    
    # ---- Summary ----
    print(f"\n{'='*60}")
    print(f"  Camera X: [{positions[:,0].min():.1f}, {positions[:,0].max():.1f}] mm")
    print(f"  Camera Y: [{positions[:,1].min():.1f}, {positions[:,1].max():.1f}] mm")
    print(f"  Camera Z: [{positions[:,2].min():.1f}, {positions[:,2].max():.1f}] mm")
    print(f"  Path length: {total_dist:.1f} mm")
    
    print(f"\n{'='*60}")
    print("OUTPUT STRUCTURE")
    print(f"{'='*60}")
    print(f"  {output_dir}/")
    print(f"  ├── images/              ({n} frames)")
    print(f"  ├── masks/               ({n} blank masks)")
    print(f"  ├── poses_bounds.npy     ({poses_bounds.shape})")
    print(f"  ├── camera_trajectory.json")
    print(f"  ├── organ_model.obj")
    print(f"  └── c3vd_meta.json")
    
    print(f"\n{'='*60}")
    print("NEXT: Run the commands below in your Endo-4DGS directory")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Convert C3VD to EndoNeRF format")
    parser.add_argument("--c3vd_dir", required=True,
                        help="Path to unzipped C3VD sequence (e.g. trans_t1_b/)")
    parser.add_argument("--organ_model", required=True,
                        help="Path to organ OBJ (e.g. trans_model.obj)")
    parser.add_argument("--output_dir", default="data/c3vd/trans_t1_b",
                        help="Output directory")
    parser.add_argument("--focal", default=None, type=float,
                        help="Focal length in pixels (default: auto)")
    parser.add_argument("--near", default=0.01, type=float)
    parser.add_argument("--far", default=100.0, type=float)
    parser.add_argument("--skip_every", default=1, type=int,
                        help="Use every N-th frame")
    parser.add_argument("--max_frames", default=None, type=int)
    args = parser.parse_args()
    
    prepare_c3vd(
        c3vd_dir=args.c3vd_dir,
        organ_model_path=args.organ_model,
        output_dir=args.output_dir,
        focal_length=args.focal,
        near=args.near, far=args.far,
        skip_every=args.skip_every,
        max_frames=args.max_frames,
    )
