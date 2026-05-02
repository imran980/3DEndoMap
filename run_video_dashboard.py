"""
End-to-end dashboard from a single endoscopic video file.

Takes an arbitrary MP4 (or any cv2-readable video), extracts frames,
estimates per-frame camera poses with monocular VO, then runs the
existing render_navigation_c3vd dashboard. No GT depth, no pre-op
mesh, no external tracker required.

What it does:
  1. ffmpeg-style frame extraction via cv2.VideoCapture.
  2. Monocular VO (ORB + RANSAC essential-matrix) -> per-frame c2w.
  3. Stages frames + pose.txt in a synthetic "C3VD-shaped" folder.
  4. Calls render_navigation_c3vd.run() with --mode dynamic by default
     (no organ mesh = TSDF-fused live mesh + cyan trajectory polyline).

Usage:
  python run_video_dashboard.py \
      --video my_bronchoscopy.mp4 \
      --output_dir output/my_bronchus \
      --hfov 90 \
      --backbone endodac \
      --endodac_repo external/EndoDAC \
      --endodac_weights external/EndoDAC/EndoDAC_fullmodel/depth_model.pth

Optional:
  --organ_mesh airway_tree.ply --mode reveal     # if you have a CT mesh
  --max_frames 400                               # subsample long videos
  --skip_every 2                                 # every other frame
"""

import os
import sys
import shutil
from argparse import ArgumentParser, Namespace
import numpy as np
import cv2
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pose_estimation import MonocularVO, write_pose_txt
import render_navigation_c3vd as rnc


def extract_frames(video_path, frames_dir, max_frames=None, skip_every=1):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.exit(f"ERROR: cannot open video {video_path}")
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"Video: {n_total} frames @ {fps:.1f} fps")

    os.makedirs(frames_dir, exist_ok=True)
    written = 0
    idx_in = 0
    pbar = tqdm(total=n_total, desc="Extracting frames")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        pbar.update(1)
        if idx_in % skip_every == 0:
            cv2.imwrite(os.path.join(frames_dir, f"{written:05d}_color.png"),
                        frame)
            written += 1
            if max_frames and written >= max_frames:
                break
        idx_in += 1
    pbar.close()
    cap.release()
    return written, fps


def estimate_poses(frames_dir, hfov_deg):
    paths = sorted(p for p in os.listdir(frames_dir)
                   if p.endswith("_color.png"))
    if len(paths) < 2:
        sys.exit("ERROR: need at least 2 frames for VO.")

    sample = cv2.imread(os.path.join(frames_dir, paths[0]))
    H, W = sample.shape[:2]
    vo = MonocularVO(hfov_deg=hfov_deg, image_hw=(H, W))

    print(f"Loading {len(paths)} frames into memory for VO ({W}x{H})...")
    frames = []
    for p in tqdm(paths, desc="Reading"):
        bgr = cv2.imread(os.path.join(frames_dir, p))
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    print("Running monocular VO (ORB + RANSAC)...")
    poses = vo.estimate(frames)

    pose_path = os.path.join(frames_dir, "pose.txt")
    write_pose_txt(poses, pose_path)
    print(f"Wrote {len(poses)} poses -> {pose_path}")
    return pose_path, (H, W)


def run(args):
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    frames_dir = os.path.join(out_dir, "video_frames")

    # 1. extract
    if args.reuse_frames and os.path.isdir(frames_dir) and \
            any(p.endswith("_color.png") for p in os.listdir(frames_dir)):
        print(f"Reusing existing frames in {frames_dir}")
    else:
        if os.path.isdir(frames_dir):
            shutil.rmtree(frames_dir)
        n, _ = extract_frames(args.video, frames_dir,
                              max_frames=args.max_frames,
                              skip_every=args.skip_every)
        print(f"Wrote {n} frames -> {frames_dir}")

    # 2. poses (skip if user supplied a pose.txt)
    if args.poses_file:
        shutil.copy2(args.poses_file,
                     os.path.join(frames_dir, "pose.txt"))
        print(f"Using user-provided poses: {args.poses_file}")
    elif not (args.reuse_frames
              and os.path.isfile(os.path.join(frames_dir, "pose.txt"))):
        estimate_poses(frames_dir, hfov_deg=args.hfov)
    else:
        print("Reusing existing pose.txt")

    # 3. hand off to the standard dashboard
    if args.mode in ("coverage", "reveal") and not args.organ_mesh:
        print(f"WARNING: --mode {args.mode} requires --organ_mesh; "
              f"falling back to 'dynamic'.")
        args.mode = "dynamic"

    dash_args = Namespace(
        c3vd_dir=frames_dir,
        output_dir=out_dir,
        mode=args.mode,
        organ_mesh=args.organ_mesh,
        backbone=args.backbone,
        variant=args.variant,
        endodac_repo=args.endodac_repo,
        endodac_weights=args.endodac_weights,
        hfov=args.hfov,
        fps=args.fps,
        skip_every=1,                # already applied during extraction
        max_frames=None,
        calibration_frames=args.calibration_frames,
        assumed_median_depth_mm=args.assumed_median_depth_mm,
        depth_trunc=args.depth_trunc,
        min_disparity_pct=args.min_disparity_pct,
        depth_smooth_ksize=args.depth_smooth_ksize,
        voxel_size=args.voxel_size,
        mesh_update_every=args.mesh_update_every,
        no_trajectory_align=True,    # no organ mesh by default; ICP NOOP
        trajectory_align_max_corr=80.0,
        tsdf_depth_min=args.tsdf_depth_min,
        tsdf_edge_margin_pct=args.tsdf_edge_margin_pct,
        tsdf_grad_thresh=args.tsdf_grad_thresh,
        tsdf_min_camera_motion_mm=args.tsdf_min_camera_motion_mm,
        tsdf_keep_top_components=args.tsdf_keep_top_components,
        tsdf_min_component_triangles=args.tsdf_min_component_triangles,
    )
    if args.organ_mesh:
        dash_args.no_trajectory_align = False

    print("\n=== Handing off to render_navigation_c3vd ===")
    rnc.run(dash_args)


if __name__ == "__main__":
    p = ArgumentParser(description="Run the dashboard on a raw endoscopy video")
    p.add_argument("--video", required=True, help="Input video file (mp4, mov, ...)")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--mode", default="dynamic",
                   choices=["coverage", "reveal", "dynamic"])
    p.add_argument("--organ_mesh", default=None,
                   help="Optional pre-op organ mesh (.ply/.obj) for "
                        "coverage/reveal modes. Required if --mode != dynamic.")
    p.add_argument("--hfov", default=90.0, type=float,
                   help="Horizontal FOV (deg). Default 90 for bronchoscopes; "
                        "use 140 for colonoscopes.")
    p.add_argument("--fps", default=30, type=int,
                   help="Output dashboard video FPS")
    p.add_argument("--max_frames", default=None, type=int)
    p.add_argument("--skip_every", default=1, type=int,
                   help="Take every Nth frame from the video.")
    p.add_argument("--reuse_frames", action="store_true",
                   help="Skip frame extraction / VO if outputs exist.")
    p.add_argument("--poses_file", default=None,
                   help="Skip VO and use this pre-computed pose.txt.")

    # Depth backbone (forwarded)
    p.add_argument("--backbone", default="endodac",
                   choices=["dav2", "endodac"])
    p.add_argument("--variant", default="vitb")
    p.add_argument("--endodac_repo", default=None)
    p.add_argument("--endodac_weights", default=None)

    # Calibration / depth filters
    p.add_argument("--calibration_frames", default=20, type=int)
    p.add_argument("--assumed_median_depth_mm", default=20.0, type=float)
    p.add_argument("--depth_trunc", default=80.0, type=float)
    p.add_argument("--min_disparity_pct", default=10.0, type=float)
    p.add_argument("--depth_smooth_ksize", default=5, type=int)

    # TSDF cleanup
    p.add_argument("--voxel_size", default=0.5, type=float)
    p.add_argument("--mesh_update_every", default=15, type=int)
    p.add_argument("--tsdf_depth_min", default=5.0, type=float)
    p.add_argument("--tsdf_edge_margin_pct", default=4.0, type=float)
    p.add_argument("--tsdf_grad_thresh", default=8.0, type=float)
    p.add_argument("--tsdf_min_camera_motion_mm", default=0.5, type=float)
    p.add_argument("--tsdf_keep_top_components", default=3, type=int)
    p.add_argument("--tsdf_min_component_triangles", default=200, type=int)

    args = p.parse_args()
    run(args)
