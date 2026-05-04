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
from tracking_backends import make_tracking_backend
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


def estimate_poses(frames_dir, hfov_deg, tracking="orb", output_dir=None,
                   endo2dtam_repo=None, endo2dtam_checkpoints=None):
    """Run a tracking backend over the frames in `frames_dir`.

    Returns (pose_path, image_hw, gs_map_path).
    """
    paths = sorted(p for p in os.listdir(frames_dir)
                   if p.endswith("_color.png"))
    if len(paths) < 2:
        sys.exit("ERROR: need at least 2 frames for tracking.")

    sample = cv2.imread(os.path.join(frames_dir, paths[0]))
    H, W = sample.shape[:2]

    print(f"Loading {len(paths)} frames into memory for tracking ({W}x{H})...")
    frames = []
    for p in tqdm(paths, desc="Reading"):
        bgr = cv2.imread(os.path.join(frames_dir, p))
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    if tracking == "orb":
        bb = make_tracking_backend("orb")
    elif tracking == "endo2dtam":
        if not endo2dtam_repo:
            sys.exit("ERROR: --tracking endo2dtam requires --endo2dtam_repo.")
        bb = make_tracking_backend(
            "endo2dtam",
            repo_dir=endo2dtam_repo,
            checkpoint_dir=endo2dtam_checkpoints,
        )
    else:
        sys.exit(f"Unknown --tracking value: {tracking}")

    print(f"Running tracking backend ({bb.name})...")
    result = bb.run(frames, hfov_deg=hfov_deg, output_dir=output_dir)
    print(f"Tracking result: {result.n_frames} poses ({result.note})")

    pose_path = os.path.join(frames_dir, "pose.txt")
    write_pose_txt(result.poses, pose_path)
    print(f"Wrote {len(result.poses)} poses -> {pose_path}")
    return pose_path, (H, W), result.gs_map_path


def _resolve_endodac_paths(args):
    """Auto-fill --endodac_repo / --endodac_weights from the conventional
    locations when the user didn't specify them. Errors clearly if the
    backbone is set to endodac but no weights can be located."""
    if args.backbone != "endodac":
        return
    here = os.path.dirname(os.path.abspath(__file__))
    candidates_repo = [
        args.endodac_repo,
        os.path.join(here, "external", "EndoDAC"),
        "external/EndoDAC",
    ]
    candidates_w = [
        args.endodac_weights,
        os.path.join(here, "external", "EndoDAC", "EndoDAC_fullmodel",
                     "depth_model.pth"),
        "external/EndoDAC/EndoDAC_fullmodel/depth_model.pth",
    ]
    if not args.endodac_repo:
        for c in candidates_repo[1:]:
            if c and os.path.isdir(c):
                args.endodac_repo = c
                print(f"Auto-detected --endodac_repo {c}")
                break
    if not args.endodac_weights:
        for c in candidates_w[1:]:
            if c and os.path.isfile(c):
                args.endodac_weights = c
                print(f"Auto-detected --endodac_weights {c}")
                break
    if not args.endodac_repo or not args.endodac_weights:
        sys.exit(
            "ERROR: --backbone endodac requires both --endodac_repo and "
            "--endodac_weights, and they couldn't be auto-detected at "
            "external/EndoDAC/...\n"
            "Either pass them explicitly, or run with --backbone dav2 "
            "(no checkpoints needed; lower quality on endoscopy)."
        )


def run(args):
    _resolve_endodac_paths(args)
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    frames_dir = os.path.join(out_dir, "video_frames")

    # ---- Atlas: build a procedural bronchial-tree mesh when the user
    # didn't supply a patient-specific one, so the GPS panel always has
    # an anatomically credible canvas. The trajectory ICP inside
    # render_navigation_c3vd then snaps the camera path onto it.
    using_atlas = False
    if args.organ_mesh is None and args.atlas == "procedural":
        from bronchus_atlas import write_atlas
        atlas_path, branches = write_atlas(out_dir)
        args.organ_mesh = atlas_path
        using_atlas = True
        print(f"[atlas] generated procedural bronchial tree: "
              f"{len(branches)} branches -> {atlas_path}")
        if args.mode == "dynamic":
            # Atlas is most useful as a reveal canvas. Switch unless the
            # user explicitly asked for the live TSDF mesh.
            print("[atlas] --mode dynamic was set; keeping it because the "
                  "atlas is informational only. Pass --mode reveal to "
                  "paint the atlas in as the camera moves through it.")

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
    gs_map_path = None
    if args.poses_file:
        shutil.copy2(args.poses_file,
                     os.path.join(frames_dir, "pose.txt"))
        print(f"Using user-provided poses: {args.poses_file}")
    elif not (args.reuse_frames
              and os.path.isfile(os.path.join(frames_dir, "pose.txt"))):
        _, _, gs_map_path = estimate_poses(
            frames_dir, hfov_deg=args.hfov,
            tracking=args.tracking, output_dir=out_dir,
            endo2dtam_repo=args.endo2dtam_repo,
            endo2dtam_checkpoints=args.endo2dtam_checkpoints,
        )
        if gs_map_path:
            print(f"Tracker also produced a 3D map: {gs_map_path}")
    else:
        print("Reusing existing pose.txt")

    # If no organ mesh and no atlas, coverage / reveal aren't valid.
    if args.mode in ("coverage", "reveal") and not args.organ_mesh:
        print(f"WARNING: --mode {args.mode} requires --organ_mesh; "
              f"falling back to 'dynamic'.")
        args.mode = "dynamic"
    # When using the atlas, default mode flips to reveal so the tree
    # paints in as the camera moves through it (the cleanest demo).
    if using_atlas and args.mode == "dynamic" and not args.force_dynamic:
        print("[atlas] auto-switching --mode dynamic -> reveal "
              "(use --force_dynamic to keep TSDF live mesh).")
        args.mode = "reveal"

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
        atlas_disclaimer=using_atlas,
        prebuilt_gs_map=gs_map_path,
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
                        "coverage/reveal modes. If unset and --atlas is "
                        "'procedural', a generic bronchial tree is generated "
                        "and used as the GPS canvas (atlas-based, approximate).")
    p.add_argument("--atlas", default="procedural",
                   choices=["procedural", "none"],
                   help="Generate a procedural bronchial-tree atlas as the "
                        "GPS canvas when --organ_mesh isn't supplied. "
                        "'none' disables this and uses the live TSDF mesh "
                        "(--mode dynamic) instead.")
    p.add_argument("--force_dynamic", action="store_true",
                   help="When using the atlas, normally we auto-switch to "
                        "--mode reveal. This flag keeps --mode dynamic "
                        "(live TSDF mesh) on top of the atlas being "
                        "saved separately to disk.")
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
    # Tracking backend
    p.add_argument("--tracking", default="orb",
                   choices=["orb", "endo2dtam"],
                   help="Camera-tracking backend. 'orb' = always-works "
                        "ORB+RANSAC VO (drifts on long clips). "
                        "'endo2dtam' = SOTA endoscopic Gaussian-SLAM, "
                        "requires --endo2dtam_repo + checkpoints.")
    p.add_argument("--endo2dtam_repo", default=None,
                   help="Path to a local clone of "
                        "https://github.com/lastbasket/Endo-2DTAM")
    p.add_argument("--endo2dtam_checkpoints", default=None,
                   help="Path to Endo-2DTAM checkpoint folder.")

    # Depth backbone
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
