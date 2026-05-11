"""
End-to-end dashboard from a single endoscopic video file.

Pipeline:
  1. cv2.VideoCapture extracts frames from the input MP4.
  2. EndoDAC (or DAv2) produces per-frame depth (mm).
  3. **Endo-2DTAM** (ICRA 2025) jointly tracks the camera and builds a
     2D-Gaussian surface map. Called in-process via rgbd_slam(config).
  4. The dashboard composites endo + depth + GPS (with the Endo-2DTAM
     map as the GPS canvas, or the procedural atlas if --atlas
     procedural).

Run inside the Endo-2DTAM conda env (their README's instructions —
python 3.10, torch 1.13.1+cu118, diff-surfel-rasterization, simple-knn).
Add our small set of deps (numpy, opencv-python, matplotlib, tqdm,
open3d, transformers, accelerate) to the same env on top.

Usage:
  python run_video_dashboard.py \\
      --video bronchoscopy.mp4 \\
      --output_dir output/bronchus \\
      --hfov 90 \\
      --endo2dtam_repo external/Endo-2DTAM \\
      --backbone endodac \\
      --endodac_repo external/EndoDAC \\
      --endodac_weights external/EndoDAC/EndoDAC_fullmodel/depth_model.pth
"""

from __future__ import annotations

import os
import sys
import shutil
from argparse import ArgumentParser, Namespace

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import render_navigation_c3vd as rnc
from endo2dtam_runner import run_endo2dtam


def _optical_flow_advance_mm(frames, hfov_deg, median_depth_mm):
    """Per-frame scope advance estimated from dense optical flow.

    Mean pixel-flow magnitude * scene_depth / focal_length gives a
    rough metric advance (mm). Used as a fallback when Endo-2DTAM
    tracking collapses to identity. Caveat: pure rotation also
    produces flow, so the estimate over-reads on rotation-only motion.
    """
    H, W = frames[0].shape[:2]
    fx = W / (2.0 * np.tan(np.radians(hfov_deg) / 2.0))
    advances = [0.0]
    prev = cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY)
    for f in tqdm(frames[1:], desc="Optical flow"):
        gray = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            prev, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        advances.append(float(np.mean(mag)) * median_depth_mm / fx)
        prev = gray
    return np.asarray(advances, dtype=np.float64)


def _synthesize_poses_from_advance(advance_mm):
    """Build a list of c2w matrices: identity rotation, position
    walking along -z by the cumulative scope advance. Magnitudes (not
    directions) are what the centerline arclength snap consumes."""
    cum = np.cumsum(advance_mm)
    poses = []
    for s in cum:
        p = np.eye(4, dtype=np.float64)
        p[2, 3] = float(s)
        poses.append(p)
    return poses


def write_pose_txt(poses, path):
    """Save 4x4 c2w matrices, one per line (16 floats, comma-separated,
    transposed before flatten so render_navigation_c3vd.parse_pose_txt
    round-trips)."""
    with open(path, "w") as f:
        for p in poses:
            flat = np.asarray(p, dtype=np.float64).T.reshape(-1).tolist()
            f.write(",".join(f"{v:.8f}" for v in flat) + "\n")


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


def _resolve_endodac_paths(args):
    """Auto-fill --endodac_repo / --endodac_weights from the conventional
    layout if the user didn't pass them explicitly."""
    if args.backbone != "endodac":
        return
    here = os.path.dirname(os.path.abspath(__file__))
    repo_candidates = [
        args.endodac_repo,
        os.path.join(here, "external", "EndoDAC"),
        "external/EndoDAC",
    ]
    weight_candidates = [
        args.endodac_weights,
        os.path.join(here, "external", "EndoDAC",
                     "EndoDAC_fullmodel", "depth_model.pth"),
        "external/EndoDAC/EndoDAC_fullmodel/depth_model.pth",
    ]
    for c in repo_candidates[1:]:
        if not args.endodac_repo and c and os.path.isdir(c):
            args.endodac_repo = c
            print(f"Auto-detected --endodac_repo {c}")
            break
    for c in weight_candidates[1:]:
        if not args.endodac_weights and c and os.path.isfile(c):
            args.endodac_weights = c
            print(f"Auto-detected --endodac_weights {c}")
            break
    if not args.endodac_repo or not args.endodac_weights:
        sys.exit(
            "ERROR: --backbone endodac needs --endodac_repo + --endodac_weights "
            "(couldn't auto-detect).\nEither pass them explicitly or run with "
            "--backbone dav2 (no checkpoints needed)."
        )


def _resolve_endo2dtam_repo(args):
    if args.endo2dtam_repo and os.path.isdir(args.endo2dtam_repo):
        return
    here = os.path.dirname(os.path.abspath(__file__))
    for c in [
        args.endo2dtam_repo,
        os.path.join(here, "external", "Endo-2DTAM"),
        "external/Endo-2DTAM",
    ]:
        if c and os.path.isdir(c):
            args.endo2dtam_repo = c
            print(f"Auto-detected --endo2dtam_repo {c}")
            return
    sys.exit(
        "ERROR: --endo2dtam_repo not provided and not found at "
        "external/Endo-2DTAM. Clone it: "
        "git clone https://github.com/lastbasket/Endo-2DTAM external/Endo-2DTAM"
    )


def _load_frames_rgb(frames_dir):
    paths = sorted(p for p in os.listdir(frames_dir)
                   if p.endswith("_color.png"))
    if len(paths) < 2:
        sys.exit("ERROR: need at least 2 frames.")
    frames = []
    for p in tqdm(paths, desc="Reading frames"):
        bgr = cv2.imread(os.path.join(frames_dir, p))
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    return frames, paths


def _predict_depths_mm(frames, args):
    """Run the depth backbone over every frame and return a list of
    HxW float32 depth maps in millimeters."""
    from depth_backbones import make_backbone
    if args.backbone == "endodac":
        bb = make_backbone("endodac",
                           repo_dir=args.endodac_repo,
                           weights_path=args.endodac_weights)
    else:
        bb = make_backbone("dav2", variant=args.variant)

    # Calibrate scale once: backbone returns disparity-like values for
    # relative variants; anchor median to --assumed_median_depth_mm.
    is_metric = bool(getattr(bb, "is_metric", False))
    a = b = None
    if not is_metric:
        sample_idx = list(range(0, len(frames),
                                max(1, len(frames) // 10)))[:10]
        med_pred = float(np.median([
            float(np.median(bb.predict(frames[i]))) for i in sample_idx
        ]))
        a = float(args.assumed_median_depth_mm) * max(med_pred, 1e-6)
        b = 0.0
        print(f"Auto-scale calibration: median pred={med_pred:.3f}, "
              f"assumed median depth={args.assumed_median_depth_mm:.1f} mm "
              f"-> a={a:.2f}, b=0")

    depths = []
    for f in tqdm(frames, desc=f"Depth ({bb.variant if hasattr(bb,'variant') else bb.__class__.__name__})"):
        pred = bb.predict(f).astype(np.float32)
        if is_metric:
            d_mm = pred * 1000.0
        else:
            eps = 1e-6
            d_mm = a / np.clip(pred, eps, None) + b
        d_mm = np.nan_to_num(d_mm, nan=0.0, posinf=0.0, neginf=0.0)
        d_mm = np.clip(d_mm, 0.0, args.depth_trunc)
        depths.append(d_mm)
    return depths


def run(args):
    _resolve_endodac_paths(args)
    _resolve_endo2dtam_repo(args)

    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    frames_dir = os.path.join(out_dir, "video_frames")

    # ---- 1. Extract frames -------------------------------------------------
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

    # ---- 2. Load frames + depth -------------------------------------------
    frames, _ = _load_frames_rgb(frames_dir)
    if args.input_scale and args.input_scale < 1.0:
        s = float(args.input_scale)
        H0, W0 = frames[0].shape[:2]
        Hs = max(2, int(round(H0 * s)))
        Ws = max(2, int(round(W0 * s)))
        # Round to multiples of 14 — DINOv2/EndoDAC patch size; avoids
        # an internal resize that wastes the savings we're trying to make.
        Hs = max(14, (Hs // 14) * 14)
        Ws = max(14, (Ws // 14) * 14)
        print(f"--input_scale {s}: resizing {W0}x{H0} -> {Ws}x{Hs} "
              f"before depth + SLAM")
        frames = [cv2.resize(f, (Ws, Hs), interpolation=cv2.INTER_AREA)
                  for f in frames]
    print(f"\nRunning depth backbone ({args.backbone}) on {len(frames)} frames...")
    depths_mm = _predict_depths_mm(frames, args)

    # ---- 3. Endo-2DTAM tracking + mapping ---------------------------------
    print("\n=== Endo-2DTAM tracking + 2D-Gaussian mapping ===")
    poses, gs_map_path, run_dir = run_endo2dtam(
        frames=frames,
        depth_mm_per_frame=depths_mm,
        hfov_deg=args.hfov,
        endo2dtam_repo=args.endo2dtam_repo,
        output_dir=out_dir,
        device=args.device,
        num_iters_tracking=args.endo2dtam_tracking_iters,
        num_iters_mapping=args.endo2dtam_mapping_iters,
        keyframe_every=args.endo2dtam_keyframe_every,
        mapping_window_size=args.endo2dtam_mapping_window,
        cleanup=not args.keep_endo2dtam_artifacts,
    )

    # Detect Endo-2DTAM tracking failure (all-identity poses) and fall
    # back to an optical-flow scope-advance estimate. On real
    # bronchoscopy + EndoDAC-predicted depth, their tracker's loss
    # landscape is flat near identity and the "save-best-iter" logic
    # commits the identity init every frame. The mapper still runs but
    # the per-frame trajectory is unusable. Flow magnitude isn't a real
    # 6-DoF pose (it conflates rotation with translation), but it gives
    # the GPS dot a believable forward-advance signal that tracks
    # whether the scope is moving.
    cam_pos = np.array([p[:3, 3] for p in poses], dtype=np.float64)
    slam_total_mm = float(
        np.linalg.norm(np.diff(cam_pos, axis=0), axis=1).sum())
    if args.motion_source == 'flow' or (
            args.motion_source == 'auto' and slam_total_mm < 0.5):
        if args.motion_source == 'auto':
            print(f"WARNING: Endo-2DTAM tracker produced "
                  f"{slam_total_mm:.3f} mm of total motion across "
                  f"{len(poses)} frames -> tracking failed. Falling "
                  f"back to optical-flow scope-advance estimate.")
        else:
            print("Using optical-flow scope-advance estimate "
                  "(--motion_source flow).")
        adv = _optical_flow_advance_mm(
            frames, hfov_deg=args.hfov,
            median_depth_mm=float(args.assumed_median_depth_mm))
        poses = _synthesize_poses_from_advance(adv)
        print(f"Flow-derived total advance: {adv.sum():.1f} mm over "
              f"{len(adv)} frames "
              f"(mean {adv.mean():.3f} mm/frame).")

    pose_path = os.path.join(frames_dir, "pose.txt")
    write_pose_txt(poses, pose_path)
    print(f"Wrote {len(poses)} poses -> {pose_path}")

    # ---- 4. Atlas (GPS canvas) when no patient-specific mesh given --------
    using_atlas = False
    atlas_centerline = None
    if args.organ_mesh is None and args.atlas == "procedural":
        from bronchus_atlas import write_atlas, airway_centerline_dfs
        atlas_path, branches = write_atlas(out_dir)
        args.organ_mesh = atlas_path
        # The DFS centerline walks every branch in topology order — this
        # is the spine the GPS panel needs for arclength-based snapping
        # (compute_organ_centerline's PCA fallback collapses a branched
        # tree to a single principal axis, which makes the lime dot
        # meaningless on the atlas).
        atlas_centerline = airway_centerline_dfs(branches)
        using_atlas = True
        print(f"[atlas] generated procedural bronchial tree "
              f"({len(branches)} branches, {len(atlas_centerline)} "
              f"centerline pts) -> {atlas_path}")

    if args.mode in ("coverage", "reveal") and not args.organ_mesh:
        print(f"WARNING: --mode {args.mode} requires --organ_mesh; "
              f"falling back to 'dynamic'.")
        args.mode = "dynamic"
    if using_atlas and args.mode == "dynamic":
        print("[atlas] auto-switching --mode dynamic -> reveal")
        args.mode = "reveal"

    # ---- 5. Dashboard ------------------------------------------------------
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
        skip_every=1,
        max_frames=None,
        calibration_frames=20,
        assumed_median_depth_mm=args.assumed_median_depth_mm,
        depth_trunc=args.depth_trunc,
        min_disparity_pct=10.0,
        depth_smooth_ksize=5,
        voxel_size=0.5,
        mesh_update_every=15,
        no_trajectory_align=(args.organ_mesh is None),
        trajectory_align_max_corr=80.0,
        atlas_disclaimer=using_atlas,
        prebuilt_gs_map=gs_map_path,
        atlas_centerline=atlas_centerline,
    )
    print("\n=== Rendering dashboard ===")
    rnc.run(dash_args)


if __name__ == "__main__":
    p = ArgumentParser(description="Run the Endo-2DTAM dashboard on a video")
    p.add_argument("--video", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--hfov", default=90.0, type=float,
                   help="Horizontal FOV in degrees (90 for bronchoscopes, "
                        "140 for colonoscopes).")

    # Endo-2DTAM
    p.add_argument("--endo2dtam_repo", default=None,
                   help="Path to a local clone of "
                        "https://github.com/lastbasket/Endo-2DTAM")
    p.add_argument("--endo2dtam_tracking_iters", default=15, type=int)
    p.add_argument("--endo2dtam_mapping_iters", default=15, type=int)
    p.add_argument("--keep_endo2dtam_artifacts", action="store_true",
                   help="Keep the staged input dataset and the full Endo-"
                        "2DTAM run dir (params.npz, checkpoints) on disk "
                        "after the run. Default: clean them up to free "
                        "multi-GB per invocation; the GS-map .ply is "
                        "always preserved next to output_dir.")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--motion_source", default="auto",
                   choices=["auto", "slam", "flow"],
                   help="Where the GPS dot's motion comes from. 'slam' = "
                        "use Endo-2DTAM poses (fails on real bronchoscopy "
                        "with predicted depth). 'flow' = optical-flow "
                        "scope-advance estimate. 'auto' (default) = slam, "
                        "fall back to flow if tracking collapsed to "
                        "identity (<0.5 mm total motion).")
    p.add_argument("--input_scale", default=1.0, type=float,
                   help="Resize frames by this factor before depth + SLAM "
                        "(memory ~ scale^2). Try 0.5 if you OOM.")
    p.add_argument("--endo2dtam_keyframe_every", default=8, type=int)
    p.add_argument("--endo2dtam_mapping_window", default=-1, type=int,
                   help="Endo-2DTAM mapping window size (-1 = unbounded). "
                        "Set to e.g. 10 to cap memory growth on long runs.")

    # Depth backbone
    p.add_argument("--backbone", default="endodac",
                   choices=["dav2", "endodac"])
    p.add_argument("--variant", default="vitb")
    p.add_argument("--endodac_repo", default=None)
    p.add_argument("--endodac_weights", default=None)
    p.add_argument("--assumed_median_depth_mm", default=20.0, type=float)
    p.add_argument("--depth_trunc", default=80.0, type=float)

    # Frame extraction
    p.add_argument("--max_frames", default=None, type=int)
    p.add_argument("--skip_every", default=1, type=int)
    p.add_argument("--reuse_frames", action="store_true",
                   help="Skip frame extraction if outputs already exist.")

    # GPS canvas
    p.add_argument("--mode", default="reveal",
                   choices=["coverage", "reveal", "dynamic"])
    p.add_argument("--organ_mesh", default=None,
                   help="Optional pre-op CT-segmented airway mesh. If unset "
                        "and --atlas procedural, a generic bronchial tree "
                        "is generated as the GPS canvas.")
    p.add_argument("--atlas", default="procedural",
                   choices=["procedural", "none"])

    # Dashboard
    p.add_argument("--fps", default=30, type=int)

    args = p.parse_args()
    run(args)
