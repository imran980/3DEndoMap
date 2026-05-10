"""
Surgical Navigation Dashboard — C3VD-direct edition.

Standalone dashboard that does NOT need a trained Endo-4DGS model.
Inputs are raw C3VD data:
  - RGB frames (NNNN_color.png)
  - 4x4 c2w poses (pose.txt)
  - optional GT depth (NNNN_depth.tiff) for backbone calibration
  - optional pre-op organ mesh (trans_model.obj) for static / reveal modes

Per-frame depth comes from a pluggable backbone (EndoDAC by default,
or DAv2). It's calibrated against C3VD GT depth when available.

Three GPS modes:
  --mode coverage   pre-op organ + red→green coverage heatmap (default if
                    --organ_mesh is given)
  --mode reveal     pre-op organ painted in by camera frustum
  --mode dynamic    organ mesh built live via TSDF fusion (no pre-op mesh
                    needed; EndoDAC depth in mm goes straight into TSDF)

Usage:
  python render_navigation_c3vd.py \
      --c3vd_dir dataset/trans_t1_b \
      --output_dir output/c3vd_dash \
      --backbone endodac \
      --endodac_repo external/EndoDAC \
      --endodac_weights external/EndoDAC/EndoDAC_fullmodel/depth_model.pth \
      --organ_mesh dataset/trans_model.obj \
      --mode reveal
"""

import os
import sys
import glob
import numpy as np
import cv2
import open3d as o3d
from tqdm import tqdm
from argparse import ArgumentParser

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
def parse_pose_txt(path):
    """Read 4x4 c2w matrices from a text file (16 floats per line)."""
    poses = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            vals = [float(v) for v in line.replace(",", " ").split()]
            if len(vals) == 16:
                poses.append(np.array(vals).reshape(4, 4).T)
            elif len(vals) == 12:
                m = np.eye(4)
                m[:3, :] = np.array(vals).reshape(3, 4)
                poses.append(m)
    return poses
from dav2_depth import fit_disparity_to_depth, disparity_to_depth


def _find_color_frames(c3vd_dir):
    """Return a sorted list of RGB frame paths under c3vd_dir.

    Accepts both `<dir>/NNNN_color.png` (run_video_dashboard layout)
    and `<dir>/rgb/NNNN_color.png` (legacy).
    """
    candidates = (
        sorted(glob.glob(os.path.join(c3vd_dir, "*_color.png")))
        or sorted(glob.glob(os.path.join(c3vd_dir, "rgb", "*_color.png")))
        or sorted(glob.glob(os.path.join(c3vd_dir, "*.png")))
    )
    return candidates
from depth_backbones import make_backbone
from dashboard_common import (
    depth_to_colormap, frustum_visibility, compute_organ_centerline,
    render_gps_frame, draw_hud,
)


def _read_c3vd_depth_mm(path, target_hw=None):
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


def _snap_trajectory_to_centerline(cam_positions, centerline):
    """Project each frame's cumulative path length onto a 1D centerline,
    so the lime dot walks along the airway at the same arclength the
    scope has traveled in 3D.

    This is the right tool for the procedural-atlas case: the atlas is a
    generic shape and we have no patient-specific registration, so the
    only meaningful "where is the scope" signal is "how far down the
    airway have we advanced." Rigid 6-DoF ICP onto a branched centerline
    has no rotational lock-in for short, drifty trajectories and was
    collapsing the lime dot to the atlas centroid.

    Returns (snapped_positions, snapped_forwards) — both Nx3, in the
    centerline's coordinate frame. Forwards are the local centerline
    tangent so the GPS panel can draw a meaningful heading arrow.
    """
    cl = np.asarray(centerline, dtype=np.float64)
    if cl.ndim != 2 or cl.shape[0] < 2 or cl.shape[1] != 3:
        raise ValueError(
            f"centerline must be (N>=2, 3); got {cl.shape}.")
    cam = np.asarray(cam_positions, dtype=np.float64)
    seg = np.linalg.norm(np.diff(cam, axis=0), axis=1)
    cam_arclen = np.concatenate([[0.0], np.cumsum(seg)])

    cl_seg = np.linalg.norm(np.diff(cl, axis=0), axis=1)
    cl_arclen = np.concatenate([[0.0], np.cumsum(cl_seg)])
    cl_total = float(cl_arclen[-1])

    snapped_pos = np.zeros_like(cam, dtype=np.float64)
    snapped_fwd = np.zeros_like(cam, dtype=np.float64)
    for i in range(len(cam)):
        s = float(min(cam_arclen[i], cl_total))
        seg_idx = int(np.searchsorted(cl_arclen, s) - 1)
        seg_idx = max(0, min(seg_idx, len(cl) - 2))
        s0, s1 = cl_arclen[seg_idx], cl_arclen[seg_idx + 1]
        t = (s - s0) / max(s1 - s0, 1e-9)
        snapped_pos[i] = cl[seg_idx] * (1 - t) + cl[seg_idx + 1] * t
        tangent = cl[seg_idx + 1] - cl[seg_idx]
        n = float(np.linalg.norm(tangent))
        snapped_fwd[i] = (tangent / n) if n > 1e-9 else np.array([0, 0, -1.0])
    return snapped_pos, snapped_fwd


def _align_trajectory_to_organ(cam_positions, centerline, max_corr=80.0):
    """Rigid (R,t) ICP fitting the camera trajectory onto the organ
    centerline, so the reveal/coverage frustum tests actually hit the
    pre-op mesh. _pose_to_organ_space() only fixes the rotation between
    C3VD's pose convention and the OBJ; this fixes the residual translation
    (and refines rotation) by aligning the trajectory curve with the
    centerline curve.

    Returns (T_4x4, fitness).
    """
    src = o3d.geometry.PointCloud()
    src.points = o3d.utility.Vector3dVector(cam_positions.astype(np.float64))
    tgt = o3d.geometry.PointCloud()
    tgt.points = o3d.utility.Vector3dVector(centerline.astype(np.float64))

    t0 = centerline.mean(axis=0) - cam_positions.mean(axis=0)
    T_init = np.eye(4)
    T_init[:3, 3] = t0

    icp = o3d.pipelines.registration.registration_icp(
        src, tgt, max_corr, T_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=300))
    return icp.transformation, float(icp.fitness)


def run(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Load RGB frames and poses ----
    # Bronchoscopy video flow has no GT depth files alongside the
    # frames; the calibration block below handles that fallback.
    rgb_paths = _find_color_frames(args.c3vd_dir)
    gt_depth_paths = [None] * len(rgb_paths)

    pose_path = os.path.join(args.c3vd_dir, "pose.txt")
    if not os.path.exists(pose_path):
        sys.exit(f"ERROR: missing {pose_path}")
    poses = parse_pose_txt(pose_path)

    n = min(len(rgb_paths), len(poses))
    rgb_paths, gt_depth_paths, poses = (rgb_paths[:n], gt_depth_paths[:n],
                                        poses[:n])
    if args.skip_every > 1:
        idx = list(range(0, n, args.skip_every))
        rgb_paths = [rgb_paths[i] for i in idx]
        gt_depth_paths = [gt_depth_paths[i] for i in idx]
        poses = [poses[i] for i in idx]
    if args.max_frames and len(rgb_paths) > args.max_frames:
        idx = np.linspace(0, len(rgb_paths) - 1, args.max_frames, dtype=int)
        rgb_paths = [rgb_paths[i] for i in idx]
        gt_depth_paths = [gt_depth_paths[i] for i in idx]
        poses = [poses[i] for i in idx]
    n = len(rgb_paths)
    if n == 0:
        sys.exit(f"ERROR: no RGB frames found under {args.c3vd_dir}")

    sample = cv2.imread(rgb_paths[0])
    H, W = sample.shape[:2]
    fx = fy = W / (2.0 * np.tan(np.radians(args.hfov) / 2.0))
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        W, H, fx, fy, W / 2.0, H / 2.0)
    print(f"Frames: {n}  Image: {W}x{H}  hFOV: {args.hfov}°  "
          f"backbone: {args.backbone}")

    # ---- Pre-op organ mesh (optional) ----
    organ_mesh = None
    if args.organ_mesh and os.path.exists(args.organ_mesh):
        organ_mesh = o3d.io.read_triangle_mesh(args.organ_mesh)
        organ_mesh.compute_vertex_normals()
        print(f"Organ mesh: {len(organ_mesh.vertices):,} verts")
    elif args.mode in ("coverage", "reveal"):
        sys.exit(f"ERROR: --mode {args.mode} requires --organ_mesh.")

    # ---- Camera trajectory (already in organ space) ----
    cam_positions = np.array([p[:3, 3] for p in poses], dtype=np.float32)
    # forward = c2w[:3, 2] in OpenCV/Open3D convention; same here since
    # _pose_to_organ_space is a rigid axis remap.
    cam_forwards = np.array([p[:3, 2] for p in poses], dtype=np.float32)
    norms = np.linalg.norm(cam_forwards, axis=1, keepdims=True)
    cam_forwards = np.where(norms > 1e-6, cam_forwards / np.maximum(norms, 1e-9),
                            np.array([0, 0, -1.0]))
    seg_len = np.linalg.norm(np.diff(cam_positions, axis=0), axis=1)
    cum_dist = np.concatenate([[0.0], np.cumsum(seg_len)])

    # ---- Depth backbone ----
    print(f"Loading depth backbone ({args.backbone})...")
    if args.backbone == "dav2":
        bb = make_backbone("dav2", variant=args.variant)
    else:
        bb = make_backbone("endodac",
                           repo_dir=args.endodac_repo,
                           weights_path=args.endodac_weights)

    # ---- Calibration (relative variants only) ----
    # If GT depth is available we fit per-frame depth = a/pred + b against
    # it. If not (e.g. arbitrary bronchoscopy video), we fall back to a
    # *scale-only* calibration that anchors median predicted disparity to
    # an assumed median scene depth (--assumed_median_depth_mm).
    a_avg = b_avg = None
    if not bb.is_metric:
        gt_idx = [i for i, p in enumerate(gt_depth_paths) if p]
        if gt_idx:
            cal_idx = gt_idx[::max(1, len(gt_idx) // args.calibration_frames)] \
                [:args.calibration_frames]
            a_list, b_list = [], []
            for i in tqdm(cal_idx, desc="Calibration"):
                rgb = cv2.cvtColor(cv2.imread(rgb_paths[i]), cv2.COLOR_BGR2RGB)
                pred = bb.predict(rgb)
                gt_mm = _read_c3vd_depth_mm(gt_depth_paths[i], target_hw=(H, W))
                a, b = fit_disparity_to_depth(pred, gt_mm)
                if a is not None:
                    a_list.append(a)
                    b_list.append(b)
            a_avg = float(np.median(a_list))
            b_avg = float(np.median(b_list))
            print(f"Global calib (median over {len(a_list)} frames): "
                  f"a={a_avg:.2f}, b={b_avg:.2f}")
        else:
            assumed = float(getattr(args, 'assumed_median_depth_mm', 20.0))
            sample_idx = list(range(0, len(rgb_paths),
                                    max(1, len(rgb_paths) // 10)))[:10]
            preds = []
            for i in tqdm(sample_idx, desc="Auto-scale (no GT)"):
                rgb = cv2.cvtColor(cv2.imread(rgb_paths[i]), cv2.COLOR_BGR2RGB)
                preds.append(float(np.median(bb.predict(rgb))))
            med_pred = float(np.median(preds)) if preds else 1.0
            a_avg = assumed * max(med_pred, 1e-6)
            b_avg = 0.0
            print(f"No GT depth available; auto-scale calibration "
                  f"(median pred={med_pred:.3f}, assumed median depth="
                  f"{assumed:.1f} mm) -> a={a_avg:.2f}, b=0.")

    # ---- TSDF (dynamic mode) ----
    # When the tracker already produced a 3D map (e.g. Endo-2DTAM's
    # 2D Gaussian map saved as a .ply), skip TSDF entirely and load
    # that as the organ mesh instead.
    organ_builder = None
    prebuilt = getattr(args, "prebuilt_gs_map", None)
    if prebuilt and os.path.isfile(prebuilt) and args.mode == "dynamic":
        print(f"Using pre-built tracker map (skipping TSDF): {prebuilt}")
        try:
            premesh = o3d.io.read_triangle_mesh(prebuilt)
            premesh.compute_vertex_normals()
            if len(premesh.vertices) > 0:
                organ_pts = np.asarray(premesh.vertices, dtype=np.float64)
                if gps_data is None:
                    gps_data = {}
                gps_data['organ_pts'] = organ_pts
                gps_data['centerline'] = compute_organ_centerline(
                    premesh, n_points=200)
                gps_data['center'] = organ_pts.mean(axis=0)
                gps_data['extent'] = (
                    organ_pts.max(axis=0) - organ_pts.min(axis=0))
                gps_data.setdefault('elev', 25)
                gps_data.setdefault('azim', 45)
                gps_data.setdefault('n_frames', n)
        except Exception as e:
            sys.exit(f"ERROR: failed to load Endo-2DTAM map "
                     f"({prebuilt}): {e}")
    elif args.mode == "dynamic":
        sys.exit(
            "ERROR: --mode dynamic requires a pre-built Gaussian map "
            "(prebuilt_gs_map). run_video_dashboard.py wires this up "
            "automatically via Endo-2DTAM; if you're calling "
            "render_navigation_c3vd directly, use --mode reveal with "
            "--organ_mesh instead.")

    # ---- GPS scene data ----
    if organ_mesh is not None:
        organ_pts = np.asarray(organ_mesh.vertices, dtype=np.float64)
        atlas_centerline = getattr(args, "atlas_centerline", None)
        use_arclength_snap = (
            atlas_centerline is not None
            and len(np.asarray(atlas_centerline)) >= 2)

        if use_arclength_snap:
            # Atlas case: we have a real DFS-walk centerline through the
            # bronchial tree. Skip the ICP — for a short, drifty SLAM
            # trajectory it has no rotational signal and collapses the
            # lime dot to the atlas centroid. Instead, snap each frame
            # along the centerline by the cumulative path length the
            # scope has traveled. The lime dot then walks the airway at
            # the same speed as the camera moves in 3D.
            centerline = np.asarray(atlas_centerline, dtype=np.float64)
            cl_arclen_total = float(
                np.linalg.norm(np.diff(centerline, axis=0), axis=1).sum())
            cam_arclen_total = float(
                np.linalg.norm(np.diff(cam_positions, axis=0), axis=1).sum())
            print(f"Trajectory->atlas arclength snap: scope traveled "
                  f"{cam_arclen_total:.1f} mm over {n} frames; atlas "
                  f"centerline is {cl_arclen_total:.1f} mm.")
            cam_positions, cam_forwards = _snap_trajectory_to_centerline(
                cam_positions, centerline)
            cam_positions = cam_positions.astype(np.float32)
            cam_forwards = cam_forwards.astype(np.float32)
            # cum_dist / seg_len computed earlier off raw cam_positions
            # are the *real* scope path lengths (in mm now that
            # endo2dtam_runner converts m->mm). Keep them — the HUD's
            # speed/path readouts should reflect actual scope motion,
            # not the snapped arclength.
        else:
            centerline = compute_organ_centerline(organ_mesh, n_points=200)
            # Snap the camera trajectory onto the organ centerline so the
            # frustum tests in coverage/reveal modes actually hit the mesh.
            if not args.no_trajectory_align:
                T_align, fit = _align_trajectory_to_organ(
                    cam_positions, centerline,
                    max_corr=args.trajectory_align_max_corr)
                print(f"Trajectory->organ ICP: fitness={fit:.3f}")
                R = T_align[:3, :3]
                t = T_align[:3, 3]
                cam_positions = (R @ cam_positions.T).T + t
                cam_forwards = (R @ cam_forwards.T).T
                norms = np.linalg.norm(cam_forwards, axis=1, keepdims=True)
                cam_forwards = np.where(
                    norms > 1e-6, cam_forwards / np.maximum(norms, 1e-9),
                    np.array([0, 0, -1.0]))
                poses = [T_align @ p for p in poses]

        gps_data = {
            'organ_pts': organ_pts,
            'centerline': centerline,
            'center': organ_pts.mean(axis=0),
            'extent': organ_pts.max(axis=0) - organ_pts.min(axis=0),
            'elev': 25, 'azim': 45,
            'n_frames': n,
        }
    else:
        # Dynamic mode without pre-op mesh — seed bounds from camera path
        margin = 30.0
        gps_data = {
            'organ_pts': np.zeros((0, 3), dtype=np.float64),
            'centerline': cam_positions.astype(np.float64),
            'center': cam_positions.mean(axis=0),
            'extent': (cam_positions.max(0) - cam_positions.min(0)
                       + 2 * margin).astype(np.float64),
            'elev': 25, 'azim': 45,
            'n_frames': n,
        }

    seen_any = (np.zeros(len(gps_data['organ_pts']), dtype=bool)
                if args.mode in ("coverage", "reveal")
                and len(gps_data['organ_pts']) > 0 else None)
    organ_pts_f = (gps_data['organ_pts'].astype(np.float32)
                   if seen_any is not None else None)
    cov_far = (float(np.median(
        np.linalg.norm(organ_pts_f - organ_pts_f.mean(0), axis=1)) * 3.0)
               if organ_pts_f is not None and len(organ_pts_f) else 120.0)

    # ---- Layout ----
    left_w = 560
    left_h = int(H * left_w / W)
    gps_h = left_h * 2
    gps_w = int(gps_h * 1.35)
    vid_w = left_w + gps_w
    vid_h = gps_h
    print(f"Video layout: {vid_w}x{vid_h}")

    video_path = os.path.join(args.output_dir, "navigation_dashboard.mp4")
    writer = cv2.VideoWriter(video_path,
                             cv2.VideoWriter_fourcc(*'mp4v'),
                             args.fps, (vid_w, vid_h))
    frames_dir = os.path.join(args.output_dir, "dashboard_frames")
    os.makedirs(frames_dir, exist_ok=True)

    print(f"Rendering {n} frames -> {video_path}")
    for idx in tqdm(range(n), desc="Dashboard"):
        # ---- depth ----
        rgb = cv2.cvtColor(cv2.imread(rgb_paths[idx]), cv2.COLOR_BGR2RGB)
        if rgb.shape[:2] != (H, W):
            rgb = cv2.resize(rgb, (W, H))
        pred = bb.predict(rgb)

        if not bb.is_metric and args.min_disparity_pct > 0:
            cutoff = float(np.percentile(pred, args.min_disparity_pct))
            valid_pred = pred > cutoff
        else:
            valid_pred = pred > 0

        if bb.is_metric:
            depth_mm = pred * 1000.0
        else:
            a_use, b_use = a_avg, b_avg
            if gt_depth_paths[idx]:
                gt_mm = _read_c3vd_depth_mm(gt_depth_paths[idx],
                                            target_hw=(H, W))
                if gt_mm is not None:
                    a_pf, b_pf = fit_disparity_to_depth(pred, gt_mm)
                    if a_pf is not None:
                        a_use, b_use = a_pf, b_pf
            depth_mm = disparity_to_depth(pred, a_use, b_use)

        depth_mm = depth_mm.astype(np.float32)
        depth_mm[~np.isfinite(depth_mm)] = 0.0
        depth_mm[~valid_pred] = 0.0
        depth_mm[depth_mm <= 0.5] = 0.0
        depth_mm[depth_mm >= args.depth_trunc] = 0.0
        if args.depth_smooth_ksize >= 3:
            depth_mm = cv2.medianBlur(depth_mm, args.depth_smooth_ksize)

        # (no online TSDF — the GS map is pre-built by Endo-2DTAM)

        # ---- coverage update ----
        if seen_any is not None:
            vis = frustum_visibility(cam_positions[idx], cam_forwards[idx],
                                     organ_pts_f, hfov_deg=args.hfov,
                                     near=1.0, far=cov_far)
            seen_any |= vis

        # ---- panels ----
        rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        endo_panel = cv2.resize(rgb_bgr, (left_w, left_h))
        depth_colored, vmin, vmax = depth_to_colormap(
            depth_mm, valid_mask=depth_mm > 0)
        depth_panel = cv2.resize(depth_colored, (left_w, left_h))

        cv2.putText(endo_panel, "ENDOSCOPIC VIEW", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 200), 2)
        cv2.putText(endo_panel, f"Frame {idx}/{n-1}",
                    (10, left_h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        cv2.putText(depth_panel, f"DEPTH ({args.backbone})", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(depth_panel, f"range: {vmin:.1f}-{vmax:.1f} mm",
                    (10, left_h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # ---- GPS panel ----
        current_cov = (seen_any.astype(np.int32)
                       if seen_any is not None else None)
        gps_img = render_gps_frame(
            gps_data, idx, coverage_counts=current_cov,
            reveal_mode=(args.mode == "reveal"),
            cam_pos=cam_positions[idx],
            cam_forward=cam_forwards[idx],
            cam_trajectory=cam_positions,
            width=gps_w, height=gps_h, dpi=80)
        gps_panel = cv2.cvtColor(gps_img, cv2.COLOR_RGB2BGR)
        if gps_panel.shape[:2] != (gps_h, gps_w):
            gps_panel = cv2.resize(gps_panel, (gps_w, gps_h))

        # ---- HUD ----
        elapsed_s = idx / float(args.fps)
        speed_mms = float(seg_len[idx - 1] * args.fps) if idx > 0 else 0.0
        dist_mm = float(cum_dist[idx])
        cov_pct = (float(seen_any.sum()) / max(len(gps_data['organ_pts']), 1)
                   * 100.0 if seen_any is not None else 0.0)
        n_mesh = len(gps_data['organ_pts']) if args.mode == "dynamic" else 0
        n_fused = len(gps_data['organ_pts']) if args.mode == "dynamic" else 0
        draw_hud(gps_panel, gps_w=gps_w, elapsed_s=elapsed_s,
                 speed_mms=speed_mms, dist_mm=dist_mm,
                 mode=args.mode, cov_pct=cov_pct,
                 n_mesh_verts=n_mesh, n_fused=n_fused,
                 atlas_disclaimer=bool(getattr(args, "atlas_disclaimer",
                                               False)))

        # ---- composite ----
        canvas = np.zeros((vid_h, vid_w, 3), dtype=np.uint8)
        canvas[:left_h, :left_w] = endo_panel
        canvas[left_h:left_h * 2, :left_w] = depth_panel
        canvas[:gps_h, left_w:left_w + gps_w] = gps_panel
        cv2.line(canvas, (left_w, 0), (left_w, vid_h), (60, 60, 60), 2)
        cv2.line(canvas, (0, left_h), (left_w, left_h), (60, 60, 60), 2)
        writer.write(canvas)
        if idx % max(1, n // 20) == 0:
            cv2.imwrite(os.path.join(frames_dir, f"frame_{idx:04d}.png"),
                        canvas)

    writer.release()
    print(f"Wrote: {video_path}")

    # No final-mesh write here — Endo-2DTAM already wrote the
    # 2D-Gaussian map (endo2dtam_gs_map.ply) into its own run dir.


if __name__ == "__main__":
    p = ArgumentParser(description="C3VD-direct surgical navigation dashboard")
    p.add_argument("--c3vd_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--mode", choices=["coverage", "reveal", "dynamic"],
                   default="reveal")
    p.add_argument("--organ_mesh", default=None,
                   help="Pre-op organ .obj/.ply (required for coverage/reveal)")
    p.add_argument("--backbone", default="endodac",
                   choices=["dav2", "endodac"])
    p.add_argument("--variant", default="vitb",
                   help="DAv2 variant (only when --backbone dav2)")
    p.add_argument("--endodac_repo", default=None)
    p.add_argument("--endodac_weights", default=None)
    p.add_argument("--hfov", default=140.0, type=float)
    p.add_argument("--fps", default=30, type=int)
    p.add_argument("--skip_every", default=1, type=int)
    p.add_argument("--max_frames", default=None, type=int)
    p.add_argument("--calibration_frames", default=20, type=int)
    p.add_argument("--assumed_median_depth_mm", default=20.0, type=float,
                   help="Used to anchor depth scale when no GT depth is "
                        "available (e.g. raw bronchoscopy video). 20 mm is "
                        "a reasonable median for bronchoscopy; "
                        "30-40 mm for colonoscopy.")
    p.add_argument("--depth_trunc", default=80.0, type=float)
    p.add_argument("--min_disparity_pct", default=10.0, type=float)
    p.add_argument("--depth_smooth_ksize", default=5, type=int)
    p.add_argument("--voxel_size", default=0.5, type=float,
                   help="TSDF voxel size mm (dynamic mode)")
    p.add_argument("--mesh_update_every", default=15, type=int)
    p.add_argument("--no_trajectory_align", action="store_true",
                   help="Skip the trajectory->centerline ICP (default: on). "
                        "Disable only if you've already aligned poses.")
    p.add_argument("--trajectory_align_max_corr", default=80.0, type=float,
                   help="Max correspondence distance (mm) for trajectory ICP")
    args = p.parse_args()
    run(args)
