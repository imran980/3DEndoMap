"""
Surgical Navigation Dashboard — Render a synchronized video showing:
  - Panel 1: Endoscopic RGB view (from inside the organ)
  - Panel 2: Depth colormap
  - Panel 3: GPS view (organ wireframe + current camera position)

Usage:
    python render_navigation.py --model_path output/endonerf/c3vd_trans \
        --configs arguments/endonerf.py --iteration 3000
"""

import os
import sys
import json
import math
import torch
import numpy as np
import cv2
from tqdm import tqdm
from argparse import ArgumentParser

# Matplotlib for GPS rendering (headless-compatible)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from arguments import ModelParams, PipelineParams, ModelHiddenParams, get_combined_args
from gaussian_renderer import render
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from utils.graphics_utils import fov2focal
from dynamic_organ import DynamicOrganBuilder

import open3d as o3d


def depth_to_colormap(depth_np, valid_mask=None, vmin=None, vmax=None,
                      colormap=cv2.COLORMAP_TURBO):
    """Convert a depth map to a colored visualization.

    Uses per-frame percentile normalization over valid pixels so each frame
    shows a real near→far gradient instead of shifting globally.
    """
    if valid_mask is None:
        valid_mask = depth_np > 0

    valid = depth_np[valid_mask]
    if valid.size < 16:
        return np.zeros((*depth_np.shape, 3), dtype=np.uint8)

    if vmin is None:
        vmin = float(np.percentile(valid, 5))
    if vmax is None:
        vmax = float(np.percentile(valid, 95))
    if vmax - vmin < 1e-4:
        vmax = vmin + 1e-4

    depth_norm = np.clip((depth_np - vmin) / (vmax - vmin), 0, 1)
    # Invert so near = red/warm, far = blue/cool (more intuitive for endoscopy)
    depth_u8 = ((1.0 - depth_norm) * 255).astype(np.uint8)
    colored = cv2.applyColorMap(depth_u8, colormap)
    colored[~valid_mask] = 0
    return colored, vmin, vmax


def auto_exposure(rgb_np, low_pct=1.0, high_pct=99.5, gamma=0.9):
    """Percentile-based auto exposure so dark endoscope renders are visible.

    Input/output: float32 (H, W, 3) expected in [0, 1+] range.
    """
    if rgb_np.size == 0:
        return rgb_np
    lum = rgb_np.max(axis=2)
    lo = float(np.percentile(lum, low_pct))
    hi = float(np.percentile(lum, high_pct))
    if hi - lo < 1e-4:
        hi = lo + 1e-4
    out = (rgb_np - lo) / (hi - lo)
    out = np.clip(out, 0, 1)
    if gamma != 1.0:
        out = np.power(out, gamma)
    return out


def frustum_visibility(cam_center, cam_forward, organ_pts,
                       hfov_deg=140.0, near=1.0, far=120.0):
    """Return a boolean array (N,) marking organ points inside this camera's
    view frustum. Cheap cone-based approximation (no occlusion): a point is
    "seen" if within hfov/2 of forward and between [near, far] along the ray.
    """
    d = organ_pts - cam_center[None, :]
    dist = np.linalg.norm(d, axis=1)
    safe = dist > 1e-6
    d_norm = np.zeros_like(d)
    d_norm[safe] = d[safe] / dist[safe, None]
    cos_sim = d_norm @ cam_forward
    cos_half = float(np.cos(np.radians(hfov_deg) / 2.0))
    return safe & (cos_sim > cos_half) & (dist >= near) & (dist <= far)


def format_time_mmss(seconds):
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"


def compute_organ_centerline(organ_mesh, n_points=100):
    """
    Compute a smooth centerline path through the organ using PCA
    to find the principal axis, then project organ vertices onto it
    and interpolate along the path.
    """
    pts = np.asarray(organ_mesh.vertices)
    center = pts.mean(axis=0)
    
    # PCA to find principal axis
    centered = pts - center
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    
    # Principal axis = eigenvector with largest eigenvalue
    principal = eigenvectors[:, -1]
    
    # Project all points onto principal axis
    projections = centered @ principal
    
    # Create centerline along principal axis
    pmin, pmax = projections.min(), projections.max()
    t_values = np.linspace(pmin, pmax, n_points)
    
    # For each t, find the centroid of nearby points
    centerline = []
    bin_width = (pmax - pmin) / n_points * 2
    for t in t_values:
        mask = np.abs(projections - t) < bin_width
        if mask.sum() > 0:
            local_center = pts[mask].mean(axis=0)
            centerline.append(local_center)
        else:
            # Fallback: linear interpolation
            centerline.append(center + principal * t)
    
    return np.array(centerline)


def render_gps_frame(gps_data, current_idx, coverage_counts=None,
                     width=640, height=480, dpi=100):
    """
    Render a single GPS frame showing the organ + current camera dot.
    The dot position is interpolated along the organ's centerline.

    If coverage_counts is provided, organ points are colored red (missed)
    → yellow → green (well covered) based on how many frames saw them.
    """
    fig = plt.figure(figsize=(width/dpi, height/dpi), dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')

    organ_pts = gps_data['organ_pts']
    centerline = gps_data['centerline']
    center = gps_data['center']
    extent = gps_data['extent']
    n_frames = gps_data['n_frames']

    # Stable subsample — reuse same indices across frames so coverage colors
    # stay consistent point-for-point throughout the video.
    n_organ = len(organ_pts)
    target_pts = 18000
    sub_idx = gps_data.get('sub_idx')
    if sub_idx is None:
        if n_organ > target_pts:
            sub_idx = np.random.RandomState(42).choice(
                n_organ, target_pts, replace=False)
        else:
            sub_idx = np.arange(n_organ)
        gps_data['sub_idx'] = sub_idx
    organ_sub = organ_pts[sub_idx]

    if coverage_counts is not None:
        cov = coverage_counts[sub_idx].astype(np.float32)
        # Normalize coverage on a soft scale so even a single hit shows color.
        scale = max(float(np.percentile(cov[cov > 0], 75)) if (cov > 0).any()
                    else 1.0, 1.0)
        t = np.clip(cov / scale, 0.0, 1.0)
        # Red (0) → Yellow (0.5) → Green (1.0)
        colors = np.stack([
            np.clip(1.0 - t, 0, 1),       # R
            np.clip(t, 0, 1),              # G
            np.zeros_like(t),              # B
        ], axis=1)
        # Missed points slightly more visible than covered ones
        alphas = np.where(cov > 0, 0.18, 0.35)
        ax.scatter(organ_sub[:, 0], organ_sub[:, 1], organ_sub[:, 2],
                   c=colors, alpha=0.25, s=4, depthshade=True)
        # Emphasize misses with an outline pass
        miss = cov == 0
        if miss.any():
            ax.scatter(organ_sub[miss, 0], organ_sub[miss, 1], organ_sub[miss, 2],
                       c='red', alpha=0.35, s=5, depthshade=True)
    else:
        ax.scatter(organ_sub[:, 0], organ_sub[:, 1], organ_sub[:, 2],
                   c='salmon', alpha=0.09, s=3, depthshade=True)
    
    # Draw centerline as faded path
    n_cl = len(centerline)
    if n_cl > 1:
        segments = []
        colors = []
        for i in range(n_cl - 1):
            segments.append([centerline[i], centerline[i+1]])
            # Progress along centerline
            cl_idx = int(i / (n_cl - 1) * (n_frames - 1))
            if cl_idx <= current_idx:
                colors.append([0.2, 0.9, 0.2, 0.6])  # green = traversed
            else:
                colors.append([0.5, 0.5, 0.5, 0.2])   # gray = future
        
        lc = Line3DCollection(segments, colors=colors, linewidths=2.0)
        ax.add_collection3d(lc)
    
    # Current position along centerline
    t = current_idx / max(1, n_frames - 1)
    cl_pos_idx = min(int(t * (n_cl - 1)), n_cl - 1)
    cur_pos = centerline[cl_pos_idx]

    # Draw current position as bright large dot (scaled up for larger panel)
    ax.scatter(*cur_pos, c='lime', s=360, zorder=10, edgecolors='white',
               linewidths=2.5, depthshade=False)

    # Start (green triangle) and end (red square)
    ax.scatter(*centerline[0], c='green', s=160, marker='^', zorder=9, depthshade=False)
    ax.scatter(*centerline[-1], c='red', s=160, marker='s', zorder=9, depthshade=False)

    # Viewpoint
    ax.view_init(elev=gps_data['elev'], azim=gps_data['azim'])

    # Axis limits — tight around organ
    max_range = extent.max() * 0.6
    ax.set_xlim(center[0] - max_range, center[0] + max_range)
    ax.set_ylim(center[1] - max_range, center[1] + max_range)
    ax.set_zlim(center[2] - max_range, center[2] + max_range)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass

    # Styling
    ax.set_facecolor('#0d1117')
    fig.patch.set_facecolor('#0d1117')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('none')
    ax.yaxis.pane.set_edgecolor('none')
    ax.zaxis.pane.set_edgecolor('none')
    ax.set_axis_off()

    # Position text
    pct = t * 100
    ax.set_title(f"GPS  •  Frame {current_idx}/{n_frames-1}  •  {pct:.0f}% through organ",
                 color='#58a6ff', fontsize=16, fontweight='bold', pad=6)

    fig.subplots_adjust(left=0.0, right=1.0, top=0.94, bottom=0.0)
    
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
    plt.close(fig)
    
    return buf


def render_navigation(dataset, hyperparam, iteration, pipeline, fps=30,
                      output_name="navigation_dashboard",
                      dynamic_organ=False, voxel_size=0.5,
                      mesh_update_every=15):
    """Main rendering pipeline."""
    print("=" * 60)
    print("SURGICAL NAVIGATION DASHBOARD")
    print("=" * 60)
    
    # ---- Load model ----
    print("\nLoading trained model...")
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, hyperparam)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        bg_color = [0, 0, 0]  # Always black bg for endo scenes
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    views = scene.getTrainCameras()
    n_frames = len(views)
    n_gaussians = gaussians.get_xyz.shape[0]
    print(f"  {n_frames} training views, {n_gaussians} Gaussians")
    
    out_dir = os.path.join(dataset.model_path, "surface_reconstruction",
                           f"iteration_{scene.loaded_iter}")
    os.makedirs(out_dir, exist_ok=True)
    
    # ---- Load organ mesh ----
    # In dynamic mode we skip loading the static organ entirely; the mesh
    # will grow from per-frame TSDF fusion instead.
    organ_mesh = None
    if dynamic_organ:
        print("  Dynamic organ mode: organ mesh will grow from depth fusion "
              f"(voxel {voxel_size} mm, updating every {mesh_update_every} frames)")
    else:
        organ_path = os.path.join(out_dir, "organ_model.ply")
        if not os.path.exists(organ_path):
            organ_path = "dataset/trans_model.obj"

        if os.path.exists(organ_path):
            organ_mesh = o3d.io.read_triangle_mesh(organ_path)
            organ_mesh.compute_vertex_normals()
            print(f"  Organ model: {len(organ_mesh.vertices):,} vertices")
        else:
            print("  WARNING: No organ model found")
    
    # ---- Compute organ centerline for GPS ----
    if organ_mesh is not None:
        print("\nComputing organ centerline for GPS tracking...")
        organ_pts = np.asarray(organ_mesh.vertices)
        centerline = compute_organ_centerline(organ_mesh, n_points=200)
        print(f"  Centerline: {len(centerline)} points")

        gps_data = {
            'organ_pts': organ_pts,
            'centerline': centerline,
            'center': organ_pts.mean(axis=0),
            'extent': organ_pts.max(axis=0) - organ_pts.min(axis=0),
            'elev': 25, 'azim': 45,
            'n_frames': n_frames,
        }
    elif dynamic_organ:
        # Placeholder; populated as the TSDF volume grows. The camera
        # trajectory bounds give us something sensible to frame the view
        # around until the first mesh snapshot is available.
        gps_data = {
            'organ_pts': np.zeros((0, 3), dtype=np.float32),
            'centerline': np.zeros((2, 3), dtype=np.float32),
            'center': np.zeros(3),
            'extent': np.ones(3),
            'elev': 25, 'azim': 45,
            'n_frames': n_frames,
            'dynamic': True,
        }
    else:
        gps_data = None

    # ---- Precompute camera trajectory (positions + forwards) ----
    cam_positions = np.zeros((n_frames, 3), dtype=np.float32)
    cam_forwards = np.zeros((n_frames, 3), dtype=np.float32)
    for i, view in enumerate(views):
        cam_positions[i] = view.camera_center.detach().cpu().numpy()
        w2c = view.world_view_transform.transpose(0, 1).detach().cpu().numpy()
        c2w = np.linalg.inv(w2c)
        f = c2w[:3, 2]
        n = np.linalg.norm(f)
        cam_forwards[i] = f / n if n > 1e-6 else np.array([0, 0, -1.0])

    # Cumulative distance (mm) along the trajectory
    seg_len = np.linalg.norm(np.diff(cam_positions, axis=0), axis=1)
    cum_dist = np.concatenate([[0.0], np.cumsum(seg_len)])

    # In dynamic mode, seed GPS bounds with the camera trajectory so the
    # first few frames render something sensible before the mesh catches up.
    if dynamic_organ and gps_data is not None:
        traj_margin = 30.0  # mm of padding around the path
        gps_data['center'] = cam_positions.mean(axis=0)
        gps_data['extent'] = (
            cam_positions.max(axis=0) - cam_positions.min(axis=0)
            + 2 * traj_margin
        )

    # ---- TSDF dynamic organ builder ----
    organ_builder = None
    tsdf_intrinsic = None
    if dynamic_organ:
        view0 = views[0]
        W0, H0 = int(view0.image_width), int(view0.image_height)
        fx = fov2focal(view0.FoVx, W0)
        fy = fov2focal(view0.FoVy, H0)
        tsdf_intrinsic = o3d.camera.PinholeCameraIntrinsic(
            W0, H0, fx, fy, W0 / 2.0, H0 / 2.0)

        # ---- Figure out the real depth scale from a few sample renders ----
        # Scene coordinates in 3DGS are normalized, so the renderer's depth
        # is NOT in mm. Measure it directly and size the TSDF voxels/bounds
        # accordingly. --voxel_size / --mesh_update_every still win if the
        # caller passes them explicitly.
        with torch.no_grad():
            samples = []
            for si in range(0, n_frames, max(1, n_frames // 8))[:8]:
                r = render(views[si], gaussians, pipeline, background,
                           mode='test')
                d = r["depth"].detach().cpu().squeeze().numpy()
                vd = d[d > 0]
                if vd.size:
                    samples.extend(vd.flatten()[:3000].tolist())
        if samples:
            d99 = float(np.percentile(samples, 99))
            d01 = float(np.percentile(samples, 1))
        else:
            d99, d01 = 10.0, 0.01
        auto_depth_trunc = max(d99 * 1.3, d01 + 1e-3)
        auto_depth_min = max(d01 * 0.5, 1e-4)
        # Target ~350 voxels across the observed depth range for a crisp
        # but memory-sane mesh.
        auto_voxel = (d99 - d01) / 350.0
        auto_voxel = max(auto_voxel, 1e-4)

        # If user left defaults (voxel_size=0.5 is the old mm default), honor
        # the auto value. Respect any explicit override that's clearly
        # scene-scale.
        final_voxel = voxel_size if voxel_size < 0.3 else auto_voxel
        organ_builder = DynamicOrganBuilder(
            voxel_size=final_voxel,
            depth_trunc=auto_depth_trunc,
            depth_min=auto_depth_min,
        )
        print(f"  TSDF volume: voxel={final_voxel:.4f} scene-units, "
              f"depth=[{auto_depth_min:.3f}, {auto_depth_trunc:.3f}]  "
              f"(raw depth p1={d01:.3f}, p99={d99:.3f})")

    # ---- Coverage setup (updated incrementally in the render loop) ----
    # Coverage is only meaningful when the organ is static and known. In
    # dynamic mode the mesh IS the observed region, so the heatmap is off.
    if gps_data is not None and not dynamic_organ:
        organ_pts_f = gps_data['organ_pts'].astype(np.float32)
        n_organ_pts = len(organ_pts_f)
        cov_near = 1.0
        cov_far = float(np.median(
            np.linalg.norm(organ_pts_f - organ_pts_f.mean(0), axis=1)) * 3.0)
        hfov_deg = 140.0  # matches prepare_c3vd default
        print(f"\nCoverage tracking enabled "
              f"(hfov={hfov_deg}°, near={cov_near:.1f}, far={cov_far:.1f})")
        seen_any = np.zeros(n_organ_pts, dtype=bool)
    else:
        organ_pts_f = None
        seen_any = None

    # ---- Layout: GPS is a big panel on the right; endo + depth stacked on the left ----
    view0 = views[0]
    H, W = int(view0.image_height), int(view0.image_width)

    # Left column panels (endo on top, depth below) — keep aspect ratio
    left_panel_w = 560
    left_panel_h = int(H * left_panel_w / W)

    # GPS panel: full height of left column (endo + depth), wider for detail
    gps_h = left_panel_h * 2
    gps_w = int(gps_h * 1.35)

    vid_w = left_panel_w + gps_w
    vid_h = gps_h  # = left_panel_h * 2

    print(f"\nVideo layout: {vid_w}x{vid_h}  "
          f"(endo/depth {left_panel_w}x{left_panel_h}, gps {gps_w}x{gps_h})")

    # ---- Compute global depth range (for stable colorbar only) ----
    print("Computing global depth range for colorbar...")
    depth_samples = []
    with torch.no_grad():
        sample_indices = list(range(0, n_frames, max(1, n_frames // 10)))
        for i in sample_indices:
            result = render(views[i], gaussians, pipeline, background, mode='test')
            d = result["depth"].cpu().squeeze().numpy()
            valid = d[d > 0]
            if len(valid) > 0:
                depth_samples.extend(valid.flatten()[:2000].tolist())

    if depth_samples:
        global_vmin = float(np.percentile(depth_samples, 2))
        global_vmax = float(np.percentile(depth_samples, 98))
    else:
        global_vmin, global_vmax = 0.0, 1.0
    print(f"  Global depth range: [{global_vmin:.3f}, {global_vmax:.3f}]")
    
    # ---- Render frames ----
    video_path = os.path.join(out_dir, f"{output_name}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(video_path, fourcc, fps, (vid_w, vid_h))
    
    frames_dir = os.path.join(out_dir, "dashboard_frames")
    os.makedirs(frames_dir, exist_ok=True)
    
    print(f"\nRendering {n_frames} frames → {video_path}")
    first_frame_stats_printed = False
    with torch.no_grad():
        for idx in tqdm(range(n_frames), desc="Dashboard"):
            view = views[idx]

            # ---- Render endo view (same path as render.py) ----
            result = render(view, gaussians, pipeline, background, mode='test')
            rgb_t = result["render"].detach().cpu()  # CxHxW, float
            depth = result["depth"].cpu().squeeze().numpy()

            # Torchvision-style conversion: exactly what render.py saves
            rgb_u8 = (rgb_t.clamp(0, 1).mul(255).add_(0.5)
                          .clamp_(0, 255).permute(1, 2, 0).byte().numpy())

            # Safety net: if the Gaussian render is essentially empty
            # (e.g. wrong iteration, unconverged model), fall back to GT so
            # the panel isn't a solid color. Can be disabled.
            render_is_blank = float(rgb_t.max()) < 0.02
            used_fallback = False
            if render_is_blank and view.original_image is not None:
                gt_t = view.original_image.detach().cpu().clamp(0, 1)
                rgb_u8 = (gt_t.mul(255).add_(0.5).clamp_(0, 255)
                              .permute(1, 2, 0).byte().numpy())
                used_fallback = True

            if not first_frame_stats_printed:
                print(f"  First render: max={float(rgb_t.max()):.3f}, "
                      f"mean={float(rgb_t.mean()):.3f}, "
                      f"depth max={float(depth.max()):.3f}, "
                      f"depth mean>0={float(depth[depth>0].mean() if (depth>0).any() else 0):.3f}")
                if render_is_blank:
                    print("  WARNING: rendered image is near-black — "
                          "check that --iteration points to a trained checkpoint. "
                          "Falling back to GT for the endo panel.")
                first_frame_stats_printed = True

            rgb_bgr = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)

            # ---- Integrate this frame into the TSDF (dynamic mode) ----
            if organ_builder is not None and not render_is_blank:
                w2c = view.world_view_transform.transpose(0, 1) \
                    .detach().cpu().numpy().astype(np.float64)
                organ_builder.integrate(rgb_u8, depth, w2c, tsdf_intrinsic)
                if (idx % mesh_update_every == 0) or (idx == n_frames - 1):
                    mesh = organ_builder.extract_mesh()
                    if mesh is not None:
                        new_pts = np.asarray(mesh.vertices)
                        gps_data['organ_pts'] = new_pts
                        gps_data['sub_idx'] = None  # re-sample for new mesh
                        # Keep view centered on the growing mesh + the camera
                        # path combined so nothing falls off-screen.
                        combined = np.vstack([new_pts, cam_positions])
                        gps_data['center'] = combined.mean(axis=0)
                        gps_data['extent'] = (
                            combined.max(axis=0) - combined.min(axis=0) + 10.0
                        )

            # Per-frame valid mask (pixels where Gaussians covered the ray)
            valid = depth > 0
            depth_colored, frame_vmin, frame_vmax = depth_to_colormap(
                depth, valid_mask=valid
            )

            endo_panel = cv2.resize(rgb_bgr, (left_panel_w, left_panel_h))
            depth_panel = cv2.resize(depth_colored, (left_panel_w, left_panel_h))

            # Labels on endo panel
            endo_label = "ENDOSCOPIC VIEW (GT fallback)" if used_fallback \
                else "ENDOSCOPIC VIEW"
            cv2.putText(endo_panel, endo_label, (10, 28),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 200), 2)
            pct = idx / max(1, n_frames - 1) * 100
            cv2.putText(endo_panel, f"Frame {idx}/{n_frames-1}",
                       (10, left_panel_h - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

            # Labels on depth panel
            cv2.putText(depth_panel, "DEPTH MAP", (10, 28),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(depth_panel,
                       f"range: {frame_vmin:.2f} - {frame_vmax:.2f}",
                       (10, left_panel_h - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            # Depth colorbar (near = red/top, far = blue/bottom)
            bar_x = left_panel_w - 30
            bar_top, bar_h_range = 40, left_panel_h - 80
            for y in range(bar_top, bar_top + bar_h_range):
                t = (y - bar_top) / bar_h_range
                val = int((1.0 - t) * 255)  # match inverted colormap (near=red on top)
                cb = cv2.applyColorMap(
                    np.array([[val]], dtype=np.uint8), cv2.COLORMAP_TURBO)[0, 0]
                cv2.line(depth_panel, (bar_x, y), (bar_x + 15, y),
                         cb.tolist(), 1)
            cv2.putText(depth_panel, "near", (bar_x - 18, bar_top - 4),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
            cv2.putText(depth_panel, "far",
                       (bar_x - 14, bar_top + bar_h_range + 14),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

            # ---- Update running coverage with this frame's frustum ----
            if seen_any is not None:
                vis = frustum_visibility(
                    cam_positions[idx], cam_forwards[idx], organ_pts_f,
                    hfov_deg=hfov_deg, near=cov_near, far=cov_far,
                )
                seen_any |= vis

            # ---- GPS panel (big, on the right) ----
            if gps_data is not None:
                current_cov = (seen_any.astype(np.int32)
                               if seen_any is not None else None)
                gps_img = render_gps_frame(
                    gps_data, idx, coverage_counts=current_cov,
                    width=gps_w, height=gps_h, dpi=110)
                gps_bgr = cv2.cvtColor(gps_img, cv2.COLOR_RGB2BGR)
                if gps_bgr.shape[1] != gps_w or gps_bgr.shape[0] != gps_h:
                    gps_bgr = cv2.resize(gps_bgr, (gps_w, gps_h))
                gps_panel = gps_bgr
            else:
                gps_panel = np.zeros((gps_h, gps_w, 3), dtype=np.uint8)
                cv2.putText(gps_panel, "GPS: No organ model",
                           (gps_w // 4, gps_h // 2),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 100, 100), 2)

            # ---- Stats overlay: withdrawal timer, speed, coverage/mesh ----
            elapsed_s = idx / float(fps)
            dist_mm = float(cum_dist[idx])
            speed_mms = float(seg_len[idx - 1] * fps) if idx > 0 else 0.0
            cov_pct = (float(seen_any.sum())
                       / max(len(gps_data['organ_pts']), 1) * 100.0
                       if seen_any is not None else 0.0)
            n_mesh_verts = (len(gps_data['organ_pts'])
                            if (gps_data is not None and dynamic_organ) else 0)

            # HUD strip at top of GPS panel
            hud_h = 74
            overlay = gps_panel.copy()
            cv2.rectangle(overlay, (0, 0), (gps_w, hud_h), (12, 12, 18), -1)
            cv2.addWeighted(overlay, 0.72, gps_panel, 0.28, 0, gps_panel)
            # Guideline: withdrawal >= 6 min. Color speed if too fast.
            good_speed = 1.0 <= speed_mms <= 6.0
            speed_color = (80, 230, 120) if good_speed else (60, 120, 240)
            time_color = (80, 230, 120) if elapsed_s >= 360 else (220, 220, 220)
            cv2.putText(gps_panel, f"Withdrawal  {format_time_mmss(elapsed_s)}",
                        (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.78,
                        time_color, 2)
            cv2.putText(gps_panel, "(target >= 06:00)",
                        (18, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (140, 140, 150), 1)
            cv2.putText(gps_panel, f"Speed  {speed_mms:5.1f} mm/s",
                        (int(gps_w * 0.32), 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.78, speed_color, 2)
            cv2.putText(gps_panel, f"Path  {dist_mm:6.1f} mm",
                        (int(gps_w * 0.32), 58),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 210), 1)
            if dynamic_organ:
                # Show reconstruction progress instead of coverage %
                mesh_color = (80, 230, 120) if n_mesh_verts > 20000 else \
                             ((60, 200, 230) if n_mesh_verts > 5000
                              else (60, 120, 240))
                cv2.putText(gps_panel,
                            f"Mesh  {n_mesh_verts:,} pts",
                            (int(gps_w * 0.62), 32),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.78, mesh_color, 2)
                cv2.putText(gps_panel,
                            f"(live TSDF fusion, {organ_builder.n_integrated}"
                            f" frames fused)",
                            (int(gps_w * 0.62), 58),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                            (160, 160, 170), 1)
            else:
                cov_color = (80, 230, 120) if cov_pct >= 80 else \
                            ((60, 200, 230) if cov_pct >= 50
                             else (60, 120, 240))
                cv2.putText(gps_panel,
                            f"Coverage  {cov_pct:5.1f}%",
                            (int(gps_w * 0.62), 32),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.78, cov_color, 2)
                bar_x1, bar_x2 = int(gps_w * 0.62), int(gps_w * 0.95)
                bar_y = 52
                cv2.rectangle(gps_panel, (bar_x1, bar_y),
                              (bar_x2, bar_y + 10), (50, 50, 60), -1)
                fill_x = int(bar_x1 + (bar_x2 - bar_x1) * cov_pct / 100.0)
                cv2.rectangle(gps_panel, (bar_x1, bar_y),
                              (fill_x, bar_y + 10), cov_color, -1)

            # Legend at the bottom of the GPS panel
            leg_y = gps_h - 28
            if dynamic_organ:
                cv2.putText(gps_panel,
                            "Organ surface built live from per-frame depth "
                            "fusion (TSDF, voxel "
                            f"{organ_builder.voxel_size:.2f} mm)",
                            (14, leg_y + 4), cv2.FONT_HERSHEY_SIMPLEX,
                            0.45, (180, 180, 190), 1)
            else:
                cv2.circle(gps_panel, (20, leg_y), 6, (60, 120, 240), -1)
                cv2.putText(gps_panel, "missed",
                            (32, leg_y + 4), cv2.FONT_HERSHEY_SIMPLEX,
                            0.45, (200, 200, 210), 1)
                cv2.circle(gps_panel, (110, leg_y), 6, (60, 230, 230), -1)
                cv2.putText(gps_panel, "partial",
                            (122, leg_y + 4), cv2.FONT_HERSHEY_SIMPLEX,
                            0.45, (200, 200, 210), 1)
                cv2.circle(gps_panel, (205, leg_y), 6, (80, 230, 120), -1)
                cv2.putText(gps_panel, "well covered",
                            (217, leg_y + 4), cv2.FONT_HERSHEY_SIMPLEX,
                            0.45, (200, 200, 210), 1)

            # ---- Composite: [endo / depth] | [gps] ----
            canvas = np.zeros((vid_h, vid_w, 3), dtype=np.uint8)
            canvas[:left_panel_h, :left_panel_w] = endo_panel
            canvas[left_panel_h:left_panel_h * 2, :left_panel_w] = depth_panel
            canvas[:gps_h, left_panel_w:left_panel_w + gps_w] = gps_panel

            # Separators
            cv2.line(canvas, (left_panel_w, 0),
                     (left_panel_w, vid_h), (60, 60, 60), 2)
            cv2.line(canvas, (0, left_panel_h),
                     (left_panel_w, left_panel_h), (60, 60, 60), 2)

            writer.write(canvas)

            if idx % max(1, n_frames // 20) == 0:
                cv2.imwrite(os.path.join(frames_dir, f"frame_{idx:04d}.png"),
                            canvas)
    
    writer.release()

    # ---- Save the final dynamic mesh ----
    final_mesh_path = None
    if organ_builder is not None:
        final_mesh = organ_builder.extract_mesh(min_vertices=0)
        if final_mesh is not None and len(final_mesh.vertices) > 0:
            final_mesh_path = os.path.join(out_dir, "dynamic_organ_mesh.ply")
            o3d.io.write_triangle_mesh(final_mesh_path, final_mesh)
            print(f"  Dynamic organ mesh: {final_mesh_path} "
                  f"({len(final_mesh.vertices):,} verts, "
                  f"{len(final_mesh.triangles):,} tris)")

    print(f"\n{'=' * 60}")
    print("DASHBOARD COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Video: {video_path}")
    print(f"  Key frames: {frames_dir}/")
    print(f"  Resolution: {vid_w}x{vid_h} @ {fps}fps")
    print(f"  Duration: {n_frames/fps:.1f}s ({n_frames} frames)")
    if final_mesh_path:
        print(f"  Final reconstructed mesh: {final_mesh_path}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Render surgical navigation dashboard video")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    hyperparam = ModelHiddenParams(parser)
    parser.add_argument("--iteration", default=3000, type=int)
    parser.add_argument("--configs", type=str, default="")
    parser.add_argument("--fps", default=30, type=int, help="Output video FPS")
    parser.add_argument("--dynamic_organ", action="store_true",
                        help="Build organ mesh live via per-frame TSDF fusion "
                             "instead of using a pre-op static mesh")
    parser.add_argument("--voxel_size", default=0.5, type=float,
                        help="TSDF voxel size in mm (dynamic_organ only)")
    parser.add_argument("--mesh_update_every", default=15, type=int,
                        help="Extract mesh every N frames (dynamic_organ only)")

    args = get_combined_args(parser)

    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)

    safe_state(False)

    render_navigation(
        model.extract(args),
        hyperparam.extract(args),
        args.iteration,
        pipeline.extract(args),
        fps=args.fps,
        dynamic_organ=args.dynamic_organ,
        voxel_size=args.voxel_size,
        mesh_update_every=args.mesh_update_every,
    )
