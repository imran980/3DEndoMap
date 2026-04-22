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

import open3d as o3d


def depth_to_colormap(depth_np, vmin=None, vmax=None):
    """Convert depth map to a colored visualization."""
    if vmin is None:
        valid = depth_np[depth_np > 0]
        if len(valid) == 0:
            return np.zeros((*depth_np.shape, 3), dtype=np.uint8)
        vmin = np.percentile(valid, 2)
        vmax = np.percentile(valid, 98)
    
    depth_norm = np.clip((depth_np - vmin) / max(vmax - vmin, 1e-6), 0, 1)
    depth_u8 = (depth_norm * 255).astype(np.uint8)
    colored = cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)
    colored[depth_np <= 0] = 0
    return colored


def boost_brightness(rgb_np, gamma=0.45, gain=2.5):
    """
    Boost brightness of dark renders using gamma correction + gain.
    Input/output: float32 [0, 1] array (H, W, 3)
    """
    boosted = np.clip(rgb_np * gain, 0, 1)
    boosted = np.power(boosted, gamma)
    return boosted


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


def render_gps_frame(gps_data, current_idx, width=640, height=480, dpi=100):
    """
    Render a single GPS frame showing the organ + current camera dot.
    The dot position is interpolated along the organ's centerline.
    """
    fig = plt.figure(figsize=(width/dpi, height/dpi), dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')
    
    organ_pts = gps_data['organ_pts']
    centerline = gps_data['centerline']
    center = gps_data['center']
    extent = gps_data['extent']
    n_frames = gps_data['n_frames']
    
    # Subsample organ for speed (8000 for better shape)
    n_organ = len(organ_pts)
    if n_organ > 8000:
        idx = np.random.RandomState(42).choice(n_organ, 8000, replace=False)
        organ_sub = organ_pts[idx]
    else:
        organ_sub = organ_pts
    
    # Draw organ as semi-transparent scatter
    ax.scatter(organ_sub[:, 0], organ_sub[:, 1], organ_sub[:, 2],
               c='salmon', alpha=0.06, s=2, depthshade=True)
    
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
    
    # Draw current position as bright large dot
    ax.scatter(*cur_pos, c='lime', s=200, zorder=10, edgecolors='white',
               linewidths=2, depthshade=False)
    
    # Start (green triangle) and end (red square)
    ax.scatter(*centerline[0], c='green', s=100, marker='^', zorder=9, depthshade=False)
    ax.scatter(*centerline[-1], c='red', s=100, marker='s', zorder=9, depthshade=False)
    
    # Viewpoint
    ax.view_init(elev=gps_data['elev'], azim=gps_data['azim'])
    
    # Axis limits — tight around organ
    max_range = extent.max() * 0.6
    ax.set_xlim(center[0] - max_range, center[0] + max_range)
    ax.set_ylim(center[1] - max_range, center[1] + max_range)
    ax.set_zlim(center[2] - max_range, center[2] + max_range)
    
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
                 color='#58a6ff', fontsize=11, fontweight='bold', pad=2)
    
    fig.tight_layout(pad=0.5)
    
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
    plt.close(fig)
    
    return buf


def render_navigation(dataset, hyperparam, iteration, pipeline, fps=30,
                      output_name="navigation_dashboard"):
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
    organ_path = os.path.join(out_dir, "organ_model.ply")
    if not os.path.exists(organ_path):
        organ_path = "dataset/trans_model.obj"
    
    organ_mesh = None
    if os.path.exists(organ_path):
        organ_mesh = o3d.io.read_triangle_mesh(organ_path)
        organ_mesh.compute_vertex_normals()
        print(f"  Organ model: {len(organ_mesh.vertices):,} vertices")
    else:
        print("  WARNING: No organ model found")
    
    # ---- Compute organ centerline for GPS ----
    print("\nComputing organ centerline for GPS tracking...")
    if organ_mesh is not None:
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
    else:
        gps_data = None
    
    # ---- Layout ----
    view0 = views[0]
    H, W = int(view0.image_height), int(view0.image_width)
    
    panel_h = min(H, 540)
    scale = panel_h / H
    panel_w = int(W * scale)
    
    gps_w = panel_w * 2
    gps_h = int(panel_h * 0.75)
    
    vid_w = panel_w * 2
    vid_h = panel_h + gps_h
    
    print(f"\nVideo layout: {vid_w}x{vid_h}")
    
    # ---- Compute depth range ----
    print("Computing depth range...")
    depth_samples = []
    rgb_max_vals = []
    with torch.no_grad():
        sample_indices = list(range(0, n_frames, max(1, n_frames // 10)))
        for i in sample_indices:
            result = render(views[i], gaussians, pipeline, background, mode='test')
            d = result["depth"].cpu().squeeze().numpy()
            r = result["render"].cpu().permute(1, 2, 0).numpy()
            valid = d[d > 0]
            if len(valid) > 0:
                depth_samples.extend(valid.flatten()[:1000].tolist())
            rgb_max_vals.append(r.max())
    
    if depth_samples:
        depth_vmin = np.percentile(depth_samples, 2)
        depth_vmax = np.percentile(depth_samples, 98)
    else:
        depth_vmin, depth_vmax = 0, 1
    
    max_rgb = max(rgb_max_vals) if rgb_max_vals else 0.01
    print(f"  Depth range: [{depth_vmin:.3f}, {depth_vmax:.3f}]")
    print(f"  Max RGB value: {max_rgb:.4f}")
    if max_rgb < 0.1:
        print(f"  ⚠ Model renders are very dark — applying brightness boost")
    
    # ---- Render frames ----
    video_path = os.path.join(out_dir, f"{output_name}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(video_path, fourcc, fps, (vid_w, vid_h))
    
    frames_dir = os.path.join(out_dir, "dashboard_frames")
    os.makedirs(frames_dir, exist_ok=True)
    
    print(f"\nRendering {n_frames} frames → {video_path}")
    with torch.no_grad():
        for idx in tqdm(range(n_frames), desc="Dashboard"):
            view = views[idx]
            
            # ---- Render endo view ----
            result = render(view, gaussians, pipeline, background, mode='test')
            rgb = result["render"].cpu().permute(1, 2, 0).numpy()
            depth = result["depth"].cpu().squeeze().numpy()
            
            # Boost if dark
            if max_rgb < 0.3:
                rgb = boost_brightness(rgb, gamma=0.4, gain=3.0)
            
            rgb_u8 = (rgb.clip(0, 1) * 255).astype(np.uint8)
            rgb_bgr = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)
            
            depth_colored = depth_to_colormap(depth, depth_vmin, depth_vmax)
            
            endo_panel = cv2.resize(rgb_bgr, (panel_w, panel_h))
            depth_panel = cv2.resize(depth_colored, (panel_w, panel_h))
            
            # Labels
            cv2.putText(endo_panel, "ENDOSCOPIC VIEW", (10, 28),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 200), 2)
            pct = idx / max(1, n_frames - 1) * 100
            cv2.putText(endo_panel, f"Frame {idx}/{n_frames-1}", (10, panel_h - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
            
            cv2.putText(depth_panel, "DEPTH MAP", (10, 28),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Depth colorbar
            bar_x = panel_w - 30
            bar_h_range = panel_h - 60
            for y in range(30, 30 + bar_h_range):
                t = (y - 30) / bar_h_range
                val = int(t * 255)
                cb = cv2.applyColorMap(np.array([[val]], dtype=np.uint8), cv2.COLORMAP_TURBO)[0, 0]
                cv2.line(depth_panel, (bar_x, y), (bar_x + 15, y), cb.tolist(), 1)
            cv2.putText(depth_panel, f"{depth_vmin:.1f}", (bar_x - 8, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
            cv2.putText(depth_panel, f"{depth_vmax:.1f}", (bar_x - 8, 30 + bar_h_range + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
            
            # ---- GPS panel ----
            if gps_data is not None:
                gps_img = render_gps_frame(gps_data, idx, width=gps_w, height=gps_h, dpi=100)
                gps_bgr = cv2.cvtColor(gps_img, cv2.COLOR_RGB2BGR)
                gps_panel = cv2.resize(gps_bgr, (gps_w, gps_h))
            else:
                gps_panel = np.zeros((gps_h, gps_w, 3), dtype=np.uint8)
                cv2.putText(gps_panel, "GPS: No organ model", (gps_w//4, gps_h//2),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 100, 100), 2)
            
            # ---- Composite ----
            canvas = np.zeros((vid_h, vid_w, 3), dtype=np.uint8)
            canvas[:panel_h, :panel_w] = endo_panel
            canvas[:panel_h, panel_w:panel_w*2] = depth_panel
            canvas[panel_h:panel_h+gps_h, :gps_w] = gps_panel
            
            # Separators
            cv2.line(canvas, (panel_w, 0), (panel_w, panel_h), (60, 60, 60), 2)
            cv2.line(canvas, (0, panel_h), (vid_w, panel_h), (60, 60, 60), 2)
            
            writer.write(canvas)
            
            if idx % max(1, n_frames // 20) == 0:
                cv2.imwrite(os.path.join(frames_dir, f"frame_{idx:04d}.png"), canvas)
    
    writer.release()
    
    print(f"\n{'=' * 60}")
    print("DASHBOARD COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Video: {video_path}")
    print(f"  Key frames: {frames_dir}/")
    print(f"  Resolution: {vid_w}x{vid_h} @ {fps}fps")
    print(f"  Duration: {n_frames/fps:.1f}s ({n_frames} frames)")


if __name__ == "__main__":
    parser = ArgumentParser(description="Render surgical navigation dashboard video")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    hyperparam = ModelHiddenParams(parser)
    parser.add_argument("--iteration", default=3000, type=int)
    parser.add_argument("--configs", type=str, default="")
    parser.add_argument("--fps", default=30, type=int, help="Output video FPS")
    
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
    )
