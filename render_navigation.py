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

    # Subsample organ for speed (more points now that the panel is larger)
    n_organ = len(organ_pts)
    target_pts = 18000
    if n_organ > target_pts:
        idx = np.random.RandomState(42).choice(n_organ, target_pts, replace=False)
        organ_sub = organ_pts[idx]
    else:
        organ_sub = organ_pts

    # Draw organ as semi-transparent scatter
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

            # ---- GPS panel (big, on the right) ----
            if gps_data is not None:
                gps_img = render_gps_frame(
                    gps_data, idx, width=gps_w, height=gps_h, dpi=110)
                gps_bgr = cv2.cvtColor(gps_img, cv2.COLOR_RGB2BGR)
                if gps_bgr.shape[1] != gps_w or gps_bgr.shape[0] != gps_h:
                    gps_bgr = cv2.resize(gps_bgr, (gps_w, gps_h))
                gps_panel = gps_bgr
            else:
                gps_panel = np.zeros((gps_h, gps_w, 3), dtype=np.uint8)
                cv2.putText(gps_panel, "GPS: No organ model",
                           (gps_w // 4, gps_h // 2),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 100, 100), 2)

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
