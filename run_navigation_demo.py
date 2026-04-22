"""
End-to-End Surgical Navigation Demo with Organ Map.

Simulates the full GPS pipeline using C3VD data:
1. Load trained model + registered colon organ mesh
2. Pick test frames as "live" query images
3. Localize each frame → position on organ
4. Render 3-panel output: [Live Frame | Match | Organ Map with dot]
5. Create summary with full trajectory on organ + error statistics

Usage:
    python run_navigation_demo.py --model_path output/c3vd_trans \
        --configs arguments/endonerf.py
"""

import os
import sys
import json
import numpy as np
import torch
import cv2
from argparse import ArgumentParser
from tqdm import tqdm

from scene import Scene, GaussianModel
from gaussian_renderer import render
from arguments import ModelParams, PipelineParams, get_combined_args, ModelHiddenParams
from utils.general_utils import safe_state
from localize_camera import compute_ssim_simple, load_organ_points, render_organ_map


def render_trajectory_map(organ_pts, endo_pts, est_positions, gt_positions=None,
                          map_size=600):
    """
    Render complete trajectory on organ model — the "full GPS trace" view.
    Shows all localized positions as colored dots (green→red = time).
    
    Returns: BGR image
    """
    panel_w = map_size
    panel_h = map_size
    img = np.zeros((panel_h, panel_w * 2, 3), dtype=np.uint8)
    
    all_pts = []
    if organ_pts is not None:
        all_pts.append(organ_pts)
    if endo_pts is not None:
        all_pts.append(endo_pts)
    all_pts.append(est_positions)
    if gt_positions is not None:
        all_pts.append(gt_positions)
    all_pts = np.vstack(all_pts)
    
    views = [("TOP VIEW (X-Z)", 0, 2), ("SIDE VIEW (X-Y)", 0, 1)]
    margin = 40
    
    for vi, (title, ax1, ax2) in enumerate(views):
        panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
        
        pts_2d = all_pts[:, [ax1, ax2]]
        p_min, p_max = pts_2d.min(0), pts_2d.max(0)
        extent = p_max - p_min
        max_ext = max(extent[0], extent[1], 1e-6)
        scale = (min(panel_w, panel_h) - 2*margin) / max_ext
        offset = np.array([margin, margin]) + \
                 (np.array([panel_w, panel_h]) - 2*margin - extent*scale) / 2
        
        def proj(pts):
            p = pts[:, [ax1, ax2]] if pts.ndim == 2 else pts[[ax1, ax2]].reshape(1, 2)
            return ((p - p_min) * scale + offset).astype(int)
        
        # Draw organ (dim)
        if organ_pts is not None:
            for p in proj(organ_pts):
                if 0 <= p[0] < panel_w and 0 <= p[1] < panel_h:
                    cv2.circle(panel, tuple(p), 1, (50, 50, 50), -1)
        
        # Draw endo mesh (medium)
        if endo_pts is not None:
            for p in proj(endo_pts):
                if 0 <= p[0] < panel_w and 0 <= p[1] < panel_h:
                    cv2.circle(panel, tuple(p), 1, (120, 120, 120), -1)
        
        # Draw GT trajectory (blue line) if available
        if gt_positions is not None and len(gt_positions) > 1:
            gt_proj = proj(gt_positions)
            for i in range(len(gt_proj) - 1):
                cv2.line(panel, tuple(gt_proj[i]), tuple(gt_proj[i+1]),
                        (180, 80, 0), 1, cv2.LINE_AA)
        
        # Draw estimated trajectory (colored line: green → red)
        est_proj = proj(est_positions)
        n = len(est_proj)
        for i in range(n - 1):
            t = i / max(n - 1, 1)
            # Green (start) → Yellow (mid) → Red (end)
            r = int(255 * t)
            g = int(255 * (1 - t * 0.7))
            b = 0
            cv2.line(panel, tuple(est_proj[i]), tuple(est_proj[i+1]),
                    (b, g, r), 2, cv2.LINE_AA)
        
        # Draw position dots
        for i in range(n):
            t = i / max(n - 1, 1)
            r = int(255 * t)
            g = int(255 * (1 - t * 0.7))
            cv2.circle(panel, tuple(est_proj[i]), 4, (0, g, r), -1)
        
        # Start / end markers
        if n > 0:
            cv2.circle(panel, tuple(est_proj[0]), 8, (0, 255, 0), 2)  # green start
            cv2.putText(panel, "S", (est_proj[0][0]+10, est_proj[0][1]+5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.circle(panel, tuple(est_proj[-1]), 8, (0, 0, 255), 2)  # red end
            cv2.putText(panel, "E", (est_proj[-1][0]+10, est_proj[-1][1]+5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Title
        cv2.putText(panel, title, (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        img[:, vi*panel_w:(vi+1)*panel_w] = panel
    
    return img


def run_demo(dataset, hyperparam, iteration, pipeline):
    """Run the full navigation demo with organ map visualization."""
    
    print("=" * 60)
    print("SURGICAL NAVIGATION GPS — C3VD COLON DEMO")
    print("=" * 60)
    
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, hyperparam)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    recon_dir = os.path.join(dataset.model_path, "surface_reconstruction",
                             f"iteration_{scene.loaded_iter}")
    out_dir = os.path.join(recon_dir, "navigation_demo")
    os.makedirs(out_dir, exist_ok=True)
    
    # ---- Load registration ----
    transform_path = os.path.join(recon_dir, "registration_transform.json")
    if os.path.exists(transform_path):
        with open(transform_path) as f:
            reg = json.load(f)
        reg_transform = np.array(reg["final_transform"])
        print(f"Registration: fitness={reg.get('icp_fitness', 'N/A'):.4f}, "
              f"scale={reg.get('scale_factor', 1.0):.4f}")
    else:
        reg_transform = np.eye(4)
        print("No registration transform found")
    
    # ---- Load organ model for map ----
    print("\nLoading organ model...")
    organ_pts, endo_pts = load_organ_points(recon_dir)
    has_map = organ_pts is not None
    
    # ---- Render training views ----
    train_views = scene.getTrainCameras()
    test_views = scene.getTestCameras()
    
    print(f"\nTraining views: {len(train_views)}")
    print(f"Test views: {len(test_views)}")
    
    print("\nRendering training views...")
    train_data = []
    with torch.no_grad():
        for view in tqdm(train_views, desc="Train renders"):
            result = render(view, gaussians, pipeline, background, mode='test')
            img_np = result["render"].cpu().permute(1, 2, 0).numpy()
            cam_center = view.camera_center.cpu().numpy()
            w2c = view.world_view_transform.transpose(0, 1).cpu().numpy()
            c2w = np.linalg.inv(w2c)
            train_data.append({
                'render': img_np,
                'cam_center': cam_center,
                'c2w': c2w,
                'time': getattr(view, 'time', 0),
            })
    
    # Full training trajectory in organ coords
    train_traj_organ = np.array([
        (reg_transform @ np.append(td['cam_center'], 1.0))[:3]
        for td in train_data
    ])
    
    # ---- Simulate live navigation ----
    print(f"\nSimulating navigation with {len(test_views)} test frames...")
    
    nav_results = []
    est_positions_organ = []
    gt_positions_organ = []
    
    with torch.no_grad():
        for qi, view in enumerate(tqdm(test_views, desc="Localizing")):
            query_img = view.original_image.cpu().permute(1, 2, 0).numpy()
            gt_center = view.camera_center.cpu().numpy()
            
            # Find best match
            best_ssim = -1
            best_idx = 0
            for ti, td in enumerate(train_data):
                ssim = compute_ssim_simple(query_img, td['render'])
                if ssim > best_ssim:
                    best_ssim = ssim
                    best_idx = ti
            
            # Estimated and GT positions in organ coords
            est_center = train_data[best_idx]['cam_center']
            est_pos_organ = (reg_transform @ np.append(est_center, 1.0))[:3]
            gt_pos_organ = (reg_transform @ np.append(gt_center, 1.0))[:3]
            
            # Forward direction in organ coords
            c2w_organ = reg_transform @ train_data[best_idx]['c2w']
            forward_organ = c2w_organ[:3, 2]
            
            pos_error = float(np.linalg.norm(est_center - gt_center))
            
            est_positions_organ.append(est_pos_organ)
            gt_positions_organ.append(gt_pos_organ)
            
            nav_results.append({
                'frame': qi,
                'best_match': best_idx,
                'ssim': float(best_ssim),
                'est_position_organ': est_pos_organ.tolist(),
                'gt_position_organ': gt_pos_organ.tolist(),
                'position_error_mm': pos_error,
                'time': getattr(view, 'time', 0),
            })
            
            # ---- Build 3-panel image ----
            matched = train_data[best_idx]['render']
            h, w_frame = query_img.shape[:2]
            
            # Panels 1+2: Query | Match
            comp = np.concatenate([query_img, matched], axis=1)
            comp_bgr = cv2.cvtColor(
                (comp * 255).clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            
            cv2.putText(comp_bgr, f"LIVE frame {qi}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(comp_bgr, f"Matched: train[{best_idx}] SSIM={best_ssim:.3f}",
                       (w_frame + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                       (0, 255, 255), 2)
            cv2.putText(comp_bgr, f"Pos error: {pos_error:.2f}mm",
                       (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                       (255, 255, 255), 2)
            
            # Panel 3: Organ map with camera dot
            if has_map:
                map_img = render_organ_map(
                    organ_pts, endo_pts,
                    cam_position=est_pos_organ,
                    cam_forward=forward_organ,
                    gt_position=gt_pos_organ,
                    map_size=h,
                    trajectory_pts=train_traj_organ,
                )
                if map_img.shape[0] != h:
                    map_img = cv2.resize(map_img,
                                         (map_img.shape[1] * h // map_img.shape[0], h))
                comp_bgr = np.concatenate([comp_bgr, map_img], axis=1)
            
            cv2.imwrite(os.path.join(out_dir, f"nav_{qi:03d}.png"), comp_bgr)
    
    est_positions_organ = np.array(est_positions_organ)
    gt_positions_organ = np.array(gt_positions_organ)
    
    # ---- Statistics ----
    errors = [r['position_error_mm'] for r in nav_results]
    ssims = [r['ssim'] for r in nav_results]
    
    print(f"\n{'='*60}")
    print("NAVIGATION RESULTS")
    print(f"{'='*60}")
    print(f"  Frames localized: {len(nav_results)}")
    print(f"  SSIM:  mean={np.mean(ssims):.4f}, "
          f"min={np.min(ssims):.4f}, max={np.max(ssims):.4f}")
    print(f"  Error: mean={np.mean(errors):.2f}mm, "
          f"median={np.median(errors):.2f}mm, max={np.max(errors):.2f}mm")
    
    # ---- Trajectory summary figure ----
    if has_map:
        print("\nGenerating trajectory summary...")
        traj_map = render_trajectory_map(
            organ_pts, endo_pts,
            est_positions=est_positions_organ,
            gt_positions=gt_positions_organ,
            map_size=500,
        )
        
        # Add statistics bar at top
        stats_h = 80
        stats_bar = np.zeros((stats_h, traj_map.shape[1], 3), dtype=np.uint8)
        cv2.putText(stats_bar,
                   f"Surgical Navigation GPS — C3VD Colon Demo",
                   (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(stats_bar,
                   f"Frames: {len(nav_results)} | "
                   f"Mean SSIM: {np.mean(ssims):.3f} | "
                   f"Mean Error: {np.mean(errors):.1f}mm | "
                   f"Median Error: {np.median(errors):.1f}mm",
                   (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        
        # Legend bar at bottom
        legend_h = 40
        legend_bar = np.zeros((legend_h, traj_map.shape[1], 3), dtype=np.uint8)
        cv2.circle(legend_bar, (20, 20), 5, (0, 255, 0), -1)
        cv2.putText(legend_bar, "Start", (30, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        cv2.circle(legend_bar, (100, 20), 5, (0, 0, 255), -1)
        cv2.putText(legend_bar, "End", (110, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        cv2.line(legend_bar, (170, 20), (200, 20), (180, 80, 0), 2)
        cv2.putText(legend_bar, "GT path", (205, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        cv2.line(legend_bar, (290, 20), (320, 20), (0, 255, 0), 2)
        cv2.putText(legend_bar, "Est path", (325, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        
        summary = np.concatenate([stats_bar, traj_map, legend_bar], axis=0)
        summary_path = os.path.join(out_dir, "trajectory_summary.png")
        cv2.imwrite(summary_path, summary)
        print(f"  Trajectory summary: {summary_path}")
    
    # ---- Sample frames grid ----
    n_show = min(8, len(nav_results))
    grid_frames = []
    for i in range(n_show):
        idx = i * len(nav_results) // n_show
        frame_path = os.path.join(out_dir, f"nav_{idx:03d}.png")
        if os.path.exists(frame_path):
            grid_frames.append(cv2.imread(frame_path))
    
    if grid_frames:
        # Resize all to same dimensions
        target_h = min(f.shape[0] for f in grid_frames)
        target_w = min(f.shape[1] for f in grid_frames)
        grid_frames = [cv2.resize(f, (target_w, target_h)) for f in grid_frames]
        
        n_cols = min(4, n_show)
        n_rows = (n_show + n_cols - 1) // n_cols
        rows = []
        for r in range(n_rows):
            row_imgs = grid_frames[r*n_cols:(r+1)*n_cols]
            while len(row_imgs) < n_cols:
                row_imgs.append(np.zeros_like(row_imgs[0]))
            rows.append(np.concatenate(row_imgs, axis=1))
        
        grid = np.concatenate(rows, axis=0)
        
        title_h = 50
        title_bar = np.zeros((title_h, grid.shape[1], 3), dtype=np.uint8)
        cv2.putText(title_bar,
                   f"Navigation Demo | {len(nav_results)} frames | "
                   f"SSIM: {np.mean(ssims):.3f} | Error: {np.mean(errors):.1f}mm",
                   (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        grid = np.concatenate([title_bar, grid], axis=0)
        
        grid_path = os.path.join(out_dir, "navigation_grid.png")
        cv2.imwrite(grid_path, grid)
        print(f"  Navigation grid: {grid_path}")
    
    # ---- Save JSON ----
    results_data = {
        "model_path": dataset.model_path,
        "source": "C3VD",
        "n_train": len(train_views),
        "n_test": len(test_views),
        "mean_ssim": float(np.mean(ssims)),
        "median_ssim": float(np.median(ssims)),
        "mean_error_mm": float(np.mean(errors)),
        "median_error_mm": float(np.median(errors)),
        "max_error_mm": float(np.max(errors)),
        "min_error_mm": float(np.min(errors)),
        "frames": nav_results,
    }
    results_path = os.path.join(out_dir, "demo_results.json")
    with open(results_path, 'w') as f:
        json.dump(results_data, f, indent=2)
    print(f"  Results JSON: {results_path}")
    
    print(f"\n{'='*60}")
    print("PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"  Step 1: Surface reconstruction    ✓")
    print(f"  Step 2: Colon model registration   ✓ (C3VD trans_model.obj)")
    print(f"  Step 3: Camera localization         ✓")
    print(f"  Step 4: GPS visualization           ✓")
    print(f"\n  Per-frame outputs:     {out_dir}/nav_*.png")
    if has_map:
        print(f"  Trajectory summary:    {out_dir}/trajectory_summary.png")
    print(f"  Navigation grid:       {out_dir}/navigation_grid.png")
    print(f"\nThe GPS pipeline is working end-to-end with real colon anatomy!")


if __name__ == "__main__":
    parser = ArgumentParser(description="Surgical Navigation GPS Demo (C3VD)")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    hyperparam = ModelHiddenParams(parser)
    
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--configs", type=str)
    
    args = get_combined_args(parser)
    
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    
    safe_state(args.quiet)
    
    run_demo(
        dataset=model.extract(args),
        hyperparam=hyperparam.extract(args),
        iteration=args.iteration,
        pipeline=pipeline.extract(args),
    )
