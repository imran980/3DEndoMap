"""
Camera Localization with Organ Map Visualization.

Given a query endoscopic frame, find where the camera is on the registered
C3VD colon model. Outputs a 3-panel image:
  [Query Frame | Best Match Render | Organ Map with Camera Dot]

Also computes localization error against C3VD ground truth poses.

Usage:
    python localize_camera.py --model_path output/c3vd_trans \
        --configs arguments/endonerf.py

    # With external query image:
    python localize_camera.py --model_path output/c3vd_trans \
        --configs arguments/endonerf.py --query_image frame.png
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


def compute_ssim_simple(img1, img2):
    """Compute SSIM between two images (0-1 range numpy arrays)."""
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)
    mu1_sq, mu2_sq, mu1_mu2 = mu1**2, mu2**2, mu1 * mu2
    sigma1_sq = cv2.GaussianBlur(img1**2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2**2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2
    ssim_map = ((2*mu1_mu2 + C1) * (2*sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return float(ssim_map.mean())


def load_organ_points(recon_dir):
    """
    Load organ model and registered endo mesh as point clouds for map rendering.
    Returns organ_pts, endo_pts (Nx3 numpy arrays) or None if not found.
    """
    try:
        import open3d as o3d
    except ImportError:
        print("  Warning: open3d not available, organ map will be limited")
        return None, None
    
    # Try multiple locations for organ mesh
    organ_candidates = [
        os.path.join(recon_dir, "organ_model.ply"),
        os.path.join(recon_dir, "organ_model.obj"),
        os.path.join(os.path.dirname(recon_dir), "organ_model.obj"),
        os.path.join(recon_dir, "synthetic_organ.ply"),
    ]
    organ_pts = None
    for path in organ_candidates:
        if os.path.exists(path):
            mesh = o3d.io.read_triangle_mesh(path)
            if len(mesh.vertices) > 0:
                if len(mesh.triangles) > 0:
                    pcd = mesh.sample_points_uniformly(number_of_points=30000)
                else:
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = mesh.vertices
                organ_pts = np.asarray(pcd.points)
                print(f"  Loaded organ model: {path} ({len(organ_pts)} pts)")
                break
    
    # Load registered endo mesh
    endo_path = os.path.join(recon_dir, "registered_endo_mesh.ply")
    endo_pts = None
    if os.path.exists(endo_path):
        mesh = o3d.io.read_triangle_mesh(endo_path)
        if len(mesh.vertices) > 0:
            if len(mesh.triangles) > 0:
                pcd = mesh.sample_points_uniformly(number_of_points=15000)
            else:
                pcd = o3d.geometry.PointCloud()
                pcd.points = mesh.vertices
            endo_pts = np.asarray(pcd.points)
            print(f"  Loaded endo mesh: {endo_path} ({len(endo_pts)} pts)")
    
    return organ_pts, endo_pts


def render_organ_map(organ_pts, endo_pts, cam_position, cam_forward=None,
                     gt_position=None, map_size=400, trajectory_pts=None):
    """
    Render a 3D organ map showing camera position from external viewpoint.
    
    Creates two orthographic projections (top + side) of the organ with:
    - Organ surface as dim points
    - Endo reconstruction as brighter points
    - Camera position as a large green dot
    - GT position as a blue dot (if available)
    - Camera forward direction as an arrow
    - Full trajectory as a faint colored path
    
    Returns: numpy image (H, W, 3) BGR
    """
    panel_w = map_size
    panel_h = map_size
    img = np.zeros((panel_h, panel_w * 2, 3), dtype=np.uint8)
    
    # Combine all points for bounding box
    all_pts = []
    if organ_pts is not None:
        all_pts.append(organ_pts)
    if endo_pts is not None:
        all_pts.append(endo_pts)
    if len(all_pts) == 0:
        return img
    
    all_pts = np.vstack(all_pts)
    
    # Two views: Top (X-Z) and Side (X-Y)
    views = [
        ("TOP (X-Z)", 0, 2),
        ("SIDE (X-Y)", 0, 1),
    ]
    
    margin = 30
    
    for vi, (title, ax1, ax2) in enumerate(views):
        panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
        
        # Compute projection bounds
        pts_2d = all_pts[:, [ax1, ax2]]
        p_min = pts_2d.min(axis=0)
        p_max = pts_2d.max(axis=0)
        extent = p_max - p_min
        max_extent = max(extent[0], extent[1], 1e-6)
        scale = (min(panel_w, panel_h) - 2 * margin) / max_extent
        offset = np.array([margin, margin]) + \
                 (np.array([panel_w, panel_h]) - 2*margin - extent*scale) / 2
        
        def project(pts_3d):
            p = pts_3d[:, [ax1, ax2]] if pts_3d.ndim == 2 else pts_3d[[ax1, ax2]]
            return ((p - p_min) * scale + offset).astype(int)
        
        # Draw organ points (dim)
        if organ_pts is not None:
            proj = project(organ_pts)
            for p in proj:
                if 0 <= p[0] < panel_w and 0 <= p[1] < panel_h:
                    cv2.circle(panel, tuple(p), 1, (60, 60, 60), -1)
        
        # Draw endo mesh points (brighter)
        if endo_pts is not None:
            proj = project(endo_pts)
            for p in proj:
                if 0 <= p[0] < panel_w and 0 <= p[1] < panel_h:
                    cv2.circle(panel, tuple(p), 1, (140, 140, 140), -1)
        
        # Draw trajectory path (if available)
        if trajectory_pts is not None and len(trajectory_pts) > 1:
            proj_traj = project(trajectory_pts)
            for i in range(len(proj_traj) - 1):
                t = i / max(len(proj_traj) - 1, 1)
                color = (int(255*(1-t)), int(100*t), int(255*t))  # cyan → red
                cv2.line(panel, tuple(proj_traj[i]), tuple(proj_traj[i+1]), color, 1)
        
        # Draw GT position (blue dot)
        if gt_position is not None:
            gt_proj = project(gt_position.reshape(1, -1))[0]
            cv2.circle(panel, tuple(gt_proj), 8, (255, 100, 0), -1)  # blue
            cv2.circle(panel, tuple(gt_proj), 8, (255, 150, 50), 1)
        
        # Draw estimated camera position (green dot)
        cam_proj = project(cam_position.reshape(1, -1))[0]
        cv2.circle(panel, tuple(cam_proj), 10, (0, 255, 0), -1)
        cv2.circle(panel, tuple(cam_proj), 10, (0, 200, 0), 2)
        
        # Draw forward direction arrow
        if cam_forward is not None:
            fwd_2d = cam_forward[[ax1, ax2]]
            fwd_norm = fwd_2d / (np.linalg.norm(fwd_2d) + 1e-8)
            arrow_end = cam_proj + (fwd_norm * 25).astype(int)
            cv2.arrowedLine(panel, tuple(cam_proj), tuple(arrow_end),
                           (0, 255, 255), 2, tipLength=0.4)
        
        # Title
        cv2.putText(panel, title, (5, 18),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        img[:, vi*panel_w:(vi+1)*panel_w] = panel
    
    # Label
    cv2.putText(img, "ORGAN MAP", (panel_w - 50, panel_h - 8),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    return img


def load_gt_poses(recon_dir):
    """
    Load C3VD ground truth poses from camera_trajectory.json or 
    registered_trajectory.json for quantitative evaluation.
    """
    # Try registered trajectory first (in organ coordinates)
    for fname in ["registered_trajectory.json", "camera_trajectory.json"]:
        # Check in recon_dir and parent
        for base in [recon_dir, os.path.dirname(recon_dir), 
                     os.path.dirname(os.path.dirname(recon_dir))]:
            path = os.path.join(base, fname)
            if os.path.exists(path):
                with open(path) as f:
                    traj = json.load(f)
                print(f"  Loaded GT poses: {path}")
                return traj
    return None


def localize(dataset, hyperparam, iteration, pipeline, query_image_path=None,
             use_test_frames=False, save_results=True):
    """
    Localize camera position and render organ map visualization.
    """
    print("=" * 60)
    print("CAMERA LOCALIZATION + ORGAN MAP")
    print("=" * 60)
    
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, hyperparam)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    recon_dir = os.path.join(dataset.model_path, "surface_reconstruction",
                             f"iteration_{scene.loaded_iter}")
    out_dir = os.path.join(recon_dir, "localization")
    os.makedirs(out_dir, exist_ok=True)
    
    # Load registration transform
    transform_path = os.path.join(recon_dir, "registration_transform.json")
    if os.path.exists(transform_path):
        with open(transform_path) as f:
            reg = json.load(f)
        reg_transform = np.array(reg["final_transform"])
        print(f"Loaded registration transform "
              f"(ICP fitness: {reg.get('icp_fitness', 'N/A'):.4f}, "
              f"scale: {reg.get('scale_factor', 1.0):.4f})")
    else:
        reg_transform = np.eye(4)
        print("No registration transform found, using identity")
    
    # Load organ + endo point clouds for map rendering
    print("\nLoading meshes for organ map...")
    organ_pts, endo_pts = load_organ_points(recon_dir)
    has_organ_map = organ_pts is not None
    if not has_organ_map:
        print("  WARNING: No organ model found. Map panel will be empty.")
        print("  Run register_to_organ.py with --organ_mesh trans_model.obj first.")
    
    # Load GT trajectory for quantitative eval
    gt_traj = load_gt_poses(recon_dir)
    
    # ---- Render all training views ----
    train_views = scene.getTrainCameras()
    print(f"\nRendering {len(train_views)} training views...")
    
    train_renders = []
    train_poses = []
    with torch.no_grad():
        for view in tqdm(train_views, desc="Rendering train"):
            result = render(view, gaussians, pipeline, background, mode='test')
            img_np = result["render"].cpu().permute(1, 2, 0).numpy()
            # Build w2c from R, T directly (avoids trans/scale issues with world_view_transform)
            R = np.array(view.R)  # c2w rotation
            T = np.array(view.T)  # w2c translation
            w2c = np.eye(4, dtype=np.float32)
            w2c[:3, :3] = R.T  # w2c rotation
            w2c[:3, 3] = T
            c2w = np.linalg.inv(w2c)
            cam_center = c2w[:3, 3]
            
            train_renders.append(img_np)
            train_poses.append({
                'cam_center': cam_center,
                'c2w': c2w,
                'w2c': w2c,
                'time': getattr(view, 'time', 0),
            })
    
    # Build trajectory in organ coords for map background
    traj_organ_pts = []
    for tp in train_poses:
        pos_h = np.append(tp['cam_center'], 1.0)
        traj_organ_pts.append((reg_transform @ pos_h)[:3])
    traj_organ_pts = np.array(traj_organ_pts)
    
    # ---- Get query images ----
    if query_image_path and os.path.exists(query_image_path):
        query_img = cv2.imread(query_image_path)
        query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        queries = [("query", query_img, None)]
    elif use_test_frames:
        test_views = scene.getTestCameras()
        print(f"\nUsing {len(test_views)} test frames as queries")
        queries = []
        with torch.no_grad():
            for i, view in enumerate(test_views):
                gt_img = view.original_image.cpu().permute(1, 2, 0).numpy()
                gt_center = view.camera_center.cpu().numpy()
                queries.append((f"test_{i:03d}", gt_img, gt_center))
    else:
        print("\nNo query specified, using training frames as sanity check")
        queries = []
        step = max(1, len(train_views) // 5)
        for idx in range(0, len(train_views), step):
            view = train_views[idx]
            gt_img = view.original_image.cpu().permute(1, 2, 0).numpy()
            gt_center = view.camera_center.cpu().numpy()
            queries.append((f"train_{idx:03d}", gt_img, gt_center))
    
    # ---- Localize each query ----
    print(f"\nLocalizing {len(queries)} queries...")
    results = []
    
    for q_name, q_img, q_gt_center in queries:
        # Compare against all training renders
        similarities = []
        for t_img in train_renders:
            if q_img.shape != t_img.shape:
                q_resized = cv2.resize(q_img, (t_img.shape[1], t_img.shape[0]))
            else:
                q_resized = q_img
            ssim = compute_ssim_simple(q_resized, t_img)
            similarities.append(ssim)
        
        similarities = np.array(similarities)
        best_idx = np.argmax(similarities)
        best_ssim = similarities[best_idx]
        
        # Get pose of best match
        best_pose = train_poses[best_idx]
        cam_pos_endo = best_pose['cam_center']
        
        # Map to organ coordinates
        pos_homo = np.append(cam_pos_endo, 1.0)
        cam_pos_organ = (reg_transform @ pos_homo)[:3]
        c2w_organ = reg_transform @ best_pose['c2w']
        forward_organ = c2w_organ[:3, 2]
        
        # Compute position error if GT available
        pos_error = None
        gt_pos_organ = None
        if q_gt_center is not None:
            pos_error = float(np.linalg.norm(cam_pos_endo - q_gt_center))
            gt_pos_organ = (reg_transform @ np.append(q_gt_center, 1.0))[:3]
        
        result = {
            'query': q_name,
            'best_match_idx': int(best_idx),
            'best_ssim': float(best_ssim),
            'top3_indices': similarities.argsort()[-3:][::-1].tolist(),
            'top3_ssims': similarities[similarities.argsort()[-3:][::-1]].tolist(),
            'position_endo': cam_pos_endo.tolist(),
            'position_organ': cam_pos_organ.tolist(),
            'forward_organ': forward_organ.tolist(),
            'time': float(best_pose['time']),
        }
        if pos_error is not None:
            result['position_error_mm'] = pos_error
            result['gt_position_organ'] = gt_pos_organ.tolist()
        
        results.append(result)
        
        err_str = f", error={pos_error:.2f}mm" if pos_error is not None else ""
        print(f"  {q_name}: matched train[{best_idx}] "
              f"(SSIM={best_ssim:.4f}{err_str})")
        
        # ---- Build 3-panel comparison image ----
        if save_results:
            best_render = train_renders[best_idx]
            if q_img.shape != best_render.shape:
                q_display = cv2.resize(q_img, (best_render.shape[1], best_render.shape[0]))
            else:
                q_display = q_img
            
            h, w = q_display.shape[:2]
            
            # Panel 1+2: Query | Match (side by side)
            comparison = np.concatenate([q_display, best_render], axis=1)
            comp_bgr = cv2.cvtColor(
                (comparison * 255).clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            
            # Labels
            cv2.putText(comp_bgr, f"Query: {q_name}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(comp_bgr, f"Matched: train[{best_idx}] SSIM={best_ssim:.3f}",
                       (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            if pos_error is not None:
                cv2.putText(comp_bgr, f"Pos error: {pos_error:.2f}mm",
                           (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                           (255, 255, 255), 2)
            
            # Panel 3: Organ map
            if has_organ_map:
                map_img = render_organ_map(
                    organ_pts, endo_pts,
                    cam_position=cam_pos_organ,
                    cam_forward=forward_organ,
                    gt_position=gt_pos_organ,
                    map_size=h,
                    trajectory_pts=traj_organ_pts,
                )
                # Resize map to match height
                if map_img.shape[0] != h:
                    map_img = cv2.resize(map_img, (map_img.shape[1] * h // map_img.shape[0], h))
                
                comp_bgr = np.concatenate([comp_bgr, map_img], axis=1)
            
            cv2.imwrite(os.path.join(out_dir, f"{q_name}_match.png"), comp_bgr)
    
    # ---- Summary ----
    if save_results:
        results_path = os.path.join(out_dir, "localization_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        errors = [r['position_error_mm'] for r in results if 'position_error_mm' in r]
        ssims = [r['best_ssim'] for r in results]
        
        print(f"\n{'='*60}")
        print("LOCALIZATION RESULTS")
        print(f"{'='*60}")
        print(f"  Queries: {len(results)}")
        print(f"  SSIM:  mean={np.mean(ssims):.4f}, "
              f"min={np.min(ssims):.4f}, max={np.max(ssims):.4f}")
        if errors:
            print(f"  Error: mean={np.mean(errors):.2f}mm, "
                  f"min={np.min(errors):.2f}mm, max={np.max(errors):.2f}mm")
        print(f"\n  Results: {results_path}")
        print(f"  Images:  {out_dir}/")
    
    return results


if __name__ == "__main__":
    parser = ArgumentParser(description="Localize camera on organ model")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    hyperparam = ModelHiddenParams(parser)
    
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--query_image", default=None, type=str,
                        help="Path to query image")
    parser.add_argument("--use_test", action="store_true",
                        help="Use test frames as queries")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--configs", type=str)
    
    args = get_combined_args(parser)
    
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    
    safe_state(args.quiet)
    
    localize(
        dataset=model.extract(args),
        hyperparam=hyperparam.extract(args),
        iteration=args.iteration,
        pipeline=pipeline.extract(args),
        query_image_path=args.query_image,
        use_test_frames=args.use_test,
    )
