"""
Register the endoscopic reconstruction to the C3VD organ model.

Uses the Gaussian point cloud (from extract_surface.py) and registers it
to the real organ model (trans_model.obj) using scaled ICP.

Pipeline:
1. Load Gaussian point cloud (endo inner surface)
2. Load real organ model (trans_model.obj from C3VD)
3. Initial alignment: scale matching + centroid translation
4. Coarse-to-fine ICP
5. Transform and save camera trajectory in organ space

Usage:
    python register_to_organ.py \\
        --model_path output/endonerf/c3vd_trans \\
        --iteration 3000 \\
        --organ_mesh dataset/trans_model.obj
"""

import os
import sys
import json
import copy
import numpy as np
import open3d as o3d
from argparse import ArgumentParser


def compute_scale_factor(source_pts, target_pts):
    """
    Compute scale factor to match source extent to target extent.
    Uses median of per-axis ratios for robustness.
    """
    src_extent = source_pts.max(axis=0) - source_pts.min(axis=0)
    tgt_extent = target_pts.max(axis=0) - target_pts.min(axis=0)
    
    # Use median of per-axis ratios (robust to one axis being degenerate)
    ratios = tgt_extent / np.maximum(src_extent, 1e-6)
    scale = float(np.median(ratios))
    
    return scale, src_extent, tgt_extent


def register_to_organ(endo_pcd_path, organ_mesh_path, output_dir,
                      traj_path=None, voxel_size=2.0):
    """
    Register endoscopic point cloud to organ model.
    """
    print("=" * 60)
    print("REGISTER ENDO RECONSTRUCTION → ORGAN MODEL")
    print("=" * 60)
    
    # ---- Step 1: Load data ----
    print("\nStep 1: Loading data")
    print(f"  Endo point cloud: {endo_pcd_path}")
    print(f"  Organ model:      {organ_mesh_path}")
    
    # Load endo: try as point cloud first, then mesh
    endo_pcd = o3d.io.read_point_cloud(endo_pcd_path)
    if len(endo_pcd.points) == 0:
        endo_mesh = o3d.io.read_triangle_mesh(endo_pcd_path)
        endo_pcd = o3d.geometry.PointCloud()
        endo_pcd.points = endo_mesh.vertices
        if len(endo_mesh.vertex_colors) > 0:
            endo_pcd.colors = endo_mesh.vertex_colors
    
    # Load organ model
    organ_mesh = o3d.io.read_triangle_mesh(organ_mesh_path)
    organ_mesh.compute_vertex_normals()
    
    if len(endo_pcd.points) == 0:
        print("ERROR: Endo point cloud has no points!")
        return None
    if len(organ_mesh.vertices) == 0:
        print("ERROR: Organ mesh has no vertices!")
        return None
    
    # Sample points from organ mesh
    n_organ_pts = min(50000, len(organ_mesh.vertices) * 2)
    if len(organ_mesh.triangles) > 0:
        organ_pcd = organ_mesh.sample_points_uniformly(number_of_points=n_organ_pts)
    else:
        organ_pcd = o3d.geometry.PointCloud()
        organ_pcd.points = organ_mesh.vertices
    
    # Downsample endo for registration
    endo_pts = np.asarray(endo_pcd.points)
    organ_pts = np.asarray(organ_pcd.points)
    
    print(f"\n  Endo:  {len(endo_pts):,} points")
    print(f"    X: [{endo_pts[:, 0].min():.2f}, {endo_pts[:, 0].max():.2f}]")
    print(f"    Y: [{endo_pts[:, 1].min():.2f}, {endo_pts[:, 1].max():.2f}]")
    print(f"    Z: [{endo_pts[:, 2].min():.2f}, {endo_pts[:, 2].max():.2f}]")
    print(f"  Organ: {len(organ_pts):,} points")
    print(f"    X: [{organ_pts[:, 0].min():.2f}, {organ_pts[:, 0].max():.2f}]")
    print(f"    Y: [{organ_pts[:, 1].min():.2f}, {organ_pts[:, 1].max():.2f}]")
    print(f"    Z: [{organ_pts[:, 2].min():.2f}, {organ_pts[:, 2].max():.2f}]")
    
    # ---- Step 2: Initial alignment (scale + translation) ----
    print("\nStep 2: Initial alignment")
    
    scale_factor, src_ext, tgt_ext = compute_scale_factor(endo_pts, organ_pts)
    print(f"  Endo spatial extent:  {src_ext.round(2)}")
    print(f"  Organ spatial extent: {tgt_ext.round(2)}")
    print(f"  Scale factor: {scale_factor:.4f}")
    
    # Build initial transform: scale around endo centroid, then translate to organ centroid
    endo_center = endo_pts.mean(axis=0)
    organ_center = organ_pts.mean(axis=0)
    
    # T_init = Translate(organ_center) @ Scale(s) @ Translate(-endo_center)
    init_transform = np.eye(4)
    init_transform[:3, :3] *= scale_factor
    init_transform[:3, 3] = organ_center - scale_factor * endo_center
    
    print(f"  Endo center:  {endo_center.round(2)}")
    print(f"  Organ center: {organ_center.round(2)}")
    
    # Apply initial transform
    endo_pcd_aligned = copy.deepcopy(endo_pcd)
    endo_pcd_aligned.transform(init_transform)
    
    aligned_center = np.asarray(endo_pcd_aligned.points).mean(axis=0)
    print(f"  After alignment → endo center: {aligned_center.round(2)}")
    
    # ---- Step 3: ICP Registration ----
    print("\nStep 3: Coarse-to-fine ICP registration")
    
    # Adaptive voxel size based on organ scale
    organ_extent = organ_pts.max(0) - organ_pts.min(0)
    auto_voxel = float(np.median(organ_extent)) / 50
    voxel_size = max(voxel_size, auto_voxel)
    print(f"  Voxel size: {voxel_size:.2f}")
    
    endo_down = endo_pcd_aligned.voxel_down_sample(voxel_size)
    organ_down = organ_pcd.voxel_down_sample(voxel_size)
    
    endo_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    organ_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    
    print(f"  Endo downsampled:  {len(endo_down.points)} points")
    print(f"  Organ downsampled: {len(organ_down.points)} points")
    
    # Coarse ICP (point-to-point)
    print("\n  Running coarse ICP (point-to-point)...")
    reg_coarse = o3d.pipelines.registration.registration_icp(
        endo_down, organ_down,
        max_correspondence_distance=voxel_size * 10,
        init=np.eye(4),
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200)
    )
    print(f"    Fitness: {reg_coarse.fitness:.4f}, RMSE: {reg_coarse.inlier_rmse:.4f}")
    
    # Fine ICP (point-to-plane)
    print("  Running fine ICP (point-to-plane)...")
    reg_fine = o3d.pipelines.registration.registration_icp(
        endo_down, organ_down,
        max_correspondence_distance=voxel_size * 3,
        init=reg_coarse.transformation,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200)
    )
    print(f"    Fitness: {reg_fine.fitness:.4f}, RMSE: {reg_fine.inlier_rmse:.4f}")
    
    # Combined: ICP @ initial alignment
    final_transform = reg_fine.transformation @ init_transform
    
    print(f"\n  Final 4x4 transform (includes scale {scale_factor:.4f}):")
    for row in final_transform:
        print(f"    [{row[0]:10.4f} {row[1]:10.4f} {row[2]:10.4f} {row[3]:10.4f}]")
    
    # ---- Step 4: Save outputs ----
    print(f"\nStep 4: Saving results to {output_dir}")
    
    # Transform and save endo point cloud
    registered_endo = copy.deepcopy(endo_pcd)
    registered_endo.transform(final_transform)
    
    reg_pcd_path = os.path.join(output_dir, "registered_endo_pcd.ply")
    o3d.io.write_point_cloud(reg_pcd_path, registered_endo)
    print(f"  Registered endo: {reg_pcd_path}")
    
    # Save organ mesh as PLY
    organ_ply_path = os.path.join(output_dir, "organ_model.ply")
    o3d.io.write_triangle_mesh(organ_ply_path, organ_mesh)
    print(f"  Organ model: {organ_ply_path}")
    
    # Save transform
    transform_data = {
        "source": "C3VD",
        "endo_input": os.path.basename(endo_pcd_path),
        "organ_mesh": os.path.basename(organ_mesh_path),
        "scale_factor": scale_factor,
        "initial_transform": init_transform.tolist(),
        "icp_coarse_transform": reg_coarse.transformation.tolist(),
        "icp_fine_transform": reg_fine.transformation.tolist(),
        "final_transform": final_transform.tolist(),
        "icp_fitness": reg_fine.fitness,
        "icp_rmse": reg_fine.inlier_rmse,
    }
    transform_path = os.path.join(output_dir, "registration_transform.json")
    with open(transform_path, 'w') as f:
        json.dump(transform_data, f, indent=2)
    print(f"  Transform: {transform_path}")
    
    # ---- Step 5: Transform camera trajectory ----
    if traj_path and os.path.exists(traj_path):
        print(f"\nStep 5: Transforming camera trajectory")
        with open(traj_path) as f:
            traj = json.load(f)
        
        registered_traj = copy.deepcopy(traj)
        reg_positions = []
        
        for frame in registered_traj["frames"]:
            pos = np.array(frame["position"] + [1.0])  # homogeneous
            new_pos = (final_transform @ pos)[:3]
            frame["position_organ"] = new_pos.tolist()
            reg_positions.append(new_pos)
            
            # Transform forward direction (rotation only, no translation)
            fwd = np.array(frame["forward"] + [0.0])  # direction, w=0
            new_fwd = (final_transform @ fwd)[:3]
            norm = np.linalg.norm(new_fwd)
            if norm > 0:
                new_fwd = new_fwd / norm
            frame["forward_organ"] = new_fwd.tolist()
        
        reg_positions = np.array(reg_positions)
        print(f"  Transformed {len(reg_positions)} camera positions to organ space")
        print(f"  Organ-space range:")
        print(f"    X: [{reg_positions[:, 0].min():.2f}, {reg_positions[:, 0].max():.2f}]")
        print(f"    Y: [{reg_positions[:, 1].min():.2f}, {reg_positions[:, 1].max():.2f}]")
        print(f"    Z: [{reg_positions[:, 2].min():.2f}, {reg_positions[:, 2].max():.2f}]")
        
        reg_traj_path = os.path.join(output_dir, "registered_trajectory.json")
        with open(reg_traj_path, 'w') as f:
            json.dump(registered_traj, f, indent=2)
        print(f"  Registered trajectory: {reg_traj_path}")
    
    # ---- Save combined scene ----
    # Organ as pink point cloud
    organ_vis = organ_pcd.voxel_down_sample(voxel_size)
    organ_vis.paint_uniform_color([0.85, 0.55, 0.50])
    
    # Registered endo as green point cloud
    reg_endo_vis = registered_endo.voxel_down_sample(voxel_size)
    reg_endo_vis.paint_uniform_color([0.3, 0.8, 0.3])
    
    combined = organ_vis + reg_endo_vis
    scene_path = os.path.join(output_dir, "registration_preview.ply")
    o3d.io.write_point_cloud(scene_path, combined)
    print(f"  Preview scene: {scene_path}")
    
    print(f"\n{'=' * 60}")
    print("REGISTRATION COMPLETE")
    print(f"{'=' * 60}")
    print(f"  ICP Fitness: {reg_fine.fitness:.4f}")
    print(f"  ICP RMSE:    {reg_fine.inlier_rmse:.4f}")
    print(f"  Scale:       {scale_factor:.4f}")
    
    return final_transform


if __name__ == "__main__":
    parser = ArgumentParser(description="Register endoscopic reconstruction to organ model")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--iteration", default=3000, type=int)
    parser.add_argument("--organ_mesh", default=None,
                        help="Path to organ model (e.g. dataset/trans_model.obj)")
    parser.add_argument("--voxel_size", default=2.0, type=float)
    args = parser.parse_args()
    
    recon_dir = os.path.join(args.model_path, "surface_reconstruction",
                             f"iteration_{args.iteration}")
    
    # Find endo point cloud (prefer pointcloud, fallback to mesh)
    endo_path = os.path.join(recon_dir, "fused_pointcloud.ply")
    if not os.path.exists(endo_path):
        endo_path = os.path.join(recon_dir, "fused_mesh.ply")
    if not os.path.exists(endo_path):
        print(f"ERROR: No endo reconstruction found in {recon_dir}")
        print("Run extract_surface.py first.")
        sys.exit(1)
    
    # Find organ mesh
    if args.organ_mesh:
        organ_path = args.organ_mesh
    else:
        candidates = [
            "dataset/trans_model.obj",
            os.path.join(recon_dir, "organ_model.ply"),
            os.path.join(recon_dir, "organ_model.obj"),
        ]
        organ_path = None
        for c in candidates:
            if os.path.exists(c):
                organ_path = c
                break
    
    if organ_path is None or not os.path.exists(organ_path):
        print(f"ERROR: Organ mesh not found. Provide --organ_mesh path.")
        sys.exit(1)
    
    # Find camera trajectory
    traj_path = os.path.join(recon_dir, "camera_trajectory.json")
    if not os.path.exists(traj_path):
        traj_path = None
    
    register_to_organ(endo_path, organ_path, recon_dir,
                      traj_path=traj_path, voxel_size=args.voxel_size)
